import copy
import datetime
import json
import logging
import math
import os
import subprocess
from dataclasses import asdict

import torch
import torch.nn as nn
import tqdm
import wandb
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lvnlp.parser.config import Arguments
from lvnlp.parser.ud_dataset import load_data, CollateFunctor
from lvnlp.parser.inference import Parser
from lvnlp.parser.model import CrossEntropySmoothingMasked
from lvnlp.parser.model import Model
from lvnlp.parser.utils import seed_everything, get_model_name, get_optimizer_param_groups

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logger = logging.getLogger(__name__)


def train(args: Arguments):
    ts = datetime.datetime.now()

    model_name = get_model_name(args.model)
    out_name = (args.prefix + '___' if args.prefix else '') + model_name.replace('/', '___')

    model_dir = os.path.join(args.out_dir, out_name)
    os.makedirs(model_dir, exist_ok=True)
    logger.info(f'{model_dir=}  {args=}')
    seed_everything(args.seed)

    if args.log_wandb:
        wandb.init(name=out_name, config=asdict(args), project=args.wandb_project, tags=[])

    logger.info(f'{model_dir=} {args=}')
    print('Arguments:\n' + ('\n'.join(f'- {k}={v}' for k, v in asdict(args).items())))

    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device >= 0 else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    use_amp = args.precision in {'fp16', 'bf16'} and device.type == 'cuda'
    amp_dtype = {'fp16': torch.float16, 'bf16': torch.bfloat16}.get(args.precision, None)
    # GradScaler is needed for fp16, but not typically for bf16
    scaler = torch.amp.GradScaler('cuda', enabled=(use_amp and args.precision == 'fp16'))

    logger.info(f'Loading model {args.model}')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    train_data, dev_data, test_data = load_data(args.treebank_path, tokenizer, min_count=args.min_count, random_mask=args.random_mask)
    print('Vocabs\n', train_data.vocabs)

    # build and pad with loaders
    train_loader = DataLoader(train_data, args.batch_size, shuffle=True, drop_last=True, num_workers=args.workers, collate_fn=CollateFunctor(train_data.pad_index))
    dev_loader = DataLoader(dev_data, args.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=CollateFunctor(train_data.pad_index))
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=CollateFunctor(train_data.pad_index))

    model = Model(args, train_data.vocabs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if args.log_wandb:
        wandb.config.update({'params': n_params})
    logger.info(f'Param count: {n_params}')

    config = {'args': asdict(args), 'vocabs': train_data.vocabs.to_state_dict(), 'stats': {'params': n_params}}
    with open(os.path.join(model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    archive_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    subprocess.run(f'tar -czf {model_dir}/udlv_{archive_ts}.tar.gz udlv', shell=True, check=True)

    if args.ema_decay > 0:
        ema_model = copy.deepcopy(model)
        for param in ema_model.parameters():
            param.requires_grad = False
    else:
        ema_model = model

    criterion = nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=args.label_smoothing).to(device)
    masked_criterion = CrossEntropySmoothingMasked(args.label_smoothing)

    optimizer_grouped_parameters = get_optimizer_param_groups(model, lr=args.lr, weight_decay=args.weight_decay, lr_ratio=args.lr_ratio)
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.99))

    def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scheduler = cosine_schedule_with_warmup(optimizer, 250, args.epochs * len(train_loader), 0.1 / 3)

    best_metric = -1
    best_dev = None
    best_test = None
    # train loop
    for epoch in range(args.epochs):
        train_iter = tqdm.tqdm(train_loader)
        model.train()
        for batch in train_iter:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):

                lemma_p, upos_p, xpos_p, feats_p, head_p, dep_p, _ = model(
                    batch['subwords'].to(device),
                    batch['alignment'].to(device),
                    batch['subword_lengths'],
                    batch['word_lengths'],
                    batch['upos'].to(device),
                    batch['arc_head'].to(device),
                )

                zero = upos_p.new_zeros(())
                lemma_loss = zero
                if lemma_p is not None:
                    lemma_loss = {
                        key: criterion(p.transpose(1, 2), batch['lemma'][key].to(device))
                        for key, p in lemma_p.items()
                    }
                    # lemma_loss = [
                    #     l * (batch['lemma'][key] != -1).float().sum().item() / (batch['feats'] != -1).float().sum().item()
                    #     for key, l in lemma_loss.items()
                    # ]
                    # lemma_loss = sum(lemma_loss) / math.sqrt(len(lemma_loss))
                    lemma_loss = (sum(lemma_loss.values()) / len(lemma_loss)) if len(lemma_loss) > 0 else zero
                upos_loss = criterion(upos_p.transpose(1, 2), batch['upos'].to(device))
                # xpos_loss = criterion(xpos_p.transpose(1, 2), batch['xpos'].to(device))
                xpos_loss = zero
                if xpos_p is not None:
                    if isinstance(xpos_p, dict):
                        xpos_loss = {
                            key: criterion(p.transpose(1, 2), batch['factored_xpos'][key].to(device))
                            for key, p in xpos_p.items()
                        }
                        xpos_loss = (sum(xpos_loss.values()) / len(xpos_loss)) if len(xpos_loss) > 0 else zero
                    else:
                        xpos_loss = criterion(xpos_p.transpose(1, 2), batch['xpos'].to(device))
                feats_loss = zero
                if feats_p is not None:
                    if isinstance(feats_p, dict):
                        feats_loss = {
                            key: criterion(p.transpose(1, 2), batch['factored_feats'][key].to(device))
                            for key, p in feats_p.items()
                        }
                        feats_loss = (sum(feats_loss.values()) / len(feats_loss)) if len(feats_loss) > 0 else zero
                    else:
                        feats_loss = criterion(feats_p.transpose(1, 2), batch['feats'].to(device))
                head_loss = masked_criterion(head_p.transpose(1, 2), batch['arc_head'].to(device)) if head_p is not None else zero
                dep_loss = criterion(dep_p.transpose(1, 2), batch['arc_dep'].to(device)) if dep_p is not None else zero

                loss = args.loss_weights_dict['lemma'] * lemma_loss + upos_loss + xpos_loss + feats_loss + head_loss + dep_loss

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            scheduler.step()

            if args.ema_decay > 0:
                with torch.no_grad():
                    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
                        param_k.data.mul_(args.ema_decay).add_((1.0 - args.ema_decay) * param_q.detach().data)
            else:
                ema_model = model

            if args.log_wandb:
                wandb.log(
                    {
                        'epoch': epoch,
                        'train/lemma_loss': lemma_loss.item(),
                        'train/upos_loss': upos_loss.item(),
                        'train/xpos_loss': xpos_loss.item(),
                        'train/feats_loss': feats_loss.item(),
                        'train/head_loss': head_loss.item(),
                        'train/dep_loss': dep_loss.item(),
                        'train/loss': loss.item(),
                        'stats/grad_norm': grad_norm.item(),
                        'stats/learning_rate': optimizer.param_groups[0]['lr'],
                    }
                )
            train_iter.set_postfix_str(f'loss: {loss.item()}')
        torch.save(model.state_dict(), os.path.join(model_dir, 'checkpoint.bin'))

        os.makedirs(os.path.join(model_dir, 'preds'), exist_ok=True)
        # dev_result = evaluate_model(ema_model, dev_loader, dev_data, device, os.path.join(model_dir, 'preds', f'dev_epoch{epoch:03d}.conllu'), add={'epoch': epoch})
        dev_result = (Parser(model=ema_model, args=args, tokenizer=tokenizer, vocabs=train_data.vocabs)
                      .parse(loader=dev_loader, dataset=dev_data, out_conll=os.path.join(model_dir, 'preds', f'dev_epoch{epoch:03d}.conllu'), do_eval=True, add_eval={'epoch': epoch}))[1]
        if args.best_metric == 'mlas_blex':
            current_metric = dev_result['MLAS'] + dev_result['BLEX']
        elif args.best_metric == 'las_xpos_lemma':
            current_metric = dev_result['LAS'] + 0.5 * dev_result['XPOS'] + 0.5 * dev_result['Lemmas']
        else:
            raise Exception(f'Unknown best metric: {args.best_metric}')
        better = current_metric > best_metric
        logger.info(f'Epoch {epoch}, dev result: {dev_result}')
        if better:
            with open(os.path.join(model_dir, 'best_dev_result.json'), 'w') as f:
                json.dump(dev_result, f, indent=2)
            best_dev = dev_result
        with open(os.path.join(model_dir, 'dev_results.jsonl'), 'a') as f:
            f.write(json.dumps(dev_result) + '\n')
        if args.log_wandb:
            wandb.log({'epoch': epoch, **{f'dev/{k}': v for k, v in dev_result.items()}})

        # test_result = evaluate_model(ema_model, test_loader, test_data, device, os.path.join(model_dir, 'preds', f'test_epoch{epoch:03d}.conllu'), add={'epoch': epoch})
        test_result = (Parser(model=ema_model, args=args, tokenizer=tokenizer, vocabs=train_data.vocabs)
                       .parse(loader=test_loader, dataset=test_data, out_conll=os.path.join(model_dir, 'preds', f'test_epoch{epoch:03d}.conllu'), do_eval=True, add_eval={'epoch': epoch}))[1]
        logger.info(f'Epoch {epoch}, test result: {test_result}')
        if better:
            with open(os.path.join(model_dir, 'best_test_result.json'), 'w') as f:
                json.dump(test_result, f, indent=2)
            best_test = test_result
        with open(os.path.join(model_dir, 'test_results.jsonl'), 'a') as f:
            f.write(json.dumps(test_result) + '\n')
        if args.log_wandb:
            wandb.log({'epoch': epoch, **{f'test/{k}': v for k, v in test_result.items()}})

        if better:
            best_metric = current_metric
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.bin'))
            if args.ema_decay is not None and args.ema_decay > 0:
                torch.save(model.state_dict(), os.path.join(model_dir, 'ema_model.bin'))

    summary = {
        'model': model_name,
        'out_name': out_name,
        'seed': args.seed,
        'group': args.group,
        'prefix': args.prefix,
        'ts': ts.isoformat(),
        'runtime': (datetime.datetime.now() - ts).total_seconds(),
        'runtime_format': str(datetime.datetime.now() - ts),
        'lr': args.lr,
        'best_dev': best_dev,
        'best_test': best_test,
    }
    with open(os.path.join(model_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    os.makedirs('runs', exist_ok=True)

    dt = ts.isoformat().replace(':', '-').replace('.', '-')
    with open(f'runs/{out_name}___summary__{dt}.json', 'w', encoding='utf8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f'Final summary: {json.dumps(summary, indent=2)}')

    del model
    del ema_model
    parser: Parser = Parser.from_pretrained(model_dir, device=args.device)
    parser.parse(loader=dev_loader, dataset=dev_data, do_eval=True, include_probs=True, out_json=os.path.join(model_dir, 'dev.probs.json'), out_conll=os.path.join(model_dir, 'dev.conllu'))
    parser.parse(loader=test_loader, dataset=test_data, do_eval=True, include_probs=True, out_json=os.path.join(model_dir, 'test.probs.json'), out_conll=os.path.join(model_dir, 'test.conllu'))
    return summary


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', level=logging.INFO)

    train(Arguments.load())
