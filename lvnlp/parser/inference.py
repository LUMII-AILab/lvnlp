import copy
import json
import logging
import os
import tempfile
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file as load_safetensors
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lvnlp.parser.analyzer import DEFAULT_ANALYZER_URL, add_analyzer, drop_analyzer_fields, rescore_with_analyzer
from lvnlp.parser.config import Arguments
from lvnlp.parser.conll18_ud_eval import load_conllu_file, evaluate
from lvnlp.parser.model import Model
from lvnlp.parser.ud_dataset import CollateFunctor, Vocabs, UDDataset, decode_factored_feats
from lvnlp.parser.utils import apply_lemma_rule
from lvnlp.parser.utils import read_conll, write_conll
from lvnlp.parser.xpos import decode_factored_xpos_logits, decode_factored_xpos, eval_xpos_err_detailed

logger = logging.getLogger(__name__)
DEFAULT_PARSER_MODEL = 'AiLab-IMCS-UL/lv_ud2.17_lv-deberta-base'


def decode(data, output, vocabs: Vocabs, in_place=True, include_probs=False):
    lemma_p, upos_p, xpos_p, feats_p, head_p, dep_p, head_liu = output

    lemma_idx = {k: p.argmax(-1) for k, p in lemma_p.items()} if lemma_p is not None else None
    upos_idx = upos_p.argmax(-1)
    xpos_idx = None
    if xpos_p is not None:
        xpos_idx = decode_factored_xpos_logits(xpos_p, vocabs.factored_xpos) if isinstance(xpos_p, dict) else xpos_p.argmax(-1)
    feats_idx = None
    if feats_p is not None:
        feats_idx = {k: p.argmax(-1) for k, p in feats_p.items()} if isinstance(feats_p, dict) else feats_p.argmax(-1)
    dep_idx = dep_p.argmax(-1) if dep_p is not None else None
    sentences = []
    for i in range(len(upos_idx)):
        tokens = []
        for j in range(len(data[i])):
            token = {
                'upos': vocabs.upos.decode(upos_idx[i, j].item()),
            }
            probs = {}
            if lemma_idx is not None:
                lemma_rule = {rt: vocabs.lemma[rt].decode(lemma_idx[rt][i, j].item()) for rt in lemma_idx}
                token['lemma'] = apply_lemma_rule(data[i][j]['text'], lemma_rule)
            if xpos_idx is not None:
                token['xpos'] = (
                    decode_factored_xpos({k: xpos_idx[k][i, j].item() for k in xpos_idx}, vocabs.factored_xpos)
                    if isinstance(xpos_idx, dict)
                    else vocabs.xpos.decode(xpos_idx[i, j].item())
                )
                if include_probs and isinstance(xpos_p, dict):
                    probs['xpos'] = {k: dict(zip(vocabs.factored_xpos[k].labels, xpos_p[k][i, j].tolist())) for k in xpos_p}
            if feats_idx is not None:
                token['feats'] = (
                    decode_factored_feats(vocabs.factored_feats, {k: feats_idx[k][i, j].item() for k in feats_idx})
                    if isinstance(feats_idx, dict)
                    else vocabs.feats.decode(feats_idx[i, j].item())
                )
                if include_probs and isinstance(feats_p, dict):
                    probs['feats'] = {k: dict(zip(vocabs.factored_feats[k].labels, feats_p[k][i, j].tolist())) for k in feats_p}
            if head_liu is not None:
                token['head'] = head_liu[i, j].item()
            if dep_idx is not None:
                token['deprel'] = vocabs.deprel.decode(dep_idx[i, j].item())

            if include_probs:
                token['probs'] = probs
            if in_place:
                data[i][j].update(**token)
            tokens.append(token)
        sentences.append(tokens)
    return sentences


class Parser:
    def __init__(self, *, model, args, tokenizer, vocabs):
        self.model = model
        self.args = args
        self.tokenizer = tokenizer
        self.vocabs = vocabs
        self.device = next(model.parameters()).device

    @classmethod
    def from_pretrained(cls, model_dir=None, device=None):
        if model_dir is None:
            model_dir = DEFAULT_PARSER_MODEL
            logger.info(f'Using default model: {model_dir}')
        model_dir = cls._resolve_model_dir(model_dir)

        if (model_dir / 'model.safetensors').exists():
            state = load_safetensors(model_dir / 'model.safetensors', device='cpu')
        elif (model_dir / 'ema_model.bin').exists():
            state = torch.load(model_dir / 'ema_model.bin', map_location='cpu', weights_only=True)
        elif (model_dir / 'model.bin').exists():
            state = torch.load(model_dir / 'model.bin', map_location='cpu', weights_only=True)
        else:
            raise ValueError(f'No model weights found in {model_dir}. Expected one of: model.safetensors, ema_model.bin, model.bin')
        if 'model' in state:
            state = state['model']
        config = json.load(open(model_dir / 'config.json'))

        args = Arguments(**config['args'])
        vocabs = Vocabs.from_state_dict(config['vocabs'])
        model = Model(args=args, vocabs=vocabs)
        model.load_state_dict(state)
        device = cls._resolve_device(device)
        model = model.to(device)
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        return cls(model=model, args=args, vocabs=vocabs, tokenizer=tokenizer)

    @staticmethod
    def _resolve_model_dir(model_dir):
        model_path = Path(model_dir)
        if model_path.is_dir():
            return model_path

        if model_path.exists():
            raise ValueError(f'Model path {model_path} exists but is not a directory')

        try:
            return Path(snapshot_download(str(model_dir)))
        except Exception as exc:
            raise ValueError(
                f'Model directory {model_path} does not exist and could not be downloaded '
                f'from Hugging Face Hub as {model_dir!r}'
            ) from exc

    @staticmethod
    def _resolve_device(device):
        if device is None:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device(f'cuda:{device}' if torch.cuda.is_available() and device >= 0 else 'cpu')
        if isinstance(device, str):
            return torch.device(device)
        raise TypeError(f'Unsupported device value: {device!r}')

    @staticmethod
    def _token(text, index):
        return {
            'id': index,
            'text': text,
            'lemma': text,
            'upos': '_',
            'xpos': '_',
            'feats': '_',
            'head': 0,
            'deprel': '_',
            'deps': '_',
            'misc': '_',
        }

    def _prepare_inference_input(self, sentences):
        upos_default = self.vocabs.upos.labels[0]
        deprel_default = self.vocabs.deprel.labels[0]
        prepared = []
        for sentence in sentences:
            tokens = []
            for index, token in enumerate(sentence, start=1):
                item = dict(token)
                item.setdefault('id', index)
                item.setdefault('text', '')
                item.setdefault('lemma', item['text'])
                item.setdefault('upos', upos_default)
                item.setdefault('xpos', '_')
                item.setdefault('feats', '_')
                item.setdefault('head', 0)
                item.setdefault('deprel', deprel_default)
                item.setdefault('deps', '_')
                item.setdefault('misc', '_')
                if item['upos'] == '_' and '_' not in self.vocabs.upos.stoi:
                    item['upos'] = upos_default
                if item['deprel'] == '_' and '_' not in self.vocabs.deprel.stoi:
                    item['deprel'] = deprel_default
                tokens.append(item)
            prepared.append(tokens)
        return prepared

    def parse(
            self,
            sentences=None,
            batch_size=1,
            out_conll=None,
            in_place=True,
            loader=None,
            dataset=None,
            do_eval=False,
            add_eval=None,
            include_probs=False,
            out_json=None,
            analyzer=False,
            analyzer_url=DEFAULT_ANALYZER_URL,
            analyzer_margin=5.0,
            analyzer_timeout=60.0,
    ):
        if loader:
            assert dataset is not None
            sentences = dataset.sentences
        elif dataset:
            sentences = dataset.sentences
            loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=CollateFunctor(dataset.pad_index))
        elif sentences:
            if isinstance(sentences, str):
                sentences = read_conll(sentences)
            dataset = UDDataset(sentences, self.tokenizer, vocabs=self.vocabs, random_mask=0)
            loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=CollateFunctor(dataset.pad_index))
        else:
            raise ValueError('Either sentences, dataset or loader must be provided')

        sentences = copy.deepcopy(sentences)
        decode_probs = include_probs or analyzer
        with torch.inference_mode():
            self.model.eval()
            for batch in loader:
                output = self.model(
                    batch['subwords'].to(self.device),
                    batch['alignment'].to(self.device),
                    batch['subword_lengths'],
                    batch['word_lengths'],
                )
                batch_data = [sentences[index] for index in batch['index']]
                decode(batch_data, output, self.vocabs, in_place=in_place, include_probs=decode_probs)
        if analyzer:
            add_analyzer(sentences, url=analyzer_url, timeout=analyzer_timeout)
            rescore_with_analyzer(sentences, margin=analyzer_margin)
            drop_analyzer_fields(sentences, drop_probs=not include_probs)
        if out_conll:
            write_conll(out_conll, sentences)
        if out_json:
            with open(out_json, 'w') as f:
                json.dump(sentences, f, indent=2, ensure_ascii=False)

        if do_eval:
            eval_res = evals(sentences, sentences, add=add_eval)
            return sentences, eval_res
        return sentences


def evals(gold_data, out_data, out_json=None, add=None):
    if isinstance(gold_data, str):
        gold_data = read_conll(gold_data)
    if isinstance(out_data, str):
        out_data = read_conll(out_data)
    with tempfile.TemporaryDirectory() as tmpdir:
        gold_path = os.path.join(tmpdir, 'gold.conllu')
        write_conll(gold_path, gold_data)
        out_path = os.path.join(tmpdir, 'out.conllu')
        write_conll(out_path, out_data)
        gold_ud = load_conllu_file(gold_path)
        out_ud = load_conllu_file(out_path)
        evaluation = evaluate(gold_ud, out_ud)
        r = {
            **(add or {}),
            'UPOS': round(evaluation['UPOS'].aligned_accuracy * 100, 3),
            'XPOS': round(evaluation['XPOS'].aligned_accuracy * 100, 3),
            'UFeats': round(evaluation['UFeats'].aligned_accuracy * 100, 3),
            'AllTags': round(evaluation['AllTags'].aligned_accuracy * 100, 3),
            'Lemmas': round(evaluation['Lemmas'].aligned_accuracy * 100, 3),
            'UAS': round(evaluation['UAS'].aligned_accuracy * 100, 3),
            'LAS': round(evaluation['LAS'].aligned_accuracy * 100, 3),
            'CLAS': round(evaluation['CLAS'].aligned_accuracy * 100, 3),
            'MLAS': round(evaluation['MLAS'].aligned_accuracy * 100, 3),
            'BLEX': round(evaluation['BLEX'].aligned_accuracy * 100, 3),

            **eval_xpos_err_detailed(out_data, gold_data)
        }
        if out_json:
            with open(out_json, 'w') as f:
                json.dump(r, f, indent=2)
        return r
