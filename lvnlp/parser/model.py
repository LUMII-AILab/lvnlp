import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from ufal.chu_liu_edmonds import chu_liu_edmonds

from lvnlp.parser.config import Arguments
from lvnlp.parser.ud_dataset import Vocabs


class MLP(nn.Module):
    def __init__(self, n_in, n_out, dropout):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.activation(self.linear(x)))


class Biaffine(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True, scale=0.0):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.scale = scale

        self.weight = nn.Parameter(
            torch.zeros(n_out, n_in + int(bias_x), n_in + int(bias_y))
        )

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat([x, torch.ones_like(x[..., :1])], dim=-1)
        if self.bias_y:
            y = torch.cat([y, torch.ones_like(y[..., :1])], dim=-1)

        # x: [B, Tx, Dx], y: [B, Ty, Dy]
        # out: [B, O, Tx, Ty]
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        s = s / (self.n_in ** self.scale)

        if self.n_out == 1:
            s = s.squeeze(1)  # [B, Tx, Ty]
        return s


class EdgeClassifier(nn.Module):
    def __init__(
        self,
        hidden_size,
        dep_vocab_size,
        dropout,
        arc_mlp_size=512,
        rel_mlp_size=128,
    ):
        super().__init__()

        self.arc_mlp_d = MLP(hidden_size, arc_mlp_size, dropout)
        self.arc_mlp_h = MLP(hidden_size, arc_mlp_size, dropout)
        self.rel_mlp_d = MLP(hidden_size, rel_mlp_size, dropout)
        self.rel_mlp_h = MLP(hidden_size, rel_mlp_size, dropout)

        self.arc_attn = Biaffine(
            n_in=arc_mlp_size,
            n_out=1,
            bias_x=True,
            bias_y=False,
            scale=0.5,
        )
        self.rel_attn = Biaffine(
            n_in=rel_mlp_size,
            n_out=dep_vocab_size,
            bias_x=True,
            bias_y=True,
            scale=0.5,
        )

        self.mask_value = float('-inf')
        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                std = math.sqrt(2.0 / (5.0 * module.in_features))
                nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.zeros_(self.arc_attn.weight)
        nn.init.zeros_(self.rel_attn.weight)

    def forward(self, x, lengths, head_gold=None):
        lengths = lengths.to(x.device)

        # x includes root at position 0
        arc_d = self.arc_mlp_d(x[:, 1:, :])  # [B, T, A]
        arc_h = self.arc_mlp_h(x)  # [B, T+1, A]

        head_prediction = self.arc_attn(arc_d, arc_h)  # [B, T, T+1]

        mask = (torch.arange(x.size(1), device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)).unsqueeze(1)
        diag = (
            torch.ones(x.size(1) - 1, x.size(1), dtype=torch.bool, device=x.device)
            .tril(1)
            .triu(1)
        )
        head_prediction = head_prediction.masked_fill(mask | diag, self.mask_value)

        if head_gold is None:
            head_logp = F.pad(head_prediction, (0, 0, 1, 0), value=torch.nan).cpu()
            head_gold = []
            for i, length in enumerate(lengths.tolist()):
                head = self.max_spanning_tree(head_logp[i, :length, :length])
                head = head + ((x.size(1) - 1) - len(head)) * [0]
                head_gold.append(torch.tensor(head))
            head_gold = torch.stack(head_gold).to(x.device)

        rel_d = self.rel_mlp_d(x[:, 1:, :])  # [B, T, R]
        rel_h_all = self.rel_mlp_h(x)  # [B, T+1, R]
        rel_h = rel_h_all.gather(
            1, head_gold.unsqueeze(-1).expand(-1, -1, rel_h_all.size(-1)).clamp(min=0)
        )  # [B, T, R]

        dep_prediction = self.rel_attn(rel_d, rel_h)  # [B, L, T, T]
        dep_prediction = dep_prediction.diagonal(dim1=2, dim2=3).transpose(1, 2)  # [B, T, L]

        return head_prediction, dep_prediction, head_gold

    def max_spanning_tree(self, weight_matrix):
        weight_matrix = weight_matrix.clone()
        weight_matrix[weight_matrix == self.mask_value] = torch.nan

        parents, _ = chu_liu_edmonds(weight_matrix.numpy().astype(float))
        assert parents[0] == -1, f'{parents}\n{weight_matrix}'
        parents = parents[1:]

        if parents.count(0) == 1:
            return parents

            best_score = float('-inf')
        best_parents = None

        for i in range(len(parents)):
            weight_matrix_mod = weight_matrix.clone()
            weight_matrix_mod[:i + 1, 0] = torch.nan
            weight_matrix_mod[i + 2:, 0] = torch.nan
            parents, score = chu_liu_edmonds(weight_matrix_mod.numpy().astype(float))
            parents = parents[1:]

            if score > best_score:
                best_score = score
                best_parents = parents

        assert best_parents is not None
        return best_parents


class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return x * F.gelu(gate)


class Classifier(nn.Module):
    def __init__(self, hidden_size, vocab_size, dropout, ffn_size=2560):
        super().__init__()

        self.transform = nn.Sequential(
            nn.Linear(hidden_size, 2 * ffn_size),
            GEGLU(),
            nn.LayerNorm(ffn_size, elementwise_affine=False),
            nn.Linear(ffn_size, hidden_size, bias=False),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, vocab_size),
        )
        self.initialize(hidden_size)

    def initialize(self, hidden_size):
        std = math.sqrt(2.0 / (5.0 * hidden_size))
        nn.init.trunc_normal_(self.transform[0].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.transform[-1].weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
        self.transform[0].bias.data.zero_()
        self.transform[-1].bias.data.zero_()

    def forward(self, x):
        return self.transform(x)


class Model(nn.Module):
    def __init__(self, args: Arguments, vocabs: Vocabs):
        super().__init__()
        self.args = args

        self.bert = AutoModel.from_pretrained(args.model, trust_remote_code=True)
        self.n_layers = self.bert.config.num_hidden_layers
        args.hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        self.layer_norm = nn.LayerNorm(args.hidden_size, elementwise_affine=False)
        self.upos_layer_score = nn.Parameter(torch.zeros(self.n_layers + 1, dtype=torch.float))
        self.xpos_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.feats_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.lemma_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.head_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))
        self.dep_layer_score = nn.Parameter(torch.zeros(self.n_layers + 2, dtype=torch.float))

        self.upos_embedding = nn.Embedding(len(vocabs.upos), args.hidden_size)
        std = math.sqrt(2.0 / (5.0 * args.hidden_size))
        nn.init.trunc_normal_(self.upos_embedding.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)

        self.lemma_classifier = nn.ModuleDict({
            k: Classifier(args.hidden_size, len(vocabs.lemma[k]), args.dropout, ffn_size=args.ffn_size)
            for k in vocabs.lemma
        })
        self.upos_classifier = Classifier(args.hidden_size, len(vocabs.upos), args.dropout, ffn_size=args.ffn_size)
        if args.mono_xpos:
            self.xpos_classifiers = Classifier(args.hidden_size, len(vocabs.xpos), args.dropout, ffn_size=args.ffn_size)
        else:
            self.xpos_classifiers = nn.ModuleDict({
                k: Classifier(args.hidden_size, len(vocabs.factored_xpos[k]), args.dropout, ffn_size=args.ffn_size)
                for k in vocabs.factored_xpos
            })
        self.edge_classifier = EdgeClassifier(hidden_size=args.hidden_size, dep_vocab_size=len(vocabs.deprel), dropout=args.dropout, arc_mlp_size=args.arc_mlp_size, rel_mlp_size=args.rel_mlp_size)
        if args.mono_feats:
            self.feats_classifiers = Classifier(args.hidden_size, len(vocabs.feats), args.dropout, ffn_size=args.ffn_size)
        else:
            self.feats_classifiers = nn.ModuleDict({
                k: Classifier(args.hidden_size, len(vocabs.factored_feats[k]), args.dropout, ffn_size=args.ffn_size)
                for k in vocabs.factored_feats
            })

    def forward(self, x, alignment_mask, subword_lengths, word_lengths, upos_gold=None, head_gold=None):
        padding_mask = (torch.arange(x.size(1)).unsqueeze(0) < subword_lengths.unsqueeze(1)).to(x.device)
        x = self.bert(x, padding_mask, output_hidden_states=True).hidden_states
        x = torch.stack(x, dim=0)
        # avg pooling
        x = torch.einsum('lbsd,bst->lbtd', x, alignment_mask) / alignment_mask.sum(1).unsqueeze(-1).unsqueeze(0).clamp(min=1.0)

        # upos_x = torch.einsum('lbtd, l -> btd', x, torch.softmax(self.upos_layer_score, dim=0))
        upos_x = (x[:, :, 1:-1, :] * torch.softmax(self.upos_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
        upos_x = self.dropout(self.layer_norm(upos_x))
        upos_preds = self.upos_classifier(upos_x)

        if upos_gold is None:
            upos_gold = upos_preds.argmax(-1)

        upos_embedding = self.upos_embedding(upos_gold.clamp(min=0))
        upos_embedding = F.pad(upos_embedding, (0, 0, 1, 1), value=0.0)
        x = torch.cat([x, upos_embedding.unsqueeze(0)], dim=0)

        lemma_preds = None
        xpos_preds = None
        feats_prediction = None
        head_prediction = None
        dep_prediction = None
        head_liu = None

        if 'lemma' in self.args.tasks_list:
            lemma_x = (x[:, :, 1:-1, :] * torch.softmax(self.lemma_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
            lemma_x = self.dropout(self.layer_norm(lemma_x))
            lemma_preds = {cls: classifier(lemma_x) for cls, classifier in self.lemma_classifier.items()}

        if 'xpos' in self.args.tasks_list:
            xpos_x = (x[:, :, 1:-1, :] * torch.softmax(self.xpos_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
            xpos_x = self.dropout(self.layer_norm(xpos_x))
            if self.args.mono_xpos:
                xpos_preds = self.xpos_classifiers(xpos_x)
            else:
                xpos_preds = {cls: classifier(xpos_x) for cls, classifier in self.xpos_classifiers.items()}

        if 'feats' in self.args.tasks_list:
            feats_x = (x[:, :, 1:-1, :] * torch.softmax(self.feats_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
            feats_x = self.dropout(self.layer_norm(feats_x))
            if self.args.mono_feats:
                feats_prediction = self.feats_classifiers(feats_x)
            else:
                feats_prediction = {cls: classifier(feats_x) for cls, classifier in self.feats_classifiers.items()}

        if 'parser' in self.args.tasks_list:
            head_x = (x[:, :, 0:-1, :] * torch.softmax(self.head_layer_score, dim=0).view(-1, 1, 1, 1)).sum(0)
            head_x = self.dropout(self.layer_norm(head_x))
            # head_prediction, dep_prediction, head_liu = self.edge_classifier(head_x, dep_x, word_lengths, head_gold)
            head_prediction, dep_prediction, head_liu = self.edge_classifier(head_x, word_lengths, head_gold)

        return lemma_preds, upos_preds, xpos_preds, feats_prediction, head_prediction, dep_prediction, head_liu


class CrossEntropySmoothingMasked:
    def __init__(self, smoothing=0.0):
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def __call__(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=1)
        nll_loss = -logprobs.gather(dim=1, index=target.unsqueeze(1).clamp(min=0)).squeeze(1)

        logprobs = logprobs.masked_fill(x == float('-inf'), 0.0)
        smooth_loss = -logprobs.sum(dim=1) / (x != float('-inf')).float().sum(dim=1)

        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        loss = loss.masked_fill(target == -1, 0.0).sum() / (target != -1).float().sum()
        return loss
