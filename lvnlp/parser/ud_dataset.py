from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset

from lvnlp.parser.utils import gen_lemma_rule, read_conll, Columns, iter_field, filter_by_len
from lvnlp.parser.utils import get_normalized_lemma_rules, LEMMA_RULE_TYPES
from lvnlp.parser.xpos import build_xpos_char_vocab, encode_factored_xpos

NONE_FEAT = '<none>'
IGNORE_INDEX = -1
MAX_SUBWORD_TOKENS = 512
MAX_SENTENCE_TOKENS = MAX_SUBWORD_TOKENS - 2


class Vocab:
    def __init__(self, labels, add_none=False):
        self.labels: list[str] = list(labels)
        if add_none and NONE_FEAT not in self.labels:
            self.labels = [NONE_FEAT] + self.labels
        self.stoi = {x: i for i, x in enumerate(self.labels)}
        self.none_idx = self.stoi[NONE_FEAT] if NONE_FEAT in self.labels else None

    def __len__(self):
        return len(self.labels)

    def encode(self, label: str, default=None) -> int:
        # default for unk values, for example -1 (should match ignore_index)
        if label in self.stoi:
            return self.stoi[label]
        if default is not None:
            return default
        if self.none_idx is not None:
            return self.none_idx
        raise KeyError(label)

    def decode(self, i: int) -> str:
        return self.labels[i]

    def to_state_dict(self):
        return self.labels

    @classmethod
    def from_state_dict(cls, x):
        return cls(list(x))

    @classmethod
    def from_data(cls, items, min_count=1, add_none=False):
        counts = Counter(items)
        labels = [x for x, c in counts.items() if c >= min_count]
        return cls(labels, add_none=add_none)

    def __repr__(self):
        return f'Vocab<len={len(self)}, values=({self.labels[:10]}>'


@dataclass
class Vocabs:
    text: Vocab
    upos: Vocab
    xpos: Vocab
    deprel: Vocab
    feats: Vocab
    factored_feats: dict[str, Vocab]
    factored_xpos: dict[str, Vocab]
    lemma: dict[str, Vocab]

    def to_state_dict(self):
        return {
            'text': self.text.to_state_dict(),
            'upos': self.upos.to_state_dict(),
            'xpos': self.xpos.to_state_dict(),
            'feats': self.feats.to_state_dict(),
            'deprel': self.deprel.to_state_dict(),
            'factored_feats': {k: v.to_state_dict() for k, v in self.factored_feats.items()},
            'factored_xpos': {k: v.to_state_dict() for k, v in self.factored_xpos.items()},
            'lemma': {k: v.to_state_dict() for k, v in self.lemma.items()},
        }

    @classmethod
    def from_state_dict(cls, d):
        return cls(
            text=Vocab.from_state_dict(d['text']),
            upos=Vocab.from_state_dict(d['upos']),
            xpos=Vocab.from_state_dict(d['xpos']),
            feats=Vocab.from_state_dict(d['feats']),
            deprel=Vocab.from_state_dict(d['deprel']),
            factored_feats={k: Vocab.from_state_dict(v) for k, v in d['factored_feats'].items()},
            factored_xpos={k: Vocab.from_state_dict(v) for k, v in d['factored_xpos'].items()},
            lemma={k: Vocab.from_state_dict(v) for k, v in d['lemma'].items()},
        )

    @classmethod
    def from_data(cls, sentences, min_count=1, lemma_rules=None):
        if lemma_rules is None:
            lemma_rules = get_normalized_lemma_rules([[(token[Columns.TEXT], token[Columns.LEMMA]) for token in sent] for sent in sentences])

        return Vocabs(
            text=Vocab.from_data(iter_field(sentences, Columns.TEXT), min_count=min_count),
            upos=Vocab.from_data(iter_field(sentences, Columns.UPOS), min_count=min_count),
            xpos=Vocab.from_data(iter_field(sentences, Columns.XPOS), min_count=min_count),
            feats=Vocab.from_data(iter_field(sentences, Columns.FEATS), min_count=min_count),
            deprel=Vocab.from_data(iter_field(sentences, Columns.DEPREL), min_count=min_count),

            factored_feats=build_factored_feats_vocabs(iter_field(sentences, Columns.FEATS)),
            factored_xpos=build_factored_xpos_vocab(),
            lemma=build_lemma_rule_vocab(lemma_rules, min_count=min_count),
        )

    def __str__(self):
        r = []
        for field, value in self.__dict__.items():
            if isinstance(value, dict):
                r.append(f'{field}: count={len(value)}')
                r += [f'{field}.{k}: {v}' for k, v in value.items()]
            else:
                r.append(f'{field}: {value}')
        return '\n'.join(r)


def build_factored_feats_vocabs(items):
    key_values = defaultdict(set)
    for item in items:
        if item == '_':
            continue
        for feat in item.split('|'):
            key, value = feat.split('=')
            key_values[key].add(value)
    # key_values = {k: v for k, v in key_values.items() if len(v) > 2}
    key_values = dict(sorted(list(key_values.items()), key=lambda x: x[0]))
    return {key: Vocab(sorted(values), add_none=True) for key, values in key_values.items()}


def _none_encoded(vocabs: dict[str, Vocab]) -> dict[str, int]:
    return {key: vocab.none_idx for key, vocab in vocabs.items()}


def encode_factored_feats(vocabs: dict[str, Vocab], feats_string: str) -> dict[str, int]:
    if not feats_string or feats_string == '_':
        return _none_encoded(vocabs)
    feats = {}
    for item in feats_string.split('|'):
        key, value = item.split('=', 1)
        feats[key] = value
    return {k: (v.encode(feats[k]) if k in feats else v.none_idx) for k, v in vocabs.items()}


def decode_factored_feats(vocabs: dict[str, Vocab], feats: dict[str, int]) -> str:
    parts = [f'{k}={vocabs[k].decode(v)}' for k, v in feats.items() if v != vocabs[k].none_idx]
    assert parts == sorted(parts)
    return '|'.join(parts) if parts else '_'


def build_factored_xpos_vocab() -> dict[str, Vocab]:
    return {key: Vocab(values) for key, values in build_xpos_char_vocab().items()}


def build_lemma_rule_vocab(lemma_rules, min_count=1):
    if not lemma_rules:
        return {key: Vocab([], add_none=True) for key in LEMMA_RULE_TYPES}
    if isinstance(lemma_rules[0], list):  # flatten
        lemma_rules = [rule for sublist in lemma_rules for rule in sublist]
    r = {k: Vocab.from_data([x[k] for x in lemma_rules if x[k] is not None], add_none=True, min_count=min_count) for k in LEMMA_RULE_TYPES}
    return r


def tokenize_words(tokenizer, words, max_length=MAX_SUBWORD_TOKENS):
    enc = tokenizer(
        words,
        is_split_into_words=True,
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
    )
    input_ids = list(enc['input_ids'])
    word_ids = enc.word_ids()
    # alignment: 0 for specials, otherwise word_index+1
    alignment = [(wi + 1) if wi is not None else 0 for wi in word_ids]
    return input_ids, word_ids, alignment


def _pad_1d(tensor: torch.Tensor, length: int, value: int) -> torch.Tensor:
    return F.pad(tensor, (0, length - tensor.size(0)), value=value)


def _pad_alignment(alignment: torch.Tensor, source_length: int, target_length: int) -> torch.Tensor:
    one_hot = F.one_hot(alignment, num_classes=target_length + 2).float()
    return F.pad(one_hot, (0, 0, 0, source_length - alignment.size(0)), value=0.0)


def _stack_padded(sentences, key: str, length: int, value: int) -> torch.Tensor:
    return torch.stack([_pad_1d(sentence[key], length, value) for sentence in sentences])


def _stack_padded_nested(sentences, key: str, subkey: str, length: int, value: int) -> torch.Tensor:
    return torch.stack([_pad_1d(sentence[key][subkey], length, value) for sentence in sentences])


class CollateFunctor:
    def __init__(self, pad_index):
        self.pad_index = pad_index

    def __call__(self, sentences):
        longest_source = max(sentence['subwords'].size(0) for sentence in sentences)
        longest_target = max(sentence['upos'].size(0) for sentence in sentences)

        return {
            'index': [sentence['index'] for sentence in sentences],
            'subwords': _stack_padded(sentences, 'subwords', longest_source, self.pad_index),
            'alignment': torch.stack([_pad_alignment(sentence['alignment'], longest_source, longest_target) for sentence in sentences]),
            'lemma': {
                key: _stack_padded_nested(sentences, 'lemma', key, longest_target, IGNORE_INDEX)
                for key in sentences[0]['lemma'].keys()
            },
            'upos': _stack_padded(sentences, 'upos', longest_target, IGNORE_INDEX),
            'xpos': _stack_padded(sentences, 'xpos', longest_target, IGNORE_INDEX),
            'feats': _stack_padded(sentences, 'feats', longest_target, IGNORE_INDEX),
            'arc_head': _stack_padded(sentences, 'arc_head', longest_target, IGNORE_INDEX),
            'arc_dep': _stack_padded(sentences, 'arc_dep', longest_target, IGNORE_INDEX),
            'subword_lengths': torch.LongTensor([sentence['subwords'].size(0) for sentence in sentences]),
            'word_lengths': torch.LongTensor([sentence['upos'].size(0) + 1 for sentence in sentences]),
            'factored_feats': {
                key: _stack_padded_nested(sentences, 'factored_feats', key, longest_target, IGNORE_INDEX)
                for key in sentences[0]['factored_feats'].keys()
            },
            'factored_xpos': {
                key: _stack_padded_nested(sentences, 'factored_xpos', key, longest_target, IGNORE_INDEX)
                for key in sentences[0]['factored_xpos'].keys()
            },
        }


def _load_sentences(path_or_sentences, max_tokens: int = MAX_SENTENCE_TOKENS):
    if isinstance(path_or_sentences, list):
        return path_or_sentences

    print(f'Load {path_or_sentences}')
    return filter_by_len(read_conll(path_or_sentences), max_tokens)


def _field(sentences, column: str) -> list[list]:
    return [[token[column] for token in sentence] for sentence in sentences]


def _initial_lemma_rules(sentences) -> list[list[dict]]:
    return [
        [gen_lemma_rule(token[Columns.TEXT], token[Columns.LEMMA], True) for token in sentence]
        for sentence in sentences
    ]


def _normalized_lemma_rules(sentences) -> list[list[dict]]:
    form_lemmas = [
        [(token[Columns.TEXT], token[Columns.LEMMA]) for token in sentence]
        for sentence in sentences
    ]
    return get_normalized_lemma_rules(form_lemmas)


def _pad_token_id(tokenizer) -> int:
    for token_id in (tokenizer.pad_token_id, tokenizer.eos_token_id, tokenizer.sep_token_id):
        if token_id is not None:
            return token_id
    return 0


def _encode_sequence(vocab: Vocab, values, default=None) -> torch.Tensor:
    return torch.LongTensor([vocab.encode(value, default=default) for value in values])


def _encode_factored_sequence(vocabs: dict[str, Vocab], values, encoder) -> dict[str, torch.Tensor]:
    encoded = [encoder(vocabs, value) for value in values]
    return {
        key: torch.LongTensor([item[key] for item in encoded])
        for key in vocabs
    }


class UDDataset(TorchDataset):
    def __init__(self, path: list | str, tokenizer, vocabs=None, random_mask=None, min_count=1):
        self.sentences = _load_sentences(path)
        self.tokenizer = tokenizer
        self.random_mask = random_mask
        self.pad_index = _pad_token_id(self.tokenizer)

        self.forms = _field(self.sentences, Columns.TEXT)
        self.lemma_rules = _initial_lemma_rules(self.sentences)
        self.upos = _field(self.sentences, Columns.UPOS)
        self.xpos = _field(self.sentences, Columns.XPOS)
        self.feats = _field(self.sentences, Columns.FEATS)
        self.arc_head = _field(self.sentences, Columns.HEAD)
        self.arc_dep = _field(self.sentences, Columns.DEPREL)

        self.vocabs: Vocabs = vocabs
        if not vocabs:
            self.lemma_rules = _normalized_lemma_rules(self.sentences)
            self.vocabs: Vocabs = Vocabs.from_data(self.sentences, min_count=min_count, lemma_rules=self.lemma_rules)

        if self.random_mask is not None and self.random_mask > 0 and (self.tokenizer.mask_token_id is not None):
            print('RANDOM MASK ENABLED', self.random_mask, self.tokenizer.mask_token_id)

    def state_dict(self):
        return {'vocabs': self.vocabs.to_state_dict()}

    def load_state_dict(self, state_dict):
        self.vocabs = Vocabs.from_state_dict(state_dict['vocabs'])

    def _apply_random_mask(self, subwords: list[int], word_ids) -> None:
        if self.random_mask is None or self.random_mask <= 0 or self.tokenizer.mask_token_id is None:
            return

        for token_index, word_index in enumerate(word_ids):
            if word_index is not None and torch.rand([]).item() < self.random_mask:
                subwords[token_index] = int(self.tokenizer.mask_token_id)

    def get_item(self, index):
        words = self.forms[index]
        subwords, word_ids, alignment = tokenize_words(self.tokenizer, words)
        self._apply_random_mask(subwords, word_ids)

        factored_feats = _encode_factored_sequence(self.vocabs.factored_feats, self.feats[index], encode_factored_feats)
        lemma_rules = {
            rule_type: torch.LongTensor([
                self.vocabs.lemma[rule_type].encode(lemma_rule[rule_type], default=IGNORE_INDEX)
                for lemma_rule in self.lemma_rules[index]
            ])
            for rule_type in LEMMA_RULE_TYPES
        }
        factored_xpos = _encode_factored_sequence(self.vocabs.factored_xpos, self.xpos[index], encode_factored_xpos)
        return {
            'index': index,
            'subwords': torch.LongTensor(subwords),
            'alignment': torch.LongTensor(alignment),
            'lemma': lemma_rules,
            'upos': _encode_sequence(self.vocabs.upos, self.upos[index]),
            'xpos': _encode_sequence(self.vocabs.xpos, self.xpos[index], default=IGNORE_INDEX),
            'feats': _encode_sequence(self.vocabs.feats, self.feats[index], default=IGNORE_INDEX),
            'arc_head': torch.LongTensor(self.arc_head[index]),
            'arc_dep': _encode_sequence(self.vocabs.deprel, self.arc_dep[index], default=IGNORE_INDEX),
            'factored_feats': factored_feats,
            'factored_xpos': factored_xpos,
        }

    def __getitem__(self, index):
        return self.get_item(index)

    def __len__(self):
        return len(self.sentences)


def load_data(treebank_path, tokenizer, min_count=3, random_mask=0.15):
    treebank_path = Path(treebank_path)
    split_paths = _find_conllu_splits(treebank_path)
    train_path = split_paths.get('train')
    dev_path = split_paths.get('dev') or split_paths.get('test')
    test_path = split_paths.get('test') or split_paths.get('dev')

    if train_path is None:
        raise ValueError(f'Train file not found in {treebank_path}')
    if dev_path is None or test_path is None:
        raise ValueError(f'Dev/test file not found in {treebank_path}')

    train_data = UDDataset(str(train_path), tokenizer=tokenizer, random_mask=random_mask, min_count=min_count)
    dev_data = UDDataset(str(dev_path), tokenizer=tokenizer, vocabs=train_data.vocabs, random_mask=0)
    test_data = UDDataset(str(test_path), tokenizer=tokenizer, vocabs=train_data.vocabs, random_mask=0)

    return train_data, dev_data, test_data


def _find_conllu_splits(treebank_path: Path) -> dict[str, Path]:
    split_paths = {}
    for path in sorted(treebank_path.iterdir()):
        if path.suffix != '.conllu':
            continue
        for split in ('train', 'dev', 'test'):
            if split in path.name:
                split_paths[split] = path
                break
    return split_paths
