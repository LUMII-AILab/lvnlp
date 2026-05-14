from collections import defaultdict
from typing import Any

import torch

MISSING_FEAT = '<none>'

POS_TO_ATTRS = {
    'noun': {
        'codifier': 'n',
        'noun_type': 'cp',  # c: common noun, p: proper noun
        'gender': 'mf0',  # m: masculine, f: feminine, 0: not applicable
        'number': 'spvd0',  # s: singular, p: plural, v: singulare tantum, d: plurale tantum, 0: not applicable
        'case': 'ngdalv0',  # n: nominative, g: genitive, d: dative, a: accusative, l: locative, v: vocative, 0: not applicable
        'declension': '1234560gr',  # Declensions: 1-6, 0: not applicable, g: genitive-only, r: reflexive
    },
    'verb': {  # 11
        'codifier': 'v',
        'verb_type': 'mopecta',  # m: main, o: modal, p: phasal, e: expression, c: "būt" auxiliary/copula, t: copulas, a: auxiliaries
        'reflexive': 'ny0',  # n: non-reflexive, y: reflexive, 0: not applicable
        'mood': 'ircdmn',  # i: indicative, r: relative, c: conditional, d: debitive, m: imperative, n: infinitive, p: participle
        'tense': 'pfs0',  # p: present, f: future, s: past, 0: not applicable
        'transitivity': 'ti0',  # t: transitive, i: intransitive, 0: not applicable
        'conjugation': '123i0',  # 1-3: conjugations, i: irregular, 0: not applicable
        'person': '1230',  # 1: 1st person, 2: 2nd person, 3: 3rd person, 0: not applicable
        'number': 'sp0',  # s: singular, p: plural, 0: not applicable
        'voice': 'ap0',  # a: active, p: passive, 0: not applicable
        'negation': 'ny',  # n: non-negated, y: negated
    },
    'participle': {  # 13
        'codifier': 'v',
        'verb_type': 'mopecta',
        'reflexive': 'ny0',
        'mood': 'p',
        'declinability': 'dpu',  # d: declinable, p: partially declinable, u: undeclinable
        'gender': 'mf0',
        'number': 'sp0',
        'case': 'ngdalv0',
        'voice': 'ap0',
        'tense': 'ps0',
        'definiteness': 'ny0',  # n: indefinite, y: definite, 0: not applicable
        'degree': 'pcs0',  # p: positive, c: comparative, s: superlative, 0: not applicable
        'negation': 'ny',
    },
    'adjective': {
        'codifier': 'a',
        'adjective_type': 'fr',  # f: qualificative, r: relative
        'gender': 'mf0',
        'number': 'sp0',
        'case': 'ngdalv0',
        'definiteness': 'ny',
        'degree': 'pcs',
    },
    'numeral': {
        'codifier': 'm',
        'numeral_type': 'cof',  # c: cardinal, o: ordinal, f: fraction
        'composition': 'scj',  # s: simple, c: compound, j: multiword
        'gender': 'mf0',
        'number': 'sp',
        'case': 'ngdalv0',
    },
    'pronoun': {
        'codifier': 'p',
        'pronoun_type': 'pxsdiqrg',  # p: personal, x: reflexive, s: possessive, d: demonstrative, i: indefinite, q: interrogative, r: relative
        'person': '1230',
        'gender': 'mf0',
        'number': 'sp0',
        'case': 'ngdal',
        'negation': 'ny',
    },
    'adverb': {
        'codifier': 'r',
        'degree': 'pcs0',  # p: positive, c: comparative, s: superlative, 0: not applicable
        'prepositional_adverb': 'yn',  # y: yes, n: no
    },
    'adposition': {
        'codifier': 's',
        'position': 'pt',  # p: preposition, t: postposition
        'governed_number': 'sp0',
        'governed_case': 'gda0',  # g: genitive, d: dative, a: accusative, 0: not applicable
    },
    'conjunction': {
        'codifier': 'c',
        'syntactic_function': 'cs',  # s: subordinating, c: coordinating
    },
    'interjection': {
        'codifier': 'i',
    },
    'particle': {
        'codifier': 'q',
    },
    'punctuation': {
        'codifier': 'z',
        'punctuation_type': 'cqsbdox',
    },
    'abbreviation': {
        'codifier': 'y',
        'abbreviation_type': 'npavrd',
    },
    'residual': {
        'codifier': 'x',
        'residual_type': 'fonux',
    },
}


POS_TO_CODE = {pos: 'V' if pos == 'participle' else x['codifier'] for pos, x in POS_TO_ATTRS.items()}
CODE_TO_POS = {v: k for k, v in POS_TO_CODE.items()}

CODE_TO_ATTRS = {code: POS_TO_ATTRS[pos] for pos, code in POS_TO_CODE.items()}


DEFAULT_FILTERED_ATTRIBUTES = {
    'noun': ['noun_type'],
    'verb': ['transitivity'],
    'adjective': ['adjective_type'],
    'abbreviation': ['abbreviation_type'],
}
DEFAULTS = DEFAULT_FILTERED_ATTRIBUTES


def filter_tag(tag, remove=None):
    remove = DEFAULT_FILTERED_ATTRIBUTES if remove is None else remove
    codifier = get_xpos_code(tag)
    remove_attrs = list(remove.get(CODE_TO_POS.get(codifier), []))
    if not remove_attrs:
        return tag
    attrs = POS_TO_ATTRS[CODE_TO_POS[codifier]]
    if len(attrs) != len(tag):
        raise ValueError(f'Invalid tag: {tag!r}, {attrs}')
    return ''.join('_' if attr in remove_attrs else value for attr, value in zip(attrs, tag))


def get_xpos_code(tag: str) -> str | None:
    if not tag or tag in ('_', '<unk>'):
        return None
    codifier = tag[0]
    return 'V' if codifier == 'v' and len(tag) > 3 and tag[3] == 'p' else codifier


def get_xpos_schema(codifier, mood=None):
    if codifier in (None, '<unk>', MISSING_FEAT):
        return None
    return CODE_TO_ATTRS.get('V' if codifier == 'v' and mood == 'p' else codifier)


def get_xpos_schema_for_tag(tag: str):
    return CODE_TO_ATTRS.get(get_xpos_code(tag))


def fix_tag(tag: str) -> str:
    schema = get_xpos_schema_for_tag(tag)
    if schema is None:
        return tag
    if len(tag) < len(schema):
        return tag.ljust(len(schema), '_')
    return tag


def get_xpos_feats(tag):
    return get_xpos_schema_for_tag(tag)


def xpos_to_dict(tag, allow_empty=False, fix=False):
    if tag == '_':
        return {}
    if fix:
        tag = fix_tag(tag)
    try:
        pos = get_xpos_schema_for_tag(tag)
        if not pos or not tag or len(pos) != len(tag):
            if allow_empty:
                return {}
            raise ValueError(f'Invalid tag: {tag!r}, {pos}')
        return {attr: value for attr, value in zip(pos, tag)}
    except Exception:
        raise


def dict_to_xpos(d):
    if not d:
        return '_'
    codifier = d['codifier']
    if codifier == 'v' and d.get('mood') == 'p':
        codifier = 'V'
    if codifier is None:
        return '_'
    pos = CODE_TO_ATTRS[codifier]
    tag = []
    for attr in pos:
        tag.append(d.get(attr, '_'))
    return ''.join(tag)


def build_xpos_char_vocab():
    values = {}
    for attrs in POS_TO_ATTRS.values():
        for attr, allowed in attrs.items():
            values.setdefault(attr, set()).update(allowed)
    values = {
        attr: [MISSING_FEAT, *sorted(allowed)]
        for attr, allowed in values.items()
    }
    # `V` is an internal selector only; model predicts surface codifiers.
    values['codifier'] = [*sorted(v for v in values['codifier'] if v != 'V')]
    return dict(sorted(values.items(), key=lambda item: item[0]))


def _none_encoded(vocabs) -> dict[str, int]:
    return {key: vocab.none_idx for key, vocab in vocabs.items()}


def encode_factored_xpos(vocabs, xpos: str) -> dict[str, int]:
    assert isinstance(xpos, str)
    if xpos in ('_', '<unk>'):
        return _none_encoded(vocabs)
    decoded = xpos_to_dict(xpos)
    return {
        key: (vocab.encode(decoded[key]) if key in decoded else vocab.none_idx)
        for key, vocab in vocabs.items()
    }


def decode_xpos_value(predicted_idx, vocab, allowed_values):
    if predicted_idx is None or predicted_idx < 0 or predicted_idx >= len(vocab):
        return None
    value = vocab.decode(predicted_idx)
    return value if value in allowed_values else None


def decode_factored_xpos(xpos_classes, vocabs):
    codifier = decode_xpos_value(xpos_classes['codifier'], vocabs['codifier'], set(vocabs['codifier'].labels))
    if codifier in (None, '<unk>', MISSING_FEAT):
        return '_'
    mood = decode_xpos_value(xpos_classes['mood'], vocabs['mood'], set(vocabs['mood'].labels))
    schema = get_xpos_schema(codifier, mood)
    if schema is None:
        return '_'

    parts = []
    for attr, allowed_values in schema.items():
        if attr == 'codifier':
            parts.append(codifier)
            continue
        value = decode_xpos_value(xpos_classes.get(attr, -1), vocabs[attr], set(allowed_values))
        if value is None:
            value = allowed_values[0]
        parts.append(value)
    return ''.join(parts)


def get_xpos_allowed_masks(vocabs, device=None):
    res = {}
    codifier_vocab = vocabs['codifier']
    codifier_none_idx = codifier_vocab.none_idx
    for attr, vocab in vocabs.items():
        if attr == 'codifier':
            continue
        mask = torch.zeros(len(codifier_vocab), 2, len(vocab), dtype=torch.bool, device=device)
        if codifier_none_idx is not None and vocab.none_idx is not None:
            mask[codifier_none_idx, :, vocab.none_idx] = True
        for code, schema in CODE_TO_ATTRS.items():
            codifier = schema['codifier']
            codifier_idx = codifier_vocab.stoi[codifier]
            is_participle = 1 if code == 'V' else 0
            allowed_values = schema.get(attr)
            if allowed_values is None:
                if vocab.none_idx is not None:
                    mask[codifier_idx, is_participle, vocab.none_idx] = True
                continue
            allowed_indices = [vocab.stoi[value] for value in allowed_values if value in vocab.stoi]
            if allowed_indices:
                mask[codifier_idx, is_participle, allowed_indices] = True
            elif vocab.none_idx is not None:
                mask[codifier_idx, is_participle, vocab.none_idx] = True
        res[attr] = mask
    return res


def decode_factored_xpos_logits(xpos_logits, vocabs):
    preds = {'codifier': xpos_logits['codifier'].argmax(dim=-1)}
    mood_logits = xpos_logits.get('mood')
    if mood_logits is not None:
        preds['mood'] = mood_logits.argmax(dim=-1)

    codifier_preds = preds['codifier']
    mood_preds = preds.get('mood')
    is_participle = torch.zeros_like(codifier_preds)
    if mood_preds is not None:
        is_participle = (
            (codifier_preds == vocabs['codifier'].stoi['v']) &
            (mood_preds == vocabs['mood'].stoi['p'])
        ).long()

    allowed_masks = get_xpos_allowed_masks(vocabs, device=codifier_preds.device)
    for attr, logits in xpos_logits.items():
        if attr == 'codifier':
            continue
        allowed = allowed_masks[attr][codifier_preds, is_participle]
        preds[attr] = logits.masked_fill(~allowed, float('-inf')).argmax(dim=-1)
    return preds


def get_xpos_attribute_values() -> dict[str, str]:
    values = defaultdict(list)
    for attrs in POS_TO_ATTRS.values():
        for attr, attr_values in attrs.items():
            values[attr] += [value for value in attr_values if value not in values[attr]]
    return {attr: ''.join(attr_values) for attr, attr_values in values.items()}


def iter_aligned_xpos_tokens(pred_sents, ref_sents):
    assert len(pred_sents) == len(
        ref_sents), f'Prediction and reference must have the same number of sentences: {len(pred_sents)} vs {len(ref_sents)}'
    for pred_sent, ref_sent in zip(pred_sents, ref_sents):
        assert len(pred_sent) == len(
            ref_sent), f'Prediction and reference must have the same number of tokens:  {len(pred_sent)} != {len(ref_sent)}\n{pred_sent}\n{ref_sent})'
        for pred_token, ref_token in zip(pred_sent, ref_sent):
            yield pred_token, ref_token


def _divide(numerator: int, denominator: int, counts=False) -> float | int:
    if counts:
        return numerator
    return float(numerator) / denominator if denominator else 0.0


def eval_xpos_err_detailed(pred_sents, ref_sents, counts=False) -> dict[str, Any]:
    """
    Return per-XPOS-attribute error rates for aligned prediction/reference tokens.

    Attribute errors are counted only when the codifier is correct, matching the
    historical metric used by this parser.
    """
    errors = defaultdict(int)
    total = defaultdict(int)

    total_count = 0
    for pred_token, ref_token in iter_aligned_xpos_tokens(pred_sents, ref_sents):
        total_count += 1
        pred_xpos = pred_token['xpos']
        ref_xpos = ref_token['xpos']
        pred_xpos_dict = xpos_to_dict(pred_xpos)
        ref_xpos_dict = xpos_to_dict(ref_xpos)

        if 'codifier' not in ref_xpos_dict or 'codifier' not in pred_xpos_dict:
            print('missing codifier', ref_xpos, pred_xpos)
        correct_codifier = ref_xpos_dict.get('codifier') == pred_xpos_dict.get('codifier')
        errors['codifier'] += 1 if not correct_codifier else 0
        total['codifier'] += 1
        total['_total'] += 1
        has_error = False
        if correct_codifier:
            for key, value in ref_xpos_dict.items():
                if key == 'codifier':
                    continue
                if key in pred_xpos_dict:
                    total[key] += 1
                    if pred_xpos_dict[key] != value:
                        errors[key] += 1
                        has_error = True
        if has_error:
            errors['_total'] += 1
    result = {
        'err_' + key:
            errors[key] if counts else round((_divide(errors[key], total_count) * 100 if total_count > 0 else 0.0), 3)
        for key in total.keys()
    }
    return dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
