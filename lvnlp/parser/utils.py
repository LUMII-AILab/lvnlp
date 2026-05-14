import logging
import os
import random
import re
from collections import Counter, defaultdict
from types import MappingProxyType

import torch
from smart_open import open
from tqdm import tqdm

logger = logging.getLogger(__name__)


def min_edit_script(source, target, allow_copy):
    a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
    for i in range(0, len(source) + 1):
        for j in range(0, len(target) + 1):
            if i == 0 and j == 0:
                a[i][j] = (0, '')
            else:
                if allow_copy and i and j and source[i - 1] == target[j - 1] and a[i - 1][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j - 1][0], a[i - 1][j - 1][1] + '→')
                if i and a[i - 1][j][0] < a[i][j][0]:
                    a[i][j] = (a[i - 1][j][0] + 1, a[i - 1][j][1] + '-')
                if j and a[i][j - 1][0] < a[i][j][0]:
                    a[i][j] = (a[i][j - 1][0] + 1, a[i][j - 1][1] + '+' + target[j - 1])
    return a[-1][-1][1]

LEMMA_RULE_TYPES = ['case', 'prefix', 'suffix', 'absolute']


def gen_lemma_rule(form, lemma, allow_copy):
    best, best_form, best_lemma = 0, 0, 0
    for l in range(len(lemma)):
        for f in range(len(form)):
            cpl = 0
            while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl].lower() == lemma[l + cpl].lower():
                cpl += 1
            if cpl > best:
                best = cpl
                best_form = f
                best_lemma = l

    if not best:
        return {'case': None, 'prefix': None, 'suffix': None, 'absolute': 'a' + lemma}

    prefix_rule = min_edit_script(form[:best_form].lower(), lemma[:best_lemma].lower(), allow_copy)
    suffix_rule = min_edit_script(form[best_form + best:].lower(), lemma[best_lemma + best:].lower(), allow_copy)

    if lemma.islower():
        return {'case': 'lower', 'prefix': prefix_rule, 'suffix': suffix_rule, 'absolute': 'relative'}

    generated_lemma = apply_lemma_rule(
        form,
        {'case': 'lower', 'prefix': prefix_rule, 'suffix': suffix_rule, 'absolute': 'relative'},
        apply_casing=False,
    )
    if generated_lemma == lemma:
        return {'case': 'keep', 'prefix': prefix_rule, 'suffix': suffix_rule, 'absolute': 'relative'}

    previous_case = -1
    lemma_casing = ''
    for i, c in enumerate(lemma):
        case = '↑' if c.lower() != c else '↓'
        if case != previous_case:
            lemma_casing += '{}{}{}'.format('¦' if lemma_casing else '', case, i if i <= len(lemma) // 2 else i - len(lemma))
        previous_case = case

    return {'case': lemma_casing, 'prefix': prefix_rule, 'suffix': suffix_rule, 'absolute': 'relative'}
    # return (lemma_casing, prefix_rule, suffix_rule, 'relative')


def get_normalized_lemma_rules(form_lemmas):
    # updates lemma rules to canonicalize based on the most common rule for each lemma, and to set case=lower if the lemma is lowercase and no other rule applies
    form_lemmas_flat = [(form, lemma) for s in form_lemmas for (form, lemma) in s]

    form_lemma_counts = Counter(form_lemmas_flat)
    form_lemma_rules_initial = {(form, lemma): gen_lemma_rule(form, lemma, True) for form, lemma in list(dict.fromkeys(form_lemmas_flat))}
    lemma_rule_counts = Counter()
    for (form, lemma), rule in form_lemma_rules_initial.items():
        lemma_rule_counts[(rule['case'], rule['prefix'], rule['suffix'], rule['absolute'])] += form_lemma_counts[(form, lemma)]

    absolute_vocab = {item[3] for item in lemma_rule_counts.keys() if item[3].startswith('a')}

    form_lemma_rules = {}
    for (form, lemma), base_rule in tqdm(form_lemma_rules_initial.items(), desc='Building lemma rules'):
        if base_rule['absolute'].startswith('a'):
            form_lemma_rules[(form, lemma)] = {'case': None, 'prefix': None, 'suffix': None, 'absolute': base_rule['absolute']}
            continue

        for (case, prefix, suffix, absolute), _ in lemma_rule_counts.most_common():
            if absolute.startswith('a'):
                continue
            rule_dict = {'case': case, 'prefix': prefix, 'suffix': suffix, 'absolute': absolute}
            applied_rule_lemma = apply_lemma_rule(form, rule_dict)
            if applied_rule_lemma == lemma:
                form_lemma_rules[(form, lemma)] = rule_dict
                # if rule_dict != base_rule: print('Updated rule for form="{}" lemma="{}": {} -> {}'.format(form, lemma, base_rule, rule_dict))
                break
        else:
            form_lemma_rules[(form, lemma)] = {'case': None, 'prefix': None, 'suffix': None, 'absolute': base_rule['absolute']}

        if ('a' + lemma) in absolute_vocab:
            form_lemma_rules[(form, lemma)]['absolute'] = 'a' + lemma
        if form_lemma_rules[(form, lemma)]['case'] is None and lemma.islower():
            form_lemma_rules[(form, lemma)]['case'] = 'lower'

    rule_to_lemma_examples = {key: defaultdict(set) for key in ['case', 'prefix', 'suffix', 'absolute']}
    for (form, lemma), rule in form_lemma_rules.items():
        for rule_type in rule_to_lemma_examples.keys():
            rule_to_lemma_examples[rule_type][rule[rule_type]].add(lemma)

    for (form, lemma), rule in form_lemma_rules.items():
        for rule_type in rule_to_lemma_examples.keys():
            if len(rule_to_lemma_examples[rule_type][rule[rule_type]]) == 1:
                form_lemma_rules[(form, lemma)][rule_type] = None
                form_lemma_rules[(form, lemma)]['absolute'] = 'a' + lemma

    gold_rules = [[form_lemma_rules[(form, lemma)] for form, lemma in s] for s in form_lemmas]
    return gold_rules


def apply_lemma_rule(form, lemma_rule, apply_casing=True):
    if isinstance(lemma_rule, tuple):
        lemma_rule = {
            'case': lemma_rule[0],
            'prefix': lemma_rule[1],
            'suffix': lemma_rule[2],
            'absolute': lemma_rule[3],
        }
    if lemma_rule['absolute'].startswith('a'):
        return lemma_rule['absolute'][1:]

    if lemma_rule['case'] == '<none>':
        lemma_rule['case'] = None
    if lemma_rule['prefix'] == '<none>':
        lemma_rule['prefix'] = None
    if lemma_rule['suffix'] == '<none>':
        lemma_rule['suffix'] = None
    if lemma_rule['absolute'] == '<none>':
        lemma_rule['absolute'] = None

    if any(rule is None for rule in lemma_rule.values()):
        return form

    rules, rule_sources = (lemma_rule['prefix'], lemma_rule['suffix']), []
    for rule in rules:
        source, i = 0, 0
        while i < len(rule):
            if rule[i] == '→' or rule[i] == '-':
                source += 1
            else:
                if rule[i] != '+':
                    print(f'Invalid rule {form=} {rule=} idx={i}', flush=True)
                assert rule[i] == '+'
                i += 1
            i += 1
        rule_sources.append(source)

    try:
        lemma, form_offset = '', 0
        for i in range(2):
            j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
            while j < len(rules[i]):
                if rules[i][j] == '→':
                    lemma += form[offset]
                    offset += 1
                elif rules[i][j] == '-':
                    offset += 1
                else:
                    assert rules[i][j] == '+'
                    lemma += rules[i][j + 1]
                    j += 1
                j += 1
            if i == 0:
                lemma += form[rule_sources[0]: len(form) - rule_sources[1]]
    except Exception:
        lemma = form

    if not apply_casing:
        return lemma

    if lemma_rule['case'] == 'lower':
        return lemma.lower()
    elif lemma_rule['case'] == 'keep':
        return lemma

    lemma = lemma.lower()
    for rule in lemma_rule['case'].split('¦'):
        if rule == '↓0':
            continue  # The lemma is lowercased initially
        if not rule:
            continue  # Empty lemma might generate empty casing rule
        case, offset = rule[0], int(rule[1:])
        lemma = lemma[:offset] + (lemma[offset:].upper() if case == '↑' else lemma[offset:].lower())

    return lemma


def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)


def get_model_name(fpath):
    if os.path.exists(fpath):
        basename = os.path.basename(fpath)
        if any(c in basename for c in ['checkpoint', 'converted']):
            basename = os.path.dirname(fpath) + '__' + basename
        elif 'models/' in fpath:
            basename = fpath.split('models/')[-1].replace('/', '__')
        return basename
    else:
        return fpath.replace('/', '__')


def get_optimizer_param_groups(model, lr, weight_decay, lr_ratio):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    norm_pat = re.compile(r'(?:^|\.)(?:LayerNorm|layer_norm|norm|.*_norm)(?:\.|$)')

    def is_no_decay(n: str) -> bool:
        if n.endswith('.bias'):
            return True
        if n.endswith('.weight') and norm_pat.search(n):
            return True
        # keep your special params out of decay
        if 'layer_score' in n:
            return True
        return False

    bert_decay_params = [(n, p) for n, p in params if n.startswith('bert.') and not is_no_decay(n)]
    bert_no_decay_params = [(n, p) for n, p in params if n.startswith('bert.') and is_no_decay(n)]
    top_decay_params = [(n, p) for n, p in params if (not n.startswith('bert.')) and not is_no_decay(n)]
    top_no_decay_params = [(n, p) for n, p in params if (not n.startswith('bert.')) and is_no_decay(n)]

    optimizer_grouped_parameters = [
        {'params': [p for _, p in bert_decay_params], 'lr': lr_ratio * lr, 'weight_decay': weight_decay},
        {'params': [p for _, p in bert_no_decay_params], 'lr': lr_ratio * lr, 'weight_decay': 0.0},
        {'params': [p for _, p in top_decay_params], 'lr': lr, 'weight_decay': weight_decay},
        {'params': [p for _, p in top_no_decay_params], 'lr': lr, 'weight_decay': 0.0},
    ]

    return optimizer_grouped_parameters


class Columns:
    ID = 'id'
    TEXT = 'text'
    LEMMA = 'lemma'
    UPOS = 'upos'
    XPOS = 'xpos'
    FEATS = 'feats'
    HEAD = 'head'
    DEPREL = 'deprel'
    DEPS = 'deps'
    MISC = 'misc'

CONLLU_COLS = (Columns.ID, Columns.TEXT, Columns.LEMMA, Columns.UPOS, Columns.XPOS, Columns.FEATS, Columns.HEAD, Columns.DEPREL, Columns.DEPS, Columns.MISC)
CONLLU_COLS_IDX = {col: idx for idx, col in enumerate(CONLLU_COLS)}

CONVERTERS = MappingProxyType({
    Columns.ID: int,
    Columns.HEAD: int
})


def filter_by_len(data, max_tokens):
    r = []
    for s in data:
        if len(s) <= max_tokens:
            r.append(s)
        else:
            print(f'Skipping sentence of length {len(s)} (tokens)')
    return r


def read_conll(f, ignore_gapping=True, cols=CONLLU_COLS, converters=CONVERTERS):
    """Load the file or string into the CoNLL-U format data.
    Input: file or string reader, where the data is in CoNLL-U format.
    Output: a tuple whose first element is a list of list of list for each token in each sentence in the data,
    where the innermost list represents all fields of a token; and whose second element is a list of lists for each
    comment in each sentence in the data.
    """
    if isinstance(f, str) and '|' in f:
        f = f.split('|')
    if isinstance(f, (list, tuple)):
        return [item for file in f for item in read_conll(file, ignore_gapping, cols, converters)]

    # f is open() or io.StringIO()
    if isinstance(f, str):
        f = open(f, 'r', encoding='utf-8')
    doc, sent = [], []
    doc_comments = []
    for line_idx, line in enumerate(f):
        # leave whitespace such as NBSP, in case it is meaningful in the conll-u doc
        line = line.lstrip().rstrip(' \n\r\t')
        if len(line) == 0:
            if len(sent) > 0:
                doc.append(sent)
                sent = []
        else:
            if line.startswith('#'):  # read comment line
                continue
            array = line.split('\t')
            if ignore_gapping and '.' in array[0]:
                continue
            if len(array) != len(cols):
                raise Exception(f'Cannot parse CoNLL line {line_idx + 1}: expecting {len(cols)} fields, {len(array)} found at line {line_idx}\n\t{array}')
            d = dict(zip(cols, array))
            for col, converter in converters.items():
                d[col] = converter(d[col])
            sent.append(d)
    if len(sent) > 0:
        doc.append(sent)
    return doc


def iter_field(data, field):
    for sent in data:
        for token in sent:
            yield token[field]


def to_conll(data, cols=CONLLU_COLS):
    res = []
    for sent in data:
        for token in sent:
            row = [token.get(col) for col in cols]
            row = [str(col) if col is not None else '_' for col in row]
            res.append('\t'.join(row))
        res.append('\n')
    return '\n'.join(res)


def write_conll(f, data):
    opened = False
    if isinstance(f, str):
        f = open(f, 'w')
        opened = True
    conll = to_conll(data)
    f.write(conll)
    if opened:
        f.close()


def tokenize_sentence(text):
    tokens = re.findall(r'\w+|\S', text)
    tokens = [{
        'id': idx,
        'text': token_text,
        'lemma': token_text,
        'upos': 'PROPN',
        'xpos': '_',
        'feats': '_',
        'head': 0,
        'deprel': '_',
        'deps': '_',
        'misc': '_'
    } for idx, token_text in enumerate(tokens, start=1)]
    return tokens
