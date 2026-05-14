import json
import math
import urllib.error
import urllib.request
from typing import Any

from lvnlp.parser.xpos import CODE_TO_ATTRS, filter_tag, fix_tag, get_xpos_attribute_values, get_xpos_schema_for_tag, xpos_to_dict


DEFAULT_ANALYZER_URL = 'http://nlp.ailab.lv:7070/analyze'


XPOS_ATTRIBUTE_VALUES = get_xpos_attribute_values()


def analyze(tokens: list[str], url: str = DEFAULT_ANALYZER_URL, timeout: float = 60.0) -> list[dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data='\n'.join(tokens).encode('utf-8'),
        headers={'Content-Type': 'text/plain; charset=utf-8'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Analyzer request failed for {url!r}: {exc}') from exc


def add_analyzer(data, url: str = DEFAULT_ANALYZER_URL, timeout: float = 60.0):
    tokens = [token['text'] for sentence in data for token in sentence]
    analyses = analyze(tokens, url=url, timeout=timeout)
    if len(analyses) != len(tokens):
        raise RuntimeError(f'Analyzer returned {len(analyses)} analyses for {len(tokens)} tokens')

    index = 0
    for sentence in data:
        for token in sentence:
            token['analysis'] = analyses[index].get('options') or []
            index += 1
    return data


def _normalize_tag(tag: str, filter_defaults: bool) -> str:
    tag = fix_tag(tag)
    if not filter_defaults:
        return tag
    try:
        return filter_tag(tag)
    except Exception:
        return tag


def simplify_options(options, add_simple=True, filter_defaults=True):
    normalized = []
    for option in options or []:
        tag = option.get('tag') or ''
        simple_tag = option.get('simplifiedTag') or ''

        if tag:
            tag = _normalize_tag(tag, filter_defaults)
            normalized.append({**option, 'tag': tag})

        if add_simple and simple_tag:
            simple_tag = _normalize_tag(simple_tag, filter_defaults)
            normalized.append({**option, 'tag': simple_tag})

    seen = set()
    result = []
    for option in normalized:
        tag = option.get('tag')
        lemma = option.get('lemma') or '_'
        key = (tag, lemma)
        if not tag or key in seen:
            continue
        seen.add(key)
        result.append({'tag': tag, 'lemma': lemma})
    return result


def is_analyzer_option_matching(option: str, xpos: str) -> bool:
    if not option or not xpos:
        return False
    if option == xpos:
        return True
    if len(xpos) != len(option):
        return False
    return all(a == b or a == '_' or b == '_' for a, b in zip(option, xpos))


def _log_softmax(values):
    if not values:
        return []
    max_value = max(values)
    log_total = max_value + math.log(sum(math.exp(value - max_value) for value in values))
    return [value - log_total for value in values]


def log_softmax_probs(probs):
    return {
        attr: dict(zip(attr_probs.keys(), _log_softmax(list(attr_probs.values()))))
        for attr, attr_probs in probs.items()
    }


def get_max_allowed(attr_probs, allowed_values):
    return max(
        ((value, attr_probs.get(value, -float('inf'))) for value in allowed_values),
        key=lambda item: item[1],
    )


def decode_max_xpos_from_log_probs(log_probs: dict[str, dict[str, float]]) -> tuple[str, float]:
    decoded = {}
    codifier = get_max_allowed(log_probs['codifier'], XPOS_ATTRIBUTE_VALUES['codifier'])[0]
    decoded['codifier'] = codifier
    if codifier == 'v':
        mood = get_max_allowed(log_probs['mood'], XPOS_ATTRIBUTE_VALUES['mood'])[0]
        decoded['mood'] = mood
        if mood == 'p':
            codifier = 'V'

    schema = CODE_TO_ATTRS[codifier]
    selected_probs = []
    for attr, allowed_values in schema.items():
        if attr in ('codifier', 'mood'):
            continue
        value, prob = get_max_allowed(log_probs[attr], allowed_values)
        decoded[attr] = value
        selected_probs.append(prob)
    return ''.join(decoded[attr] for attr in schema), sum(selected_probs)


def decode_from_analysis_and_log_probs(analysis, log_probs) -> tuple[dict[str, str], float]:
    best = None
    best_prob = -float('inf')

    for option in analysis:
        tag = option['tag']
        try:
            decoded = xpos_to_dict(tag)
        except Exception:
            continue

        schema = get_xpos_schema_for_tag(tag)
        if schema is None:
            continue

        for attr, allowed_values in schema.items():
            if decoded[attr] == '_' or decoded[attr] not in allowed_values:
                decoded[attr] = get_max_allowed(log_probs[attr], allowed_values)[0]

        option['tag'] = ''.join(decoded[attr] for attr in schema)
        attr_probs = [log_probs[attr].get(value, -float('inf')) for attr, value in decoded.items()]
        option_prob = sum(attr_probs) if attr_probs else -float('inf')
        option['prob'] = option_prob
        if option_prob > best_prob:
            best = option
            best_prob = option_prob

    if best is None:
        return {'tag': decode_max_xpos_from_log_probs(log_probs)[0], 'lemma': '_'}, best_prob
    return best, best_prob


def rescore_with_analyzer(data, margin: float = 5.0, overwrite_lemma: bool = True):
    for sentence in data:
        for token in sentence:
            raw_probs = token.get('probs', {}).get('xpos')
            if not raw_probs:
                continue

            analysis = simplify_options(token.get('analysis') or [], add_simple=False, filter_defaults=True)
            if not analysis:
                continue

            log_probs = log_softmax_probs(raw_probs)
            best, best_prob = decode_from_analysis_and_log_probs(analysis, log_probs)
            greedy_tag, greedy_prob = decode_max_xpos_from_log_probs(log_probs)
            token['analysis'] = analysis

            if best_prob > greedy_prob - margin:
                token['xpos'] = best['tag']
                lemma = best.get('lemma')
                if overwrite_lemma and lemma and lemma != '_' and token.get('lemma', '').lower() != lemma.lower():
                    token['lemma'] = lemma
            elif not token.get('xpos'):
                token['xpos'] = greedy_tag
    return data


def drop_analyzer_fields(data, drop_probs=False):
    for sentence in data:
        for token in sentence:
            token.pop('analysis', None)
            if drop_probs:
                token.pop('probs', None)
    return data


if __name__ == '__main__':
    print(analyze(['Jānis', 'gāja', 'uz', 'veikalu', '.']))
