# author = liuwei
# date = 2023-12-30

import os
import json
import re
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import torch
import math
import random
import nltk
from collections import defaultdict
from conn_filter import filter_so, filter_or, filter_as, filter_and, filter_after
from conn_filter import filter_because, filter_before, filter_but, filter_if
from conn_filter import filter_when, filter_since, filter_then
random.seed(106524)



# Candidate-generation constants mirrored from rknaebel/discopy/discopy/utils.py.
# The base and BERT connective components both rely on the same
# get_connective_candidates() logic from discopy.components.connective.base.
_SINGLE_CONNECTIVES = {
    'accordingly', 'additionally', 'after', 'afterward', 'afterwards', 'also',
    'alternatively', 'although', 'and', 'because', 'besides', 'but',
    'consequently', 'conversely', 'earlier', 'else', 'except', 'finally',
    'further', 'furthermore', 'hence', 'however', 'indeed', 'instead', 'later',
    'lest', 'likewise', 'meantime', 'meanwhile', 'moreover', 'nevertheless',
    'next', 'nonetheless', 'nor', 'once', 'or', 'otherwise', 'overall', 'plus',
    'previously', 'rather', 'regardless', 'separately', 'similarly',
    'simultaneously', 'since', 'specifically', 'still', 'then', 'thereafter',
    'thereby', 'therefore', 'though', 'thus', 'till', 'ultimately', 'unless',
    'until', 'whereas', 'while', 'yet'
}
_MULTI_CONNECTIVES = list(map(lambda s: s.split(' '), [
    'as a result', 'as an alternative', 'as if', 'as long as', 'as soon as',
    'as much as', 'as though', 'as well', 'as', 'before and after', 'before',
    'by comparison', 'by contrast', 'by then', 'for example', 'for instance',
    'for', 'if then', 'if and when', 'if', 'in addition', 'in contrast',
    'in fact', 'in other words', 'in particular', 'in short', 'in sum',
    'in the end', 'in turn', 'insofar as', 'much as', 'now that',
    'on the contrary', 'on the other hand', 'so that', 'so', 'when and if',
    'when', 'neither nor', 'either or'
]))
_DISTANT_CONNECTIVES = list(map(lambda s: s.split(' '), [
    'if then', 'neither nor', 'either or'
]))
_MULTI_CONNECTIVES_FIRST = {
    'as', 'before', 'by', 'for', 'either', 'if', 'in', 'insofar', 'much',
    'neither', 'now', 'on', 'so', 'when'
}

_TREEBANK_TOKENIZER = nltk.tokenize.TreebankWordTokenizer()
_SEPARATOR_TOKENS = {',', ';', ':'}


def _tokenize_with_spans(text):
    spans = list(_TREEBANK_TOKENIZER.span_tokenize(text))
    tokens = [text[start:end] for start, end in spans]
    lowered = [tok.lower().strip("'") for tok in tokens]
    return tokens, spans, lowered


def _is_punct_token(token):
    return bool(re.fullmatch(r"[^\w]+", token)) or token in {"''", '``'}


def _first_content_idx(tokens):
    for idx, tok in enumerate(tokens):
        if not _is_punct_token(tok):
            return idx
    return 0


def _last_content_idx(tokens):
    for idx in range(len(tokens) - 1, -1, -1):
        if not _is_punct_token(tokens[idx]):
            return idx
    return max(len(tokens) - 1, 0)


def _generate_connective_candidates(text):
    """Replicate discopy.components.connective.base.get_connective_candidates.

    Returns a list of dicts with the candidate kind so callers can distinguish
    distant connectives (e.g. ``if..then``) from contiguous multi-word ones
    (e.g. ``if then``).
    """
    tokens, spans, lowered = _tokenize_with_spans(text)
    chosen_indices = set()
    candidates = []
    for word_idx, word in enumerate(lowered):
        if word_idx in chosen_indices:
            continue

        for conn in _DISTANT_CONNECTIVES:
            if word == conn[0]:
                try:
                    i = lowered.index(conn[1], word_idx)
                    candidates.append({
                        'kind': 'distant',
                        'tokens': [(word_idx, conn[0]), (i, conn[1])],
                        'source_tokens': tokens,
                        'source_spans': spans,
                    })
                    break
                except ValueError:
                    continue

        if word in _MULTI_CONNECTIVES_FIRST:
            for multi_conn in _MULTI_CONNECTIVES:
                if (word_idx + len(multi_conn)) < len(lowered) and all(
                    c == lowered[word_idx + i] for i, c in enumerate(multi_conn)
                ):
                    chosen_indices.update(word_idx + i for i in range(len(multi_conn)))
                    candidates.append({
                        'kind': 'multi',
                        'tokens': [(word_idx + i, c) for i, c in enumerate(multi_conn)],
                        'source_tokens': tokens,
                        'source_spans': spans,
                    })
                    break

        if word in _SINGLE_CONNECTIVES:
            chosen_indices.add(word_idx)
            candidates.append({
                'kind': 'single',
                'tokens': [(word_idx, word)],
                'source_tokens': tokens,
                'source_spans': spans,
            })

    return candidates


def _candidate_key(candidate):
    words = [w for _, w in candidate['tokens']]
    if candidate['kind'] == 'distant':
        return '..'.join(words)
    return ' '.join(words)


def _candidate_position(candidate, tokens):
    idxs = [i for i, _ in candidate['tokens']]
    first_idx = _first_content_idx(tokens)
    last_idx = _last_content_idx(tokens)
    is_start = idxs[0] == first_idx
    is_end = idxs[-1] == last_idx
    return is_start, is_end


def _clean_segment(text, start, end):
    start = max(0, start)
    end = min(len(text), end)
    while start < end and text[start].isspace():
        start += 1
    while start < end and text[end - 1].isspace():
        end -= 1

    while start < end and text[start] in ',;:':
        start += 1
        while start < end and text[start].isspace():
            start += 1
    while start < end and text[end - 1] in ',;:':
        end -= 1
        while start < end and text[end - 1].isspace():
            end -= 1

    return text[start:end], (start, end)


def _first_separator_after(token_idx, tokens, spans):
    for i in range(token_idx + 1, len(tokens)):
        if tokens[i] in _SEPARATOR_TOKENS:
            return i, spans[i]
    return None, None


def _last_separator_before(token_idx, tokens, spans):
    for i in range(token_idx - 1, -1, -1):
        if tokens[i] in _SEPARATOR_TOKENS:
            return i, spans[i]
    return None, None


def _build_within_args(text, candidate):
    tokens = candidate['source_tokens']
    spans = candidate['source_spans']
    idxs = [i for i, _ in candidate['tokens']]
    start_char = spans[idxs[0]][0]
    end_char = spans[idxs[-1]][1]

    if candidate['kind'] == 'distant':
        second_start = spans[idxs[1]][0]
        second_end = spans[idxs[1]][1]
        arg1, span1 = _clean_segment(text, 0, second_start)
        arg2, span2 = _clean_segment(text, second_end, len(text))
        if arg1 and arg2:
            return arg1, arg2, span1, span2
        return None

    is_start, is_end = _candidate_position(candidate, tokens)

    if is_start:
        sep_idx, sep_span = _first_separator_after(idxs[-1], tokens, spans)
        if sep_idx is None:
            return None
        arg1, span1 = _clean_segment(text, end_char, sep_span[0])
        arg2, span2 = _clean_segment(text, sep_span[1], len(text))
    elif is_end:
        sep_idx, sep_span = _last_separator_before(idxs[0], tokens, spans)
        if sep_idx is None:
            return None
        arg1, span1 = _clean_segment(text, 0, sep_span[0])
        arg2, span2 = _clean_segment(text, sep_span[1], start_char)
    else:
        arg1, span1 = _clean_segment(text, 0, start_char)
        arg2, span2 = _clean_segment(text, end_char, len(text))

    if arg1 and arg2:
        return arg1, arg2, span1, span2
    return None


def _build_between_args(sent1, sent2, candidate):
    tokens = candidate['source_tokens']
    spans = candidate['source_spans']
    idxs = [i for i, _ in candidate['tokens']]

    is_start, _ = _candidate_position(candidate, tokens)
    if not is_start:
        return None

    if candidate['kind'] == 'distant':
        arg2_start = spans[idxs[1]][1]
    else:
        arg2_start = spans[idxs[-1]][1]
        sep_idx, sep_span = _first_separator_after(idxs[-1], tokens, spans)
        if sep_idx is not None:
            arg2_start = sep_span[1]

    arg1, span1 = _clean_segment(sent1, 0, len(sent1))
    arg2, span2 = _clean_segment(sent2, arg2_start, len(sent2))
    if arg1 and arg2:
        return arg1, arg2, span1, span2
    return None


def find_conn_within_sent(sent, conn, start_sent=False, in_sent=True, end_sent=False):
    """Find a connective inside one sentence using DisCoPy candidate generation.

    The candidate-identification stage mirrors the logic used by the DisCoPy base
    and BERT connective components. Argument boundaries are derived heuristically
    from the matched connective position because the original DisCoPy connective
    modules only identify connective spans, not argument spans.
    """
    conn = conn.lower().strip()
    candidates = _generate_connective_candidates(sent)
    for candidate in candidates:
        if _candidate_key(candidate) != conn:
            continue

        is_start, is_end = _candidate_position(candidate, candidate['source_tokens'])
        in_middle = not is_start and not is_end
        if (is_start and not start_sent) or (in_middle and not in_sent) or (is_end and not end_sent):
            continue

        args = _build_within_args(sent, candidate)
        if args is not None:
            return True, args
    return False, None


def find_conn_between_sent(sent1, sent2, conn):
    """Find a sentence-initial connective in sent2 using DisCoPy candidates."""
    conn = conn.lower().strip()
    candidates = _generate_connective_candidates(sent2)
    for candidate in candidates:
        if _candidate_key(candidate) != conn:
            continue
        args = _build_between_args(sent1, sent2, candidate)
        if args is not None:
            return True, args
    return False, None


def labels_from_file(label_file):
    label_list = []
    with open(label_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                label_list.append(line.strip().lower())

    return label_list


def is_exp_intra(sent, conn_list):
    sent_words = nltk.word_tokenize(sent)
    special_conns = {
        "although": (True, True, False), "even if": (True, True, False),
        "even though": (True, True, False), "if": (True, True, False),
        "instead": (False, False, True), "later": (False, True, True),
        "though": (False, True, True), "yet": (True, False, True),
    }
    filter_dict = {
        "so": filter_so, "or": filter_or, "as": filter_as,
        "and": filter_and, "after": filter_after, "because": filter_because,
        "before": filter_before, "but": filter_but, "if": filter_if,
        "when": filter_when, "since": filter_since, "then": filter_then
    }
    for conn in conn_list:
        if ".." in conn:
            items = conn.split("..")
            if items[0].lower() in sent_words and items[1].lower() in sent_words:
                if "," in sent:
                    comma_pos = sent.find(",")  # the first one
                    arg1 = sent[:comma_pos + 1].strip()
                    arg2 = sent[comma_pos + 1:].strip()
                    return True, (arg1, arg2, (0, comma_pos+1), (comma_pos+1, len(sent)))
        if conn in special_conns:
            start_sent, in_sent, end_sent = special_conns[conn]
            has_conn, args = find_conn_within_sent(sent, conn, start_sent, in_sent, end_sent)
        else:
            has_conn, args = find_conn_within_sent(sent, conn, False, True, False)
        if has_conn:
            if conn in filter_dict:
                filter_func = filter_dict[conn]

                # filter_flag = filter_func(args[0], args[1])
                # if not filter_flag:
                #     return True, args

                try:
                    filter_flag = filter_func(args[0], args[1])
                except (IndexError, KeyError):
                    filter_flag = True
                except Exception:
                    filter_flag = True
                if not filter_flag:
                    return True, args

            else:
                return True, args

    return False, None


def is_exp_inter(sent1, sent2, conn_list):
    special_conns = ["although", "even if", "even though", "if", "yet"]
    filter_dict = {
        "so": filter_so, "or": filter_or, "as": filter_as,
        "and": filter_and, "after": filter_after, "because": filter_because,
        "before": filter_before, "but": filter_but, "if": filter_if,
        "when": filter_when, "since": filter_since, "then": filter_then
    }
    for conn in conn_list:
        if conn not in special_conns:
            has_conn, args = find_conn_between_sent(sent1, sent2, conn)
            if has_conn:
                if conn in filter_dict:
                    filter_func = filter_dict[conn]

                    # filter_flag = filter_func(args[0], args[1])
                    # if not filter_flag:
                    #     return True, args

                    try:
                        filter_flag = filter_func(args[0], args[1])
                    except (IndexError, KeyError):
                        filter_flag = True
                    except Exception:
                        filter_flag = True
                    if not filter_flag:
                        return True, args

                else:
                    return True, args

    return False, None


def split_into_sentences(text):
    alphabets= "([A-Za-z])"
    prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
    suffixes = "(Inc|Ltd|Jr|Sr|Co)"
    starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
    acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
    websites = "[.](com|net|org|io|gov|me|edu)"
    digits = "([0-9])"

    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = text.replace("e.g.","e<prd>g<prd>")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "..." in text: text = text.replace("...","<prd><prd><prd>")
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]

    return sentences


def pack_batch_data(arg_pairs, tokenizer, max_length):
    """
    Args:
        arg_pairs:
        tokenizer:
        max_length:
    """
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    for item in arg_pairs:
        arg1 = item[0]
        arg2 = item[1]
        tmp_text_res = tokenizer(
            text=arg1,
            text_pair=arg2,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = tmp_text_res.input_ids
        attention_mask = tmp_text_res.attention_mask
        if "token_type_ids" in tmp_text_res:
            token_type_ids = tmp_text_res["token_type_ids"]
        else:
            token_type_ids = torch.zeros_like(attention_mask)
        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
    # all_input_ids = np.array(all_input_ids)
    # all_attention_mask = np.array(all_attention_mask)
    # all_token_type_ids = np.array(all_token_type_ids)
    all_input_ids = torch.cat(all_input_ids, dim=0)
    all_attention_mask = torch.cat(all_attention_mask, dim=0)
    all_token_type_ids = torch.cat(all_token_type_ids, dim=0)

    return (all_input_ids, all_attention_mask, all_token_type_ids)


def split_into_sentences_stanza(text, stanza_nlp):
    nlp_text = stanza_nlp(text)
    sentences = []
    for sentence in nlp_text.sentences:
        sentences.append(sentence.text)

    return sentences