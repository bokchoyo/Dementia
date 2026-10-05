# author = liuwei
# date = 2023-l1-30

import logging
import os
import json
from tqdm import trange

import argparse
import torch
from transformers.models.roberta import RobertaConfig, RobertaTokenizer
import stanza

from pdtb_model import BaseClassifier

logging.disable(logging.WARNING)
from pdtb_utils import labels_from_file, is_exp_inter, is_exp_intra, split_into_sentences, pack_batch_data, \
    split_into_sentences_stanza

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()
chlr.setFormatter(formatter)
logger.addHandler(chlr)


def get_argparse():
    parser = argparse.ArgumentParser()

    # I/O paths
    parser.add_argument("--input_file", required=True, type=str,
                        help="Path to the input JSONL file to parse.")
    parser.add_argument("--output_file", required=True, type=str,
                        help="Path to the output JSONL file.")
    parser.add_argument("--parser_dir", default="data/parser/pdtb3", type=str,
                        help="Directory containing the pretrained PDTB parser checkpoints.")

    # model settings
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str, help="only large")
    parser.add_argument("--label_level", default=1, type=int)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--keep_text", action="store_true",
                        help="If set, keep the original text field in the output. By default, all original columns except text are preserved.")

    return parser


def parse_doc(input_doc, parse_params):
    """
    1. split the doc into sentences
    2. judge if connectives exist between sentences
    3. if yes, then use explicit model, else use implicit model
    4. save the results into doc

    requires:
        stanza,
        connective_list
    """
    tokenizer = parse_params["tokenizer"]
    exp_model = parse_params["exp_model"]
    imp_model = parse_params["imp_model"]
    conn_list = parse_params["conn_list"]
    exp_rel_list = parse_params["exp_rel_list"]
    imp_rel_list = parse_params["imp_rel_list"]
    max_seq_length = parse_params.get("max_seq_length", 256)
    use_stanza = parse_params.get("use_stanza", False)

    ## 1. split doc into sents, and judge if sent has connectives or not
    if use_stanza:
        stanza_dir = "/hits/basement/nlp/liuwi/resources/stanza_resources"
        stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize', dir=stanza_dir, download_method=None)
        doc_sents = split_into_sentences_stanza(input_doc, stanza_nlp)
    else:
        doc_sents = split_into_sentences(input_doc)
    sent_num = len(doc_sents)
    all_exp_items = []
    all_imp_items = []
    for idx in range(sent_num):
        # intra-sentence explicit relation
        has_conn, arg_pair = is_exp_intra(doc_sents[idx], conn_list=conn_list)
        if has_conn:
            all_exp_items.append((arg_pair[0], arg_pair[1], idx, idx, arg_pair[2], arg_pair[3]))

        # inter-sentence explicit relation
        if idx < sent_num - 1:
            has_conn, _ = is_exp_inter(doc_sents[idx], doc_sents[idx + 1], conn_list=conn_list)
            if has_conn:
                all_exp_items.append((
                    doc_sents[idx], doc_sents[idx + 1], idx, idx + 1,
                    (0, len(doc_sents[idx])), (0, len(doc_sents[idx + 1]))
                ))
            else:
                all_imp_items.append((
                    doc_sents[idx], doc_sents[idx + 1], idx, idx + 1,
                    (0, len(doc_sents[idx])), (0, len(doc_sents[idx + 1]))
                ))

    ## 2. parse both explicit and implicit
    pred_rels = []
    if len(all_exp_items) > 0:
        batch_data = pack_batch_data(all_exp_items, tokenizer, max_length=max_seq_length)
        batch_data = tuple(t.to(exp_model.device) for t in batch_data)
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = exp_model(**inputs)
            exp_pred_ids = list(outputs[0])
        # recover the position of relations
        assert len(all_exp_items) == len(exp_pred_ids), (len(all_exp_items), len(exp_pred_ids))
        for item, rel_id in zip(all_exp_items, exp_pred_ids):
            rel = exp_rel_list[rel_id]
            arg1_pos = item[2]
            arg2_pos = item[3]
            arg1_span = item[4]
            arg2_span = item[5]
            pred_rels.append((rel, arg1_pos, arg2_pos, "exp", arg1_span, arg2_span))
    if len(all_imp_items) > 0:
        batch_data = pack_batch_data(all_imp_items, tokenizer, max_length=max_seq_length)
        batch_data = tuple(t.to(imp_model.device) for t in batch_data)
        inputs = {
            "input_ids": batch_data[0],
            "attention_mask": batch_data[1],
            "token_type_ids": batch_data[2],
            "flag": "Eval"
        }
        with torch.no_grad():
            outputs = imp_model(**inputs)
            imp_pred_ids = list(outputs[0])
        assert len(all_imp_items) == len(imp_pred_ids), (len(all_imp_items), len(imp_pred_ids))
        for item, rel_id in zip(all_imp_items, imp_pred_ids):
            rel = imp_rel_list[rel_id]
            arg1_pos = item[2]
            arg2_pos = item[3]
            arg1_span = item[4]
            arg2_span = item[5]
            pred_rels.append((rel, arg1_pos, arg2_pos, "imp", arg1_span, arg2_span))

    ## 3. sorted and merge relation
    pred_rels = sorted(pred_rels, key=lambda x: (x[1], x[2]))
    all_rels = []
    all_rel_spans = []
    for item in pred_rels:
        all_rels.append(item[0])
        all_rel_spans.append((item[1], item[2], item[3], item[4], item[5]))

    return doc_sents, all_rels, all_rel_spans


def parse_file(in_file, parse_params, out_file, keep_text=False):
    """
    Preserve all original columns from the input JSONL except ``text`` by default,
    then append the parser outputs: ``sents``, ``rels``, and ``spans``.
    Set ``keep_text=True`` to also keep the original text field.
    """
    ## 1. read out raw rows
    all_rows = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "text" not in sample:
                raise KeyError(f"Missing required 'text' field in input row: {sample}")
            all_rows.append(sample)

    ## 2. parse docs
    all_parsed_rows = []
    doc_num = len(all_rows)
    doc_iter = trange(1, doc_num + 1, desc="Num")
    for doc_idx in doc_iter:
        sample = all_rows[doc_idx - 1]
        doc_text = sample["text"]
        doc_sents, doc_rels, doc_rel_spans = parse_doc(doc_text, parse_params)

        output_sample = {k: v for k, v in sample.items() if keep_text or k != "text"}
        output_sample["sents"] = doc_sents
        output_sample["rels"] = doc_rels
        output_sample["spans"] = doc_rel_spans
        all_parsed_rows.append(output_sample)

    ## 3. write into file
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_file, "w", encoding="utf-8") as f:
        for sample in all_parsed_rows:
            f.write("%s\n" % (json.dumps(sample, ensure_ascii=False)))


def parse():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    print("Training/evaluation parameters %s", args)

    ## 1. init model
    parser_dir = args.parser_dir
    config = RobertaConfig.from_pretrained(args.model_name_or_path)
    config.HP_dropout = 0.1
    tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)

    map_loc = args.device

    # 1.1 explicit model
    exp_dir = os.path.join(parser_dir, "exp", "l{}".format(args.label_level))
    conn_list = labels_from_file(os.path.join(exp_dir, "conn_list.txt"))
    exp_rel_list = labels_from_file(os.path.join(exp_dir, "labels.txt"))
    args.num_labels = len(exp_rel_list)
    exp_model = BaseClassifier(config=config, args=args)
    exp_model = exp_model.to(args.device)
    exp_checkpoint_file = os.path.join(exp_dir, "pytorch_model.bin")
    exp_state = torch.load(exp_checkpoint_file, map_location=map_loc)
    exp_model.load_state_dict(exp_state, strict=False)

    # 1.2 implicit model
    imp_dir = os.path.join(parser_dir, "imp", "l{}".format(args.label_level))
    imp_rel_list = labels_from_file(os.path.join(imp_dir, "labels.txt"))
    args.num_labels = len(imp_rel_list)
    imp_model = BaseClassifier(config=config, args=args)
    imp_model = imp_model.to(args.device)
    imp_checkpoint_file = os.path.join(imp_dir, "pytorch_model.bin")
    imp_state = torch.load(imp_checkpoint_file, map_location=map_loc)
    imp_model.load_state_dict(imp_state, strict=False)

    ## 2. parse one file
    parse_params = {
        "tokenizer": tokenizer,
        "exp_model": exp_model,
        "imp_model": imp_model,
        "conn_list": conn_list,
        "exp_rel_list": exp_rel_list,
        "imp_rel_list": imp_rel_list,
        "use_stanza": False,
        "max_seq_length": args.max_seq_length,
    }
    parse_file(args.input_file, parse_params, args.output_file, keep_text=args.keep_text)


if __name__ == "__main__":
    parse()
