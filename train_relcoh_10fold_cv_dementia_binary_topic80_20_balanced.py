"""
Patient/control classifier stacked on a frozen coherence prediction model.
Inputs to the binary head:
1) the frozen coherence model's final hidden representation, and
2) the frozen coherence model's softmax probabilities.
Only the new binary head is trainable. The original coherence model is kept
in eval mode and wrapped in torch.no_grad(), so gradients cannot update it.
"""


# author = liuwei
# date = 2024-05-07

import logging
import os
import json
import pickle
import math
import random
import time
import datetime
from tqdm import tqdm

from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader

from torch.optim import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from transformers import RobertaConfig, RobertaTokenizer, RobertaModel, BertModel
from transformers import XLNetConfig, XLNetTokenizer, XLNetModel
from transformers import LlamaConfig, LlamaTokenizer, LlamaModel
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, AutoModel
from dataset import SentDataset
from model import SentTransformer, BaseClassifer
try:
    from model import FrozenCoherenceBinaryClassifier
except ImportError:
    class FrozenCoherenceBinaryClassifier(nn.Module):
        """
        Patient/control classifier stacked on a frozen coherence prediction model.

        Inputs to the binary head:
          1) final hidden representation feeding the original coherence classifier,
          2) frozen coherence softmax probabilities.

        This fallback keeps the script runnable even if model.py has not been
        updated with FrozenCoherenceBinaryClassifier/extract_features.
        """
        def __init__(self, coherence_model, args):
            super(FrozenCoherenceBinaryClassifier, self).__init__()
            self.coherence_model = coherence_model
            for param in self.coherence_model.parameters():
                param.requires_grad = False
            self.coherence_model.eval()

            self.num_labels = int(getattr(args, "num_labels", 2))
            self.coherence_num_labels = int(getattr(self.coherence_model, "num_labels", 3))
            if hasattr(self.coherence_model, "feature_dim"):
                self.feature_dim = int(self.coherence_model.feature_dim)
            else:
                self.feature_dim = int(self.coherence_model.classifier.in_features)
            self.binary_input_dim = self.feature_dim + self.coherence_num_labels

            hidden_size = int(getattr(args, "binary_head_hidden_size", 128))
            if hidden_size <= 0:
                hidden_size = max(16, self.binary_input_dim // 2)

            dropout_p = float(getattr(args, "dropout", 0.1))
            self.binary_classifier = nn.Sequential(
                nn.Dropout(dropout_p),
                nn.Linear(self.binary_input_dim, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_size, self.num_labels),
            )
            self._init_binary_head()

        def _init_binary_head(self):
            for module in self.binary_classifier.modules():
                if isinstance(module, nn.Linear):
                    module.weight.data.normal_(mean=0.0, std=0.02)
                    if module.bias is not None:
                        module.bias.data.zero_()

        def train(self, mode=True):
            super(FrozenCoherenceBinaryClassifier, self).train(mode)
            self.coherence_model.eval()
            return self

        def _extract_features(self, sent_vectors=None, sent_mask=None, doc_vectors=None):
            self.coherence_model.eval()
            with torch.no_grad():
                if hasattr(self.coherence_model, "extract_features"):
                    if doc_vectors is not None:
                        hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(doc_vectors=doc_vectors)
                    else:
                        hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(
                            sent_vectors=sent_vectors, sent_mask=sent_mask
                        )
                elif doc_vectors is not None:
                    if doc_vectors.dtype == torch.bfloat16:
                        doc_vectors = doc_vectors.float()
                    features = self.coherence_model.dropout(doc_vectors)
                    features = features.float()
                    features = self.coherence_model.fc1(features)
                    features = self.coherence_model.dropout(features)
                    hidden = self.coherence_model.fc2(features)
                    classifier_input = self.coherence_model.dropout(hidden)
                    coherence_logits = self.coherence_model.classifier(classifier_input)
                    coherence_probs = torch.softmax(coherence_logits, dim=-1)
                else:
                    if sent_vectors.dtype == torch.bfloat16:
                        sent_vectors = sent_vectors.float()
                    sent_vectors = self.coherence_model.proj(sent_vectors)
                    input_vectors = self.coherence_model.abs_position_embedding(sent_vectors)
                    output = self.coherence_model.transformer(input_vectors, sent_mask)
                    output = self.coherence_model.dropout(output)
                    hidden = self.coherence_model.fc(output)
                    classifier_input = self.coherence_model.dropout(hidden)
                    coherence_logits = self.coherence_model.classifier(classifier_input)
                    coherence_probs = torch.softmax(coherence_logits, dim=-1)
            return hidden.detach(), coherence_logits.detach(), coherence_probs.detach()

        def forward(self, sent_vectors=None, sent_mask=None, doc_vectors=None, labels=None, flag="Train", **kwargs):
            hidden, coherence_logits, coherence_probs = self._extract_features(
                sent_vectors=sent_vectors, sent_mask=sent_mask, doc_vectors=doc_vectors
            )
            binary_features = torch.cat([hidden.float(), coherence_probs.float()], dim=-1)
            binary_logits = self.binary_classifier(binary_features)
            binary_probs = torch.softmax(binary_logits, dim=-1)
            _, preds = torch.max(binary_logits, dim=-1)

            outputs = (preds, binary_logits, binary_probs, coherence_probs)
            if flag.upper() == "TRAIN":
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                loss = loss_fct(binary_logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs
            return outputs

from utils import labels_from_file


class EnhancedFrozenCoherenceBinaryClassifier(nn.Module):
    """
    Patient/control classifier stacked on a frozen coherence model.

    The binary head uses:
      [last hidden representation from frozen coherence model,
       frozen coherence softmax probabilities]

    Unlike older wrapper versions, this class supports class-weighted binary
    CrossEntropyLoss, which helps avoid majority-class collapse.
    """
    def __init__(self, coherence_model, args):
        super(EnhancedFrozenCoherenceBinaryClassifier, self).__init__()
        self.coherence_model = coherence_model
        for param in self.coherence_model.parameters():
            param.requires_grad = False
        self.coherence_model.eval()

        self.num_labels = int(getattr(args, "num_labels", 2))
        self.coherence_num_labels = int(getattr(self.coherence_model, "num_labels", 3))
        if hasattr(self.coherence_model, "feature_dim"):
            self.feature_dim = int(self.coherence_model.feature_dim)
        else:
            self.feature_dim = int(self.coherence_model.classifier.in_features)
        self.binary_input_dim = self.feature_dim + self.coherence_num_labels

        hidden_size = int(getattr(args, "binary_head_hidden_size", 128))
        if hidden_size <= 0:
            hidden_size = max(16, self.binary_input_dim // 2)

        dropout_p = float(getattr(args, "dropout", 0.1))
        self.binary_classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(self.binary_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden_size, self.num_labels),
        )
        self._init_binary_head()

        class_weights = getattr(args, "binary_class_weights", None)
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            if class_weights.numel() != self.num_labels:
                raise ValueError(
                    f"binary_class_weights length {class_weights.numel()} does not match num_labels {self.num_labels}"
                )
            self.register_buffer("binary_class_weights", class_weights)
        else:
            self.binary_class_weights = None

    def _init_binary_head(self):
        for module in self.binary_classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()

    def train(self, mode=True):
        super(EnhancedFrozenCoherenceBinaryClassifier, self).train(mode)
        self.coherence_model.eval()
        return self

    def _extract_features(self, sent_vectors=None, sent_mask=None, doc_vectors=None):
        self.coherence_model.eval()
        with torch.no_grad():
            if hasattr(self.coherence_model, "extract_features"):
                if doc_vectors is not None:
                    hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(doc_vectors=doc_vectors)
                else:
                    hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(
                        sent_vectors=sent_vectors, sent_mask=sent_mask
                    )
            elif doc_vectors is not None:
                if doc_vectors.dtype == torch.bfloat16:
                    doc_vectors = doc_vectors.float()
                features = self.coherence_model.dropout(doc_vectors)
                features = features.float()
                features = self.coherence_model.fc1(features)
                features = self.coherence_model.dropout(features)
                hidden = self.coherence_model.fc2(features)
                classifier_input = self.coherence_model.dropout(hidden)
                coherence_logits = self.coherence_model.classifier(classifier_input)
                coherence_probs = torch.softmax(coherence_logits, dim=-1)
            else:
                if sent_vectors.dtype == torch.bfloat16:
                    sent_vectors = sent_vectors.float()
                sent_vectors = self.coherence_model.proj(sent_vectors)
                input_vectors = self.coherence_model.abs_position_embedding(sent_vectors)
                output = self.coherence_model.transformer(input_vectors, sent_mask)
                output = self.coherence_model.dropout(output)
                hidden = self.coherence_model.fc(output)
                classifier_input = self.coherence_model.dropout(hidden)
                coherence_logits = self.coherence_model.classifier(classifier_input)
                coherence_probs = torch.softmax(coherence_logits, dim=-1)
        return hidden.detach(), coherence_logits.detach(), coherence_probs.detach()

    def forward(self, sent_vectors=None, sent_mask=None, doc_vectors=None, labels=None, flag="Train", **kwargs):
        hidden, coherence_logits, coherence_probs = self._extract_features(
            sent_vectors=sent_vectors, sent_mask=sent_mask, doc_vectors=doc_vectors
        )
        binary_features = torch.cat([hidden.float(), coherence_probs.float()], dim=-1)
        binary_logits = self.binary_classifier(binary_features)
        binary_probs = torch.softmax(binary_logits, dim=-1)
        _, preds = torch.max(binary_logits, dim=-1)

        outputs = (preds, binary_logits, binary_probs, coherence_probs)
        if flag.upper() == "TRAIN":
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-1,
                weight=self.binary_class_weights if self.binary_class_weights is not None else None,
            )
            loss = loss_fct(binary_logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs



# logging.disable(logging.WARNING)

# set logger, print to console and write to file
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
BASIC_FORMAT = "%(asctime)s:%(levelname)s: %(message)s"
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
chlr = logging.StreamHandler()  # 输出到控制台的handler
chlr.setFormatter(formatter)
logger.addHandler(chlr)

# for output
dt = datetime.datetime.now()
TIME_CHECKPOINT_DIR = "checkpoint_{}-{}-{}_{}:{}".format(dt.year, dt.month, dt.day, dt.hour, dt.minute)
PREFIX_CHECKPOINT_DIR = "checkpoint"


def get_argparse():
    parser = argparse.ArgumentParser()

    # for data
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--dataset", default="test", type=str, help="toefl1234, gcdc")
    parser.add_argument("--fold_id", default=-1, type=int, help="[1-5] for toefl1234, [1-10] for gcdc")
    parser.add_argument("--output_dir", default="data/result", type=str, help="path to save checkpoint")
    parser.add_argument("--label_list", default="low, medium, high", type=str)
    parser.add_argument("--model_type", default="transformer_sent", type=str,
                        help="Supported: transformer_sent (or sent), flat (or base).")
    parser.add_argument("--pooler_type", default="avg", type=str, help="cls, avg")
    parser.add_argument("--embed_dim", default=300, type=int, help="the dimension size of fc layer")
    parser.add_argument("--embed_file", default="glove.840B.300d.txt", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str, help="roberta-base")

    # for training
    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--test_file", default=r"data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl", type=str)
    parser.add_argument("--prediction_output_file", default="", type=str)
    parser.add_argument("--checkpoint_file", default="", type=str)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--eval_batch_size", default=48, type=int)
    parser.add_argument("--max_text_length", default=512, type=int)
    parser.add_argument("--max_arg_num", default=32, type=int)
    parser.add_argument("--max_sent_num", default=24, type=int, help="number of sents in a text")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="training epoch")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--max_grad_norm", default=2.0, type=float)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)
    parser.add_argument("--seed", default=106524, type=int, help="random seed")
    parser.add_argument(
        "--train_file",
        default=r"data/dataset/gcdc/combined/1/train_full.json",
        type=str,
        help="training json/jsonl file"
    )
    # for paper-style 10-fold CV on the full GCDC training file
    parser.add_argument("--do_cv", default=False, action="store_true",
                        help="Run 10-fold cross-validation over --cv_file/--train_file for model selection.")
    parser.add_argument("--cv_file", default="", type=str,
                        help="Full GCDC training json/jsonl file used to create CV folds. Defaults to --train_file.")
    parser.add_argument("--cv_folds", default=10, type=int, help="Number of CV folds; paper uses 10 for GCDC.")
    parser.add_argument("--cv_metric", default="acc", choices=["acc", "f1"],
                        help="Validation metric used to select the best CV checkpoint/epoch. The paper reports Acc.")
    parser.add_argument("--cv_split_dir", default="", type=str,
                        help="Directory where generated CV train/dev jsonl files are saved.")
    parser.add_argument("--cv_select_epoch_strategy", default="mean", choices=["mean", "best_fold"],
                        help="How to choose the final full-data training epoch after CV.")
    parser.add_argument("--skip_final_train_after_cv", default=False, action="store_true",
                        help="Use the best CV fold checkpoint directly instead of retraining on the full GCDC training file.")

    # for dementia patient/control binary fine-tuning on top of a frozen coherence model
    parser.add_argument("--do_binary_train", default=False, action="store_true",
                        help="Train a patient/control binary classifier on top of a frozen coherence model.")
    parser.add_argument("--do_binary_test", default=False, action="store_true",
                        help="Run patient/control prediction with a trained binary classifier.")
    parser.add_argument("--binary_train_file", default="", type=str,
                        help="Dementia training json/jsonl file. Labels are read from source_label/source_group/score when possible.")
    parser.add_argument("--binary_dev_file", default="", type=str,
                        help="Optional dementia dev json/jsonl file. If omitted, --binary_dev_ratio is split from train.")
    parser.add_argument("--binary_checkpoint_file", default="", type=str,
                        help="Optional checkpoint for the frozen-coherence + binary-head model.")
    parser.add_argument("--binary_label_list", default="control,patient", type=str,
                        help="Binary labels in id order. Default makes control=0 and patient=1.")
    parser.add_argument("--binary_num_train_epochs", default=-1, type=int,
                        help="Epochs for binary dementia fine-tuning. Uses --num_train_epochs when <= 0.")
    parser.add_argument("--binary_learning_rate", default=-1.0, type=float,
                        help="Learning rate for binary dementia fine-tuning. Uses --learning_rate when <= 0.")
    parser.add_argument("--binary_dev_ratio", default=0.0, type=float,
                        help="Optional dev split ratio taken from the 80% dementia training split. Default 0 disables dev selection so the 20% holdout remains a true test set.")
    parser.add_argument("--binary_test_ratio", default=0.2, type=float,
                        help="Held-out dementia test ratio selected within each dialogue topic. Default 0.2 gives 80/20 per topic.")
    parser.add_argument("--binary_topic_field", default="dialogue_topic", type=str,
                        help="Field used for per-topic 80/20 dementia splitting. Falls back to topic, then unknown.")
    parser.add_argument("--binary_metric", default="f1", choices=["acc", "f1"],
                        help="Metric for selecting the binary head checkpoint when a binary dev split is used. Default f1 avoids majority-class selection.")
    parser.add_argument("--binary_loss_weighting", default="balanced", choices=["none", "balanced"],
                        help="Use inverse-frequency class weights for the binary loss. Default balanced helps prevent all-control collapse.")
    parser.add_argument("--binary_class_weight_power", default=1.0, type=float,
                        help="Power applied to balanced class weights. 1.0 = full inverse-frequency weighting; 0.5 = milder weighting.")
    parser.add_argument("--binary_head_hidden_size", default=128, type=int,
                        help="Hidden size of the new binary classifier head.")

    # for transformer encoder
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--scaled", default=True)

    return parser


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_model_type(model_type: str) -> str:
    key = model_type.lower().strip()
    alias_map = {
        "transformer_sent": "transformer_sent",
        "sent": "transformer_sent",
        "flat": "flat",
        "base": "flat",
    }
    if key not in alias_map:
        raise ValueError(
            "Unsupported model_type: {}. "
            "This train_dimentia.py matches the uploaded model.py and currently supports "
            "'transformer_sent'/'sent' and 'flat'/'base' only."
            .format(model_type)
        )
    return alias_map[key]


def build_model_inputs(args, batch, flag="Train"):
    model_type = normalize_model_type(args.model_type)
    if model_type == "transformer_sent":
        return {
            "sent_vectors": batch[1],
            "sent_mask": batch[3],
            "labels": batch[8],
            "flag": flag,
        }
    if model_type == "flat":
        return {
            "doc_vectors": batch[0],
            "labels": batch[8],
            "flag": flag,
        }
    raise ValueError(f"Unsupported model_type: {args.model_type}")


def get_dataloader(dataset, args, mode="train"):
    if mode.lower() == "train":
        sampler = RandomSampler(dataset)
        batch_size = args.train_batch_size
    else:
        sampler = SequentialSampler(dataset)
        batch_size = args.eval_batch_size

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        sampler=sampler
    )

    return data_loader


def get_trainable_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def get_optimizer(model, args, num_training_steps):
    trainable_named_params = get_trainable_parameters(model)
    if not trainable_named_params:
        raise ValueError("No trainable parameters found. Check that the binary classifier head was created correctly.")

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in trainable_named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in trainable_named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    # Drop empty groups, which can happen for very small heads.
    optimizer_grouped_parameters = [g for g in optimizer_grouped_parameters if len(g["params"]) > 0]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio),
        num_training_steps=num_training_steps
    )
    return optimizer, scheduler


def train(model, args, train_dataloader):
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs

    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    progress_desc = getattr(args, "progress_desc", "Training")
    with tqdm(total=t_total, desc=progress_desc, dynamic_ncols=True, leave=True) as progress_bar:
        for epoch in range(1, int(num_train_epochs) + 1):
            model.train()
            model.zero_grad()

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)

                inputs = build_model_inputs(args, batch, flag="Train")

                outputs = model(**inputs)
                loss = outputs[0]

                loss.backward()

                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            output_dir = os.path.join(args.output_dir, "good")
            output_dir = os.path.join(output_dir, f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    print("Training finished.")


def evaluate(model, args, dataloader, epoch, desc="dev", write_file=False):
    all_label_ids = None
    all_pred_ids = None
    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        inputs = build_model_inputs(args, batch, flag="Eval")
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        label_ids = batch[8].detach().cpu().numpy()
        pred_ids = preds.detach().cpu().numpy()
        if all_label_ids is None:
            all_label_ids = label_ids
            all_pred_ids = pred_ids
        else:
            all_label_ids = np.append(all_label_ids, label_ids)
            all_pred_ids = np.append(all_pred_ids, pred_ids)

    # print(all_label_ids.shape, all_pred_ids.shape)
    acc = accuracy_score(y_true=all_label_ids, y_pred=all_pred_ids)
    f1 = f1_score(y_true=all_label_ids, y_pred=all_pred_ids, average="macro")
    """
    res = classification_report(
        y_true=all_label_ids,
        y_pred=all_pred_ids,
        target_names=args.label_list
    )
    print(res)
    """

    if write_file:
        res_file = "{}_res_{}.txt".format(desc, args.model_type)
        data_dir = os.path.join(args.data_dir, "preds")
        os.makedirs(data_dir, exist_ok=True)
        res_file = os.path.join(data_dir, res_file)
        error_num = 0
        with open(res_file, "w", encoding="utf-8") as f:
            for l, p in zip(all_label_ids, all_pred_ids):
                if l == p:
                    f.write("%s\t%s\n" % (args.label_list[l], args.label_list[p]))
                else:
                    error_num += 1
                    f.write("%s\t%s\t%d\n" % (args.label_list[l], args.label_list[p], error_num))

    return acc, f1


def normalize_source_group(source_label):
    """Map dementia source labels to readable group names when possible."""
    key = str(source_label).strip().lower()
    if key in {"0", "control", "controls", "cn"}:
        return "control"
    if key in {"1", "patient", "patients", "dementia", "ad", "alzheimers", "alzheimer"}:
        return "patient"
    return key if key else "unknown"


def prepare_inference_jsonl(test_file, label_list):
    """
    Create a temp jsonl for inference where every item has a valid score label,
    while preserving source metadata for grouped counting.
    """
    dummy_label = label_list[0]
    raw_items = []
    temp_file = os.path.join(
        os.path.dirname(test_file),
        os.path.splitext(os.path.basename(test_file))[0] + "_inference_ready.jsonl"
    )

    with open(test_file, "r", encoding="utf-8") as fr, open(temp_file, "w", encoding="utf-8") as fw:
        for idx, line in enumerate(fr):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)

            source_label = str(item.get("source_label", "")).strip()
            if not source_label:
                item_id = str(item.get("id", ""))
                if "_" in item_id:
                    source_label = item_id.split("_", 1)[0]
                else:
                    source_label = "unknown"
            source_group = normalize_source_group(source_label)

            dialogue_topic = str(item.get("dialogue_topic", item.get("topic", "unknown"))).strip()
            if not dialogue_topic:
                dialogue_topic = "unknown"

            score = str(item.get("score", dummy_label)).lower().strip()
            if score not in label_list:
                item["score"] = dummy_label
            else:
                item["score"] = score

            item["source_label"] = source_label
            item["source_group"] = source_group
            item["dialogue_topic"] = dialogue_topic

            raw_items.append({
                "id": item.get("id", f"item_{idx}"),
                "source_label": source_label,
                "source_group": source_group,
                "dialogue_topic": dialogue_topic,
                "source_file": item.get("source_file", ""),
                "text": item.get("text", "")
            })
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")

    return temp_file, raw_items


def predict_only(model, args, dataloader, raw_items, desc="test"):
    all_pred_ids = []
    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        inputs = build_model_inputs(args, batch, flag="Eval")

        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]

        pred_ids = preds.detach().cpu().numpy().tolist()
        all_pred_ids.extend(pred_ids)

    if len(all_pred_ids) != len(raw_items):
        raise ValueError(f"Prediction count mismatch: got {len(all_pred_ids)} predictions for {len(raw_items)} input rows")

    # counts[dialogue_topic][source_group][predicted_label] = count
    counts = {}
    output_rows = []
    for item, pred_id in zip(raw_items, all_pred_ids):
        pred_label = args.label_list[int(pred_id)]
        source_label = str(item.get("source_label", "unknown"))
        source_group = normalize_source_group(item.get("source_group", source_label))
        dialogue_topic = str(item.get("dialogue_topic", "unknown")).strip() or "unknown"

        if dialogue_topic not in counts:
            counts[dialogue_topic] = {}
        if source_group not in counts[dialogue_topic]:
            counts[dialogue_topic][source_group] = {label: 0 for label in args.label_list}
        counts[dialogue_topic][source_group][pred_label] += 1

        output_rows.append({
            "id": item.get("id", ""),
            "source_label": source_label,
            "source_group": source_group,
            "dialogue_topic": dialogue_topic,
            "source_file": item.get("source_file", ""),
            "predicted_label": pred_label
        })

    prediction_output_file = args.prediction_output_file.strip()
    if not prediction_output_file:
        base = os.path.splitext(os.path.basename(args.test_file))[0]
        prediction_output_file = os.path.join(os.path.dirname(args.test_file), f"{base}_predictions.jsonl")

    output_dir = os.path.dirname(prediction_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(prediction_output_file, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved predictions to: {prediction_output_file}")
    print("\nPrediction counts by dialogue_topic: control vs patient")

    for dialogue_topic in sorted(counts.keys()):
        topic_counts = counts[dialogue_topic]
        control_counts = topic_counts.get("control", {label: 0 for label in args.label_list})
        patient_counts = topic_counts.get("patient", {label: 0 for label in args.label_list})

        print(f"\ndialogue_topic={dialogue_topic}")
        print(f"{'label':<15}{'control':>10}{'patient':>10}{'patient-control':>18}")

        control_total = 0
        patient_total = 0
        for label in args.label_list:
            control_value = control_counts.get(label, 0)
            patient_value = patient_counts.get(label, 0)
            control_total += control_value
            patient_total += patient_value
            print(f"{label:<15}{control_value:>10}{patient_value:>10}{patient_value - control_value:>18}")

        print(f"{'total':<15}{control_total:>10}{patient_total:>10}{patient_total - control_total:>18}")

        # Print any non-control/patient groups so they are not silently hidden.
        extra_groups = sorted(g for g in topic_counts.keys() if g not in {'control', 'patient'})
        for source_group in extra_groups:
            print(f"  Additional group: {source_group}")
            extra_total = 0
            for label in args.label_list:
                value = topic_counts[source_group].get(label, 0)
                extra_total += value
                print(f"    {label}: {value}")
            print(f"    total: {extra_total}")

    return counts, prediction_output_file



def load_json_records(data_file):
    """Load either JSONL, a JSON list, or a JSON object containing a list."""
    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file not found: {data_file}")

    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read().strip()

    if not raw_text:
        return []

    if raw_text[0] in ["[", "{"]:
        try:
            obj = json.loads(raw_text)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                for key in ["data", "records", "examples", "items"]:
                    if key in obj and isinstance(obj[key], list):
                        return obj[key]
                return [obj]
        except json.JSONDecodeError:
            pass

    records = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def save_jsonl_records(records, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_record_label(record):
    if isinstance(record, dict):
        for key in ["score", "labelA", "label", "target", "gold"]:
            if key in record and record[key] is not None:
                return str(record[key]).lower().strip()
    return "__missing__"


def clone_args(args, **overrides):
    cloned = argparse.Namespace(**vars(args))
    for key, value in overrides.items():
        setattr(cloned, key, value)
    return cloned


def get_checkpoint_file(args, epoch):
    return os.path.join(args.output_dir, "good", f"{PREFIX_CHECKPOINT_DIR}_{epoch}", "pytorch_model.bin")


def get_best_checkpoint_info_path(args):
    return os.path.join(args.output_dir, "good", "best_checkpoint.json")


def save_best_checkpoint_info(args, epoch, score, acc, f1):
    info_path = get_best_checkpoint_info_path(args)
    os.makedirs(os.path.dirname(info_path), exist_ok=True)
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": int(epoch),
                "metric": getattr(args, "cv_metric", "acc"),
                "best_score": float(score),
                "best_acc": float(acc),
                "best_f1": float(f1),
                "checkpoint_file": get_checkpoint_file(args, epoch),
            },
            f,
            indent=2,
        )


def build_fresh_model(args):
    args.model_type = normalize_model_type(args.model_type)
    if args.model_type == "transformer_sent":
        model = SentTransformer(args=args)
    elif args.model_type == "flat":
        model = BaseClassifer(args=args)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")
    return model.to(args.device)


def create_cv_split_files(args, cv_file):
    records = load_json_records(cv_file)
    if len(records) < 2:
        raise ValueError(f"Need at least 2 records for cross-validation; got {len(records)} from {cv_file}")

    n_splits = min(int(args.cv_folds), len(records))
    labels = [get_record_label(record) for record in records]

    from collections import Counter
    label_counts = Counter(labels)
    can_stratify = len(label_counts) > 1 and min(label_counts.values()) >= n_splits
    indices = np.arange(len(records))

    if can_stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices, labels)
        split_kind = "StratifiedKFold"
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(indices)
        split_kind = "KFold"

    split_dir = args.cv_split_dir.strip()
    if not split_dir:
        split_dir = os.path.join(os.path.dirname(cv_file), f"folds_{n_splits}cv")
    os.makedirs(split_dir, exist_ok=True)

    split_rows = []
    for fold_idx, (train_idx, dev_idx) in enumerate(split_iter, start=1):
        fold_dir = os.path.join(split_dir, f"fold_{fold_idx}")
        train_path = os.path.join(fold_dir, "train.jsonl")
        dev_path = os.path.join(fold_dir, "dev.jsonl")
        save_jsonl_records([records[i] for i in train_idx], train_path)
        save_jsonl_records([records[i] for i in dev_idx], dev_path)
        split_rows.append(
            {
                "fold": fold_idx,
                "train_file": train_path,
                "dev_file": dev_path,
                "train_size": int(len(train_idx)),
                "dev_size": int(len(dev_idx)),
            }
        )

    metadata_path = os.path.join(split_dir, "cv_split_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": cv_file,
                "n_records": len(records),
                "n_splits": n_splits,
                "splitter": split_kind,
                "label_counts": dict(label_counts),
                "folds": split_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved {n_splits}-fold CV files to: {split_dir}")
    return split_rows, metadata_path


def train_with_dev_selection(model, args, train_dataloader, dev_dataloader):
    t_total = int(len(train_dataloader) * args.num_train_epochs)
    num_train_epochs = args.num_train_epochs

    optimizer, scheduler = get_optimizer(model, args, t_total)
    logger.info("***** Running CV training *****")
    logger.info("  Num train examples = %d", len(train_dataloader.dataset))
    logger.info("  Num dev examples = %d", len(dev_dataloader.dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Batch size per device = %d", args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)

    best_score = float("-inf")
    best_epoch = 0
    best_acc = 0.0
    best_f1 = 0.0

    progress_desc = getattr(args, "progress_desc", f"Fold {getattr(args, 'fold_id', '?')}")
    with tqdm(total=t_total, desc=progress_desc, dynamic_ncols=True, leave=True) as progress_bar:
        for epoch in range(1, int(num_train_epochs) + 1):
            model.train()
            model.zero_grad()

            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(args.device) for t in batch)
                inputs = build_model_inputs(args, batch, flag="Train")
                outputs = model(**inputs)
                loss = outputs[0]

                loss.backward()

                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            output_dir = os.path.join(args.output_dir, "good", f"{PREFIX_CHECKPOINT_DIR}_{epoch}")
            os.makedirs(output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

            model.eval()
            dev_acc, dev_f1 = evaluate(model, args, dev_dataloader, epoch, desc="cv_dev", write_file=False)
            score = dev_f1 if getattr(args, "cv_metric", "acc").lower() == "f1" else dev_acc

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_acc = dev_acc
                best_f1 = dev_f1
                save_best_checkpoint_info(args, epoch=epoch, score=score, acc=dev_acc, f1=dev_f1)

            progress_bar.set_postfix(
                epoch=epoch,
                best_acc=f"{best_acc:.4f}",
                best_f1=f"{best_f1:.4f}",
            )

    best_checkpoint_file = get_checkpoint_file(args, best_epoch)
    return {
        "best_epoch": int(best_epoch),
        "best_score": float(best_score),
        "best_acc": float(best_acc),
        "best_f1": float(best_f1),
        "best_checkpoint": best_checkpoint_file,
    }


def print_cv_results_summary(fold_results, args, selected_epoch, best_fold):
    print("\nCV results summary:")
    print(f"{'fold':>4}{'best_epoch':>12}{'best_acc':>12}{'best_f1':>12}{'selected':>14}")
    for row in fold_results:
        selected_value = row['best_f1'] if getattr(args, "cv_metric", "acc").lower() == "f1" else row['best_acc']
        print(
            f"{row['fold']:>4}"
            f"{row['best_epoch']:>12}"
            f"{row['best_acc']:>12.4f}"
            f"{row['best_f1']:>12.4f}"
            f"{selected_value:>14.4f}"
        )
    print(
        "Best fold by %s: fold %d, epoch %d, Acc=%.4f, F1=%.4f"
        % (args.cv_metric, best_fold['fold'], best_fold['best_epoch'], best_fold['best_acc'], best_fold['best_f1'])
    )
    print("Selected full-training epoch from CV: %d" % selected_epoch)


def run_10fold_cv_model_selection(args, dataset_params):
    cv_file = args.cv_file.strip() if getattr(args, "cv_file", "").strip() else args.train_file
    if not cv_file or not os.path.exists(cv_file):
        raise FileNotFoundError(f"CV training file not found: {cv_file}")

    original_output_dir = args.output_dir
    split_rows, metadata_path = create_cv_split_files(args, cv_file)
    cv_root = os.path.join(original_output_dir, f"cv_{len(split_rows)}fold")
    os.makedirs(cv_root, exist_ok=True)

    fold_results = []
    for split in split_rows:
        fold_idx = split["fold"]
        fold_output_dir = os.path.join(cv_root, f"fold_{fold_idx}")
        fold_args = clone_args(
            args,
            output_dir=fold_output_dir,
            train_file=split["train_file"],
            test_file=split["dev_file"],
            fold_id=fold_idx,
            progress_desc=f"Fold {fold_idx}/{len(split_rows)}",
        )
        set_seed(args.seed + fold_idx)
        model = build_fresh_model(fold_args)
        train_dataset = SentDataset(split["train_file"], params=dataset_params)
        dev_dataset = SentDataset(split["dev_file"], params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, fold_args, mode="train")
        dev_dataloader = get_dataloader(dev_dataset, fold_args, mode="dev")
        result = train_with_dev_selection(model, fold_args, train_dataloader, dev_dataloader)
        result.update({
            "fold": fold_idx,
            "train_file": split["train_file"],
            "dev_file": split["dev_file"],
            "train_size": split["train_size"],
            "dev_size": split["dev_size"],
            "output_dir": fold_output_dir,
        })
        fold_results.append(result)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_key = "best_f1" if getattr(args, "cv_metric", "acc").lower() == "f1" else "best_acc"
    best_fold = max(fold_results, key=lambda row: row[metric_key])
    best_epochs = [row["best_epoch"] for row in fold_results if row["best_epoch"] > 0]
    if getattr(args, "cv_select_epoch_strategy", "mean") == "best_fold":
        selected_epoch = int(best_fold["best_epoch"])
    else:
        selected_epoch = int(round(float(np.mean(best_epochs)))) if best_epochs else int(args.num_train_epochs)
    selected_epoch = max(1, min(int(args.num_train_epochs), selected_epoch))

    summary = {
        "paper_process": "10-fold CV over the full GCDC training file for model selection, then dementia prediction.",
        "cv_file": cv_file,
        "split_metadata": metadata_path,
        "cv_metric": args.cv_metric,
        "cv_select_epoch_strategy": args.cv_select_epoch_strategy,
        "selected_epoch_for_full_training": selected_epoch,
        "best_fold_by_metric": best_fold,
        "folds": fold_results,
    }
    summary_path = os.path.join(cv_root, "cv_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print_cv_results_summary(fold_results, args, selected_epoch, best_fold)
    print(f"\nSaved CV summary to: {summary_path}")

    if getattr(args, "skip_final_train_after_cv", False):
        print("Skipping final full-data training; using the best CV fold checkpoint directly.")
        return best_fold["best_checkpoint"]

    print("\n========== Final training on the full GCDC training file ==========")
    final_output_dir = os.path.join(original_output_dir, "final_full_train")
    final_args = clone_args(
        args,
        output_dir=final_output_dir,
        train_file=cv_file,
        num_train_epochs=selected_epoch,
        fold_id=-1,
        progress_desc="Final train",
    )
    set_seed(args.seed)
    final_model = build_fresh_model(final_args)
    final_train_dataset = SentDataset(cv_file, params=dataset_params)
    final_train_dataloader = get_dataloader(final_train_dataset, final_args, mode="train")
    train(final_model, final_args, final_train_dataloader)
    final_checkpoint = get_checkpoint_file(final_args, selected_epoch)
    print(f"Final full-data checkpoint selected for dementia prediction: {final_checkpoint}")
    del final_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return final_checkpoint



def parse_label_list(label_list_value):
    if isinstance(label_list_value, (list, tuple)):
        return [str(v).lower().strip() for v in label_list_value if str(v).strip()]
    return [v.lower().strip() for v in str(label_list_value).split(",") if v.strip()]


def infer_binary_label_from_record(record):
    """Infer control/patient label from common dementia dataset fields."""
    if not isinstance(record, dict):
        return None

    # Prefer explicit binary/group labels over coherence score.
    candidate_keys = [
        "binary_label", "source_group", "source_label", "diagnosis",
        "group", "class", "target", "label", "gold", "score"
    ]
    for key in candidate_keys:
        if key not in record or record[key] is None:
            continue
        normalized = normalize_source_group(record[key])
        if normalized in {"control", "patient"}:
            return normalized

    item_id = str(record.get("id", "")).strip()
    if item_id:
        prefix = item_id.split("_", 1)[0]
        normalized = normalize_source_group(prefix)
        if normalized in {"control", "patient"}:
            return normalized

    return None


def prepare_binary_dementia_jsonl(data_file, binary_label_list, output_file=None, require_label=False):
    """
    Convert a dementia json/jsonl file into the format expected by SentDataset.
    The converted `score` is `control` or `patient`; control=0 and patient=1
    when using the default --binary_label_list "control,patient".
    """
    if not data_file or not os.path.exists(data_file):
        raise FileNotFoundError(f"Dementia binary data file not found: {data_file}")

    binary_label_list = parse_label_list(binary_label_list)
    if set(binary_label_list) != {"control", "patient"}:
        raise ValueError(
            "--binary_label_list must contain exactly control and patient. "
            f"Got: {binary_label_list}"
        )

    records = load_json_records(data_file)
    converted_records = []
    raw_items = []
    missing_label_examples = []
    dummy_label = binary_label_list[0]

    for idx, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        item = dict(record)
        binary_label = infer_binary_label_from_record(item)
        if binary_label is None:
            if require_label:
                missing_label_examples.append(item.get("id", f"row_{idx}"))
                continue
            binary_label = dummy_label

        item["score"] = binary_label
        item["source_group"] = normalize_source_group(item.get("source_group", binary_label))
        if not item.get("source_label"):
            item["source_label"] = binary_label
        if not item.get("dialogue_topic"):
            item["dialogue_topic"] = item.get("topic", "unknown") or "unknown"

        converted_records.append(item)
        raw_items.append({
            "id": item.get("id", f"item_{idx}"),
            "source_label": item.get("source_label", ""),
            "source_group": normalize_source_group(item.get("source_group", binary_label)),
            "true_label": binary_label if infer_binary_label_from_record(record) is not None else "",
            "dialogue_topic": str(item.get("dialogue_topic", "unknown")).strip() or "unknown",
            "source_file": item.get("source_file", ""),
            "text": item.get("text", ""),
        })

    if missing_label_examples:
        preview = ", ".join(map(str, missing_label_examples[:10]))
        raise ValueError(
            f"Could not infer control/patient labels for {len(missing_label_examples)} rows "
            f"in {data_file}. Examples: {preview}. Add source_label/source_group/score with "
            "values control/patient or 0/1."
        )

    if not converted_records:
        raise ValueError(f"No usable records found in {data_file}")

    if output_file is None:
        base = os.path.splitext(os.path.basename(data_file))[0]
        output_file = os.path.join(os.path.dirname(data_file), f"{base}_binary_ready.jsonl")

    save_jsonl_records(converted_records, output_file)
    return output_file, raw_items


def get_dialogue_topic_from_record(record, topic_field="dialogue_topic"):
    """Return the dialogue topic used for the required per-topic 80/20 split."""
    if not isinstance(record, dict):
        return "unknown"
    candidate_keys = []
    if topic_field:
        candidate_keys.append(topic_field)
    candidate_keys.extend(["dialogue_topic", "topic"])
    seen = set()
    for key in candidate_keys:
        if key in seen:
            continue
        seen.add(key)
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def summarize_binary_records(records):
    """Summarize counts by dialogue topic and binary label."""
    from collections import Counter, defaultdict
    summary = defaultdict(Counter)
    for record in records:
        topic = get_dialogue_topic_from_record(record)
        label = normalize_source_group(record.get("score", record.get("source_group", "unknown")))
        summary[topic][label] += 1
    return summary


def print_binary_topic_split_summary(train_records, test_records, metadata_path):
    train_summary = summarize_binary_records(train_records)
    test_summary = summarize_binary_records(test_records)
    topics = sorted(set(train_summary.keys()) | set(test_summary.keys()))
    print("\nDementia binary 80/20 split by dialogue_topic:")
    print(f"{'topic':<30}{'train':>8}{'test':>8}{'train_control':>16}{'train_patient':>16}{'test_control':>14}{'test_patient':>14}")
    for topic in topics:
        tr = train_summary.get(topic, {})
        te = test_summary.get(topic, {})
        train_total = sum(tr.values())
        test_total = sum(te.values())
        print(
            f"{topic:<30}{train_total:>8}{test_total:>8}"
            f"{tr.get('control', 0):>16}{tr.get('patient', 0):>16}"
            f"{te.get('control', 0):>14}{te.get('patient', 0):>14}"
        )
    print(f"Saved split metadata to: {metadata_path}")


def split_binary_records_by_dialogue_topic(records, test_ratio=0.2, seed=106524, topic_field="dialogue_topic"):
    """
    Split dementia records so every dialogue topic contributes approximately
    80% to binary training and 20% to held-out testing.

    For each topic with at least 2 rows, at least 1 row is held out for test.
    Very small one-row topics are kept in train because they cannot be split.
    When feasible, the split is additionally stratified by control/patient
    within the topic; otherwise it falls back to deterministic shuffling.
    """
    from collections import Counter, defaultdict

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for record in records:
        topic = get_dialogue_topic_from_record(record, topic_field=topic_field)
        record["dialogue_topic"] = topic
        grouped[topic].append(record)

    train_records = []
    test_records = []
    split_metadata = []

    for topic in sorted(grouped.keys()):
        topic_records = list(grouped[topic])
        n_total = len(topic_records)
        if n_total <= 1 or test_ratio <= 0:
            train_part = topic_records
            test_part = []
            split_method = "all_train_too_few_rows"
        else:
            n_test = int(round(n_total * float(test_ratio)))
            n_test = max(1, min(n_total - 1, n_test))
            labels = [normalize_source_group(r.get("score", r.get("source_group", "unknown"))) for r in topic_records]
            label_counts = Counter(labels)
            n_classes = len(label_counts)
            can_stratify = (
                n_classes > 1
                and min(label_counts.values()) >= 2
                and n_test >= n_classes
                and (n_total - n_test) >= n_classes
            )
            if can_stratify:
                train_part, test_part = train_test_split(
                    topic_records,
                    test_size=n_test,
                    random_state=seed,
                    stratify=labels,
                )
                split_method = "topic_plus_label_stratified"
            else:
                shuffled = list(topic_records)
                rng.shuffle(shuffled)
                test_part = shuffled[:n_test]
                train_part = shuffled[n_test:]
                split_method = "topic_only_shuffled"

        train_records.extend(train_part)
        test_records.extend(test_part)
        split_metadata.append(
            {
                "dialogue_topic": topic,
                "total": int(n_total),
                "train": int(len(train_part)),
                "test": int(len(test_part)),
                "train_label_counts": dict(Counter(normalize_source_group(r.get("score", r.get("source_group", "unknown"))) for r in train_part)),
                "test_label_counts": dict(Counter(normalize_source_group(r.get("score", r.get("source_group", "unknown"))) for r in test_part)),
                "split_method": split_method,
            }
        )

    rng.shuffle(train_records)
    rng.shuffle(test_records)
    return train_records, test_records, split_metadata


def make_binary_train_dev_test_files(args, binary_label_list):
    binary_root = os.path.join(args.output_dir, "binary_dementia")
    prepared_dir = os.path.join(binary_root, "prepared")
    split_dir = os.path.join(binary_root, "topic_80_20_split")
    os.makedirs(prepared_dir, exist_ok=True)
    os.makedirs(split_dir, exist_ok=True)

    binary_train_file = args.binary_train_file.strip()
    if not binary_train_file:
        # Allows quick experiments on a labeled dementia file previously passed as --test_file.
        binary_train_file = args.test_file.strip()
    if not binary_train_file:
        raise ValueError("Provide --binary_train_file for dementia patient/control fine-tuning.")

    all_ready_file = os.path.join(
        prepared_dir,
        os.path.splitext(os.path.basename(binary_train_file))[0] + "_binary_all_ready.jsonl"
    )
    all_ready_file, _ = prepare_binary_dementia_jsonl(
        binary_train_file,
        binary_label_list,
        output_file=all_ready_file,
        require_label=True,
    )

    all_records = load_json_records(all_ready_file)
    test_ratio = float(getattr(args, "binary_test_ratio", 0.2))
    train_records, heldout_test_records, split_metadata = split_binary_records_by_dialogue_topic(
        all_records,
        test_ratio=test_ratio,
        seed=args.seed,
        topic_field=getattr(args, "binary_topic_field", "dialogue_topic"),
    )

    if not train_records:
        raise ValueError("The 80% dementia training split is empty. Check --binary_train_file and --binary_test_ratio.")
    if not heldout_test_records:
        raise ValueError(
            "The 20% dementia test split is empty. Need at least two rows in at least one dialogue_topic "
            "or lower --binary_test_ratio."
        )

    train_ready_file = os.path.join(split_dir, "train_binary_topic80.jsonl")
    heldout_test_file = os.path.join(split_dir, "test_binary_topic20.jsonl")
    save_jsonl_records(train_records, train_ready_file)
    save_jsonl_records(heldout_test_records, heldout_test_file)

    metadata_path = os.path.join(split_dir, "topic_80_20_split_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source_file": binary_train_file,
                "all_ready_file": all_ready_file,
                "topic_field": getattr(args, "binary_topic_field", "dialogue_topic"),
                "test_ratio": test_ratio,
                "train_file": train_ready_file,
                "test_file": heldout_test_file,
                "n_total": len(all_records),
                "n_train": len(train_records),
                "n_test": len(heldout_test_records),
                "topics": split_metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print_binary_topic_split_summary(train_records, heldout_test_records, metadata_path)

    dev_ready_file = ""
    if args.binary_dev_file.strip():
        dev_ready_file = os.path.join(
            prepared_dir,
            os.path.splitext(os.path.basename(args.binary_dev_file.strip()))[0] + "_binary_dev_ready.jsonl"
        )
        dev_ready_file, _ = prepare_binary_dementia_jsonl(
            args.binary_dev_file.strip(),
            binary_label_list,
            output_file=dev_ready_file,
            require_label=True,
        )
    else:
        # Optional dev selection comes only from the 80% training portion.
        # The 20% topic-balanced split remains untouched for final testing.
        dev_ratio = float(getattr(args, "binary_dev_ratio", 0.0))
        if dev_ratio > 0.0 and len(train_records) >= 3:
            labels = [record["score"] for record in train_records]
            from collections import Counter
            label_counts = Counter(labels)
            stratify = labels if len(label_counts) > 1 and min(label_counts.values()) >= 2 else None
            final_train_records, dev_records = train_test_split(
                train_records,
                test_size=dev_ratio,
                random_state=args.seed,
                stratify=stratify,
            )
            train_ready_file = os.path.join(split_dir, "train_binary_topic80_minus_dev.jsonl")
            dev_ready_file = os.path.join(split_dir, "dev_binary_from_topic80.jsonl")
            save_jsonl_records(final_train_records, train_ready_file)
            save_jsonl_records(dev_records, dev_ready_file)
            print(
                f"Optional binary dev split from the 80% training portion: "
                f"train={len(final_train_records)}, dev={len(dev_records)}, heldout_test={len(heldout_test_records)}"
            )

    return train_ready_file, dev_ready_file, heldout_test_file


def get_latest_checkpoint_from_output_dir(output_dir):
    ckpt_root = os.path.join(output_dir, "good")
    if not os.path.exists(ckpt_root):
        return ""
    ckpts = [d for d in os.listdir(ckpt_root) if d.startswith(PREFIX_CHECKPOINT_DIR + "_")]
    if not ckpts:
        return ""
    ckpts.sort(key=lambda x: int(x.split("_")[-1]))
    return os.path.join(ckpt_root, ckpts[-1], "pytorch_model.bin")


def resolve_coherence_checkpoint(args, selected_checkpoint=""):
    for candidate in [
        getattr(args, "checkpoint_file", "").strip(),
        selected_checkpoint,
        get_latest_checkpoint_from_output_dir(args.output_dir),
    ]:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a coherence checkpoint. Run --do_cv/--do_train first or pass "
        "--checkpoint_file pointing to the trained GCDC coherence model."
    )



def compute_binary_class_weights_from_file(train_file, binary_label_list, power=1.0):
    """
    Return inverse-frequency class weights in label-list order.
    Formula before power: total / (num_classes * class_count).
    """
    records = load_json_records(train_file)
    labels = [str(record.get("score", "")).lower().strip() for record in records]
    from collections import Counter
    counts = Counter(labels)
    total = sum(counts.get(label, 0) for label in binary_label_list)
    if total <= 0:
        return None, counts

    weights = []
    num_classes = len(binary_label_list)
    for label in binary_label_list:
        count = counts.get(label, 0)
        if count <= 0:
            # Avoid inf; this will also make the missing class obvious in the printed counts.
            weights.append(1.0)
        else:
            weights.append((total / float(num_classes * count)) ** float(power))
    return weights, counts


def print_binary_train_label_summary(train_file, binary_label_list):
    records = load_json_records(train_file)
    from collections import Counter, defaultdict
    counts = Counter(str(record.get("score", "")).lower().strip() for record in records)
    topic_counts = defaultdict(Counter)
    for record in records:
        topic = str(record.get("dialogue_topic", record.get("topic", "unknown"))).strip() or "unknown"
        label = str(record.get("score", "")).lower().strip()
        topic_counts[topic][label] += 1

    print("\nBinary training label counts:")
    print(f"{'label':<15}{'count':>10}")
    for label in binary_label_list:
        print(f"{label:<15}{counts.get(label, 0):>10}")
    print(f"{'total':<15}{sum(counts.get(label, 0) for label in binary_label_list):>10}")

    print("\nBinary training label counts by dialogue_topic:")
    header = f"{'dialogue_topic':<25}" + "".join(f"{label:>12}" for label in binary_label_list) + f"{'total':>12}"
    print(header)
    for topic in sorted(topic_counts.keys()):
        total = sum(topic_counts[topic].get(label, 0) for label in binary_label_list)
        print(f"{topic:<25}" + "".join(f"{topic_counts[topic].get(label, 0):>12}" for label in binary_label_list) + f"{total:>12}")



def build_binary_args(args, binary_label_list):
    binary_args = clone_args(args)
    binary_args.label_list = parse_label_list(binary_label_list)
    binary_args.num_labels = len(binary_args.label_list)
    binary_args.output_dir = os.path.join(args.output_dir, "binary_dementia")
    binary_args.num_train_epochs = (
        int(args.binary_num_train_epochs) if int(getattr(args, "binary_num_train_epochs", -1)) > 0
        else int(args.num_train_epochs)
    )
    binary_args.learning_rate = (
        float(args.binary_learning_rate) if float(getattr(args, "binary_learning_rate", -1.0)) > 0.0
        else float(args.learning_rate)
    )
    binary_args.progress_desc = "Binary dementia"
    binary_args.cv_metric = getattr(args, "binary_metric", "f1")
    os.makedirs(binary_args.output_dir, exist_ok=True)
    return binary_args


def build_frozen_binary_model(args, coherence_checkpoint, binary_args):
    coherence_args = clone_args(
        args,
        label_list=list(getattr(args, "coherence_label_list", args.label_list)),
        num_labels=len(getattr(args, "coherence_label_list", args.label_list)),
    )
    coherence_model = build_fresh_model(coherence_args)
    if coherence_checkpoint and os.path.exists(coherence_checkpoint):
        print(f"Loading frozen coherence checkpoint: {coherence_checkpoint}")
        coherence_model.load_state_dict(torch.load(coherence_checkpoint, map_location=args.device), strict=False)
    for param in coherence_model.parameters():
        param.requires_grad = False
    coherence_model.eval()
    return EnhancedFrozenCoherenceBinaryClassifier(coherence_model, binary_args).to(args.device)


def train_binary_classifier(args, dataset_params, selected_coherence_checkpoint=""):
    binary_label_list = parse_label_list(args.binary_label_list)
    binary_args = build_binary_args(args, binary_label_list)
    train_ready_file, dev_ready_file, heldout_test_file = make_binary_train_dev_test_files(args, binary_label_list)
    coherence_checkpoint = resolve_coherence_checkpoint(args, selected_coherence_checkpoint)

    binary_dataset_params = dict(dataset_params)
    binary_dataset_params["label_list"] = binary_label_list

    print_binary_train_label_summary(train_ready_file, binary_label_list)
    if getattr(args, "binary_loss_weighting", "balanced") == "balanced":
        class_weights, label_counts = compute_binary_class_weights_from_file(
            train_ready_file,
            binary_label_list,
            power=float(getattr(args, "binary_class_weight_power", 1.0)),
        )
        binary_args.binary_class_weights = class_weights
        print("\nUsing balanced binary loss weights:")
        for label, weight in zip(binary_label_list, class_weights):
            print(f"  {label}: weight={float(weight):.4f}, train_count={int(label_counts.get(label, 0))}")
    else:
        binary_args.binary_class_weights = None
        print("\nUsing unweighted binary CrossEntropyLoss.")

    binary_model = build_frozen_binary_model(args, coherence_checkpoint, binary_args)
    trainable_params = sum(p.numel() for p in binary_model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in binary_model.parameters() if not p.requires_grad)
    print(f"Trainable binary-head parameters: {trainable_params:,}")
    print(f"Frozen coherence-model parameters: {frozen_params:,}")
    print(f"Binary training file 80% topic split: {train_ready_file}")
    print(f"Binary held-out test file 20% topic split: {heldout_test_file}")

    train_dataset = SentDataset(train_ready_file, params=binary_dataset_params)
    train_dataloader = get_dataloader(train_dataset, binary_args, mode="train")

    if dev_ready_file:
        dev_dataset = SentDataset(dev_ready_file, params=binary_dataset_params)
        dev_dataloader = get_dataloader(dev_dataset, binary_args, mode="dev")
        result = train_with_dev_selection(binary_model, binary_args, train_dataloader, dev_dataloader)
        binary_checkpoint = result["best_checkpoint"]
    else:
        train(binary_model, binary_args, train_dataloader)
        binary_checkpoint = get_latest_checkpoint_from_output_dir(binary_args.output_dir)

    print(f"Binary dementia checkpoint selected: {binary_checkpoint}")
    return binary_checkpoint, binary_args, heldout_test_file


def _safe_binary_metric(y_true, y_pred, label_to_id, label_name):
    label_id = label_to_id[label_name]
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label_id and p == label_id)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label_id and p != label_id)
    denom = tp + fn
    return float(tp) / float(denom) if denom > 0 else float("nan")


def print_binary_results_by_dialogue_topic(output_rows, binary_label_list):
    from collections import defaultdict, Counter

    label_to_id = {label: idx for idx, label in enumerate(binary_label_list)}
    by_topic = defaultdict(list)
    for row in output_rows:
        true_label = normalize_source_group(row.get("true_label", ""))
        pred_label = normalize_source_group(row.get("predicted_label", ""))
        topic = str(row.get("dialogue_topic", "unknown")).strip() or "unknown"
        if true_label in label_to_id and pred_label in label_to_id:
            by_topic[topic].append((true_label, pred_label))

    if not by_topic:
        print("\nNo known true labels available for per-dialogue-topic binary metrics.")
        return

    print("\nBinary prediction results by dialogue_topic")
    print(
        f"{'dialogue_topic':<25}"
        f"{'true_control':>13}{'true_patient':>13}"
        f"{'pred_control':>13}{'pred_patient':>13}"
        f"{'acc':>9}{'macro_f1':>10}"
        f"{'control_rec':>13}{'patient_rec':>13}"
    )

    summary_rows = []
    for topic in sorted(by_topic.keys()):
        pairs = by_topic[topic]
        y_true = [label_to_id[t] for t, p in pairs]
        y_pred = [label_to_id[p] for t, p in pairs]

        true_counts = Counter(t for t, p in pairs)
        pred_counts = Counter(p for t, p in pairs)
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        macro_f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", labels=list(range(len(binary_label_list))), zero_division=0)
        control_rec = _safe_binary_metric(y_true, y_pred, label_to_id, "control")
        patient_rec = _safe_binary_metric(y_true, y_pred, label_to_id, "patient")
        print(
            f"{topic:<25}"
            f"{true_counts.get('control', 0):>13}{true_counts.get('patient', 0):>13}"
            f"{pred_counts.get('control', 0):>13}{pred_counts.get('patient', 0):>13}"
            f"{acc:>9.4f}{macro_f1:>10.4f}"
            f"{control_rec:>13.4f}{patient_rec:>13.4f}"
        )
        summary_rows.append({
            "dialogue_topic": topic,
            "true_control": int(true_counts.get("control", 0)),
            "true_patient": int(true_counts.get("patient", 0)),
            "pred_control": int(pred_counts.get("control", 0)),
            "pred_patient": int(pred_counts.get("patient", 0)),
            "acc": float(acc),
            "macro_f1": float(macro_f1),
            "control_recall": float(control_rec),
            "patient_recall": float(patient_rec),
        })
    return summary_rows


def predict_binary_only(model, args, dataloader, raw_items, desc="binary_test"):
    all_pred_ids = []
    all_prob_rows = []
    all_coherence_prob_rows = []
    all_true_ids = []
    label_to_id = {label: idx for idx, label in enumerate(args.label_list)}
    coherence_label_list = list(getattr(args, "coherence_label_list", []))

    for batch in dataloader:
        batch = tuple(t.to(args.device) for t in batch)
        inputs = build_model_inputs(args, batch, flag="Eval")
        with torch.no_grad():
            outputs = model(**inputs)
            preds = outputs[0]
            binary_probs = outputs[2] if len(outputs) > 2 else None
            coherence_probs = outputs[3] if len(outputs) > 3 else None

        all_pred_ids.extend(preds.detach().cpu().numpy().tolist())
        if binary_probs is not None:
            all_prob_rows.extend(binary_probs.detach().cpu().numpy().tolist())
        else:
            all_prob_rows.extend([[float("nan")] * len(args.label_list) for _ in range(preds.size(0))])
        if coherence_probs is not None:
            coherence_rows = coherence_probs.detach().cpu().numpy().tolist()
            all_coherence_prob_rows.extend(coherence_rows)
            if not coherence_label_list and coherence_rows:
                coherence_label_list = [f"coherence_{i}" for i in range(len(coherence_rows[0]))]
        else:
            all_coherence_prob_rows.extend([])

    if len(all_pred_ids) != len(raw_items):
        raise ValueError(f"Prediction count mismatch: got {len(all_pred_ids)} predictions for {len(raw_items)} input rows")

    output_rows = []
    counts = {"control": {label: 0 for label in args.label_list}, "patient": {label: 0 for label in args.label_list}, "unknown": {label: 0 for label in args.label_list}}
    if len(all_coherence_prob_rows) != len(all_pred_ids):
        all_coherence_prob_rows = [[float("nan")] * len(coherence_label_list) for _ in all_pred_ids]

    for item, pred_id, prob_row, coherence_prob_row in zip(raw_items, all_pred_ids, all_prob_rows, all_coherence_prob_rows):
        pred_label = args.label_list[int(pred_id)]
        true_label = normalize_source_group(item.get("true_label", ""))
        source_group = normalize_source_group(item.get("source_group", true_label))
        count_group = true_label if true_label in {"control", "patient"} else "unknown"
        counts[count_group][pred_label] += 1
        if true_label in label_to_id:
            all_true_ids.append(label_to_id[true_label])
        else:
            all_true_ids.append(None)

        row = {
            "id": item.get("id", ""),
            "source_label": item.get("source_label", ""),
            "source_group": source_group,
            "true_label": true_label if true_label in {"control", "patient"} else "",
            "dialogue_topic": item.get("dialogue_topic", "unknown"),
            "source_file": item.get("source_file", ""),
            "predicted_label": pred_label,
            "predicted_id": int(pred_id),
        }
        for label, prob in zip(args.label_list, prob_row):
            row[f"prob_{label}"] = float(prob)
        for label, prob in zip(coherence_label_list, coherence_prob_row):
            row[f"coherence_prob_{label}"] = float(prob)
        output_rows.append(row)

    prediction_output_file = args.prediction_output_file.strip()
    if not prediction_output_file:
        base = os.path.splitext(os.path.basename(args.test_file))[0]
        prediction_output_file = os.path.join(os.path.dirname(args.test_file), f"{base}_binary_predictions.jsonl")

    output_dir = os.path.dirname(prediction_output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(prediction_output_file, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nSaved binary predictions to: {prediction_output_file}")
    print("\nBinary prediction counts by true/source group")
    print(f"{'true_group':<15}" + "".join(f"{label:>12}" for label in args.label_list) + f"{'total':>12}")
    for group in ["control", "patient", "unknown"]:
        total = sum(counts[group].values())
        if total == 0:
            continue
        print(f"{group:<15}" + "".join(f"{counts[group][label]:>12}" for label in args.label_list) + f"{total:>12}")

    known_pairs = [(true_id, pred_id) for true_id, pred_id in zip(all_true_ids, all_pred_ids) if true_id is not None]
    if known_pairs:
        y_true = [p[0] for p in known_pairs]
        y_pred = [p[1] for p in known_pairs]
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro", labels=list(range(len(args.label_list))), zero_division=0)
        print(f"\nBinary test Acc={acc:.4f}, macro-F1={f1:.4f}")

        topic_summary_rows = print_binary_results_by_dialogue_topic(output_rows, args.label_list)
        if topic_summary_rows:
            topic_summary_file = os.path.splitext(prediction_output_file)[0] + "_by_dialogue_topic.json"
            with open(topic_summary_file, "w", encoding="utf-8") as f:
                json.dump(topic_summary_rows, f, indent=2, ensure_ascii=False)
            print(f"Saved per-topic binary summary to: {topic_summary_file}")

    return counts, prediction_output_file


def run_binary_prediction(args, dataset_params, binary_checkpoint_file="", selected_coherence_checkpoint="", test_file_override=""):
    binary_label_list = parse_label_list(args.binary_label_list)
    binary_args = build_binary_args(args, binary_label_list)
    binary_args.num_train_epochs = int(args.binary_num_train_epochs) if int(getattr(args, "binary_num_train_epochs", -1)) > 0 else int(args.num_train_epochs)

    coherence_checkpoint = ""
    try:
        coherence_checkpoint = resolve_coherence_checkpoint(args, selected_coherence_checkpoint)
    except FileNotFoundError:
        # A full binary checkpoint contains coherence_model.* weights, so this can still work.
        coherence_checkpoint = ""

    binary_model = build_frozen_binary_model(args, coherence_checkpoint, binary_args)
    checkpoint = binary_checkpoint_file or args.binary_checkpoint_file.strip() or get_latest_checkpoint_from_output_dir(binary_args.output_dir)
    if not checkpoint or not os.path.exists(checkpoint):
        raise FileNotFoundError(
            "Could not find a binary classifier checkpoint. Run --do_binary_train first or pass --binary_checkpoint_file."
        )
    print(f"Loading binary checkpoint: {checkpoint}")
    binary_model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=False)
    binary_model.eval()

    binary_dataset_params = dict(dataset_params)
    binary_dataset_params["label_list"] = binary_label_list

    source_test_file = test_file_override or args.test_file
    if not source_test_file:
        raise ValueError("No binary test file is available. Pass --test_file or train with the automatic 80/20 topic split.")
    test_ready_file = os.path.join(
        binary_args.output_dir,
        "prepared",
        os.path.splitext(os.path.basename(source_test_file))[0] + "_binary_test_ready.jsonl"
    )
    os.makedirs(os.path.dirname(test_ready_file), exist_ok=True)
    print(f"Using binary test file: {source_test_file}")
    test_ready_file, raw_items = prepare_binary_dementia_jsonl(
        source_test_file,
        binary_label_list,
        output_file=test_ready_file,
        require_label=False,
    )
    test_dataset = SentDataset(test_ready_file, params=binary_dataset_params)
    test_dataloader = get_dataloader(test_dataset, binary_args, mode="test")
    return predict_binary_only(binary_model, binary_args, test_dataloader, raw_items, desc="binary_test")


def main():
    args = get_argparse().parse_args()
    if torch.cuda.is_available():
        args.n_gpu = 1
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        args.n_gpu = 0
    args.device = device
    logger.info(" Training/evaluation parameters %s", args)
    logger.info(" ####### Acc, F1 for fold %d ########", args.fold_id)
    # print("Training/evaluation parameters %s", args)
    set_seed(args.seed)

    ## 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    saved_file = "rel_embed.bin"
    args.saved_embed_file = os.path.join(data_dir, saved_file)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    args.data_dir = data_dir
    output_dir = os.path.join(args.output_dir, args.dataset)
    output_dir = os.path.join(
        output_dir,
        "fast_flat_{}+{}".format(args.model_type, args.model_name_or_path.split("/")[-1])
    )
    if args.fold_id > 0:
        output_dir = os.path.join(output_dir, str(args.fold_id))
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    exp_rel_list = labels_from_file("data/parser/pdtb3/exp/l1/labels.txt")
    imp_rel_list = labels_from_file("data/parser/pdtb3/imp/l1/labels.txt")
    rel_list = set()
    _ = [rel_list.add(l) for l in exp_rel_list]
    _ = [rel_list.add(l) for l in imp_rel_list]
    rel_list = list(rel_list)
    rel_list = sorted(rel_list)
    args.rel_list = rel_list
    label_list = args.label_list.split(",")
    label_list = [l.lower().strip() for l in label_list]
    args.label_list = label_list
    args.coherence_label_list = list(label_list)
    args.num_labels = len(label_list)

    # args.embed_file = os.path.join(
    #     args.embed_file
    # )
    #
    # ## 2. define models
    config = AutoConfig.from_pretrained(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if "llama" in args.model_name_or_path:
        """
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
            torch_dtype=torch.bfloat16
        ) 
        """
        encoder = None
    else:
        encoder = AutoModel.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
    if encoder is not None:
        for name, param in encoder.named_parameters():
            param.requires_grad = False
        encoder.eval()
        encoder = encoder.to(args.device)

    config.HP_dropout = args.dropout
    args.input_size = config.hidden_size
    args.model_type = normalize_model_type(args.model_type)

    if args.model_type == "transformer_sent":
        model = SentTransformer(args=args)
    elif args.model_type == "flat":
        model = BaseClassifer(args=args)
    else:
        raise ValueError(f"Unsupported model_type: {args.model_type}")

    model = model.to(args.device)

    ## 3. prepare dataset
    dataset_params = {
        "tokenizer": tokenizer,
        "encoder": encoder,
        "pooler_type": args.pooler_type,
        "max_text_length": args.max_text_length,
        "max_arg_num": args.max_arg_num,
        "max_sent_num": args.max_sent_num,
        "label_list": label_list,
        "rel_list": rel_list,
    }

    selected_checkpoint = ""
    if args.do_cv:
        selected_checkpoint = run_10fold_cv_model_selection(args, dataset_params)
        if selected_checkpoint:
            args.checkpoint_file = selected_checkpoint
            print(f"Using frozen coherence checkpoint for downstream dementia classifier: {args.checkpoint_file}")
    elif args.do_train:
        args.progress_desc = "Training"
        train_dataset = SentDataset(args.train_file, params=dataset_params)
        train_dataloader = get_dataloader(train_dataset, args, mode="train")
        train(model, args, train_dataloader)
        selected_checkpoint = get_latest_checkpoint_from_output_dir(args.output_dir)
        if selected_checkpoint:
            args.checkpoint_file = selected_checkpoint
            print(f"Using frozen coherence checkpoint for downstream dementia classifier: {args.checkpoint_file}")

    if args.do_binary_train:
        binary_checkpoint, binary_args, heldout_test_file = train_binary_classifier(args, dataset_params, selected_checkpoint)
        args.binary_checkpoint_file = binary_checkpoint
        if args.do_test or args.do_binary_test:
            # By default, evaluate on the held-out 20% from each dialogue topic.
            # Passing --test_file is still useful for external prediction-only runs,
            # but after --do_binary_train the requested 20% topic holdout is used.
            run_binary_prediction(
                args,
                dataset_params,
                binary_checkpoint_file=binary_checkpoint,
                selected_coherence_checkpoint=selected_checkpoint,
                test_file_override=heldout_test_file,
            )
    elif args.do_binary_test:
        run_binary_prediction(args, dataset_params, selected_coherence_checkpoint=selected_checkpoint)
    elif args.do_test:
        inference_test_file, raw_items = prepare_inference_jsonl(args.test_file, args.label_list)
        print(f"Using test file: {args.test_file}")
        print(f"Inference-ready file: {inference_test_file}")

        test_dataset = SentDataset(inference_test_file, params=dataset_params)
        test_dataloader = get_dataloader(test_dataset, args, mode="test")

        checkpoint_file = args.checkpoint_file.strip()
        if checkpoint_file:
            print(f"Loading checkpoint: {checkpoint_file}")
            model.load_state_dict(torch.load(checkpoint_file, map_location=args.device), strict=False)
            model.eval()
        else:
            checkpoint_file = get_latest_checkpoint_from_output_dir(args.output_dir)
            if checkpoint_file:
                print(f"Loading checkpoint: {checkpoint_file}")
                model.load_state_dict(torch.load(checkpoint_file, map_location=args.device), strict=False)
                model.eval()
            else:
                raise FileNotFoundError(f"No checkpoints found under: {os.path.join(args.output_dir, 'good')}")

        predict_only(model, args, test_dataloader, raw_items, desc="test")

if __name__ == "__main__":
    main()

