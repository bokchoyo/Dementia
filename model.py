# date=2024-08-18
# author=liuwei
# Modified: frozen coherence feature extractor + dementia binary classifier head

import math
import os
import json

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu

from module import Embedding, Absolute_Position_Embedding, FusionTransformer, Transformer_Encoder
from module import Doc_Pooler, Relative_Position_Embedding


class BaseClassifer(nn.Module):
    def __init__(self, args):
        super(BaseClassifer, self).__init__()

        self.num_labels = args.num_labels
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.dropout = nn.Dropout(p=args.dropout)
        self.fc1.weight.data.normal_(mean=0.0, std=0.02)
        self.fc1.bias.data.zero_()
        self.fc2.weight.data.normal_(mean=0.0, std=0.02)
        self.fc2.bias.data.zero_()
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    @property
    def feature_dim(self):
        return self.classifier.in_features

    def extract_features(self, doc_vectors):
        """
        Return the final hidden representation immediately before the original
        coherence classifier, plus the original coherence logits/probabilities.
        This is used by the frozen dementia binary classifier wrapper.
        """
        if doc_vectors.dtype == torch.bfloat16:
            doc_vectors = doc_vectors.float()
        input_vectors = self.dropout(doc_vectors)
        input_vectors = input_vectors.float()
        input_vectors = self.fc1(input_vectors)
        input_vectors = self.dropout(input_vectors)
        hidden = self.fc2(input_vectors)
        classifier_input = self.dropout(hidden)
        logits = self.classifier(classifier_input)
        probs = F.softmax(logits, dim=-1)
        return hidden, logits, probs

    def forward(
        self,
        doc_vectors,
        labels=None,
        flag="Train"
    ):
        hidden, logits, probs = self.extract_features(doc_vectors=doc_vectors)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class SentTransformer(nn.Module):
    def __init__(self, args):
        super(SentTransformer, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels

        self.proj = nn.Linear(self.input_size, self.hidden_size)
        self.abs_position_embedding = Absolute_Position_Embedding(
            self.hidden_size, learnable=False
        )
        self.transformer = Transformer_Encoder(
            {
                "num_layers": 1, "hidden_size": args.hidden_size,
                "num_heads": 8, "scaled": True
            }
        )
        self.fc = nn.Linear(args.hidden_size, args.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    @property
    def feature_dim(self):
        return self.classifier.in_features

    def extract_features(self, sent_vectors, sent_mask):
        """
        Return the final hidden representation immediately before the original
        coherence classifier, plus the original coherence logits/probabilities.
        The hidden representation is the last trainable hidden layer feeding the
        original coherence classifier.
        """
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.proj(sent_vectors)
        input_vectors = self.abs_position_embedding(sent_vectors)
        output = self.transformer(input_vectors, sent_mask)
        output = self.dropout(output)
        hidden = self.fc(output)
        classifier_input = self.dropout(hidden)
        logits = self.classifier(classifier_input)
        probs = F.softmax(logits, dim=-1)
        return hidden, logits, probs

    def forward(
        self,
        sent_vectors,
        sent_mask,
        labels=None,
        flag="Train"
    ):
        hidden, logits, probs = self.extract_features(
            sent_vectors=sent_vectors,
            sent_mask=sent_mask,
        )

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class FrozenCoherenceBinaryClassifier(nn.Module):
    """
    Patient/control classifier stacked on a frozen coherence prediction model.

    Inputs to the binary head:
      1) the frozen coherence model's final hidden representation, and
      2) the frozen coherence model's softmax probabilities.

    Only the new binary head is trainable. The original coherence model is kept
    in eval mode and wrapped in torch.no_grad(), so gradients cannot update it.
    """
    def __init__(self, coherence_model, args):
        super(FrozenCoherenceBinaryClassifier, self).__init__()
        self.coherence_model = coherence_model
        for param in self.coherence_model.parameters():
            param.requires_grad = False
        self.coherence_model.eval()

        self.num_labels = int(getattr(args, "num_labels", 2))
        self.coherence_num_labels = int(getattr(self.coherence_model, "num_labels", 3))
        self.feature_dim = int(getattr(self.coherence_model, "feature_dim", self.coherence_model.classifier.in_features))
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
        # Keep the frozen coherence model deterministic even while the wrapper trains.
        self.coherence_model.eval()
        return self

    def _extract_from_frozen_coherence_model(self, **kwargs):
        self.coherence_model.eval()
        with torch.no_grad():
            if "doc_vectors" in kwargs and kwargs["doc_vectors"] is not None:
                hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(
                    doc_vectors=kwargs["doc_vectors"]
                )
            else:
                hidden, coherence_logits, coherence_probs = self.coherence_model.extract_features(
                    sent_vectors=kwargs["sent_vectors"],
                    sent_mask=kwargs["sent_mask"],
                )
        return hidden.detach(), coherence_logits.detach(), coherence_probs.detach()

    def forward(
        self,
        sent_vectors=None,
        sent_mask=None,
        doc_vectors=None,
        labels=None,
        flag="Train",
        **kwargs,
    ):
        hidden, coherence_logits, coherence_probs = self._extract_from_frozen_coherence_model(
            sent_vectors=sent_vectors,
            sent_mask=sent_mask,
            doc_vectors=doc_vectors,
        )
        binary_features = torch.cat([hidden.float(), coherence_probs.float()], dim=-1)
        binary_logits = self.binary_classifier(binary_features)
        binary_probs = F.softmax(binary_logits, dim=-1)
        _, preds = torch.max(binary_logits, dim=-1)

        outputs = (preds, binary_logits, binary_probs, coherence_probs)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(binary_logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs


class FusionClassifier(nn.Module):
    def __init__(self, args):
        super(FusionClassifier, self).__init__()

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.embed_dim = args.embed_dim
        self.max_node_len = args.max_sent_num + args.max_rel_num + 1
        self.dropout = nn.Dropout(args.dropout)
        self.num_labels = args.num_labels
        self.rel_embedding = Embedding(
            vocab=args.rel_list,
            embed_dim=args.embed_dim
        )
        self.sent_proj = nn.Linear(self.input_size, self.hidden_size)
        self.rel_proj = nn.Linear(self.embed_dim, self.hidden_size)
        self.rel_pos_embedding = Relative_Position_Embedding(
            self.hidden_size, self.max_node_len*2
        )
        self.fusion_transformer = FusionTransformer(
            {
                "num_layers": 1, "hidden_size": args.hidden_size,
                "num_heads": 8, "scaled": True
            }
        )
        self.fc = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.classifier = nn.Linear(args.hidden_size // 4, self.num_labels)
        self.classifier.weight.data.normal_(mean=0.0, std=0.02)
        self.classifier.bias.data.zero_()

    def forward(
        self,
        sent_vectors,
        sent_mask,
        rel_ids,
        pos_start,
        pos_end,
        sent_rel_mask,
        labels=None,
        flag="Train"
    ):
        # for sent
        if sent_vectors.dtype == torch.bfloat16:
            sent_vectors = sent_vectors.float()
        sent_vectors = self.dropout(sent_vectors)
        sent_vectors = self.sent_proj(sent_vectors)

        # for rel
        rel_vectors = self.rel_embedding(rel_ids)
        rel_vectors = self.dropout(rel_vectors)
        rel_vectors = self.rel_proj(rel_vectors)

        input_vectors = torch.cat(
            (sent_vectors, rel_vectors), dim=1
        )
        input_vectors = self.dropout(input_vectors)

        # pos embedding
        rel_pos_vectors = self.rel_pos_embedding(
            pos_start, pos_end
        )

        output = self.fusion_transformer(
            hidden_states=input_vectors,
            attention_mask=sent_mask,
            fusion_mask=sent_rel_mask,
            rel_pos_input=rel_pos_vectors
        )

        output = self.dropout(output)
        output = self.fc(output)
        output = self.dropout(output)
        logits = self.classifier(output)

        _, preds = torch.max(logits, dim=-1)
        outputs = (preds,)
        if flag.upper() == 'TRAIN':
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
