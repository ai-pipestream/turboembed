"""Exports the 2-layer BERT weights of testdata/tiny-bert-bundle to the two
ONNX graphs that OpenNLP's SentenceVectorsDL runs in
analysis/embed/reference/SentenceVectorsDlGolden.java:

  mean_l2.onnx
      The 2-layer BERT encoder followed by masked mean pooling over all live
      tokens ([CLS] + pieces + [SEP]) and L2 normalization, shaped
      [batch, 1, 32] so SentenceVectorsDL.getVectors returns the whole-row
      unit mean vector.

  cls.onnx
      The raw 2-layer BERT last_hidden_state shaped [batch, seq, 32] so
      SentenceVectorsDL.getVectors returns last_hidden_state[0][0], the
      [CLS] position vector.

Run in the reference container (bundle/reference/Dockerfile) via
analysis/embed/reference/regenerate.sh.

Arguments: <weights.safetensors> <mean_l2.onnx> <cls.onnx>
"""

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import BertConfig, BertModel


class MeanPooledBert(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        h = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        m = attention_mask.unsqueeze(-1).to(h.dtype)
        s = (h * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1e-9)
        return F.normalize(s, p=2, dim=-1, eps=1e-12)


class RawBert(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state


def main(weights_path, mean_out, cls_out):
    cfg = BertConfig(
        vocab_size=30522,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=512,
    )
    bert = BertModel(cfg)
    bert.load_state_dict(load_file(weights_path), strict=True)
    bert.eval()

    dummy = (
        torch.tensor([[101, 2023, 2003, 1037, 3231, 102]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
        torch.tensor([[0, 0, 0, 0, 0, 0]], dtype=torch.long),
    )
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "token_type_ids": {0: "batch", 1: "seq"},
        "output": {0: "batch", 1: "seq"},
    }
    for out_path, mod in [(mean_out, MeanPooledBert(bert)), (cls_out, RawBert(bert))]:
        torch.onnx.export(
            mod,
            dummy,
            out_path,
            input_names=["input_ids", "attention_mask", "token_type_ids"],
            output_names=["output"],
            dynamic_axes=dynamic_axes,
            opset_version=14,
            dynamo=False,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
