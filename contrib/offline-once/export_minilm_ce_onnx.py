#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""One-off ONNX export of the pinned MiniLM-L6 cross-encoder.

Not on the score path. TurboRerank loads SHA-pinned OpenVINO IR produced
from this ONNX (C++ `onnx_to_ir`). See native/turborerank/tools/onnx_to_ir.cpp.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        default="models/rerank/ms-marco-minilm-l6",
        help="Directory with the SHA-pinned safetensors + config",
    )
    p.add_argument(
        "--out",
        default="models/ov-rerank/ms-marco-minilm-l6",
        help="Destination for model.onnx plus vocab/config copies",
    )
    args = p.parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = AutoModelForSequenceClassification.from_pretrained(src, local_files_only=True)
    model.eval()

    ids = torch.ones(1, 16, dtype=torch.long)
    mask = torch.ones(1, 16, dtype=torch.long)
    types = torch.zeros(1, 16, dtype=torch.long)
    onnx_path = out / "model.onnx"
    torch.onnx.export(
        model,
        (ids, mask, types),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "token_type_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "token_type_ids": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,
    )
    for name in ("vocab.txt", "config.json", "tokenizer.json", "tokenizer_config.json"):
        src_f = src / name
        if src_f.is_file():
            shutil.copy2(src_f, out / name)
    print(f"wrote {onnx_path} ({onnx_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
