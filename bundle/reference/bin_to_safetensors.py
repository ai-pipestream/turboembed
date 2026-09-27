"""The tensors of a PyTorch checkpoint (pytorch_model.bin) written as a
safetensors file, for a model whose upstream repository ships no
safetensors file. The bundle tool runs it in the reference container,
with no network, on the checkpoint it fetched; the bundle carries the
result and never the checkpoint, which the library does not read.

Every tensor comes out as stored: its dtype, its shape, its values, made
contiguous, under its name. Nothing is renamed, cast, tied or dropped.
Two runs give the same bytes: the file is deterministic in its tensors.

Arguments: the checkpoint, the file to write, and the JSON file that
says what ran.
"""

import json
import sys

import safetensors
import torch
from safetensors.torch import save_file

SETTINGS = ["tensors=as stored, contiguous", "weights_only=True"]


def main(src, out, produced_by_path):
    state = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        sys.exit("the checkpoint is not a flat state dict of tensors")
    save_file({name: t.contiguous() for name, t in state.items()}, out)
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "safetensors",
                "tool_version": f"{safetensors.__version__} (torch {torch.__version__})",
                "settings": SETTINGS,
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
