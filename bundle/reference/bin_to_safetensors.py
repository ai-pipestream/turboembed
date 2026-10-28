"""The tensors of a PyTorch checkpoint (pytorch_model.bin) written as a
safetensors file, for a model whose upstream repository ships no
safetensors file. The bundle tool runs it in the reference container,
with no network, on the checkpoint it fetched; the bundle carries the
result and never the checkpoint, which the library does not read.

Every tensor comes out as stored: its dtype, its shape, its values, made
contiguous, under its name. Nothing is renamed, tied or dropped, and the
only cast is the one the optional `float32` argument asks for: floating
tensors written as float32, an exact widening of BF16/F16 values, for a
checkpoint whose stored dtype no backend computes in at MODEL precision.
Two runs give the same bytes: the file is deterministic in its tensors.

Arguments: the checkpoint, the file to write, the JSON file that says
what ran, and optionally `float32`.
"""

import json
import sys

import safetensors
import torch
from safetensors.torch import save_file, load_file

def main(src, out, produced_by_path, *rest):
    if rest not in ((), ("float32",)):
        sys.exit(f"unknown arguments {rest}")
    cast = rest == ("float32",)
    settings = ["tensors=as stored, contiguous" + (", cast to float32" if cast else ""), "weights_only=True"]
    state = load_file(src, device="cpu") if src.endswith(".safetensors") else torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        sys.exit("the checkpoint is not a flat state dict of tensors")
    save_file(
        {name: (t.float() if cast and t.is_floating_point() else t).contiguous() for name, t in state.items()}, out
    )
    with open(produced_by_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tool": "safetensors",
                "tool_version": f"{safetensors.__version__} (torch {torch.__version__})",
                "settings": settings,
            },
            f,
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
