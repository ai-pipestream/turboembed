"""Small torch.save checkpoints for the bundle tool's reader.

Usage: torch_fixture.py <output directory>

Writes, with seeded random values:

- state.pt: an OrderedDict state dict of F32 tensors, one a view at an
  offset into a storage another tensor shares, one a scalar and one with
  a zero-length dimension; state.json: each tensor's name, shape and
  values, as torch.load(weights_only=True) reads them back.
- noncontiguous.pt: one transposed tensor, which the reader refuses.
- float64.pt: one F64 tensor, which the reader refuses.
- legacy.pt: the pre-zip format (_use_new_zipfile_serialization=False),
  which the reader refuses.

Run with the pinned torch of analysis/relation/reference/glirel.
"""
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(7)
base = torch.randn(10)
state = OrderedDict([
    ('layer.weight', torch.randn(3, 4)),
    ('layer.bias', torch.randn(3)),
    ('shared.whole', base),
    ('shared.tail', base[4:]),
    ('scalar', torch.tensor(2.5)),
    ('empty', torch.zeros(0, 5)),
])
torch.save(state, OUT / 'state.pt')
back = torch.load(OUT / 'state.pt', weights_only=True)
(OUT / 'state.json').write_text(json.dumps(
    {k: dict(shape=list(v.shape), values=v.flatten().tolist()) for k, v in back.items()}, indent=1) + '\n')
torch.save(OrderedDict([('t', torch.randn(3, 2).t())]), OUT / 'noncontiguous.pt')
torch.save(OrderedDict([('d', torch.randn(2, dtype=torch.float64))]), OUT / 'float64.pt')
torch.save(OrderedDict([('w', torch.randn(2))]), OUT / 'legacy.pt', _use_new_zipfile_serialization=False)
print('written', OUT, torch.__version__)
