"""Seeded random GLiNER head weights for a small, offline head parity fixture.
Arguments: a directory holding gliner-0.2.21-py3-none-any.whl, new output
directory. The shapes and tensor names are those of urchade/gliner_small-v2.1
reduced to width 16; no value comes from a checkpoint (GLiNER v2.1 is
accept-first since 2026-10-28, docs/licences.md).
"""
import json
from pathlib import Path
import sys
import types
import zipfile
import torch
from safetensors.torch import save_file
root, out = map(Path, sys.argv[1:])
out.mkdir(exist_ok=False)
SEED=20261028
generator=torch.Generator().manual_seed(SEED)
pkg=types.ModuleType("reference_heads");pkg.__path__=[];sys.modules[pkg.__name__]=pkg
with zipfile.ZipFile(root / "gliner-0.2.21-py3-none-any.whl") as z:
    for name in ["layers","span_rep"]:
        m=types.ModuleType(f"reference_heads.{name}");m.__package__=pkg.__name__;sys.modules[m.__name__]=m
        exec(compile(z.read(f"gliner/modeling/{name}.py"),name,"exec"),m.__dict__)
from reference_heads.layers import LstmSeq2SeqEncoder,create_projection_layer
from reference_heads.span_rep import SpanMarkerV0
h=16
projection=torch.nn.Linear(32,h).eval()
rnn=LstmSeq2SeqEncoder(types.SimpleNamespace(hidden_size=h)).eval()
span=SpanMarkerV0(h,3).eval()
prompt=create_projection_layer(h,0.4).eval()
modules={"token_rep_layer.projection":projection,"rnn":rnn,"span_rep_layer.span_rep_layer":span,"prompt_rep_layer":prompt}
sliced={}
for stem,m in modules.items():
    local={}
    for name,value in m.state_dict().items():
        t=(0.05*torch.randn(value.shape,generator=generator,dtype=torch.float32)).contiguous()
        local[name]=t;sliced[f"{stem}.{name}"]=t
    m.load_state_dict(local,strict=True)
save_file(sliced,str(out/"heads.safetensors"))
# Seeded input rows at the reduced input width.
embedding=torch.randn((7,32),generator=generator,dtype=torch.float32)
with torch.no_grad():
    x=projection(embedding).unsqueeze(0)
    words=rnn(x[:,2:],torch.ones((1,5),dtype=torch.bool))
    p=prompt(x[:,:2])
    idx=torch.tensor([[(i,min(i+k,4)) for i in range(5) for k in range(3)]])
    logits=torch.einsum("blkd,bcd->blkc",span(words,idx),p)[0]
    flat=[float(logits[i,k,j]) for i in range(5) for k in range(min(3,5-i)) for j in range(2)]
(out/"reference.json").write_text(json.dumps(dict(input=embedding.tolist(),logits=flat),indent=2)+"\n")
(out/"provenance.json").write_text(json.dumps(dict(weights="seeded random",seed=SEED,shapes="urchade/gliner_small-v2.1 head tensor names, reduced to width 16",hidden=16,input_width=32,reference="GLiNER 0.2.21 SpanMarkerV0 and LstmSeq2SeqEncoder, eval mode"),indent=2)+"\n")
