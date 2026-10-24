"""Generate real-checkpoint GLiNER parity outputs, with upstream GLiNER 0.2.21 heads.

Only fixture generation uses Python. Run with a local checkpoint directory and
an output JSON path. Requires the wheel, checkpoint, spm.model, encoder-config.json.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import zipfile
import torch
from safetensors.torch import load_file
from transformers import DebertaV2Config, DebertaV2Model, DebertaV2TokenizerFast

root = Path(sys.argv[1])
torch.set_num_threads(1)
wheel = root / "gliner-0.2.21-py3-none-any.whl"
# Load the actual upstream mathematical components without importing its optional
# ONNX serving frontend. No operators or weights are replaced.
pkg = types.ModuleType("gliner_reference")
pkg.__path__ = []
sys.modules[pkg.__name__] = pkg
with zipfile.ZipFile(wheel) as z:
    for name in ["layers", "span_rep", "utils", "decoder"]:
        mod = types.ModuleType(f"gliner_reference.{name}")
        mod.__package__ = "gliner_reference"
        sys.modules[mod.__name__] = mod
        exec(compile(z.read(f"gliner/{'decoding' if name in ['utils','decoder'] else 'modeling'}/{name}.py"), f"gliner/{name}.py", "exec"), mod.__dict__)
from gliner_reference.layers import create_projection_layer, LstmSeq2SeqEncoder
from gliner_reference.span_rep import SpanMarkerV0
from gliner_reference.decoder import SpanDecoder

if (root / "model.safetensors").exists():
    weights = load_file(root / "model.safetensors")
else:
    weights = torch.load(root / "pytorch_model.bin", map_location="cpu", weights_only=True)
ec = json.loads((root / "encoder-config.json").read_text())
vocab_size = weights["token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight"].shape[0]
ec["vocab_size"] = vocab_size
cfg = DebertaV2Config(**ec)
encoder = DebertaV2Model(cfg).eval()
encoder.load_state_dict({k.removeprefix("token_rep_layer.bert_layer.model."): v for k, v in weights.items() if k.startswith("token_rep_layer.bert_layer.model.")}, strict=True)
projection = torch.nn.Linear(768, 512).eval()
projection.load_state_dict({k.removeprefix("token_rep_layer.projection."): v for k,v in weights.items() if k.startswith("token_rep_layer.projection.")}, strict=True)
rnn = LstmSeq2SeqEncoder(types.SimpleNamespace(hidden_size=512)).eval()
rnn.load_state_dict({k.removeprefix("rnn."):v for k,v in weights.items() if k.startswith("rnn.")}, strict=True)
span = SpanMarkerV0(512, 12).eval()
span.load_state_dict({k.removeprefix("span_rep_layer.span_rep_layer."):v for k,v in weights.items() if k.startswith("span_rep_layer.span_rep_layer.")}, strict=True)
prompt = create_projection_layer(512, 0.4).eval()
prompt.load_state_dict({k.removeprefix("prompt_rep_layer."):v for k,v in weights.items() if k.startswith("prompt_rep_layer.")}, strict=True)
tok = DebertaV2TokenizerFast(vocab_file=str(root / "spm.model"))
tok.add_tokens(["[FLERT]", "<<ENT>>", "<<SEP>>"])
ent_id = vocab_size - 2
assert tok.convert_tokens_to_ids("<<ENT>>") == ent_id
if vocab_size == 250105:
    checkpoint_id = "urchade/gliner_multi-v2.1"
    revision = "443d26d654e0324125a96bebd8e796c14ff2efe6"
    cases = [
        (["person", "organization", "location"], ["Alice", "travaille", "chez", "Microsoft", "à", "Paris", "."]),
        (["person", "city", "university"], ["Zoë", "besuchte", "die", "Technische", "Universität", "München", "."]),
        (["person", "city", "country"], ["太郎", "は", "東京", "と", "日本", "を", "訪れた", "。"]),
        (["emoji", "person"], ["Zoë", "😀🧬", "𐍈", "."]),
        (["organization", "location", "university"], ["University", "of", "California", "is", "in", "California", "."]),
    ]
else:
    checkpoint_id = "urchade/gliner_small-v2.1"
    revision = "4e091416cf7c3481db542c2a3d26156916f3a47f"
    cases = [
        (["person", "organization", "location"], ["Alice", "works", "at", "Microsoft", "in", "Paris", "."]),
        (["person", "city", "country"], ["Zoë", "visited", "New", "York", "in", "the", "United", "States", "."]),
        (["emoji", "person"], ["Zoë", "😀🧬", "𐍈", "."]),
        (["organization", "location", "university"], ["University", "of", "California", "is", "in", "California", "."]),
    ]
out=[]
with torch.no_grad():
    for labels, words in cases:
        pieces=sum((["<<ENT>>", l] for l in labels), [])+["<<SEP>>"]+words
        batch=tok(pieces,is_split_into_words=True,return_tensors="pt",truncation=False)
        ids=batch["input_ids"][0].tolist()
        word_ids=batch.word_ids()
        positions=[word_ids.index(i+2*len(labels)+1) for i in range(len(words))]
        markers=[i for i,t in enumerate(ids) if t==ent_id]
        h=projection(encoder(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"]).last_hidden_state)
        w=rnn(h[:,positions], torch.ones((1,len(words)),dtype=torch.bool))
        p=prompt(h[:,markers])
        indices=torch.tensor([[(i,min(i+k,len(words)-1)) for i in range(len(words)) for k in range(12)]])
        logits=torch.einsum("blkd,bcd->blkc",span(w,indices),p)[0]
        scores=[]
        for i in range(len(words)):
            for k in range(min(12,len(words)-i)):
                for l in range(len(labels)):
                    scores.append(float(logits[i,k,l]))
        decoded={}
        for mode in ["flat","nested"]:
            selected=SpanDecoder(None).decode([words], {i+1:l for i,l in enumerate(labels)}, logits.unsqueeze(0), flat_ner=mode=="flat",threshold=0.5)[0]
            decoded[mode]=[[a,b+1,labels.index(l),score] for a,b,l,score in selected]
        out.append(dict(labels=labels,words=words,ids=ids,prompts=markers,positions=positions,logits=scores,decoded=decoded))
result=dict(checkpoint=checkpoint_id,revision=revision,license="Apache-2.0",wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),cases=out)
Path(sys.argv[2]).write_text(json.dumps(result,indent=2)+"\n")
print("reference cases",len(out),"logits",sum(len(c['logits']) for c in out))
