"""Small, reproducible slices of the Apache-2.0 GLiNER2 checkpoint."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from safetensors.torch import save_file
from transformers import DebertaV2Config, DebertaV2Model

REVISION = 'f9634218e53580c56edf0de97ca1a7d3f1c2354e'
ROOT = Path(sys.argv[1])
ROOT.mkdir(parents=True, exist_ok=True)
BASE = f'https://huggingface.co/fastino/gliner2-base-v1/resolve/{REVISION}/model.safetensors'

def fetch(start, end):
    return subprocess.check_output(['curl', '-fsSL', '--retry', '3', '-r', f'{start}-{end}', f'{BASE}?range={start}-{end}'])

n = int.from_bytes(fetch(0, 7), 'little')
header_bytes = fetch(8, n + 7)
header = json.loads(header_bytes)
config = DebertaV2Config(vocab_size=16, hidden_size=32, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
torch.set_num_threads(1)
model = DebertaV2Model(config).eval()
state = {}
records = []
for key, value in model.state_dict().items():
    source = 'encoder.' + key
    info = header[source]
    shape = info['shape']
    assert info['dtype'] == 'F32'
    # All dimensions are prefixes except vocabulary and relative position rows.
    offset = 1000 if key == 'embeddings.word_embeddings.weight' else 248 if key == 'encoder.rel_embeddings.weight' else 0
    width = shape[1] if len(shape) == 2 else 1
    start = 8 + n + info['data_offsets'][0] + offset * width * 4
    end = start + value.shape[0] * width * 4 - 1
    data = fetch(start, end)
    assert len(data) == end - start + 1
    arr = np.frombuffer(data, dtype='<f4').copy()
    if len(shape) == 2:
        arr = arr.reshape(-1, width)[:, :value.shape[1]].copy()
    state[key] = torch.from_numpy(arr)
    records.append(dict(tensor=source, source_shape=shape, row_start=offset,
                        shape=list(value.shape), range_sha256=hashlib.sha256(data).hexdigest()))
model.load_state_dict(state, strict=True)
save_file(state, ROOT / 'weights.safetensors')
(ROOT / 'config.json').write_text(config.to_json_string())
cases = []
for ids, mask in [([0,1,2,3],[1,1,1,1]), ([1,2,3,4,5,6],[1,1,1,1,0,0]),
                  (list(range(16)) + [3,2,1,0],[1]*20), ([15],[1]), ([2,3,4,5],[0,1,0,1])]:
    with torch.no_grad():
        out = model(torch.tensor([ids]), attention_mask=torch.tensor([mask]), output_hidden_states=True)
    final = out.last_hidden_state
    projections = {}
    for role, stem in [('token', 'query'), ('prompt', 'key')]:
        prefix = f'encoder.layer.0.attention.self.{stem}_proj'
        result = torch.nn.functional.linear(final, state[prefix+'.weight'], state[prefix+'.bias'])
        projections[role] = result.squeeze(0).flatten().tolist()
    cases.append(dict(ids=ids, mask=mask, layers=[s.squeeze(0).flatten().tolist() for s in out.hidden_states], **projections))
(ROOT / 'reference.json').write_text(json.dumps(cases, separators=(',', ':')) + '\n')
(ROOT / 'provenance.json').write_text(json.dumps(dict(repository='fastino/gliner2-base-v1', revision=REVISION,
    license='Apache-2.0', transformers='4.57.1', torch=torch.__version__,
    header_sha256=hashlib.sha256(header_bytes).hexdigest(), slices=records), indent=2) + '\n')
print('fixture complete', ROOT)
