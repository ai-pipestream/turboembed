"""A small, reproducible DeBERTa-v2 fixture on seeded random weights.

The shapes and tensor names are those of GLiNER2's encoder reduced to two
layers of width 32; no value comes from a checkpoint (GLiNER2 is
accept-first since 2026-10-28, docs/licences.md).
"""
import json
from pathlib import Path
import sys
import torch
from safetensors.torch import save_file
from transformers import DebertaV2Config, DebertaV2Model

SEED = 20261028
ROOT = Path(sys.argv[1])
ROOT.mkdir(parents=True, exist_ok=True)
config = DebertaV2Config(vocab_size=16, hidden_size=32, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
torch.set_num_threads(1)
model = DebertaV2Model(config).eval()
generator = torch.Generator().manual_seed(SEED)
state = {}
for key, value in model.state_dict().items():
    noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
    # LayerNorm scales near 1, everything else at a trained model's scale.
    state[key] = (1 + 0.1 * noise) if 'LayerNorm.weight' in key else 0.1 * noise
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
(ROOT / 'provenance.json').write_text(json.dumps(dict(weights='seeded random', seed=SEED,
    shapes='GLiNER2 encoder tensor names, reduced to 2 layers of width 32', transformers='4.57.1',
    torch=torch.__version__), indent=2) + '\n')
print('fixture complete', ROOT)
