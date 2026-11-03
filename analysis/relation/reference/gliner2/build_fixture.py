"""Reference outputs for the native GLiNER2 relation scorer.

Usage: build_fixture.py <output directory> [<gliner2-base-v1 checkout>]

Two fixtures are written:

- reduced.safetensors and reduced.json: seeded random weights in the
  shapes and under the tensor names of a GLiNER2 checkpoint, the encoder
  the one core/tests/fixtures/deberta holds (hidden 32, 2 layers) and the
  relation heads at that width, with the outputs of GLiNER2's own layer
  classes on fixed token sequences. No value comes from a checkpoint
  (GLiNER2 is accept-first since 2026-10-28, docs/licences.md).
- real.json, only when the checkout is given: what fastino/gliner2-base-v1
  at REVISION gives through GLiNER2's own extract_relations on fixed
  texts: the input ids, each relation's instance count and the extracted
  relations with their confidences and byte offsets. Only outputs are
  written, no weights.

Only fixture production uses Python; the tests are native Rust.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from transformers import DebertaV2Config, DebertaV2Model

from gliner2 import GLiNER2
from gliner2.layers import CountLSTMv2, SpanMarkerV0, create_mlp

REVISION = 'f9634218e53580c56edf0de97ca1a7d3f1c2354e'
SEED = 20261028
OUT = Path(sys.argv[1])
SRC = Path(sys.argv[2]) if len(sys.argv) > 2 else None
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
D = 32
ROOT = Path(__file__).resolve().parents[4]

# ---- the reduced encoder: core/tests/fixtures/deberta's seeded weights ----
config = DebertaV2Config(vocab_size=16, hidden_size=D, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
encoder = DebertaV2Model(config).eval()
state = load_file(str(ROOT / 'core/tests/fixtures/deberta/weights.safetensors'))
encoder.load_state_dict(state, strict=True)

# ---- the relation heads at width D, seeded random ----
generator = torch.Generator().manual_seed(SEED)
count_pred = create_mlp(D, [2 * D], 20, dropout=0., activation='relu').eval()
count_embed = CountLSTMv2(hidden_size=D).eval()
marker = SpanMarkerV0(D, 8, 0.1).eval()
span = 'span_rep.span_rep_layer'
heads = {}
for prefix, module in [('count_pred', count_pred), ('count_embed', count_embed), (span, marker)]:
    local = {}
    for key, value in module.state_dict().items():
        noise = torch.randn(value.shape, generator=generator, dtype=torch.float32)
        if key.endswith(('norm1.weight', 'norm2.weight')):
            local[key] = 1 + 0.1 * noise
        elif value.ndim == 2:
            # Fan-in scaling keeps every layer's outputs near unit size, so
            # counts and span choices vary as a trained model's do.
            local[key] = noise / value.shape[1] ** 0.5
        else:
            local[key] = 0.02 * noise
        heads[f'{prefix}.{key}'] = local[key]
    module.load_state_dict(local, strict=True)

tensors = {'encoder.' + k: v.contiguous() for k, v in state.items()}
tensors.update({k: v.contiguous() for k, v in heads.items()})
save_file(tensors, OUT / 'reduced.safetensors')

FIXED_COUNT = 3
cases = []
# ids, two labels' [P], head [R] and tail [R] positions, first subword of each word
for ids, markers, words in [
    ([5, 3, 9, 1, 4, 2, 4, 6, 7, 8, 11, 12, 13, 14, 10], [[1, 4, 6]], [10, 11, 12, 13, 14]),
    ([5, 3, 9, 1, 4, 2, 4, 6, 0, 5, 3, 7, 1, 4, 2, 4, 6, 8, 11, 12, 15, 13, 14, 10, 9, 12, 11],
     [[1, 4, 6], [10, 13, 15]], [18, 19, 20, 22, 23, 24, 25, 26]),
    ([3, 1, 4, 2, 4, 9, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [[0, 2, 4]],
     list(range(5, 22))),
]:
    with torch.no_grad():
        h = encoder(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
        w = h[words]
        labels = []
        for p, r1, r2 in markers:
            count_logits = count_pred(h[p])
            count = int(count_logits.argmax())
            pc = torch.stack([h[r1], h[r2]])
            fixed = count_embed(pc, FIXED_COUNT)
            logits = marker.score_queries(w.unsqueeze(0), fixed.reshape(1, -1, D), mask_invalid=True)[0]
            actual = count_embed(pc, count) if count > 0 else torch.zeros(0, 2, D)
            # The two [R] rows of one schema are nearly equal at this width,
            # so attention across them barely shows; the [P] row and the
            # first word are unlike, and make it count.
            crossed = count_embed(torch.stack([h[p], h[words[0]]]), FIXED_COUNT)
            labels.append(dict(count=count, count_logits=count_logits.tolist(),
                               queries=fixed.flatten().tolist(),
                               logits=[x if np.isfinite(x) else None for x in logits.flatten().tolist()],
                               actual_queries=actual.flatten().tolist(),
                               crossed_queries=crossed.flatten().tolist()))
    cases.append(dict(ids=ids, markers=markers, words=words, labels=labels))
(OUT / 'reduced.json').write_text(json.dumps(dict(
    weights='seeded random', seed=SEED, fixed_count=FIXED_COUNT,
    torch=torch.__version__, cases=cases), separators=(',', ':')) + '\n')

# ---- the full checkpoint through GLiNER2 itself ----
if SRC is None:
    print('fixtures complete (reduced only)', OUT)
    sys.exit(0)
model = GLiNER2.from_pretrained(str(SRC))
model.eval()
seen = {}
model.encoder.register_forward_pre_hook(lambda mod, args, kwargs: seen.__setitem__('ids', kwargs.get('input_ids', args[0] if args else None)[0].tolist()), with_kwargs=True)
counts = []
model.count_pred.register_forward_hook(lambda mod, args, out: counts.append(out.reshape(-1, 20).argmax(-1).tolist()))
real = []
for text, rels in [
    ('Steve Jobs founded Apple in Cupertino. Tim Cook works for Apple.', ['founded', 'works for', 'located in']),
    ('Marie Curie was born in Warsaw and married Pierre Curie in Paris', ['born in', 'spouse of']),
    ('The Amazon river flows through Brazil, Peru and Colombia!', ['flows through', 'capital of']),
]:
    counts.clear()
    out = model.extract_relations(text, rels, include_confidence=True, include_spans=True)
    norm = text if text.endswith(('.', '!', '?')) else text + '.'
    byte = lambda i: len(norm[:i].encode('utf-8'))
    found = []
    for label, items in out['relation_extraction'].items():
        for it in items:
            found.append(dict(label=rels.index(label),
                head=[byte(it['head']['start']), byte(it['head']['end']), it['head']['confidence']],
                tail=[byte(it['tail']['start']), byte(it['tail']['end']), it['tail']['confidence']]))
    real.append(dict(text=text, labels=rels, ids=seen['ids'], counts=[c for cs in counts for c in cs], relations=found))
(OUT / 'real.json').write_text(json.dumps(dict(
    source=f'fastino/gliner2-base-v1@{REVISION}', license='Apache-2.0', gliner2='ca4f68235b50fe4c3ecf3876851d748fe4b15c6c',
    torch=torch.__version__, cases=real), indent=1) + '\n')
print('fixtures complete', OUT)
