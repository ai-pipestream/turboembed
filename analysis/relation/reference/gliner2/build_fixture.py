"""Reference outputs for the native GLiNER2 relation scorer.

Usage: build_fixture.py <gliner2-base-v1 checkout> <output directory>

The checkout is fastino/gliner2-base-v1 at REVISION (Apache-2.0). Two
fixtures are written:

- reduced.safetensors and reduced.json: the encoder sliced as
  core/tests/fixtures/deberta slices it (hidden 32, 2 layers, vocabulary
  rows 1000..1015, relative rows 248..263) and the relation heads sliced
  to that width, saved under the checkpoint's own tensor names, with the
  outputs of GLiNER2's own layer classes on fixed token sequences. This is
  a numerical fixture, not a usable model.
- real.json: what the full checkpoint gives through GLiNER2's own
  extract_relations on fixed texts: the input ids, each relation's
  instance count and the extracted relations with their confidences and
  byte offsets. Only outputs are written, no weights.

Only fixture production uses Python; the tests are native Rust.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import load_file
from safetensors.torch import save_file
from transformers import DebertaV2Config, DebertaV2Model

from gliner2 import GLiNER2
from gliner2.layers import CountLSTMv2, SpanMarkerV0, create_mlp

REVISION = 'f9634218e53580c56edf0de97ca1a7d3f1c2354e'
SRC = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
full = load_file(str(SRC / 'model.safetensors'))
D, BIG = 32, 768

# ---- reduced encoder, sliced as the core encoder fixture slices it ----
config = DebertaV2Config(vocab_size=16, hidden_size=D, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
encoder = DebertaV2Model(config).eval()
state = {}
for key, value in encoder.state_dict().items():
    src = full['encoder.' + key]
    offset = 1000 if key == 'embeddings.word_embeddings.weight' else 248 if key == 'encoder.rel_embeddings.weight' else 0
    if src.ndim == 2:
        part = src[offset:offset + value.shape[0], :value.shape[1]]
    else:
        part = src[:value.shape[0]]
    state[key] = torch.from_numpy(np.ascontiguousarray(part))
encoder.load_state_dict(state, strict=True)


def rows_cols(name, rows, cols):
    """A 2-D tensor's rows and columns, each a list of index ranges."""
    a = full[name]
    r = np.concatenate([np.arange(s, e) for s, e in rows])
    c = np.concatenate([np.arange(s, e) for s, e in cols])
    return np.ascontiguousarray(a[np.ix_(r, c)])


def vec(name, ranges):
    a = full[name]
    return np.ascontiguousarray(np.concatenate([a[s:e] for s, e in ranges]))


heads = {}
def lin(name, rows, cols):
    heads[name + '.weight'] = rows_cols(name + '.weight', rows, cols)
    heads[name + '.bias'] = vec(name + '.bias', rows)

span = 'span_rep.span_rep_layer'
for side in ('project_start', 'project_end'):
    lin(f'{span}.{side}.0', [(0, 4 * D)], [(0, D)])
    lin(f'{span}.{side}.3', [(0, D)], [(0, 4 * D)])
lin(f'{span}.out_project.0', [(0, 4 * D)], [(0, D), (BIG, BIG + D)])
lin(f'{span}.out_project.3', [(0, D)], [(0, 4 * D)])
lin('count_pred.0', [(0, 2 * D)], [(0, D)])
lin('count_pred.2', [(0, 20)], [(0, 2 * D)])
heads['count_embed.pos_embedding.weight'] = rows_cols('count_embed.pos_embedding.weight', [(0, 20)], [(0, D)])
gates = [(g * BIG, g * BIG + D) for g in range(3)]
for side in ('ih', 'hh'):
    heads[f'count_embed.gru.weight_{side}_l0'] = rows_cols(f'count_embed.gru.weight_{side}_l0', gates, [(0, D)])
    heads[f'count_embed.gru.bias_{side}_l0'] = vec(f'count_embed.gru.bias_{side}_l0', gates)
t = 'count_embed.transformer'
lin(f'{t}.in_projector', [(0, 128)], [(0, D)])
for name in full:
    if name.startswith(f'{t}.transformer.layers.'):
        heads[name] = np.ascontiguousarray(full[name])
lin(f'{t}.out_projector.0', [(0, D)], [(0, 128), (128, 128 + D)])
lin(f'{t}.out_projector.2', [(0, D)], [(0, D)])
lin(f'{t}.out_projector.4', [(0, D)], [(0, D)])

count_pred = create_mlp(D, [2 * D], 20, dropout=0., activation='relu').eval()
count_pred.load_state_dict({k[len('count_pred.'):]: torch.from_numpy(v) for k, v in heads.items() if k.startswith('count_pred.')})
count_embed = CountLSTMv2(hidden_size=D).eval()
count_embed.load_state_dict({k[len('count_embed.'):]: torch.from_numpy(v) for k, v in heads.items() if k.startswith('count_embed.')})
marker = SpanMarkerV0(D, 8, 0.1).eval()
marker.load_state_dict({k[len(span) + 1:]: torch.from_numpy(v) for k, v in heads.items() if k.startswith(span)})

tensors = {'encoder.' + k: v.contiguous() for k, v in state.items()}
tensors.update({k: torch.from_numpy(v) for k, v in heads.items()})
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
            labels.append(dict(count=count, count_logits=count_logits.tolist(),
                               queries=fixed.flatten().tolist(),
                               logits=[x if np.isfinite(x) else None for x in logits.flatten().tolist()],
                               actual_queries=actual.flatten().tolist()))
    cases.append(dict(ids=ids, markers=markers, words=words, labels=labels))
(OUT / 'reduced.json').write_text(json.dumps(dict(
    source=f'fastino/gliner2-base-v1@{REVISION}', license='Apache-2.0', fixed_count=FIXED_COUNT,
    torch=torch.__version__, cases=cases), separators=(',', ':')) + '\n')

# ---- the full checkpoint through GLiNER2 itself ----
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
