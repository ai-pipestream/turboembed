"""Reference outputs for the native GLiREL relation scorer.

Usage: build_fixture.py <glirel checkout> <deberta-v3-large dir> <output directory>

<glirel checkout> holds jackboyla/glirel-large-v0 at REVISION
(`pytorch_model.bin`, `glirel_config.json`; CC BY-NC-SA 4.0), and
<deberta-v3-large dir> holds microsoft/deberta-v3-large's `config.json`,
`spm.model` and `tokenizer_config.json` (MIT). Two fixtures are written:

- reduced.safetensors and reduced.json: a small encoder and GLiREL heads
  with seeded random weights (no checkpoint values: GLiREL's weights are
  non-commercial and stay out of the repository), saved under the
  checkpoint's own tensor names, and the outputs of GLiREL's own layer
  classes on fixed token sequences.
- real.json: what the full checkpoint gives through GLiREL's own
  predict_relations on fixed texts: the words, the input ids, every
  ordered entity pair's logit for every label, and the relations at 0.5.
  Only outputs are written, no weights.

The checkpoint is loaded with weights_only=True, so its pickle builds
tensors and nothing else. Only fixture production uses Python; the tests
are native Rust.
"""
import json
import re
import sys
import tempfile
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModel, DebertaV2Config, DebertaV2Model

REVISION = '40a523e12a8432d6da364cf2a195a28755ff04d3'
GLIREL = '1f485a268c2144cdc0221bcf0b24152f93c0a155'
SRC, BASE, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.manual_seed(20261021)

# The base encoder's architecture from its config only: GLiREL replaces every
# weight with the checkpoint's, so the base weights are never fetched.
AutoModel.from_pretrained = lambda name, **kw: AutoModel.from_config(AutoConfig.from_pretrained(name))
_load = torch.load
torch.load = lambda *a, **kw: _load(*a, **{**kw, 'weights_only': True})

from glirel import GLiREL  # noqa: E402
from glirel.modules.base import generate_entity_pairs_indices  # noqa: E402
from glirel.modules.layers import LstmSeq2SeqEncoder, ScorerLayer  # noqa: E402
from glirel.modules.rel_rep import RelRepLayer  # noqa: E402

# ---- reduced: seeded random encoder and heads, GLiREL's own classes ----
E, H = 32, 16
# The vocabulary is as long as testdata/sentencepiece/unigram-identity.model,
# so the boundary tests can tokenize real text with it.
config = DebertaV2Config(vocab_size=800, hidden_size=E, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
encoder = DebertaV2Model(config).eval()
projection = torch.nn.Linear(E, H)
rnn = LstmSeq2SeqEncoder(H, H // 2, num_layers=1, bidirectional=True).eval()
rel = RelRepLayer(hidden_size=H, max_width=12, rel_mode='marker', span_mode='markerv1', dropout=0.4).eval()
prompt = torch.nn.Sequential(torch.nn.Linear(H, 4 * H), torch.nn.Dropout(0.4), torch.nn.ReLU(),
                             torch.nn.Linear(4 * H, H)).eval()
scorer = ScorerLayer('dot', hidden_size=H)
# Biases away from zero, so a dropped bias shows.
with torch.no_grad():
    for m in [projection, rnn, rel, prompt]:
        for name, p in m.named_parameters():
            if name.endswith('bias') or 'bias_' in name:
                p.uniform_(-0.5, 0.5)
tensors = {f'token_rep_layer.bert_layer.model.{k}': v.contiguous() for k, v in encoder.state_dict().items()}
for prefix, m in [('token_rep_layer.projection.', projection), ('rnn.', rnn),
                  ('span_rep_layer.', rel), ('prompt_rep_layer.', prompt)]:
    tensors.update({prefix + k: v.contiguous() for k, v in m.state_dict().items()})
save_file(tensors, OUT / 'reduced.safetensors')

cases = []
# ids; per label its [REL] and label positions; each word's first subword;
# entity spans as inclusive word ranges.
for ids, labels, words, spans in [
    ([1, 9, 5, 9, 7, 2, 4, 6, 8, 10, 11, 12, 13, 2], [[1, 2], [3, 4]], [6, 7, 8, 9, 10, 11, 12],
     [[0, 1], [3, 3], [5, 6]]),
    ([1, 9, 3, 2, 14, 15, 4, 6, 5, 8, 7, 2], [[1, 2]], [4, 5, 6, 7, 8, 9, 10], [[0, 0], [2, 4], [6, 6], [1, 1]]),
]:
    with torch.no_grad():
        h = encoder(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
        p = projection(h)
        types = torch.stack([(p[r] + p[l]) / 2 for r, l in labels])
        w = p[words].unsqueeze(0)
        lstm = rnn(w, torch.ones(1, len(words), dtype=torch.long))
        span_idx = torch.tensor(spans)
        pairs = generate_entity_pairs_indices(span_idx)
        entities = rel.rel_rep_layer.span_marker(lstm, span_idx.unsqueeze(0))[0]
        pair_rep = rel(lstm, span_idx=span_idx.unsqueeze(0), relations_idx=pairs.unsqueeze(0))[0]
        type_rep = prompt(types)
        logits = scorer(pair_rep.unsqueeze(0), type_rep.unsqueeze(0))[0]
    index = {tuple(s): i for i, s in enumerate(spans)}
    cases.append(dict(ids=ids, labels=labels, words=words, spans=spans,
        pairs=[[index[tuple(a)], index[tuple(b)]] for a, b in pairs.tolist()],
        projected=p.flatten().tolist(), lstm=lstm[0].flatten().tolist(),
        entities=entities.flatten().tolist(), pair_reps=pair_rep.flatten().tolist(),
        types=type_rep.flatten().tolist(), logits=logits.flatten().tolist()))
(OUT / 'reduced.json').write_text(json.dumps(dict(
    seed=20261021, glirel=GLIREL, torch=torch.__version__, cases=cases), separators=(',', ':')) + '\n')

# ---- the full checkpoint through GLiREL itself ----
cfg = json.loads((SRC / 'glirel_config.json').read_text())
cfg['model_name'] = str(BASE)
local = Path(tempfile.mkdtemp())
(local / 'glirel_config.json').write_text(json.dumps(cfg))
(local / 'pytorch_model.bin').symlink_to((SRC / 'pytorch_model.bin').resolve())
model = GLiREL.from_pretrained(str(local))
model.eval()
seen = {}
model.token_rep_layer.bert_layer.model.register_forward_pre_hook(
    lambda mod, args, kwargs: seen.__setitem__('ids', (args[0] if args else kwargs['input_ids'])[0].tolist()),
    with_kwargs=True)
model.scorer.register_forward_hook(lambda mod, args, out: seen.__setitem__('logits', out[0].tolist()))
model.span_rep_layer.register_forward_hook(
    lambda mod, args, kwargs, out: seen.__setitem__('pairs', kwargs['relations_idx'][0].tolist()), with_kwargs=True)
def split(text):
    """GLiREL's words as UTF-8 byte ranges."""
    b = lambda i: len(text[:i].encode('utf-8'))
    return [[b(m.start()), b(m.end())] for m in re.finditer(r'\w+(?:[-_]\w+)*|\S', text)]


real = []
for text, ner, labels in [
    ('Derren Nesbitt had a history of being cast in "Doctor Who", having played villainous warlord Tegana '
     'in the 1964 First Doctor serial "Marco Polo".',
     [[0, 1], [10, 11], [18, 18], [22, 23], [26, 27]],
     ['country of origin', 'licensed to broadcast to', 'father', 'followed by', 'characters']),
    ('Steve Jobs founded Apple in Cupertino. Tim Cook works for Apple.',
     [[0, 1], [3, 3], [5, 5], [7, 8], [11, 11]], ['founded by', 'employer', 'headquarters location']),
    ('Marie Curie, born in Warsaw, married Pierre Curie in Paris.',
     [[0, 1], [5, 5], [8, 9], [11, 11]], ['spouse', 'place of birth', 'residence', 'sibling']),
]:
    seen.clear()
    found = model.predict_relations(text, labels, threshold=0.5, ner=[n + ['ENT', ''] for n in ner], top_k=-1)
    words = split(text)
    index = {tuple(s): i for i, s in enumerate(ner)}
    pairs = [[index[tuple(a)], index[tuple(b)]] for a, b in seen['pairs']]
    real.append(dict(text=text, words=words, ner=ner, labels=labels, ids=seen['ids'], pairs=pairs,
        logits=seen['logits'],
        relations=[dict(head=r['head_pos'], tail=r['tail_pos'], label=labels.index(r['label']), score=r['score'])
                   for r in found]))
# The word splitter against Python's `re`, on characters it treats unlike
# a naive reading of `\w`.
splits = ['cafe\u0301 a\u200db x\u00b2 \u0663 \u2167 \u00bd under_score well-known a--b _x',
          'Gr\u00f6\u00dfe \u6771\u4eac na\u00efve', 'e-mail@x.org http://a.b ok!?']
(OUT / 'real.json').write_text(json.dumps(dict(
    source=f'jackboyla/glirel-large-v0@{REVISION}', license='CC-BY-NC-SA-4.0', glirel=GLIREL,
    transformers=transformers.__version__, torch=torch.__version__, cases=real,
    splits=[dict(text=s, words=split(s)) for s in splits]), indent=1, ensure_ascii=False) + '\n')
print('fixtures complete', OUT)
