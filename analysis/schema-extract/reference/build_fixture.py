"""Reference outputs for native schema-driven extraction.

Usage: build_fixture.py <gliner2-base-v1 checkout> <relation reduced.safetensors> <output directory>

The checkout is fastino/gliner2-base-v1 at REVISION (Apache-2.0). Two
fixtures are written:

- classifier.safetensors and reduced.json: the classification head
  sliced to width 32 to go with the relation scorer's reduced fixture
  (the encoder and heads sliced to width 32), and the outputs of
  GLiNER2's own layer classes: count queries across three and four
  fields, and classifier logits.
- real.json: what the full checkpoint gives through GLiNER2's own
  `extract` on fixed texts and schemas (entities, classifications and
  structures together): the input ids, each span task's instance count,
  each classification's probabilities, and the decoded results with
  confidences and character offsets. Only outputs are written.

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
from gliner2.layers import CountLSTMv2, create_mlp

REVISION = 'f9634218e53580c56edf0de97ca1a7d3f1c2354e'
GLINER2 = 'ca4f68235b50fe4c3ecf3876851d748fe4b15c6c'
SRC, REDUCED, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
full = load_file(str(SRC / 'model.safetensors'))
D = 32

# ---- reduced: the relation fixture plus the classifier at width 32 ----
tensors = load_file(str(REDUCED))
tensors['classifier.0.weight'] = np.ascontiguousarray(full['classifier.0.weight'][:2 * D, :D])
tensors['classifier.0.bias'] = np.ascontiguousarray(full['classifier.0.bias'][:2 * D])
tensors['classifier.2.weight'] = np.ascontiguousarray(full['classifier.2.weight'][:, :2 * D])
tensors['classifier.2.bias'] = np.ascontiguousarray(full['classifier.2.bias'])
# Only the classifier is written; the rest is the relation fixture's.
save_file({k: torch.from_numpy(v) for k, v in tensors.items() if k.startswith('classifier.')},
          OUT / 'classifier.safetensors')

config = DebertaV2Config(vocab_size=16, hidden_size=D, intermediate_size=48,
    num_hidden_layers=2, num_attention_heads=2, max_position_embeddings=32,
    relative_attention=True, position_buckets=8, max_relative_positions=32,
    norm_rel_ebd='layer_norm', share_att_key=True, pos_att_type=['c2p', 'p2c'],
    position_biased_input=False, type_vocab_size=0, layer_norm_eps=1e-7,
    hidden_dropout_prob=0, attention_probs_dropout_prob=0)
encoder = DebertaV2Model(config).eval()
encoder.load_state_dict({k[len('encoder.'):]: torch.from_numpy(v) for k, v in tensors.items()
                         if k.startswith('encoder.')}, strict=True)
part = lambda prefix: {k[len(prefix):]: torch.from_numpy(v) for k, v in tensors.items() if k.startswith(prefix)}
count_embed = CountLSTMv2(hidden_size=D).eval()
count_embed.load_state_dict(part('count_embed.'))
classifier = create_mlp(D, [2 * D], 1, dropout=0., activation='relu').eval()
classifier.load_state_dict(part('classifier.'))

cases = []
for ids, fields, rows in [
    ([5, 3, 9, 1, 4, 2, 4, 6, 7, 8, 11, 12, 13, 14, 10], [1, 4, 6], [0, 7, 9, 12]),
    ([3, 1, 4, 2, 4, 9, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], [0, 2, 5, 9], [3, 5, 8]),
]:
    with torch.no_grad():
        h = encoder(torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
        queries = {str(k): count_embed(h[fields], k).flatten().tolist() for k in (1, 3)}
        logits = classifier(h[rows]).squeeze(-1).tolist()
    cases.append(dict(ids=ids, fields=fields, queries=queries, rows=rows, logits=logits))
(OUT / 'reduced.json').write_text(json.dumps(dict(
    source=f'fastino/gliner2-base-v1@{REVISION}', license='Apache-2.0', gliner2=GLINER2,
    torch=torch.__version__, cases=cases), separators=(',', ':')) + '\n')

# ---- the full checkpoint through GLiNER2 itself ----
model = GLiNER2.from_pretrained(str(SRC))
model.eval()
seen = {}
model.encoder.register_forward_pre_hook(
    lambda mod, args, kwargs: seen.__setitem__('ids', kwargs.get('input_ids', args[0] if args else None)[0].tolist()),
    with_kwargs=True)
model.count_pred.register_forward_hook(
    lambda mod, args, out: seen.setdefault('counts', []).extend(out.reshape(-1, 20).argmax(-1).tolist()))
model.classifier.register_forward_hook(
    lambda mod, args, out: seen.setdefault('classifier', []).append(out.reshape(-1).tolist()))


def build(spec):
    """GLiNER2's schema from the fixture's schema description."""
    s = model.create_schema()
    for st in spec.get('structures', []):
        b = s.structure(st['name'])
        for f in st['fields']:
            b.field(f['name'], dtype='str' if f.get('single') else 'list', choices=f.get('choices'),
                    description=f.get('description'), threshold=f.get('threshold'))
    if spec.get('entities'):
        s.entities({e['name']: {k: v for k, v in [('description', e.get('description')),
                                                   ('threshold', e.get('threshold')),
                                                   ('dtype', 'str' if e.get('single') else 'list')]
                                if v is not None} for e in spec['entities']})
    for c in spec.get('classifications', []):
        s.classification(c['task'], [l['name'] for l in c['labels']], multi_label=c.get('multi', False),
                         cls_threshold=c.get('threshold', 0.5), top_k=c.get('top_k'))
        # Descriptions only for the labels that have one.
        descs = {l['name']: l['description'] for l in c['labels'] if l.get('description')}
        if descs:
            s.schema['classifications'][-1]['label_descriptions'] = descs
    return s


def byte(text, i):
    return len(text[:i].encode('utf-8'))


def bytes_of(text, value):
    """Character offsets in a decoded value as UTF-8 byte offsets."""
    if isinstance(value, dict):
        if 'start' in value and 'end' in value:
            out = dict(value)
            out['start'], out['end'] = byte(text, out['start']), byte(text, out['end'])
            return out
        return {k: bytes_of(text, v) for k, v in value.items()}
    if isinstance(value, list):
        return [bytes_of(text, v) for v in value]
    return value


SCHEMA_A = {
    'entities': [{'name': 'person'}, {'name': 'company', 'description': 'a business or firm'},
                 {'name': 'city', 'threshold': 0.3}, {'name': 'product', 'single': True}],
    'classifications': [
        {'task': 'sentiment', 'labels': [{'name': 'positive'}, {'name': 'negative'}, {'name': 'neutral'}]},
        {'task': 'topics', 'labels': [{'name': 'technology', 'description': 'computers and software'},
                                      {'name': 'business'}, {'name': 'sports'}],
         'multi': True, 'threshold': 0.4, 'top_k': 2}],
    'structures': [{'name': 'product', 'fields': [
        {'name': 'name', 'single': True}, {'name': 'price', 'single': True, 'description': 'the cost in money'},
        {'name': 'features'}, {'name': 'condition', 'single': True, 'choices': ['new', 'used', 'refurbished']}]}],
}
SCHEMA_B = {
    'entities': [{'name': 'person'}, {'name': 'organization'}, {'name': 'location'}, {'name': 'date'}],
    'structures': [{'name': 'event', 'fields': [{'name': 'what'}, {'name': 'when', 'single': True},
                                                 {'name': 'where'}]},
                   {'name': 'employment', 'fields': [{'name': 'employee'}, {'name': 'employer', 'single': True},
                                                      {'name': 'role', 'choices': ['engineer', 'manager', 'founder']}]}],
}
SCHEMA_C = {
    'classifications': [{'task': 'language', 'labels': [{'name': 'english'}, {'name': 'french'},
                                                         {'name': 'german'}]}],
    'entities': [{'name': 'dish', 'description': 'a food or meal'}, {'name': 'place'}],
}
real = []
for text, spec in [
    ('Apple released the new iPhone 16 in Cupertino on Monday. Tim Cook said it costs $799 and has a '
     'faster chip, a brighter screen and a longer battery. Reviewers loved it.', SCHEMA_A),
    ('Marie Dubois joined Renault as an engineer in Paris on 3 March 2021, after the company opened its '
     'new research centre.', SCHEMA_B),
    ('We ate a café crème and a croque-monsieur at a naïve little bistro near Montmartre.', SCHEMA_C),
    ('Steve Jobs founded Apple in 1976 with Steve Wozniak; the company later sold the refurbished Mac at a '
     'discount of 20 percent.', SCHEMA_A),
]:
    seen.clear()
    out = model.extract(text, build(spec), include_confidence=True, include_spans=True)
    real.append(dict(text=text, schema=spec, ids=seen['ids'], counts=seen.get('counts', []),
                     classifier=seen.get('classifier', []), result=bytes_of(text, out)))
(OUT / 'real.json').write_text(json.dumps(dict(
    source=f'fastino/gliner2-base-v1@{REVISION}', license='Apache-2.0', gliner2=GLINER2,
    torch=torch.__version__, cases=real), indent=1, ensure_ascii=False) + '\n')
print('fixtures complete', OUT)
