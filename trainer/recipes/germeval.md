# GermEval 2014 native NER recipes

Part of #1524. `germeval2014-outer` and `germeval2014-inner` train
separate CRF heads (`ner_outer`, `ner_inner`) on a frozen encoder. Both
use `turboembed:entities`; neither replaces or flattens the other level.
An assembled runtime pipeline must run both heads on shared word
features and retain overlapping entities from different levels. This
recipe does not yet implement that combined runtime or publish a model.

## Source, rights and pins

The [University of Hamburg publisher page](https://www.inf.uni-hamburg.de/en/inst/ab/lt/resources/data/germaner.html)
states CC BY 4.0. Credit Darina Benikova, Chris Biemann and Marc Reznicek
(NoSta-D, 2014), and Darina Benikova, Chris Biemann, Max Kisselew and
Sebastian Pado (GermEval companion paper, 2014), and link the source with
any distributed head. This is default-tier data under the #1524 ruling.

Use the corrected 2022-04-28 files linked by the
[original publisher](https://sites.google.com/site/germeval2014ner/data)
in its [download folder](https://drive.google.com/drive/folders/1kC0I2UGl2ltrluI9NqDjaQJGw5iliw_J).
The publisher corrected encoding artifacts; the recipe does not rewrite
the data. Download and retain the three files in caller-owned storage.
The older Hamburg zip has different bytes and is refused by these pins.

| File | Bytes | SHA-256 |
|---|---:|---|
| `NER-de-train.tsv` | 7882321 | `98682c21700e5bd7959ac361150679929e4af5d91e4dd7c379cc89133a810994` |
| `NER-de-dev.tsv` | 723854 | `2ffe8405acc6bb6111b62a88623aa1e1f2e6db16b77fc005c54163541807aa05` |
| `NER-de-test.tsv` | 1682725 | `ec6e6adce5ed71cee119ae758e569ea4be54c6be5319b3b49dfcdcb05e51301b` |

The native reader accepts the documented token ID, token, outer BIO,
inner BIO columns, checking both label columns even when training only
one. IDs are consecutive within each sentence; original spelling and
boundaries are preserved. One optional trailing tab is framing (the
pinned test file contains one); additional or nonempty fields are errors.
Malformed BIO sequences, unknown types, empty tokens and comments inside
sentences fail instead of silently dropping rows.

Both heads declare all 25 BIO labels: O plus B/I for PER, ORG, LOC, OTH
and their `deriv` and `part` variants. Types do not disappear when a
training split lacks their I label. The vocabulary never inspects dev
or test labels. This matters for the published dev data, which includes
valid labels absent from training. Neither compounds nor derivations
are silently folded into the four main types.

Complete pinned reads give 24,000/2,200/5,100 sentences and
452,853/41,653/96,499 tokens for train/dev/test, with zero skipped
sentences for either level.

## Train and evaluate

```sh
turbo-train-head check --corpus germeval2014-outer --dir corpora/germeval2014
turbo-train-head train --corpus germeval2014-outer --dir corpora/germeval2014 \
  --bundle bundles/german-encoder --out runs/germeval-outer --crf --threads 2 --commit COMMIT
turbo-train-head train --corpus germeval2014-inner --dir corpora/germeval2014 \
  --bundle bundles/german-encoder --out runs/germeval-inner --crf --threads 2 --commit COMMIT
turbo-train-head export --corpus germeval2014-outer --dir corpora/germeval2014 --out german-test.tsv
turbo-train-head peer --corpus germeval2014-outer --dir corpora/germeval2014 \
  --peer PEER-MODEL-AND-VERSION --column bio --predictions peer.tsv \
  --out runs/german-peer --commit COMMIT
```

Export/import uses the same strict original-token BIO protocol as
[WNUT](wnut17.md), with explicit `--map` spelling conversions when needed.
Report each level separately with its exact-span F1 and majority
baseline. A flat peer cannot establish nested-entity parity; do not
invent inner predictions or treat an unsupported subtype as O to make
scores comparable. A coarse or combined official-task metric would need
its own explicit scoring protocol and tests; it is not reported here.

Book full accelerator training with krick-1 in #1524 using the exact
commit, encoder digest and command. CPU tests use authored fixtures and
tiny BERT to prove real fitting, complete dev inclusion and saved-head
inference agreement. They do not establish German accuracy. Full
training, paired spaCy/Stanza measurements where schemas map, combined
runtime packaging, quality gates and release still remain in #1524.
