# GLiNER Mixed-Corpus Evaluation Receipts (#1947)

Committed receipts for mixed-corpus (`wnut17,multiconer2-en`, 39 labels)
zero-shot evaluation with held-out schema
`location,VisualWork,Scientist,Clothing`:

- `small-v2.5-dev.json`, `small-v2.5-dev-thresholds.json`, `small-v2.5-test.json`:
  `gliner-community/gliner_small-v2.5` via `eval-gliner` at `e7b0ad4cf`
- `medium-v2.5-dev.json`, `medium-v2.5-dev-thresholds.json`, `medium-v2.5-test.json`:
  `gliner-community/gliner_medium-v2.5` via `eval-gliner` at `e7b0ad4cf`
- `gliner2-base-v1-dev.json`, `gliner2-base-v1-dev-thresholds.json`, `gliner2-base-v1-test.json`:
  `fastino/gliner2-base-v1` via `eval-gliner2`

Provenance note: the GLiNER2 receipts (`gliner2-base-v1-dev.json` and
`gliner2-base-v1-test.json`) record `"commit": "e7b0ad4cf"`; they were made
with `eval-gliner2` as this change adds it, on the tree of `e7b0ad4cf`.
