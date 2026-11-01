# Open-Span NER Receipts (#1735)

Committed receipts for the `HEAD_OPEN_SPAN` (`turbo::head::OpenSpanMatcher`)
evaluation on WNUT-17 (`wnut17`, CC BY 4.0) over frozen `ModernBERT-base`
layer 12 (`modernbert-base-l12`, Apache-2.0):

| Receipt | Setting | Test All-Type F1 | Majority F1 | Dictionary F1 | Seen Types F1 | Zero-Shot Held-Out (`location`) F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `wnut17.json` | All 6 WNUT-17 entity types | **0.2708** | 0.0594 | 0.0000 | **0.2708** (majority 0.0594, dictionary 0.0000) | n/a |
| `wnut17-holdout-location.json` | 5 seen types; `location` held out zero-shot | **0.2601** | 0.0575 | 0.0000 | **0.2829** (majority 0.0653, dictionary 0.0000) | **0.0460** (majority 0.0000, dictionary 0.0000) |

Both runs use `--epochs 15 --patience 5 --batch 32 --lr 0.003 --threads 8`
and record `eval.dictionary` and `eval.open_span` (`seen_types`,
`held_out_types`, `seen_model`, `seen_baseline`, `seen_dictionary`,
`held_out_model`, `held_out_baseline`, `held_out_dictionary`) alongside the
paired bootstrap comparison in `receipt.json`.

External peer comparison rows (`spaCy`, `Stanza`, and local-only `GLiNER`
reference checkpoints) remain open under #1735.
