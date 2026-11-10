# Served relation steps on the Re-DocRED test split (#1952)

Both relation steps of `AnalyzeDocumentProgressive`, each run through a
real server on a loopback port, over the 500 Re-DocRED test documents
(`data/test_revised.json`, SHA-256 `ea673b7e...723c`, the file the
`re-docred` recipe pins; 17,448 gold triples). Each request carries the
document's words as `Tokens` and every gold entity mention as one
`Entities` row, typed by the release's entity type, in its initial graph.
No `Sentences` layer is sent, so each step reads the whole document
(document scope). Mention relations are read back as their mentions'
entities, and a triple counts when its entities, direction and property
are gold's, the rule the in-process recipe scores by. 69 of 13,018 gold
mentions are not sent: 68 repeat a span of their own entity and one
repeats another entity's span.

Machine B, CPU (AMD Ryzen 9 9950X), receipt `document.json`.

| step | model | P | R | micro F1 | found | right | seconds |
|---|---|---|---|---|---|---|---|
| `turbo-tagger:relations` | Re-DocRED head (#1734) in its ModernBERT-base bundle, manifest `1a25ecb2...d3b7` | 0.6260 | 0.3502 | 0.4492 | 9,762 | 6,111 | 689 |
| `turbo:zero-shot-relations` | GLiNER2 base v1 package, model `845fc4bd...c007` | 0.2841 | 0.0364 | 0.0645 | 2,235 | 635 | 59 |

On the ten properties commonest in training, which GLiNER2 is given
under their Wikidata English labels (gold and predictions restricted to
them, 10,842 gold triples):

| step | P | R | F1 | vs the head (paired document bootstrap, 95%) |
|---|---|---|---|---|
| `turbo-tagger:relations` | 0.6242 | 0.4817 | 0.5438 | |
| `turbo:zero-shot-relations` | 0.2841 | 0.0586 | 0.0971 | -0.4467 [-0.4785, -0.4145] |

## Reading note

- The served head scores 0.4492 against 0.4589 for the same head in the
  recipe's in-process receipt. The served step treats each `Entities` row
  as its own entity and scores mention pairs. The recipe scores entity
  pairs, pooling each entity's mentions. Mapping mention pairs back to
  entities does not recover the pooling.
- GLiNER2 refused 18 of the 500 documents. Their prompt is longer than
  the encoder's 512 pieces ("528 exceeds encoder max_sequence of 512",
  read in process), and the step does not window. Those documents are
  scored as finding nothing. Over the wire the refusal arrives as
  `TURBO_E_INTERNAL` with no detail, though it is a length limit.
- GLiNER2 is zero-shot and reads label names. The head was trained on
  this corpus. The comparison says how far the served zero-shot step is
  from a trained head on this release, not how good GLiNER2 is in
  general.

## Reproduce

The head's run directory predates the 1-based `best_epoch` of the
receipt schema (#1804). It was assembled from a copy with `host` removed,
`best_epoch` 6 renumbered 7, and the checkpoint header's `best_epoch`
renumbered to match. The head's tensor bytes are unchanged (SHA-256
`8b4277b3...fc6c` over the data section). The GLiNER2 package is the
checkpoint directory with its encoder package as `encoder/`, the tree
`analysis/pins/models.txt` pins as the progressive graph release test.

```sh
turbo-train-head assemble --bundle modernbert-base --run re-docred-head --out redocred-head-bundle
cargo run --release -p turbo-kserve --example relations_corpus -- --corpus Re-DocRED \
  --head-bundle redocred-head-bundle --gliner2 gliner2-package --scope document --out document.json
```
