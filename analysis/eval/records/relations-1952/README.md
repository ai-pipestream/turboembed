# Served relation steps on the Re-DocRED test split (#1952)

Both relation steps of `AnalyzeDocumentProgressive`, each run through a
real server on a loopback port, over the 500 Re-DocRED test documents
(`data/test_revised.json`, SHA-256 `ea673b7e...723c`, the file the
`re-docred` recipe pins; 17,448 gold triples). Each request carries the
document's words as `Tokens` and every gold entity mention as one
`Entities` row, typed by the release's entity type, in its initial graph.
At document scope no `Sentences` layer is sent, so a pair may join
mentions anywhere in the document. At sentence scope the request also
carries the corpus's sentences as `Sentences` and sets
`within_sentence`, so only mentions in one sentence are paired. Mention relations are read back as their mentions'
entities, and a triple counts when its entities, direction and property
are gold's, the rule the in-process recipe scores by. 69 of 13,018 gold
mentions are not sent: 68 repeat a span of their own entity and one
repeats another entity's span.

Machine B, CPU (AMD Ryzen 9 9950X), receipts `document.json` and
`sentence.json`. Each step's predictions hash the same over two runs
(`predictions_sha256`).

### Document scope

| step | model | P | R | micro F1 | found | right | seconds |
|---|---|---|---|---|---|---|---|
| `turbo-tagger:relations` | Re-DocRED head (#1734) in its ModernBERT-base bundle, manifest `1a25ecb2...d3b7` | 0.6260 | 0.3502 | 0.4492 | 9,762 | 6,111 | 603 |
| `turbo:zero-shot-relations` | GLiNER2 base v1 package, model `845fc4bd...c007` | 0.2841 | 0.0364 | 0.0645 | 2,235 | 635 | 49 |

On the ten properties commonest in training, which GLiNER2 is given
under their Wikidata English labels (gold and predictions restricted to
them, 10,842 gold triples):

| step | P | R | F1 | vs the head (paired document bootstrap, 95%) |
|---|---|---|---|---|
| `turbo-tagger:relations` | 0.6242 | 0.4817 | 0.5438 | |
| `turbo:zero-shot-relations` | 0.2841 | 0.0586 | 0.0971 | -0.4467 [-0.4785, -0.4145] |

### Sentence scope

| step | P | R | micro F1 | found | right | seconds |
|---|---|---|---|---|---|---|
| `turbo-tagger:relations` | 0.7473 | 0.0927 | 0.1650 | 2,165 | 1,618 | 597 |
| `turbo:zero-shot-relations` | 0.3615 | 0.0260 | 0.0485 | 1,256 | 454 | 46 |

On the ten properties:

| step | P | R | F1 | vs the head (paired document bootstrap, 95%) |
|---|---|---|---|---|
| `turbo-tagger:relations` | 0.7426 | 0.1296 | 0.2207 | |
| `turbo:zero-shot-relations` | 0.3615 | 0.0419 | 0.0751 | -0.1456 [-0.1653, -0.1271] |

Pairing only within a sentence raises both steps' precision and cuts
their recall: most Re-DocRED triples join entities whose mentions never
share a sentence, so no in-sentence pair can find them.

## Reading note

- The served head scores 0.4492 against 0.4589 for the same head in the
  recipe's in-process receipt. The served step treats each `Entities` row
  as its own entity and scores mention pairs. The recipe scores entity
  pairs, pooling each entity's mentions. Mapping mention pairs back to
  entities does not recover the pooling.
- GLiNER2 refused the same 18 of the 500 documents at both scopes. The
  step encodes the whole document in one prompt, and `within_sentence`
  only limits which pairs it scores, so a prompt over the encoder's 512
  pieces is over at either scope. The step does not window. Each refusal
  arrives as `TURBO_E_CAPACITY` (771) naming the limit and both counts
  (for example "over its encoder max_sequence 512, got 528"), is listed
  in the receipt, and is scored as finding nothing.
- GLiNER2 is zero-shot and reads label names. The head was trained on
  this corpus. The comparison says how far the served zero-shot step is
  from a trained head on this release, not how good GLiNER2 is in
  general.

## Gates

Quality is measured here on Re-DocRED with gold mentions given. The
redistribution class is the one docs/licences.md records for each model
and its lineage.

| model | step | quality on this release | redistribution | stage |
|---|---|---|---|---|
| Re-DocRED head on ModernBERT-base (#1734) | `turbo-tagger:relations` | micro F1 0.4492 (document), 0.1650 (sentence) | share-alike: trained on Re-DocRED (CC BY-SA 4.0); offered for download once the user accepts, never in a default bundle | experimental |
| GLiNER2 base v1 (`fastino/gliner2-base-v1`) | `turbo:zero-shot-relations` | F1 0.0971 (document) and 0.0751 (sentence) on ten properties; refuses documents over 512 pieces | local-only by its training data; fetched from its publisher after the user accepts the card's terms | preview |
| GLiREL large v0 (`jackboyla/glirel-large-v0`) | none served | not measured on this corpus | local-only, CC BY-NC-SA 4.0; fetched after the user accepts that licence | experimental, in process only |

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
cargo run --release -p turbo-kserve --example relations_corpus -- --corpus Re-DocRED \
  --head-bundle redocred-head-bundle --gliner2 gliner2-package --scope sentence --out sentence.json
```
