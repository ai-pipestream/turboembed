# MASC named entity recipe

Part of #1730. `masc-ne` trains a separate BIO head over person, location,
organization and date. The MASC `org` annotation maps to `organization`.
The fixed nine-label vocabulary never depends on development or test gold,
and WNUT labels are never merged into this head.

The source is MASC 3.0.0, CC BY 3.0 US. The recipe reuses the publisher
attribution, terms and extracted-corpus SHA-256 recorded in
`turbo_train_recipes::masc::RELEASE`. It refuses an unrecorded directory
unless explicitly passed `--unrecorded-corpus`; such runs are not releases.
The data stays in caller-owned storage, outside the repository.

Test documents are exactly the existing MASC recipe split, with the Twitter
genre excluded from all partitions. Every tenth remaining training document
(position 9 modulo 10) supplies development data before annotation filtering.
Documents without an NE layer supply no negative examples. Unsupported
annotations, overlaps, discontinuous entities and incomplete pieces are
errors rather than silently discarded positives.

The reader preserves sentence-local pieces in `entity_pieces()`: matching
IDs identify one original entity in a document, and `continues` is true on
every piece whose sentence lacks any original token. `entities()` remains
the complete-entity view. Sentence-only training refuses a piece with
`MascError::CutEntity`, naming its document, original entity ID and source
UTF-16 offsets. The recipe's `sentences` API retains this typed error.

```sh
turbo-train-head check --corpus masc-ne --dir corpora/MASC-3.0.0
turbo-train-head train --corpus masc-ne --dir corpora/MASC-3.0.0 \
  --bundle bundles/modernbert-base --out runs/masc-ne --crf --commit COMMIT
```

The intended encoder is `answerdotai/ModernBERT-base`, matching the Penn
POS head from #1649. Book the accelerator run in #1730 with the exact
commit, encoder digest and command before starting. This change provides
no training receipt. A corpus containing unsupported cut annotations will
refuse training until a document-aware training path can represent them.

Release still requires exact-span F1 per type against gold, the majority
baseline, spaCy `en_core_web_trf` and Stanza on the same held-out documents,
with explicit type mappings and paired bootstrap over documents. Existing
sentence-bootstrap output is not that document-level gate. Those receipts
and the default-tier English bundle shipment through #1726 remain open.
