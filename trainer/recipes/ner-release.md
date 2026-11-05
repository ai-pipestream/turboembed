# Native NER prediction and paired comparison

`predict-ner` evaluates one explicitly named BIO head from an assembled
bundle on the corpus's original held-out words. It uses the native
`HeadRunner` encoder and CRF/argmax decoder, without re-tokenizing or
joining the input into synthetic sentences. Training labels do not enter
inference. WNUT, GermEval outer and GermEval inner remain distinct heads
(`ner`, `ner_outer`, `ner_inner`). MultiCoNER uses its own share-alike
bundle, never a bundle containing default-tier heads.

Assembly and zoo fetching belong to #1726 / #1748. Resolve a zoo ID with
that shared interface, including the Drive source from #1740, and pass
its verified cache directory below. These commands implement neither an
assembler nor a transport/authentication provider.

```sh
turbo-train-head predict-ner --corpus wnut17 --dir <pinned-corpus> \
  --bundle <assembled-bundle> --head ner --device cpu --precision exact \
  --commit <source-commit> --out <new-prediction-directory>
```

The new directory contains `predictions.bio` and `prediction.json`.
The receipt records corpus pins, the bundle manifest digest, the selected
head spec, actual backend/device/precision, and prediction bytes' size
and SHA-256. It contains no raw command or input directory. A receipt
identifies the inference inputs and output; it does not establish model
accuracy or authorize distribution. Altered bundles are refused by the
existing verified loader. An incompatible label schema is refused.

## Peer exchange

Export test words without gold using the existing `export` command.
Run each peer on those exact words and sentence boundaries; retain its
version and model identifiers and prediction digest. Convert BIOES or
BILOU only by validated span-preserving conversion before import.
`compare-ner` expects strict BIO and refuses malformed sequences before
mapping or dropping any type. Peer systems are evaluation references,
not training teachers or gold generators.

```sh
turbo-train-head compare-ner --corpus wnut17 --dir <pinned-corpus> \
  --predictions <native-predictions.bio> --prediction-record <prediction.json> \
  --peer-predictions <peer.bio> --peer <system-and-model-version> \
  --types person --map PERSON=person,ORG=-,GPE=-,DATE=- \
  --commit <source-commit> --out <new-comparison-directory>
```

Mappings are explicit type names, not BIO labels. `-` explicitly excludes
a peer type with no valid mapping; add mappings for every unknown type
actually present. Do not map generic organizations to WNUT corporations
merely to increase apparent coverage. The example is a person-only
comparison, not full-schema parity. The receipt retains the selected
types, mappings, every excluded span count, full native metrics and
mapped peer metrics. Explicitly dropped peer types are omitted from the
mapped peer metrics as well. No automatic acceptance threshold is used.

The candidate receipt must match the corpus record and actual prediction
bytes. Both files must preserve every original word and sentence.
The comparator requires at least 30 sentences and gold support for each
selected type. It reports exact-span micro/per-type scores and paired
whole-sentence bootstrap deltas for native versus peer and native versus
the global training-token-majority baseline. The baseline uses training
labels only, with deterministic ties; unlike the older train/peer receipts,
it is not the per-word majority baseline. A winning `I-` label that cannot
form valid BIO is refused. Bootstrap defaults to 1,000 resamples and a
fixed seed. A fixed seed reproduces the same sampled comparisons.

Use the same procedure separately for German outer and inner heads with
explicit German peer mappings. An outer-only peer does not establish
nested-entity parity. Production receipts, encoder/corpus digests,
held-out scores and zoo publication remain #1729 acceptance work; CPU
fixtures establish integration only. Do not publish raw host-bearing
training receipts or corpus text as portable release records.
