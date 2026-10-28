# Native MASC NER preparation

`native_ner::prepare` reads local MASC-layout files. It performs no network
access and reads no model archive. `native_ner::fit_cpu` runs a linear BIO
CRF head on word features from a supplied bundle with CPU Exact precision
and one encoder thread. It returns trained tensors and a report in memory.

The supported MASC labels are person, location, organization (MASC `org`)
and date. Documents without NE annotations supply no examples. Unknown
labels, discontinuous or overlapping entities, annotation warnings,
unplaced entities and invalid token bounds are refused. The recipe reads
`MascSentence::entities`, not the legacy lossy `names` projection.

The existing MASC test split is unchanged: document position 9 modulo 10,
in the MASC reader's order. Twitter exclusions retain their existing
positions. Development takes position 8 modulo 10 of the remaining
training-side documents, before filtering for NE annotations. These are
the project's established test split and a new development rule, not an
upstream official three-way split. Fitting reads train/dev only; test data
cannot affect early stopping. Relative document names, text digests and
partitions are retained. The report hashes all source files, partition
membership, training settings and the encoder manifest. A hash identifies
bytes; it does not certify rights, publisher provenance or model quality.

The CPU smoke uses project-authored text and the repository's generated
tiny-BERT fixture. Its historical manifest labels do not establish
pretrained encoder provenance. The test trains twice, compares tensor
bytes, requires decreasing loss and compares core decoding with the
loaded EntityTagger on Unicode source spans. The standard bundle written
by this test is only a test artifact. Production packaging will use the
shared trainer exporter, not a new format in this recipe.

## Before production training or release

- The current MASC reader can divide an entity across sentences without
  exposing the original identity or a partial-entity marker. The formats
  API must expose that case so this recipe can refuse it before a real
  training run. Sentence-local validation cannot reconstruct lost IDs.
- Verify MASC publisher and licence provenance and match input digests to
  the approved release. The existing release record documents a publisher
  link that was not independently checked.
- Select and verify a licence-clean production encoder. The generated
  test fixture is not a candidate pretrained encoder.
- Integrate WNUT17 after its reader and pins merge. Keep separate dataset
  heads until a partial-label objective or suitable shared annotations
  prevents unannotated types from becoming negative labels. WNUT's
  corporation does not automatically mean all MASC organizations.
- Supply verified time, money and percentage annotations. MASC and WNUT
  do not establish those missing classes.
- Run real held-out gold and named-peer evaluations, complete training,
  package through the shared exporter, and publish release evidence.

This preparation does not close #1524 and supplies no corpus accuracy,
GPU throughput or release claim.
