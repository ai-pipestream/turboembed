# CRF entity decoding

The current numerical API is `turbo_sequence::crf::EntityDecoder`. It accepts
a typed `EntityEncoding`, kind `TagSet`, ordered `EntityLabel` states and
native `CrfScores`. BIO and BILOU constraints apply to partition functions,
best paths and span marginals. See `analysis/sequence/README.md` for the
schema, ownership, resource limits and calibration identity requirements.

Archive-based `CrfNameFinder` and document annotator construction cannot load
models after #1511. The former MASC fixture example was removed. A custom CRF
payload inside a classical model archive does not bypass this restriction.
Native document annotation is separate work in #1540; replacement heads are
tracked in #1524. Numerical decoding support does not imply that this model
integration or its calibration has been delivered.

`analysis/sequence/tests` checks constrained decoding by enumerating complete
small lattices and comparing partitions, paths and marginals. This proof uses
native score arrays, no model archives or JVM expected output.

The earlier [CRF receipt](receipts/crf-726/README.md) and
[full MASC evaluation](namefinder-crf-evaluation.md) are historical records.
Their accuracy and timing results do not apply to a replacement model.
