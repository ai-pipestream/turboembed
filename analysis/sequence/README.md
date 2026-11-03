# turbo-sequence

Numerical sequence decoding, typed BIO/BILOU entity decoding, and Unicode
pattern classification. Classical OpenNLP model loading is disabled under
#1511. Archive-based constructors return an error; downloading a model cannot
make them usable. Replacement entity heads are tracked in #1524. Removal of
the remaining classical feature generators and training dependencies is
tracked in #1532.

## Typed entity scores

`crf::EntityDecoder` accepts an `EntityEncoding`, an ordered `TagSet` of entity
kinds, and an ordered list of `EntityLabel` states. Emission columns must use
that state order. A complete schema has one `Outside` state and either
`Begin`/`Inside` or `Begin`/`Inside`/`Last`/`Unit` for each kind. Empty kind sets
are supported. Invalid, missing, repeated or out-of-range states are errors.

The caller supplies `CrfScores`: token emissions, previous-label-major
transitions, start scores and end scores. The lattice validates dimensions
and finite values, then owns centered scores and forward/backward tables.
The input slices can be released after lattice construction.

`EntityLattice::best_paths(k)` returns constrained paths and their entities.
`ScoredEntity` contains a kind `Tag`, a sentence-local `IxRange<Tokens>`, and
an exact span marginal over the complete legal lattice. Map those token
indices through the existing input `Tokens` layer when creating document
entities. Do not tokenize or join text again.

`span_distribution(range)` returns one probability for no exact entity over
the supplied interval, followed by probabilities in kind order. An entity
that overlaps but does not equal the interval belongs to the first outcome.
BIO spans exclude a continuation immediately following their last token;
BILOU paths must close an entity before sentence end.

`schema_hash()` covers the encoding, kind-set name, ordered kind names and
ordered typed states. It is only a schema identity. Calibration consumers
must also bind the native model parameters and encoder identity, decoder
settings, candidate count, and alignment policy. Old archive-bound
calibrations are not compatible by implication.

## Resource limits

The numerical implementation uses dense transitions. Forward/backward work
is quadratic in label count and linear in sequence length. The lattice and
k-best APIs reject oversized logical cell counts; these limits are not a
caller-specific byte or latency budget. Callers serving untrusted workloads
must enforce their own input and concurrency limits.

K-best decoding keeps one candidate scratch buffer and only the selected
backpointers per saved cell. It does not retain discarded candidate capacity
in the saved lattice. The score tables are owned copies; this is not a
zero-copy model interface.

## Validation

The Rust tests enumerate complete small BIO/BILOU lattices independently and
compare legal paths, partition functions, token marginals and exact entity
span probabilities. They cover reordered state columns, empty sequences,
outside-only schemas, ties, large common score offsets and invalid inputs.
A separate test measures real allocator traffic to catch retained candidate
capacity growth. No model archive or JVM output supplies these test results.

Run `cargo nextest run -p turbo-sequence` for the retained suite. Classical
model downloads and JVM golden comparisons are no longer acceptance checks.
