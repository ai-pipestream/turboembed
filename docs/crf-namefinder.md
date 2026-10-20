# CRF name finder

`CrfSequenceModel` in `turbo_sequence::crf` implements constrained BIO/BILOU
forward/backward and exact k-best decoding. `CrfTrainer` in `turbo_train::crf`
optimizes sequence likelihood with L2 and shared QN L-BFGS. BIOES maps to
BILOU start/cont/last/unit outcomes. Identical constraints apply to training
and inference. Entity probabilities are span marginals, without a claim of
empirical calibration.

Use `train_crf_name_model` with `TokenNameFinderFactory` and `CrfTrainer`.
`CrfNameFinder::read_model` loads the resulting package;
`CrfNameFinder::from_file` builds the descriptor. `NameFinder::from_crf` accepts
a context generator directly. `FeatureRow` supports binary and signed numeric
features with additive duplicates.

Emission extraction resets adaptive state per sentence, gives generators
constant `other` history, and omits po/pow/powf/ppo context features. Thus
label history cannot leak into emissions. Transitions model label dependence.

The versioned `nameFinder.crf` payload has manifest type
`turbo-linear-chain-crf-v1`. It uses the existing zip container, but is not an
OpenNLP Maxent binary. Loading validates counts, shapes, labels, duplicate
keys, finite weights and trailing bytes. Arithmetic beyond f64 support fails;
Java arbitrary-precision path sums are not implemented.

`CrfNameAnnotator` validates View/Sentences/Tokens and produces Entities plus
Alternatives<Entities>. The latter combines candidate spans and unit-token
fallback edges, with Best marking the first path. Scores sum probabilities
for mutually exclusive kinds at identical spans; fallback scores are zero.
This union permits recombinations and cannot retain ranks or entity kinds.
Use `CrfNameSequence` for exact ranked, typed paths and log probabilities.
Discontinuous sentence/token input is rejected.

Evidence and limitations: `docs/receipts/crf-726/README.md`.
Reproduce MASC with `analysis/train-recipes/examples/crf_namefind.rs`:

```sh
cargo run --release --locked -p turbo-train-recipes -- fetch-masc --home target/crf-data
# Extract the verified archive, then supply its directory:
cargo run --release --locked -p turbo-train-recipes --example crf_namefind -- DATA_DIRECTORY bio
cargo run --release --locked -p turbo-train-recipes --example crf_namefind -- DATA_DIRECTORY bilou
```

The CPU harnesses are `analysis/sequence/examples/crf_speed.rs` and
`analysis/sequence/tests/crf-reference/Speed.java`. Compile Speed.java with
CrfScorer.java and CrfScoreTable.java from the reference commit in the receipt.
Both calculate marginals and the best path on identical bounded scores.
