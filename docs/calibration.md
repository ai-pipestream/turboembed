# Calibrated confidence and abstention

`turbo-calibration` adds opt-in probability calibration and split conformal
prediction. Existing categorizer and name finder APIs keep their raw outputs.
The adapters in `analysis/doccat/src/calibrated.rs` and
`analysis/namefind/src/calibrated.rs` write an aligned `Calibrated<L>` sidecar
containing the raw distribution, calibrated distribution, label-index set,
artifact identity, policy, quantile, and optional abstention reason. Reasons
are `EmptySet`, `SetTooLarge`, and `BelowThreshold`, in that precedence order.
Consumers must inspect the sidecar before delivering a category or entity.
Raw output presence is not acceptance. For `NameFinderME`
(`CalibratedNameAnnotator`, `Calibrated<Tokens>`), inspect every covered token
because token sets are prefix-conditioned and do not form a jointly valid BIO
sequence. For `CrfNameFinder` (`CalibratedCrfNameAnnotator`,
`Calibrated<Entities>`), each decoded entity span carries its own calibrated
sequence-marginal distribution over `["other", kind_0, ...]`, aligned 1-to-1
with `Entities`.

## Fit, bind, and evaluate

1. Train the underlying model without using any calibration or test samples.
2. Collect full model probability distributions and gold label indices on
   three separate splits: probability fitting, conformal calibration, and
   final evaluation. Label order must be the model's outcome order (or
   `CrfSequenceModel::span_outcomes()` for span-level CRF calibration).
3. Construct the actual producer identity with `turbo_doccat::calibrated::identity`,
   `turbo_namefind::calibrated::identity` (`Task::TokenTagging`), or
   `turbo_namefind::calibrated::crf_identity` (`Task::SpanTagging`). Supply a
   domain identifier.
4. Call `turbo_train::calibration::fit` with a `Method` and two `Samples`
   collections. The frozen transform is fitted only on the first collection;
   sorted `1 - p(gold)` scores come only from the second. Split names and SHA-256
   digests of the probability/target records are stored in the artifact.
5. Persist `Calibration::to_bytes()`. Reload with `Calibration::from_bytes`
   and the expected producer identity. Model, revision, SHA-256, task, domain,
   ordered labels, and inference configuration must match. A mismatch names
   the artifact digest. In-memory doccat models without a model digest, and
   custom feature generators without an identifiable configuration, are refused.
6. Construct `Predictor`, `CalibratedCategorizer`, `CalibratedNameAnnotator`,
   or `CalibratedCrfNameAnnotator` with
   `Policy { coverage, max_set_size, min_confidence }`. All policy fields
   enter `config_hash()`, together with the complete artifact. Built-in doccat
   feature settings and CRF sequence codec settings also enter the producer
   identity.
7. Evaluate the untouched test split with `turbo_eval::calibration::reliability`
   and `coverage`. ECE is equal-width top-label ECE; Brier is the sum of squared
   errors across classes. Set coverage and delivered coverage are separate:
   an abstained example never counts as delivered, even if its set covers gold.

The binary artifact uses version `TCAL0001`, little-endian numbers, u32
lengths, and fixed field order. Decoders validate counts against available
bytes before allocation, reject trailing bytes, and validate parameters.
`DocumentGraph` codecs register the sidecars for document `Categories`,
`Tokens`, and `Entities`; the imported policy, set, and reason must be
consistent.

## Span-level CRF calibration

`NameFinderME` decodes with beam search and returns per-token MaxEnt
distributions conditioned on the winning prefix (`Task::TokenTagging`). In
contrast, `CrfSequenceModel` (`analysis/sequence/src/crf/model.rs`) defines a
globally normalized probability distribution over legal BIO or BILOU paths.
For any contiguous token span `start..end`:

- Each entity kind `k` in `CrfSequenceModel::entity_kinds()` corresponds to a
  unique legal BIO or BILOU state sequence on `start..end` (with `{k}-cont`
  excluded at `end` under BIO). Because distinct entity kinds require distinct
  start or unit states at `start`, the events that `start..end` is a complete
  entity of kind `k` are mutually exclusive across kinds.
- `CrfSequenceModel::span_distribution(&lattice, start, end)` evaluates each
  kind's exact forward-backward marginal via `CrfLattice::span_probability` and
  places the complement probability `1 - sum_k p(k)` at index `0` (`"other"`,
  not a complete entity on `start..end`), yielding a normalized distribution
  over `CrfSequenceModel::span_outcomes()` (`["other", kind_0, ...]`).
- `CrfNameFinder::find_scored_spans` pairs each top-1 decoded entity with its
  span marginal distribution, and `CrfNameFinder::candidate_spans(tokens, k)`
  returns the ordered unique candidate spans from the top-`k` constrained paths
  with their span marginal distributions for calibration fitting and conformal
  scoring.
- `CalibratedCrfNameAnnotator` writes `(Entities, Calibrated<Entities>)`, where
  row `i` of `Calibrated<Entities>` aligns to entity `i` of `Entities`, and
  `Entities` aligns both to `View` byte spans and `Tokens` index ranges. When a
  candidate span is ambiguous between an entity kind and `"other"`, both indices
  enter the conformal prediction set and a single-label policy (`max_set_size: 1`)
  abstains with `Abstention::SetTooLarge`.

## Methods and coverage interpretation

- Temperature scaling applies softmax to `log(p) / T`, preserving exact zero
  input probabilities. Fitting minimizes held-out negative log likelihood.
- Platt scaling fits a pooled one-vs-rest sigmoid of clipped log odds, then
  normalizes the class scores. Slope is nonnegative.
- Isotonic regression groups equal inputs before pool-adjacent-violators.
  Inference uses right-closed steps, clamps outside the observed range, and
  normalizes across classes. Zero fitted scores receive a positive floor.

All input distributions must be finite, in `[0, 1]`, and sum to one within
`1e-6`. Raw SVM margins are not probabilities and are not accepted.
Temperature is bounded to `[1e-6, 1e6]`; sigmoid parameters are bounded to
avoid infinite arithmetic. Fitting and conformal splits with identical names
or identical probability/target digests are refused. Distinct digests do not
prove that the caller used disjoint examples; dataset separation remains a
caller responsibility.

The conformal rank is `ceil((n + 1) * coverage)`. Rank `n + 1` uses an infinite
cutoff, producing every label instead of incorrectly clamping to the last
observed score. Ties can give conservative coverage. The requested coverage
is a marginal guarantee under exchangeability, not a per-example correctness
promise. A confidence threshold or set-size cap may abstain and reduce
**delivered** coverage. Domain shift, adaptive token histories, and dependent
tokens can invalidate exchangeability. NameFinderME returns local model
probabilities conditioned on the winning decoded prefix, not sequence marginals.
Fit and evaluate with the same adaptive-state boundaries as deployment.

The temperature method follows [Guo et al.](https://proceedings.mlr.press/v70/guo17a.html).
The finite-sample conformal quantile follows
[Angelopoulos and Bates](https://arxiv.org/abs/2107.07511).

## Validation receipts

Tests train small real GIS models and execute the real SVM evaluator and
NameFinderME decoder. They do not mock inference. The synthetic domains use
independent seeds for fitting, conformal calibration, and evaluation; they
are reproducible API/numerical checks, not corpus quality claims.

On 4,000 continuous binary test examples with requested coverage 90% and
absolute tolerance 4 percentage points:

| Transform | ECE before | ECE after | Test coverage |
| --- | ---: | ---: | ---: |
| Temperature | 0.150675 | 0.016772 | 0.886250 |
| Platt | 0.150675 | 0.016772 | 0.886250 |
| Isotonic | 0.150675 | 0.026911 | 0.894250 |

The per-model tests use 300 fitting, 301 conformal, and 1,000 evaluation rows.
Their deliberately coarse probability values have ties: coverage is 100%
for a 99% request, within the stated absolute tolerance of 3 percentage points.
At this high requested coverage the sets contain both labels. This is
conservative set coverage, not 100% classification accuracy; a one-label
set-size policy would abstain. The continuous test above also checks a
less conservative 90% request.

| Producer | ECE before | ECE after | Brier before | Brier after |
| --- | ---: | ---: | ---: | ---: |
| DocumentCategorizerME | 0.274769 | 0.016000 | 0.566124 | 0.415640 |
| SVM categorizer | 0.164858 | 0.056439 | 0.478147 | 0.421837 |
| NameFinderME | 0.254385 | 0.124000 | 0.546524 | 0.463920 |

Run the numerical test with `cargo test --locked -p turbo-train --test calibration -- --nocapture`
and the model integration receipts with
`cargo test --locked -p turbo-doccat -p turbo-namefind --test calibrated -- --nocapture`.

## Real-corpus ECE and coverage records

### MASC 3.0.0 (`analysis/train-recipes/examples/masc_calibration.rs`)

`analysis/train-recipes/examples/masc_calibration.rs` evaluates all three
transforms (`Temperature`, `Platt`, `Isotonic`) at requested coverage `0.90` on
the verified MASC 3.0.0 release (`corpus_sha256`
`445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd`, CC BY 3.0
US, Twitter genre excluded by `masc::SPLIT`; see `docs/licences.md`). Reproduce
with:

```sh
cargo run --release --locked -p turbo-train-recipes --example masc_calibration -- DATA_DIRECTORY
```

Models are trained only on `split.train` (351 documents). The 39 held-out
documents in `split.held_out` are partitioned deterministically (`i % 3`) into
disjoint `masc-fit`, `masc-conformal`, and `masc-test` splits.
`Single-label delivered` reports `delivered_coverage` under the stricter
decision policy `Policy { coverage: 0.90, max_set_size: 1, min_confidence: 0.50 }`.

1. **`DocumentCategorizerME` (4 genres: `prose-reference`, `news-press`, `correspondence`, `spoken`)**:
   40-token passages grouped per document (`fit_n = 424`, `conformal_n = 763`, `test_n = 530`, raw accuracy `0.700000`).

| Transform | ECE | Brier | Set coverage (`0.90`) | Mean set size | Single-label delivered | Abstention rate | Artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw | 0.087442 | 0.441689 | n/a | n/a | n/a | n/a | n/a |
| Temperature | 0.039741 | 0.431322 | 0.807547 | 1.372 | 0.511321 | 0.373585 | `20052b0d6027a3d7619a01c1524ace15df63e3979143033ecf90ebbdb080b0cd` |
| Platt | 0.037501 | 0.431160 | 0.801887 | 1.360 | 0.511321 | 0.371698 | `717cd7f30aaebe1a9b3d4c5062ad44b7aecf4c732d8fa23e1050389f37435447` |
| Isotonic | 0.051353 | 0.435257 | 0.800000 | 1.342 | 0.537736 | 0.333962 | `ef6a4ecfee0ac6d644df4a639d7dc31ad96f169e35b31d731a28a42fae208a7b` |

   Because passages are partitioned by whole document (13 held-out documents
   per split), passage-level exchangeability is only approximate across
   documents, and realized set coverage (`0.800` to `0.808`) reflects
   document-level genre clustering.

2. **`NameFinderME` token-level (`Task::TokenTagging`, BIO, 9 outcomes across `person`, `location`, `organization`, `date`)**:
   Trained on 2,048 genre-spread training sentences; evaluated on 900 held-out
   sentences split `i % 3` (`fit_n = 4069`, `conformal_n = 3724`, `test_n = 4078` tokens, raw accuracy `0.967141`).

| Transform | ECE | Brier | Set coverage (`0.90`) | Mean set size | Single-label delivered | Abstention rate | Artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw | 0.017955 | 0.053879 | n/a | n/a | n/a | n/a | n/a |
| Temperature | 0.004669 | 0.051509 | 0.896027 | 0.902 | 0.896027 | 0.097842 | `8c2f14558bedd0595e614700a332dc910297a621dccf2d0a62351db992713ac2` |
| Platt | 0.007387 | 0.051755 | 0.895782 | 0.902 | 0.895782 | 0.098087 | `bb92597bd277e8d765a15599267029d4d71683d1def725b859dd588b97cf0e01` |
| Isotonic | 0.003900 | 0.051084 | 0.896273 | 0.902 | 0.896273 | 0.097597 | `6f46983328f399809aaeb4f98f34e59229177bfbc3e2638e0c666a7a6454c945` |

3. **`CrfNameFinder` span-level (`Task::SpanTagging`, BIO, 5 outcomes `["other", "date", "location", "organization", "person"]`)**:
   Candidate spans from `CrfNameFinder::candidate_spans(..., 4)` bound via
   `crf_identity_with_candidates(..., 4)` and `CalibratedCrfNameAnnotator::with_candidates(..., 4)`
   on the same held-out sentence splits (`fit_n = 550`, `conformal_n = 558`, `test_n = 542` candidate spans, raw accuracy `0.850554`).

| Transform | ECE | Brier | Set coverage (`0.90`) | Mean set size | Single-label delivered | Abstention rate | Artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw | 0.036268 | 0.213593 | n/a | n/a | n/a | n/a | n/a |
| Temperature | 0.038892 | 0.211607 | 0.898524 | 1.120 | 0.784133 | 0.132841 | `5de64ae1508fe7cf59ed8ad0edb9eb602779e251464f7d886454d74a291ff6eb` |
| Platt | 0.045838 | 0.211462 | 0.898524 | 1.120 | 0.782288 | 0.134686 | `c2b0523a7a356a2ce6fcf4af3ddc1ff7dbdfb1d48adbdebfc6178e60574178b6` |
| Isotonic | 0.042270 | 0.209870 | 0.904059 | 1.133 | 0.776753 | 0.154982 | `0580a9c01865b12fd34144cbe83a027235abc0914d022912c1ab0b467855f4bc` |

4. **`CrfNameFinder` span-level (`Task::SpanTagging`, BILOU, 5 outcomes `["other", "date", "location", "organization", "person"]`)**:
   Candidate spans from `CrfNameFinder::candidate_spans(..., 4)` bound via
   `crf_identity_with_candidates(..., 4)` and `CalibratedCrfNameAnnotator::with_candidates(..., 4)`
   on the same held-out sentence splits (`fit_n = 551`, `conformal_n = 553`, `test_n = 548` candidate spans, raw accuracy `0.868613`).

| Transform | ECE | Brier | Set coverage (`0.90`) | Mean set size | Single-label delivered | Abstention rate | Artifact SHA-256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Raw | 0.026169 | 0.185469 | n/a | n/a | n/a | n/a | n/a |
| Temperature | 0.023087 | 0.183377 | 0.914234 | 1.109 | 0.804745 | 0.124088 | `9b60cfe431e886e8522a3cb7673c5ff1291504c8e8a6be34ec281d7022a7a812` |
| Platt | 0.026345 | 0.183358 | 0.912409 | 1.108 | 0.802920 | 0.125912 | `32284f429c373887ce8041bb4f2166aa89a693c9fe0a8dfb478a1819dc39f973` |
| Isotonic | 0.024549 | 0.185042 | 0.901460 | 1.091 | 0.810219 | 0.105839 | `6640a979ad41e6c4e08d3b2b654b492509a42b80803ce2316cf26b00b958d5e3` |

### Checked-in GermEval 2014 and passage excerpts

`analysis/namefind/tests/calibrated.rs`
(`germeval_real_corpus_token_and_crf_span_calibration_receipts`) and
`analysis/doccat/tests/calibrated.rs`
(`real_passage_corpus_receipts_across_all_transforms`) run in CI on the
checked-in 105-sentence GermEval 2014 excerpt
(`analysis/goldens/data/inputs/ner/germeval-excerpt.txt`, SHA-256
`0bc6a217acfc21773d49b2ab3861e0cd318c2ce825374a4aba5b242571eaa53c`, CC BY 4.0)
and checked-in passage fixtures (`analysis/goldens/data/inputs/svm-doccat-train/topics.txt`
and `analysis/goldens/data/inputs/sentiment-train/feedback.txt`):

| Producer and corpus | Transform | ECE before | ECE after | Brier before | Brier after | Set coverage (`0.90`) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `CrfNameFinder` span-level (GermEval 2014 excerpt, `test_n = 50`) | Temperature | 0.078290 | 0.149059 | 0.317316 | 0.316092 | 0.980000 |
| `CrfNameFinder` span-level (GermEval 2014 excerpt, `test_n = 50`) | Platt | 0.078290 | 0.139214 | 0.317316 | 0.315288 | 0.980000 |
| `CrfNameFinder` span-level (GermEval 2014 excerpt, `test_n = 50`) | Isotonic | 0.078290 | 0.114776 | 0.317316 | 0.324982 | 0.920000 |
| `DocumentCategorizerME` (checked-in passages, `test_n = 26`) | Temperature | 0.189156 | 0.283442 | 0.530043 | 0.581107 | 0.846154 |
| `DocumentCategorizerME` (checked-in passages, `test_n = 26`) | Platt | 0.189156 | 0.253670 | 0.530043 | 0.570071 | 0.846154 |
| `DocumentCategorizerME` (checked-in passages, `test_n = 26`) | Isotonic | 0.189156 | 0.373917 | 0.530043 | 0.718390 | 1.000000 |

## Allocation and validation contract

`Predictor::predict_into` writes scores and prediction-set indices into caller
buffers. `Calibrated<L>` stores binary single-row outputs inline and larger
batches in flat columns with CSR prediction sets. Its identity and policy are
immutable and checked once; public `push` checks every model distribution,
and `from_rows` checks every imported decision against its policy. The wire
codec uses that checked constructor. Consequently `Layer::check` does not
repeat an identity walk or revalidate unchanged rows.

Use `row(i)` or `rows()` to read decisions and `metadata()` for the identity,
calibration digest and policy. Mutable public columns are no longer exposed.
The allocation regression verifies zero allocations for binary single-row
construction, prediction and validation, and fewer than 64 allocations for a
1,024-row batch. Name-finder capture uses one flat score vector rather than
one vector per token; its type tags are prepared once per annotator. Its
calibration identity now includes the parsed beam size and pattern mode.

## Four-feature microbenchmark

The driver is `analysis/doccat/examples/calibration_timing.rs`. Build with
`cargo build --locked --release -p turbo-doccat --example calibration_timing`.
Arguments: `maxent`, `svm`, or `names`; token count; `raw` or `calibrated`;
optional `checked` to include output `Layer::check` for categorizers. Both
variants use the same release executable and the same tiny two-label fixtures.
Model construction and artifact binding are outside the timed loop. Each
process warms up for 100 calls, then measures 500,000 classifier calls or
3,000 name-finder calls. Four CPU-4-pinned ABBA rounds yield eight observations
per variant. Names measures decoding plus token calibration, without the
adapter's input validation and entity assembly. These are not deployment-model
or complete Document/Analyzer timings.

The updated measurements below include output validation for classifiers.

| Producer | Tokens | Raw ns/call | Calibrated ns/call | Change |
| --- | ---: | ---: | ---: | ---: |
| maxent | 16 | 385.05 | 531.91 | +38.14% |
| maxent | 128 | 3097.80 | 3167.01 | +2.23% |
| svm | 16 | 355.11 | 400.81 | +12.87% |
| svm | 128 | 2278.77 | 2327.43 | +2.14% |
| names | 16 | 22302.62 | 22969.38 | +2.99% |
| names | 128 | 227152.73 | 238230.23 | +4.88% |

The short classifier cases expose fixed overhead; they are not the
size fixture below. The MaxEnt short-input run was noisy (raw 365-669 ns,
calibrated 458-705 ns). Keep these costs visible when choosing calibration
for very small models and short inputs.

## Deployment-size inference timing

`analysis/doccat/examples/calibration_deployment.rs` constructs deterministic
models with 16,384 vocabulary features. The vocabulary uses real words from
the checked-in MiniLM vocabulary. The MaxEnt models have two or eight labels;
the probabilistic linear SVM has 64 support vectors with 32 sparse values
each; NameFinderME has seven BIO labels. These are size fixtures through real
inference, not trained deployment checkpoints or accuracy benchmarks.

The workload cycles through 64 different 128-token documents. Classifiers run
the full annotator and output layer validation, asserting equal raw results
before measurement. NameFinder runs decoding, flat token probability capture,
and calibration, clearing adaptive state per document. It does not include
the layer adapter's span checks and entity assembly. Temperature is 2, coverage
is 0.9, and maximum set size is the number of labels. Other fitted transforms
and complete Analyzer pipelines require their own measurements.

Build with `cargo build --locked --release -p turbo-doccat --example calibration_deployment`.
Run `taskset -c 14 target/release/examples/calibration_deployment maxent 128 2`
and substitute `svm 128`, `names 128`, or `maxent 128 8`. On an AMD Ryzen 9
9950X3D, each process warms both variants 256 times and runs four ABBA rounds
with 200,000 classifier calls or 3,000 name-finder calls per observation.
The table uses the median of eight observations per variant. Construction
and binding are outside the loop. Timings ran sequentially without this
checkout compiling at the same time.

| Producer | Labels | Raw ns/call | Calibrated ns/call | Change |
| --- | ---: | ---: | ---: | ---: |
| MaxEnt | 2 | 5189.14 | 5203.45 | +0.28% |
| SVM | 2 | 7799.83 | 7786.81 | -0.17% |
| NameFinderME | 7 | 378011.46 | 382521.56 | +1.19% |
| MaxEnt | 8 | 6120.88 | 6229.04 | +1.77% |

All four median changes are within two percent on this workload. The small
negative SVM result is timing noise, not a speedup claim. Raw observations
are in `docs/receipts/calibration-718/deployment.txt`; host contention remains
visible in some rounds. An earlier CPU-4 run before the multiclass square-root
fast path gave +8.32% for names with substantial timing variation. These
results support this deployment-size workload's speed bar, not a blanket
latency guarantee for every model, transform, input length or complete pipeline.

Batch sidecars retain raw probabilities intentionally: graph export/import
and policy validation remain self-contained if the producer is unavailable
or changes. Binary single-row storage still needs no heap allocation.
