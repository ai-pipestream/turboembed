# Calibrated confidence and abstention

`turbo-calibration` adds opt-in probability calibration and split conformal
prediction. Existing categorizer and name finder APIs keep their raw outputs.
The adapters in `analysis/doccat/src/calibrated.rs` and
`analysis/namefind/src/calibrated.rs` write an aligned `Calibrated<L>` sidecar
containing the raw distribution, calibrated distribution, label-index set,
artifact identity, policy, quantile, and optional abstention reason. Reasons
are `EmptySet`, `SetTooLarge`, and `BelowThreshold`, in that precedence order.
Consumers must inspect the sidecar before delivering a category or entity.
Raw output presence is not acceptance. For entities, inspect every covered
token; the token sets do not form a jointly valid BIO sequence.

## Fit, bind, and evaluate

1. Train the underlying model without using any calibration or test samples.
2. Collect full model probability distributions and gold label indices on
   three separate splits: probability fitting, conformal calibration, and
   final evaluation. Label order must be the model's outcome order.
3. Construct the actual producer identity with `turbo_doccat::calibrated::identity`
   or `turbo_namefind::calibrated::identity`. Supply a domain identifier.
4. Call `turbo_train::calibration::fit` with a `Method` and two `Samples`
   collections. The frozen transform is fitted only on the first collection;
   sorted `1 - p(gold)` scores come only from the second. Split names and SHA-256
   digests of the probability/target records are stored in the artifact.
5. Persist `Calibration::to_bytes()`. Reload with `Calibration::from_bytes`
   and the expected producer identity. Model, revision, SHA-256, task, domain,
   ordered labels, and inference configuration must match. A mismatch names
   the artifact digest. In-memory doccat models without a model digest, and
   custom feature generators without an identifiable configuration, are refused.
6. Construct `Predictor`, `CalibratedCategorizer`, or `CalibratedNameAnnotator`
   with `Policy { coverage, max_set_size, min_confidence }`. All policy fields
   enter `config_hash()`, together with the complete artifact. Built-in doccat
   feature settings also enter the producer identity.
7. Evaluate the untouched test split with `turbo_eval::calibration::reliability`
   and `coverage`. ECE is equal-width top-label ECE; Brier is the sum of squared
   errors across classes. Set coverage and delivered coverage are separate:
   an abstained example never counts as delivered, even if its set covers gold.

The binary artifact uses version `TCAL0001`, little-endian numbers, u32
lengths, and fixed field order. Decoders validate counts against available
bytes before allocation, reject trailing bytes, and validate parameters.
`DocumentGraph` codecs register the sidecars for document `Categories` and
`Tokens`; the imported policy, set, and reason must be consistent.

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
