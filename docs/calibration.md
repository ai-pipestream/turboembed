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
for a 90% request. This is conservative set coverage, not 100% classification
accuracy. The acceptance tolerance for these discrete smoke tests is a
one-sided shortfall of 3 percentage points; they do not establish closeness
to 90% from above.

| Producer | ECE before | ECE after | Brier before | Brier after |
| --- | ---: | ---: | ---: | ---: |
| DocumentCategorizerME | 0.274769 | 0.016000 | 0.566124 | 0.415640 |
| SVM categorizer | 0.164858 | 0.056439 | 0.478147 | 0.421837 |
| NameFinderME | 0.254385 | 0.124000 | 0.546524 | 0.463920 |

Run the numerical test with `cargo test --locked -p turbo-train --test calibration -- --nocapture`
and the model integration receipts with
`cargo test --locked -p turbo-doccat -p turbo-namefind --test calibrated -- --nocapture`.
