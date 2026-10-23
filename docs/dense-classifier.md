# Dense classifiers and document feature selection

Issue #747 adds training to the existing fixed-weight classifier heads.
`analysis/embed/src/dense/mod.rs` exposes `DenseClassifierModel`,
`DenseClassifierTrainer` and `DenseClassifierAnnotator<L>`.

A trainer consumes `Vectors<L>` and one target row per vector. The entire
encoder `ModelRef` and vector dimension identify the feature space. Validation
and inference reject another encoder even when its dimension matches. Missing
vector rows and non-finite values return errors, rather than receiving labels.

`Architecture::Logistic` has no hidden layer. `Architecture::Mlp` has one
hidden layer with `HiddenActivation::Relu` or `HiddenActivation::Gelu`.
`OutputKind::Softmax` expects targets summing to one; `OutputKind::Sigmoid`
accepts independent targets in [0,1], including a single binary output.
For `OutputKind::Softmax`, `Categories<L>` carries all scores and the single
highest-scoring label. For `OutputKind::Sigmoid`, `DenseClassifierModel`
maintains one decision threshold per label in `(0.0, 1.0)` (defaulting to `0.5`
for every label), and `DenseClassifierAnnotator<L>` writes every label `i`
whose probability reaches `thresholds()[i]` into `Categories<L>::selected(item)`
with `Categories<L>::category[item]` set to the highest-scoring selected label
or `Tag::NONE` when no label clears its threshold (abstention).

`AdamParameters` controls the local RNG seed, learning rate, coupled L2 decay,
batch size, epochs and optional validation patience. Biases are not decayed.
Adam uses beta1=0.9, beta2=0.999, epsilon=1e-8 and bias correction. Each epoch
shuffles rows with seeded SplitMix64. A fixed seed, input order and executable
produce the same bytes. Cross-platform bit identity is not promised because
floating-point transcendental functions can differ. Numerical failures return
errors. Training refuses more than four million weights and biases before
allocating optimizer state.

With validation, training returns the checkpoint with the smallest validation
loss, including the initial checkpoint. `TrainingResult` records epochs run,
best epoch, and the returned checkpoint's training and validation losses.
Patience requires validation and a positive count. Validation does not contribute
gradients or alter the training shuffle.

`calibrate_temperature` leaves weights unchanged and searches 81 temperatures
in exp(-4)..=exp(4), including 1, against held-out cross-entropy. It also considers
the model's current temperature, so calibration cannot increase that loss.
Use a separate calibration split when reporting final test performance.

The `TDENSE01` binary wrapper records temperature and the `THEAD001` head,
including encoder identity, labels, architecture and parameters. When a
`OutputKind::Sigmoid` model has calibrated per-label thresholds that differ
from `0.5`, `to_bytes` writes a `TDENSE02` header containing temperature, the
threshold count, the little-endian `f64` thresholds in `(0.0, 1.0)`, and the
`THEAD001` payload (`TDENSE01` is preserved whenever thresholds are at their
`0.5` default or the head uses `OutputKind::Softmax`). Decoding checks counts
against remaining bytes before allocation and rejects truncation, invalid
dimensions, non-finite parameters, out-of-range thresholds, and trailing bytes.
Annotation provenance hashes the complete classifier bytes, so training,
temperature, and threshold changes invalidate cached output.

## Multi-label threshold calibration and evaluation (#980)

`DenseClassifierModel` re-exports `ThresholdCalibrator`, `MultiLabelReport`,
`Prf`, and `evaluate_multilabel_sets` from `turbo_doccat::multilabel` and
provides:

- `thresholds(&self) -> &[f64]` and `with_thresholds(self, Vec<f64>) -> Result<Self, String>`
  for inspecting or setting per-label sigmoid decision thresholds in `(0.0, 1.0)`.
- `select_tags(&self, probabilities: &[f64]) -> Result<(Tag, Vec<Tag>), String>`
  and `predicted_tags(&self, vector: &[f32]) -> Result<Vec<Tag>, String>`
  for applying per-label thresholds to a probability slice or scoring a single
  input vector and returning the emitted [`Tag`]s in label order.
- `calibrate_thresholds(&self, vectors: &Vectors<L>, truth: &[BTreeSet<usize>], calibrator: &ThresholdCalibrator) -> Result<Self, String>`
  and `calibrate_thresholds_from_targets(&self, vectors: &Vectors<L>, targets: &[Vec<f64>], calibrator: &ThresholdCalibrator) -> Result<Self, String>`
  to tune each label's threshold independently on held-out validation vectors
  by sweeping the label's observed validation scores as candidate cut points and
  maximizing per-label $F_\beta$ (optionally subject to `min_precision`, with
  ties broken toward the highest threshold), clamping the fitted threshold into
  `[1e-6, 1.0 - 1e-6]` and retaining `DEFAULT_THRESHOLD` (`0.5`) when a label has
  zero positive validation examples.
- `evaluate_multilabel(&self, vectors: &Vectors<L>, gold: &[BTreeSet<usize>]) -> Result<MultiLabelReport, String>`
  to compute micro-averaged, macro-averaged (averaged over all `0..num_labels`
  labels under `f_beta`'s zero-division conventions: precision `1.0` when no
  predictions are made, recall `1.0` when a label has no gold positives, and F1
  `0.0` when both precision and recall are `0.0`), and sample-averaged
  precision, recall, F1, plus subset exact-match accuracy.

On the WILDS Civil Comments multi-label benchmark (`269,038` training,
`45,180` validation, and `133,782` test comments across five independent toxicity
facets: `insult`, `obscene`, `threat`, `identity_attack`, and `severe_toxicity`,
with a rater fraction $\ge 0.5$ marking a positive label), fitting a linear
`OutputKind::Sigmoid` head (`AdamParameters { seed: 42, learning_rate: 0.01, weight_decay: 1e-5, batch_size: 256, epochs: 8, patience: None }`)
on sentence embeddings and calibrating per-label thresholds on the validation
split with `ThresholdCalibrator::default()` (`beta = 1.0`,
`min_precision = None`) yields:

| Encoder | Split | Thresholds (`insult`, `obscene`, `threat`, `identity_attack`, `severe_toxicity`) | Micro P / R / F1 | Macro P / R / F1 | Exact match |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `sentence-transformers/all-MiniLM-L6-v2` | val (`45,180`) | flat `[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]` | `0.6845 / 0.1135 / 0.1947` | `0.8424 / 0.2397 / 0.2666` | `0.9228` |
| `sentence-transformers/all-MiniLM-L6-v2` | val (`45,180`) | calibrated `[0.1928, 0.0429, 0.0248, 0.1319, 0.5000]` | `0.3496 / 0.4862 / 0.4068` | `0.4230 / 0.4775 / 0.4409` | `0.8634` |
| `sentence-transformers/all-MiniLM-L6-v2` | test (`133,782`) | flat `[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]` | `0.6788 / 0.1103 / 0.1898` | `0.8468 / 0.0401 / 0.0681` | `0.9226` |
| `sentence-transformers/all-MiniLM-L6-v2` | test (`133,782`) | calibrated `[0.1928, 0.0429, 0.0248, 0.1319, 0.5000]` | `0.3452 / 0.4714 / 0.3985` | `0.4220 / 0.2735 / 0.2390` | `0.8628` |
| `BAAI/bge-small-en-v1.5` | val (`45,180`) | flat `[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]` | `0.7602 / 0.1101 / 0.1923` | `0.8706 / 0.2377 / 0.2642` | `0.9234` |
| `BAAI/bge-small-en-v1.5` | val (`45,180`) | calibrated `[0.2018, 0.0476, 0.0186, 0.1369, 0.5000]` | `0.3899 / 0.4568 / 0.4207` | `0.4439 / 0.4602 / 0.4472` | `0.8780` |
| `BAAI/bge-small-en-v1.5` | test (`133,782`) | flat `[0.5000, 0.5000, 0.5000, 0.5000, 0.5000]` | `0.7308 / 0.1043 / 0.1825` | `0.8689 / 0.0364 / 0.0624` | `0.9232` |
| `BAAI/bge-small-en-v1.5` | test (`133,782`) | calibrated `[0.2018, 0.0476, 0.0186, 0.1369, 0.5000]` | `0.3856 / 0.4475 / 0.4142` | `0.4321 / 0.2571 / 0.2401` | `0.8777` |

## Sparse feature selection

`analysis/doccat/src/selection.rs` provides `FeatureSelection` with
`SelectionMetric::MutualInformation` (natural-log units) or `SelectionMetric::Chi2`
(Pearson's statistic). Ranking uses binary feature presence versus all classes,
including absence counts. Repeated occurrences count once for ranking. Zero-valued
features count as absent; negative or non-finite values are rejected. Equal scores
are ordered by the feature's UTF-16 ordering for reproducibility.

Fit the selector on training events only, then call `SelectedFeatures::filter`.
Filtering preserves outcome labels, predicate multiplicities and numeric values.
The output feeds MaxEnt or `turbo_train::liblinear::train_events`; Liblinear's
usual sorted/unique predicate requirement still applies. Unselected features
have no learned weights, so inference ignores them without an extra filter.

`turbo_doccat::DoccatTrainer` prepares selected events from document samples and
a `DoccatFactory`. Existing `turbo_train_components::doccat` training entry points
also accept opt-in parameters `FeatureSelection=MI` or `FeatureSelection=Chi2`
and positive `FeatureSelectionTopK`. Filtering runs before indexing and cutoff.
Without those parameters, the previous training behavior is unchanged.

Tests cover separable data, XOR (MLP versus linear), binary and independent
sigmoid targets, per-label sigmoid threshold calibration and `TDENSE02`
serialization, end-to-end multi-label annotation on `testdata/tiny-bert-bundle`,
deterministic training, numerical gradients, early stopping, temperature
calibration, codec boundaries, encoder mismatches, and MI/Chi2 feature reduction
with retained classification accuracy for MaxEnt and Liblinear.
