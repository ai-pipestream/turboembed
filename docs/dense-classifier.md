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
`Categories<L>` carries all scores and their highest scoring label. This API
does not implement multi-label threshold selection.

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
including encoder identity, labels, architecture and parameters. Decoding checks
counts against remaining bytes before allocation and rejects truncation, invalid
dimensions, non-finite parameters and trailing bytes. Annotation provenance hashes
the complete classifier bytes, so training and temperature changes invalidate
cached output.

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
sigmoid targets, deterministic training, numerical gradients, early stopping,
calibration, codec boundaries, encoder mismatches, and MI/Chi2 feature reduction
with retained classification accuracy for MaxEnt and Liblinear. These are
synthetic correctness checks, not accuracy claims for a production corpus.
