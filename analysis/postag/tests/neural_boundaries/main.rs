//! Boundary regressions for the CRF scorer and the BiLSTM-CRF model (#649):
//! dimension mismatches, empty tag sets, and out-of-range gold indices must
//! fail with typed errors, never panic, and `BilstmPosModel::new` must hold
//! the shape, value, and vocabulary contract `load` enforces. Gradient
//! accumulators handed to `CrfScorer::loss_and_gradients` are checked before
//! any of them is written.

mod crf;
mod model;
