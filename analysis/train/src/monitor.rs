//! Training progress monitors and stop criteria: `opennlp.tools.monitoring`
//! and `TrainingConfiguration`.
//!
//! A trainer tells its [`ProgressMonitor`] about every finished iteration
//! and about the end of training, and asks its [`StopCriteria`] after each
//! iteration whether to stop early. [`DefaultProgressMonitor`] keeps the
//! lines `DefaultTrainingProgressMonitor` logs, formatted as it formats
//! them, so a run's log can be compared with the JVM's.

use crate::error::Result;
use crate::params::{DEFAULT_LL_THRESHOLD, DEFAULT_TOLERANCE, Key, TrainingParameters};
use turbo_jvm::text::double_to_string;

/// `TrainingMeasure`: what a trainer reports per iteration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingMeasure {
    Accuracy,
    LogLikelihood,
}

impl TrainingMeasure {
    /// `getMeasureName()`.
    pub fn name(self) -> &'static str {
        match self {
            TrainingMeasure::Accuracy => "Training Accuracy",
            TrainingMeasure::LogLikelihood => "Log Likelihood",
        }
    }
}

/// `StopCriteria<Double>`: a test on the change of a measure between two
/// iterations, and the message logged when it stops training.
pub trait StopCriteria {
    /// `test(value)`: whether training should stop.
    fn test(&self, value: f64) -> bool;

    /// `getMessageIfSatisfied()`.
    fn message_if_satisfied(&self) -> String;
}

/// `StopCriteria.FINISHED`, with the iteration count filled in.
pub fn finished_message(iterations: i32) -> String {
    format!("Training finished after completing {iterations} iterations successfully.")
}

/// `LogLikelihoodThresholdBreached`: stop when the log likelihood rose by
/// less than the threshold (`LLThreshold`, default 0.0001). The GIS
/// trainer's default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogLikelihoodThresholdBreached {
    pub threshold: f64,
}

impl LogLikelihoodThresholdBreached {
    /// With the threshold from `LLThreshold`, or the default.
    pub fn from_parameters(params: &TrainingParameters) -> Result<LogLikelihoodThresholdBreached> {
        Ok(LogLikelihoodThresholdBreached {
            threshold: params.get_double(Key::LL_THRESHOLD)?.unwrap_or(DEFAULT_LL_THRESHOLD),
        })
    }
}

impl Default for LogLikelihoodThresholdBreached {
    fn default() -> LogLikelihoodThresholdBreached {
        LogLikelihoodThresholdBreached { threshold: DEFAULT_LL_THRESHOLD }
    }
}

impl StopCriteria for LogLikelihoodThresholdBreached {
    fn test(&self, difference: f64) -> bool {
        difference < self.threshold
    }

    fn message_if_satisfied(&self) -> String {
        format!(
            "Stopping: Difference between log likelihood of current and previous iteration is less than threshold {} .",
            double_to_string(self.threshold)
        )
    }
}

/// `IterDeltaAccuracyUnderTolerance`: stop when the training accuracy
/// changed by less than the tolerance in absolute value (`Tolerance`,
/// default 0.00001). The perceptron trainer's default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IterDeltaAccuracyUnderTolerance {
    pub tolerance: f64,
}

impl IterDeltaAccuracyUnderTolerance {
    /// With the tolerance from `Tolerance`, or the default.
    pub fn from_parameters(params: &TrainingParameters) -> Result<IterDeltaAccuracyUnderTolerance> {
        Ok(IterDeltaAccuracyUnderTolerance {
            tolerance: params.get_double(Key::TOLERANCE)?.unwrap_or(DEFAULT_TOLERANCE),
        })
    }
}

impl Default for IterDeltaAccuracyUnderTolerance {
    fn default() -> IterDeltaAccuracyUnderTolerance {
        IterDeltaAccuracyUnderTolerance { tolerance: DEFAULT_TOLERANCE }
    }
}

impl StopCriteria for IterDeltaAccuracyUnderTolerance {
    fn test(&self, delta_accuracy: f64) -> bool {
        delta_accuracy.abs() < self.tolerance
    }

    fn message_if_satisfied(&self) -> String {
        format!("Stopping: change in training set accuracy less than {{{}}}", double_to_string(self.tolerance))
    }
}

/// `TrainingProgressMonitor`.
pub trait ProgressMonitor {
    /// `finishedIteration`: iteration `iteration` (1-based) classified
    /// `correct` of `total` events, and measured `value` of `measure`.
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64);

    /// `finishedTraining`: training ended after `iterations` were asked
    /// for, early when `stop` is the criterion that stopped it.
    fn finished_training(&mut self, iterations: i32, stop: Option<&dyn StopCriteria>);

    /// `isTrainingFinished`.
    fn is_training_finished(&self) -> bool;

    /// `display(clear)`: show what was recorded, and forget it when `clear`.
    fn display(&mut self, clear: bool);
}

/// `DefaultTrainingProgressMonitor`: records one line per iteration,
/// `<iteration>: (<correct>/<total>) <measure> : <value>` with the value as
/// `Double.toString` writes it, and a line for the end of training. The
/// JVM logs the lines on `display`; this one keeps them for the caller.
#[derive(Debug, Clone, Default)]
pub struct DefaultProgressMonitor {
    finished: bool,
    progress: Vec<String>,
    /// What `display` showed, in order, across calls.
    displayed: Vec<String>,
}

impl DefaultProgressMonitor {
    pub fn new() -> DefaultProgressMonitor {
        DefaultProgressMonitor::default()
    }

    /// The lines recorded and not yet cleared by `display(true)`.
    pub fn progress(&self) -> &[String] {
        &self.progress
    }

    /// Every line `display` has shown.
    pub fn displayed(&self) -> &[String] {
        &self.displayed
    }
}

impl ProgressMonitor for DefaultProgressMonitor {
    fn finished_iteration(&mut self, iteration: i32, correct: i32, total: i32, measure: TrainingMeasure, value: f64) {
        self.progress.push(format!(
            "{iteration}: ({correct}/{total}) {} : {}",
            measure.name(),
            double_to_string(value)
        ));
    }

    fn finished_training(&mut self, iterations: i32, stop: Option<&dyn StopCriteria>) {
        self.progress.push(match stop {
            Some(s) => s.message_if_satisfied(),
            None => finished_message(iterations),
        });
        self.finished = true;
    }

    fn is_training_finished(&self) -> bool {
        self.finished
    }

    fn display(&mut self, clear: bool) {
        self.displayed.extend(self.progress.iter().cloned());
        if clear {
            self.progress.clear();
        }
    }
}

/// `TrainingConfiguration(progMon, stopCriteria)`: what a trainer is given
/// to report to and to stop by. A trainer uses its own defaults for a
/// missing part (`DefaultTrainingProgressMonitor`, and its own criterion).
#[derive(Default)]
pub struct TrainingConfiguration {
    pub monitor: Option<Box<dyn ProgressMonitor>>,
    pub stop: Option<Box<dyn StopCriteria>>,
}

impl TrainingConfiguration {
    pub fn new(
        monitor: Option<Box<dyn ProgressMonitor>>,
        stop: Option<Box<dyn StopCriteria>>,
    ) -> TrainingConfiguration {
        TrainingConfiguration { monitor, stop }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_monitor_formats_like_the_jvm() {
        let mut m = DefaultProgressMonitor::new();
        m.finished_iteration(1, 150, 230, TrainingMeasure::LogLikelihood, -159.42379571219436);
        m.finished_iteration(2, 230, 230, TrainingMeasure::Accuracy, 1.0);
        assert!(!m.is_training_finished());
        m.finished_training(100, None);
        assert!(m.is_training_finished());
        assert_eq!(
            m.progress(),
            [
                "1: (150/230) Log Likelihood : -159.42379571219436",
                "2: (230/230) Training Accuracy : 1.0",
                "Training finished after completing 100 iterations successfully."
            ]
        );
        m.display(false);
        assert_eq!(m.progress().len(), 3);
        m.display(true);
        assert!(m.progress().is_empty());
        assert_eq!(m.displayed().len(), 6);
    }

    #[test]
    fn stop_criteria_test_and_report_as_the_jvm_does() {
        let ll = LogLikelihoodThresholdBreached::default();
        assert!(ll.test(0.00009));
        assert!(!ll.test(0.0001));
        assert!(ll.test(-5.0));
        assert_eq!(
            ll.message_if_satisfied(),
            "Stopping: Difference between log likelihood of current and previous iteration is less than threshold 1.0E-4 ."
        );
        let acc = IterDeltaAccuracyUnderTolerance::default();
        assert!(acc.test(-0.000001));
        assert!(!acc.test(0.00001));
        assert_eq!(acc.message_if_satisfied(), "Stopping: change in training set accuracy less than {1.0E-5}");
        let mut m = DefaultProgressMonitor::new();
        m.finished_training(100, Some(&ll));
        assert_eq!(m.progress()[0], ll.message_if_satisfied());
    }

    #[test]
    fn stop_criteria_read_their_parameters() {
        let mut p = TrainingParameters::new();
        assert_eq!(LogLikelihoodThresholdBreached::from_parameters(&p).unwrap().threshold, DEFAULT_LL_THRESHOLD);
        assert_eq!(IterDeltaAccuracyUnderTolerance::from_parameters(&p).unwrap().tolerance, DEFAULT_TOLERANCE);
        p.put(Key::LL_THRESHOLD, "0.01");
        p.put(Key::TOLERANCE, 0.5);
        assert_eq!(LogLikelihoodThresholdBreached::from_parameters(&p).unwrap().threshold, 0.01);
        assert_eq!(IterDeltaAccuracyUnderTolerance::from_parameters(&p).unwrap().tolerance, 0.5);
        p.put(Key::LL_THRESHOLD, 1);
        assert!(LogLikelihoodThresholdBreached::from_parameters(&p).is_err());
    }
}
