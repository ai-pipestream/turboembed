//! `TrainerFactory.getEventTrainer(params).train(...)` and
//! `getEventModelSequenceTrainer(params).train(...)`: the trainer the
//! `Algorithm` parameter names, and the model it produces. Without an
//! `Algorithm` the event trainer is `GISTrainer`, as
//! `Parameters.ALGORITHM_DEFAULT_VALUE` is `MAXENT`; the sequence trainer
//! has no default.

use crate::error::{Error, Result};
use crate::event::Event;
use crate::gis::{self, GisModel, TrainingReport};
use crate::index::DataIndex;
use crate::monitor::TrainingConfiguration;
use crate::naive_bayes::{self, NaiveBayesModel};
use crate::params::{Algorithm, TrainerType, TrainingParameters};
use crate::perceptron::{self, PerceptronModel, SequenceSource};
use crate::qn::{self, QnModel};
use crate::writer::{WritableModel, write_gis, write_naive_bayes, write_perceptron, write_qn};

/// A model an event trainer produced, by the trainer that produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainedModel {
    Gis(GisModel),
    Qn(QnModel),
    /// From the perceptron or the sequence perceptron trainer.
    Perceptron(PerceptronModel),
    NaiveBayes(NaiveBayesModel),
}

impl TrainedModel {
    /// The algorithm whose trainer made the model.
    pub fn algorithm(&self) -> Algorithm {
        match self {
            TrainedModel::Gis(_) => Algorithm::Maxent,
            TrainedModel::Qn(_) => Algorithm::MaxentQn,
            TrainedModel::Perceptron(_) => Algorithm::Perceptron,
            TrainedModel::NaiveBayes(_) => Algorithm::NaiveBayes,
        }
    }

    /// `GenericModelWriter.persist`: the bytes of the `.model` entry.
    pub fn write(&self) -> Result<Vec<u8>> {
        match self {
            TrainedModel::Gis(m) => write_gis(m),
            TrainedModel::Qn(m) => write_qn(m),
            TrainedModel::Perceptron(m) => write_perceptron(m),
            TrainedModel::NaiveBayes(m) => write_naive_bayes(m),
        }
    }

    /// The model's labels, patterns and parameters.
    pub fn as_writable(&self) -> &dyn WritableModel {
        match self {
            TrainedModel::Gis(m) => m,
            TrainedModel::Qn(m) => m,
            TrainedModel::Perceptron(m) => m,
            TrainedModel::NaiveBayes(m) => m,
        }
    }
}

/// The event trainer `Algorithm` names: `MAXENT` when absent. The sequence
/// perceptron is [`Error::WrongTrainerType`].
fn event_trainer(params: &TrainingParameters) -> Result<Algorithm> {
    let algorithm = match params.algorithm()? {
        None => Algorithm::Maxent,
        Some(name) => Algorithm::from_name(name)?,
    };
    match algorithm {
        Algorithm::Maxent | Algorithm::MaxentQn | Algorithm::Perceptron | Algorithm::NaiveBayes => Ok(algorithm),
        Algorithm::PerceptronSequence => Err(Error::WrongTrainerType { algorithm, wanted: TrainerType::Event }),
    }
}

/// `getEventModelSequenceTrainer(params, report).train(stream)`: the
/// sequence perceptron, the one sequence trainer OpenNLP has built in. An
/// absent `Algorithm` is refused as the JVM refuses it
/// (`IllegalArgumentException("Trainer type couldn't be determined!")`,
/// [`Error::NoAlgorithm`]).
pub fn train_sequences(
    source: &dyn SequenceSource,
    params: &TrainingParameters,
) -> Result<(TrainedModel, TrainingReport)> {
    let algorithm = match params.algorithm()? {
        None => return Err(Error::NoAlgorithm),
        Some(name) => Algorithm::from_name(name)?,
    };
    match algorithm {
        Algorithm::PerceptronSequence => {
            let (model, report) = perceptron::train_sequences(source, params)?;
            Ok((TrainedModel::Perceptron(model), report))
        }
        other => Err(Error::WrongTrainerType { algorithm: other, wanted: TrainerType::EventModelSequence }),
    }
}

/// `getEventTrainer(params, report, config).train(events)`.
pub fn train_events(
    events: &[Event],
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
) -> Result<(TrainedModel, TrainingReport)> {
    Ok(match event_trainer(params)? {
        Algorithm::MaxentQn => {
            let (model, report) = qn::train_events_with(events, params, config)?;
            (TrainedModel::Qn(model), report)
        }
        Algorithm::Perceptron => {
            let (model, report) = perceptron::train_events(events, params, config)?;
            (TrainedModel::Perceptron(model), report)
        }
        Algorithm::NaiveBayes => {
            let (model, report) = naive_bayes::train_events(events, params)?;
            (TrainedModel::NaiveBayes(model), report)
        }
        _ => {
            let (model, report) = gis::train_events(events, params, config)?;
            (TrainedModel::Gis(model), report)
        }
    })
}

/// `getEventTrainer(params, report, config).train(indexer)`.
pub fn train(
    index: &DataIndex,
    params: &TrainingParameters,
    config: &mut TrainingConfiguration,
) -> Result<TrainedModel> {
    Ok(match event_trainer(params)? {
        Algorithm::MaxentQn => TrainedModel::Qn(qn::train_with(index, params, config)?),
        Algorithm::Perceptron => TrainedModel::Perceptron(perceptron::train(index, params, config)?),
        Algorithm::NaiveBayes => TrainedModel::NaiveBayes(naive_bayes::train(index, params)?),
        _ => TrainedModel::Gis(gis::train(index, params, config)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::Key;

    fn events() -> Vec<Event> {
        vec![
            Event::from_strs("yes", &["a", "b"]),
            Event::from_strs("no", &["c", "b"]),
            Event::from_strs("maybe", &["d"]),
            Event::from_strs("yes", &["a", "d"]),
        ]
    }

    #[test]
    fn the_algorithm_picks_the_trainer() {
        let mut p = TrainingParameters::new();
        p.put(Key::CUTOFF, 0);
        p.put(Key::ITERATIONS, 5);
        let mut config = TrainingConfiguration::default();
        let (gis, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(gis.algorithm(), Algorithm::Maxent);
        assert_eq!(&gis.write().unwrap()[..13], b"\x0b\0TURBO-GIS-1");
        p.put(Key::ALGORITHM, Algorithm::MaxentQn.name());
        let (qn, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(qn.algorithm(), Algorithm::MaxentQn);
        assert_eq!(&qn.write().unwrap()[..12], b"\x0a\0TURBO-QN-1");
        assert_eq!(qn.as_writable().num_predicates(), 4);
        p.put(Key::ALGORITHM, Algorithm::Perceptron.name());
        let (perceptron, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(perceptron.algorithm(), Algorithm::Perceptron);
        assert_eq!(&perceptron.write().unwrap()[..20], b"\x12\0TURBO-PERCEPTRON-1");
        p.put(Key::ALGORITHM, Algorithm::NaiveBayes.name());
        let (nb, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(nb.algorithm(), Algorithm::NaiveBayes);
        assert_eq!(&nb.write().unwrap()[..21], b"\x13\0TURBO-NAIVE-BAYES-1");
        assert_eq!(nb.as_writable().num_predicates(), 4);
        p.put(Key::ALGORITHM, Algorithm::PerceptronSequence.name());
        assert_eq!(
            train_events(&events(), &p, &mut config).err(),
            Some(Error::WrongTrainerType { algorithm: Algorithm::PerceptronSequence, wanted: TrainerType::Event })
        );
        p.put(Key::ALGORITHM, "Mystery");
        assert_eq!(train_events(&events(), &p, &mut config).err(), Some(Error::UnknownAlgorithm("Mystery".into())));
    }

    /// Stops whenever it is asked.
    struct Stop;

    impl crate::monitor::StopCriteria for Stop {
        fn test(&self, _: f64) -> bool {
            true
        }

        fn message_if_satisfied(&self) -> String {
            "stop".into()
        }
    }

    /// The configuration's stop criterion reaches the quasi-Newton trainer
    /// through both entry points: a run told to stop after its first
    /// iteration is the one-iteration run, not the full one.
    #[test]
    fn qn_hears_the_configuration_s_stop_criterion() {
        let qn_params = |iterations: Option<i32>| {
            let mut p = TrainingParameters::new();
            p.put(Key::CUTOFF, 0);
            p.put(Key::ALGORITHM, Algorithm::MaxentQn.name());
            if let Some(n) = iterations {
                p.put(Key::ITERATIONS, n);
            }
            p
        };
        let parameters = |m: &TrainedModel| match m {
            TrainedModel::Qn(q) => q.all_parameters().to_vec(),
            other => panic!("{:?}", other.algorithm()),
        };
        let mut plain = TrainingConfiguration::default();
        let full = parameters(&train_events(&events(), &qn_params(None), &mut plain).unwrap().0);
        let one = parameters(&train_events(&events(), &qn_params(Some(1)), &mut plain).unwrap().0);
        assert_ne!(full, one);
        let mut stop = TrainingConfiguration::new(None, Some(Box::new(Stop)));
        assert_eq!(parameters(&train_events(&events(), &qn_params(None), &mut stop).unwrap().0), one);
        let index = crate::index::DataIndex::index(
            &events(),
            &crate::index::IndexParams::new(crate::index::IndexerKind::OnePass, 0, true),
        )
        .unwrap();
        let full = parameters(&train(&index, &qn_params(None), &mut plain).unwrap());
        let one = parameters(&train(&index, &qn_params(Some(1)), &mut plain).unwrap());
        assert_ne!(full, one);
        assert_eq!(parameters(&train(&index, &qn_params(None), &mut stop).unwrap()), one);
    }
}
