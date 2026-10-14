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
use crate::params::{Algorithm, TrainerType, TrainingParameters};
use crate::perceptron::{self, PerceptronModel, SequenceSource};
use crate::qn::{self, QnModel};
use crate::writer::{WritableModel, write_gis, write_perceptron, write_qn};

/// A model an event trainer produced, by the trainer that produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainedModel {
    Gis(GisModel),
    Qn(QnModel),
    /// From the perceptron or the sequence perceptron trainer.
    Perceptron(PerceptronModel),
}

impl TrainedModel {
    /// The algorithm whose trainer made the model.
    pub fn algorithm(&self) -> Algorithm {
        match self {
            TrainedModel::Gis(_) => Algorithm::Maxent,
            TrainedModel::Qn(_) => Algorithm::MaxentQn,
            TrainedModel::Perceptron(_) => Algorithm::Perceptron,
        }
    }

    /// `GenericModelWriter.persist`: the bytes of the `.model` entry.
    pub fn write(&self) -> Result<Vec<u8>> {
        match self {
            TrainedModel::Gis(m) => write_gis(m),
            TrainedModel::Qn(m) => write_qn(m),
            TrainedModel::Perceptron(m) => write_perceptron(m),
        }
    }

    /// The model's labels, patterns and parameters.
    pub fn as_writable(&self) -> &dyn WritableModel {
        match self {
            TrainedModel::Gis(m) => m,
            TrainedModel::Qn(m) => m,
            TrainedModel::Perceptron(m) => m,
        }
    }
}

/// The event trainer `Algorithm` names: `MAXENT` when absent. An algorithm
/// OpenNLP has but this crate does not train yet is [`Error::NotPorted`];
/// the sequence perceptron is [`Error::WrongTrainerType`].
fn event_trainer(params: &TrainingParameters) -> Result<Algorithm> {
    let algorithm = match params.algorithm()? {
        None => Algorithm::Maxent,
        Some(name) => Algorithm::from_name(name)?,
    };
    match algorithm {
        Algorithm::Maxent | Algorithm::MaxentQn | Algorithm::Perceptron => Ok(algorithm),
        Algorithm::PerceptronSequence => Err(Error::WrongTrainerType { algorithm, wanted: TrainerType::Event }),
        other => Err(Error::NotPorted(other)),
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
            let (model, report) = qn::train_events(events, params)?;
            (TrainedModel::Qn(model), report)
        }
        Algorithm::Perceptron => {
            let (model, report) = perceptron::train_events(events, params, config)?;
            (TrainedModel::Perceptron(model), report)
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
        Algorithm::MaxentQn => TrainedModel::Qn(qn::train(index, params)?),
        Algorithm::Perceptron => TrainedModel::Perceptron(perceptron::train(index, params, config)?),
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
        assert_eq!(&gis.write().unwrap()[..5], b"\0\x03GIS");
        p.put(Key::ALGORITHM, Algorithm::MaxentQn.name());
        let (qn, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(qn.algorithm(), Algorithm::MaxentQn);
        assert_eq!(&qn.write().unwrap()[..4], b"\0\x02QN");
        assert_eq!(qn.as_writable().num_predicates(), 4);
        p.put(Key::ALGORITHM, Algorithm::Perceptron.name());
        let (perceptron, _) = train_events(&events(), &p, &mut config).unwrap();
        assert_eq!(perceptron.algorithm(), Algorithm::Perceptron);
        assert_eq!(&perceptron.write().unwrap()[..12], b"\0\x0aPerceptron");
        p.put(Key::ALGORITHM, Algorithm::PerceptronSequence.name());
        assert_eq!(
            train_events(&events(), &p, &mut config).err(),
            Some(Error::WrongTrainerType { algorithm: Algorithm::PerceptronSequence, wanted: TrainerType::Event })
        );
        p.put(Key::ALGORITHM, Algorithm::NaiveBayes.name());
        assert_eq!(train_events(&events(), &p, &mut config).err(), Some(Error::NotPorted(Algorithm::NaiveBayes)));
        p.put(Key::ALGORITHM, "Mystery");
        assert_eq!(train_events(&events(), &p, &mut config).err(), Some(Error::UnknownAlgorithm("Mystery".into())));
    }
}
