//! `TrainerFactory.getEventTrainer(params).train(...)`: the event trainer
//! the `Algorithm` parameter names, and the model it produces. Without an
//! `Algorithm` the trainer is `GISTrainer`, as `Parameters.ALGORITHM_DEFAULT_VALUE`
//! is `MAXENT`.

use crate::error::{Error, Result};
use crate::event::Event;
use crate::gis::{self, GisModel, TrainingReport};
use crate::index::DataIndex;
use crate::monitor::TrainingConfiguration;
use crate::params::{Algorithm, TrainingParameters};
use crate::qn::{self, QnModel};
use crate::writer::{WritableModel, write_gis, write_qn};

/// A model an event trainer produced, by the trainer that produced it.
#[derive(Debug, Clone, PartialEq)]
pub enum TrainedModel {
    Gis(GisModel),
    Qn(QnModel),
}

impl TrainedModel {
    /// The algorithm whose trainer made the model.
    pub fn algorithm(&self) -> Algorithm {
        match self {
            TrainedModel::Gis(_) => Algorithm::Maxent,
            TrainedModel::Qn(_) => Algorithm::MaxentQn,
        }
    }

    /// `GenericModelWriter.persist`: the bytes of the `.model` entry.
    pub fn write(&self) -> Result<Vec<u8>> {
        match self {
            TrainedModel::Gis(m) => write_gis(m),
            TrainedModel::Qn(m) => write_qn(m),
        }
    }

    /// The model's labels, patterns and parameters.
    pub fn as_writable(&self) -> &dyn WritableModel {
        match self {
            TrainedModel::Gis(m) => m,
            TrainedModel::Qn(m) => m,
        }
    }
}

/// The event trainer `Algorithm` names: `MAXENT` when absent. An algorithm
/// OpenNLP has but this crate does not train yet is [`Error::NotPorted`].
fn event_trainer(params: &TrainingParameters) -> Result<Algorithm> {
    let algorithm = match params.algorithm()? {
        None => Algorithm::Maxent,
        Some(name) => Algorithm::from_name(name)?,
    };
    match algorithm {
        Algorithm::Maxent | Algorithm::MaxentQn => Ok(algorithm),
        other => Err(Error::NotPorted(other)),
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
        p.put(Key::ALGORITHM, Algorithm::NaiveBayes.name());
        assert_eq!(train_events(&events(), &p, &mut config).err(), Some(Error::NotPorted(Algorithm::NaiveBayes)));
        p.put(Key::ALGORITHM, "Mystery");
        assert_eq!(train_events(&events(), &p, &mut config).err(), Some(Error::UnknownAlgorithm("Mystery".into())));
    }
}
