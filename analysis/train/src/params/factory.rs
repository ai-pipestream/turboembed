use super::{Key, TrainingParameters};
use crate::error::{Error, Result};

/// `opennlp.tools.ml.AlgorithmType`: the trainers OpenNLP has built in,
/// as the `Algorithm` parameter names them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// `MAXENT`: `GISTrainer`, model type `GIS`.
    Maxent,
    /// `MAXENT_QN`: `QNTrainer`, model type `QN`.
    MaxentQn,
    /// `PERCEPTRON`: `PerceptronTrainer`, model type `Perceptron`.
    Perceptron,
    /// `PERCEPTRON_SEQUENCE`: `SimplePerceptronSequenceTrainer`, an event model sequence
    /// trainer, model type `Perceptron`.
    PerceptronSequence,
    /// `NAIVEBAYES`: `NaiveBayesTrainer`, model type `NaiveBayes`.
    NaiveBayes,
}

impl Algorithm {
    pub const ALL: [Algorithm; 5] = [
        Algorithm::Maxent,
        Algorithm::MaxentQn,
        Algorithm::Perceptron,
        Algorithm::PerceptronSequence,
        Algorithm::NaiveBayes,
    ];

    /// `getAlgorithmType()`: the `Algorithm` parameter value.
    pub fn name(self) -> &'static str {
        match self {
            Algorithm::Maxent => "MAXENT",
            Algorithm::MaxentQn => "MAXENT_QN",
            Algorithm::Perceptron => "PERCEPTRON",
            Algorithm::PerceptronSequence => "PERCEPTRON_SEQUENCE",
            Algorithm::NaiveBayes => "NAIVEBAYES",
        }
    }

    /// `getModelType()`: the type string at the start of the model file.
    pub fn model_type(self) -> &'static str {
        match self {
            Algorithm::Maxent => "GIS",
            Algorithm::MaxentQn => "QN",
            Algorithm::Perceptron | Algorithm::PerceptronSequence => "Perceptron",
            Algorithm::NaiveBayes => "NaiveBayes",
        }
    }

    /// `fromAlgorithmType`, matched exactly.
    pub fn from_name(name: &str) -> Result<Algorithm> {
        Algorithm::ALL.into_iter().find(|a| a.name() == name).ok_or_else(|| Error::UnknownAlgorithm(name.to_owned()))
    }

    /// `fromModelType`: the first algorithm with this model type
    /// (`Perceptron` is [`Algorithm::Perceptron`]).
    pub fn from_model_type(model_type: &str) -> Option<Algorithm> {
        Algorithm::ALL.into_iter().find(|a| a.model_type() == model_type)
    }

    /// Which kind of trainer the algorithm's trainer class is, as
    /// `TrainerFactory.getTrainerType` finds it: `SimplePerceptronSequenceTrainer`
    /// extends `AbstractEventModelSequenceTrainer`, an
    /// `EventModelSequenceTrainer`, and every other built in trainer is an
    /// `EventTrainer`. No built in trainer is a `SequenceTrainer`.
    pub fn trainer_type(self) -> TrainerType {
        match self {
            Algorithm::PerceptronSequence => TrainerType::EventModelSequence,
            _ => TrainerType::Event,
        }
    }
}

/// `TrainerFactory.TrainerType`, with the `TrainerType` parameter value
/// each trainer interface writes into the report (`EventTrainer.EVENT_VALUE`
/// and the others).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainerType {
    /// `EVENT_MODEL_TRAINER`, value `Event`.
    Event,
    /// `EVENT_MODEL_SEQUENCE_TRAINER`, value `EventModelSequence`.
    EventModelSequence,
    /// `SEQUENCE_TRAINER`, value `Sequence`.
    Sequence,
}

impl TrainerType {
    /// The `TrainerType` parameter value.
    pub fn value(self) -> &'static str {
        match self {
            TrainerType::Event => "Event",
            TrainerType::EventModelSequence => "EventModelSequence",
            TrainerType::Sequence => "Sequence",
        }
    }
}

/// `TrainerFactory.getTrainerType`: `Event` with no `Algorithm` parameter,
/// the algorithm's type when it names a built in one, and `None` for a name
/// OpenNLP would try to load as an extension class. A non-string value is
/// an error, as the string getter throws.
pub fn trainer_type(params: &TrainingParameters) -> Result<Option<TrainerType>> {
    Ok(match params.algorithm()? {
        None => Some(TrainerType::Event),
        Some(name) => Algorithm::from_name(name).ok().map(Algorithm::trainer_type),
    })
}

/// `TrainerFactory.isValid`: false when `Algorithm` names no built in
/// trainer or `Cutoff` or `Iterations` is a string that is not an
/// integer. Reads `DataIndexer` as a string too, so a non-string value
/// there, or under `Algorithm`, is an error.
pub fn is_valid(params: &TrainingParameters) -> Result<bool> {
    if let Some(name) = params.algorithm()?
        && Algorithm::from_name(name).is_err()
    {
        return Ok(false);
    }
    for key in [Key::CUTOFF, Key::ITERATIONS] {
        match params.get_int(key) {
            Ok(_) => {}
            Err(Error::ParameterFormat { .. }) => return Ok(false),
            Err(e) => return Err(e),
        }
    }
    params.get_str(Key::DATA_INDEXER)?;
    Ok(true)
}
