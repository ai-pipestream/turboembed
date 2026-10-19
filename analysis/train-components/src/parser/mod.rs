//! Training OpenNLP's bottom-up constituency parsers (`opennlp.tools.parser.chunking` and
//! `opennlp.tools.parser.treeinsert`): the event streams, the `train` methods and the
//! `ParserModel` file they write, and `ParserEvaluator` and `ParserCrossValidator`.
//!
//! - [`chunking`][]: the events of `chunking.ParserEventStream` and `chunking.Parser.train`.
//! - [`treeinsert`][]: the events of `treeinsert.ParserEventStream` and
//!   `treeinsert.Parser.train`.
//! - `stream` (private): what the two event streams share, the `TAG` events of
//!   `AbstractParserEventStream.addTagEvents`, and the events the parser's chunker is trained on.
//! - [`model`][]: the `ParserModel` file `new ParserModel(...)` writes and the copies
//!   `ParserModel.update*Model` make.
//! - [`evaluator`][]: `ParserEvaluator` and `ParserCrossValidator`.
//!
//! Parsing, reading a model file and the context generators are `turbo-parser`'s; the event
//! streams call the same generators the parsers do, so training and parsing share one
//! implementation of each. The parser's tagger is trained with [`train_pos_model`], as a
//! part-of-speech model is, and its chunker and its build, check and attach models with the event
//! trainer the parameters name. These live here and not in `turbo-parser` because `turbo-formats` reads
//! treebanks into `turbo_parser::Parse` trees, while training needs `turbo-formats` (the
//! part-of-speech samples, `FMeasure`) and this crate's tagger trainer.

pub mod chunking;
pub mod evaluator;
pub mod model;
mod stream;
pub mod treeinsert;

use turbo_analysis::Span;
use turbo_formats::{Field, PosSample};
use turbo_parser::model::{HeadRulesArtifact, PARSER_CHUNKER_FACTORY, ParserModel, ParserType};
use turbo_parser::{Parse, ParserPosSample};
use turbo_train::{Event, ModelPackage, TrainingConfiguration, TrainingParameters, trainer};

use crate::error::Result;
use crate::package::{FACTORY_KEY, Stamp, add_artifacts};
use crate::postag::{BEAM_SIZE_KEY, PosTaggerFactory, train_pos_model};

pub use evaluator::{ParserCrossValidator, ParserEvaluator};
pub use turbo_parser::ParserEventType;

/// Training parameters for the sub-models of a constituency parser ([`chunking::train`] and
/// [`treeinsert::train`]).
#[derive(Clone, Debug)]
pub struct ParserTrainingParameters {
    /// Dictionary n-gram frequency cutoff (`dict.Cutoff`, default `5`).
    pub dict_cutoff: usize,
    /// Parameters for `build.model`.
    pub build: TrainingParameters,
    /// Parameters for `check.model`.
    pub check: TrainingParameters,
    /// Parameters for `attach.model` (`TREEINSERT` only).
    pub attach: TrainingParameters,
    /// Parameters for `parsertager.postagger`.
    pub tagger: TrainingParameters,
    /// Parameters for `parserchunker.chunker`.
    pub chunker: TrainingParameters,
}

impl Default for ParserTrainingParameters {
    fn default() -> Self {
        Self::with_iterations_and_cutoff(100, 5)
    }
}

impl ParserTrainingParameters {
    /// Creates [`ParserTrainingParameters`] where every sub-model uses `iterations` and `cutoff`.
    pub fn with_iterations_and_cutoff(iterations: i32, cutoff: i32) -> Self {
        let make = || {
            let mut p = TrainingParameters::new();
            p.put(turbo_train::Key::ITERATIONS, iterations);
            p.put(turbo_train::Key::CUTOFF, cutoff);
            p
        };
        let mut tagger = make();
        tagger.put(BEAM_SIZE_KEY, 10);
        Self {
            dict_cutoff: cutoff.max(0) as usize,
            build: make(),
            check: make(),
            attach: make(),
            tagger,
            chunker: make(),
        }
    }
}

/// Trains a [`ParserModel`] of `parser_type` from `samples`: [`chunking::train`] or
/// [`treeinsert::train`].
pub fn train_parser(
    language: &str,
    samples: &[Parse],
    rules: HeadRulesArtifact,
    params: &ParserTrainingParameters,
    parser_type: ParserType,
) -> Result<ParserModel> {
    match parser_type {
        ParserType::Chunking => chunking::train(language, samples, rules, params),
        ParserType::TreeInsert => treeinsert::train(language, samples, rules, params),
    }
}

/// The manifest entries a parser's `train` collects from the reports of the maxent models it
/// trains (`chunking.Parser.mergeReportIntoManifest`): each report entry under its model's name and
/// a `.`, as `build.Training-Eventhash`.
#[derive(Debug, Default)]
pub(crate) struct TrainingReports(Vec<(String, String)>);

impl TrainingReports {
    /// Trains the maxent model `namespace` names on `events`, keeps its report, and returns the
    /// model's bytes.
    pub(crate) fn train(&mut self, namespace: &str, events: &[Event], params: &TrainingParameters) -> Result<Vec<u8>> {
        let (model, report) = trainer::train_events(events, params, &mut TrainingConfiguration::default())?;
        for (key, value) in report.manifest_entries() {
            self.0.push((format!("{namespace}.{key}"), value));
        }
        Ok(model.write()?)
    }

    /// The entries, for [`model::from_serialized_parts`].
    pub(crate) fn manifest_entries(&self) -> Vec<(&str, &str)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect()
    }
}

/// The part-of-speech samples `PosSampleStream` makes of `samples`, without spans.
fn collect_pos_samples(samples: &[Parse]) -> Vec<PosSample<'static>> {
    let mut out = Vec::with_capacity(samples.len());
    for p in samples {
        let pos = ParserPosSample::from_parse(p);
        let toks: Vec<Field<'static>> = pos.sentence.into_iter().map(|s| Field::made(s, Span::new(0, 0))).collect();
        let tags: Vec<Field<'static>> = pos.tags.into_iter().map(|s| Field::made(s, Span::new(0, 0))).collect();
        if let Ok(sample) = PosSample::new(toks, tags, Span::new(0, 0)) {
            out.push(sample);
        }
    }
    out
}

/// The parser's tagger (`parsertager.postagger`): `POSTaggerME.train` with a `POSTaggerFactory`
/// for `language` without a descriptor or a tag dictionary, over the part-of-speech samples of
/// `samples`.
pub(crate) fn train_parser_tagger_bytes(
    language: &str,
    samples: &[Parse],
    params: &TrainingParameters,
) -> Result<Vec<u8>> {
    let pos_samples = collect_pos_samples(samples);
    let pos_factory = PosTaggerFactory::new(language, None, None)?;
    let trained =
        train_pos_model(&pos_samples, &pos_factory, params, &mut TrainingConfiguration::default(), Stamp::at(0))?;
    Ok(trained.package.to_bytes())
}

/// The parser's chunker (`parserchunker.chunker`): `ChunkerME.train` with a
/// `ParserChunkerFactory` over the chunk samples of `samples` ([`stream::parser_chunker_events`]).
pub(crate) fn train_parser_chunker_bytes(
    language: &str,
    samples: &[Parse],
    params: &TrainingParameters,
) -> Result<Vec<u8>> {
    crate::chunk::check_trainer_type(params)?;
    let beam = crate::chunk::beam_size(params)?;
    let events = stream::parser_chunker_events(samples);
    let (model, report) = trainer::train_events(&events, params, &mut TrainingConfiguration::default())?;
    let model_bytes = model.write()?;

    let mut manifest = Stamp::at(0).manifest("ChunkerME", language);
    manifest.set_training(&report);
    manifest.set(FACTORY_KEY, PARSER_CHUNKER_FACTORY);
    manifest.set(BEAM_SIZE_KEY, &beam.to_string());
    let mut package = ModelPackage::new(manifest);
    add_artifacts(&mut package, vec![(turbo_sequence::chunker::MODEL_ENTRY_NAME, model_bytes)])?;
    Ok(package.to_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use std::str::FromStr;
    use turbo_parser::{EnglishHeadRules, ParseSampleStream};
    use turbo_train::{Key, TrainerType};

    fn tiny_samples_and_rules() -> (Vec<Parse>, HeadRulesArtifact) {
        let treebank = "\
(TOP (S (NP-SBJ (DT The) (NN dog)) (VP (VBD ran)) (. .)))
(TOP (S (NP-SBJ (DT A) (NN cat)) (VP (VBD slept)) (. .)))
";
        let samples = ParseSampleStream::new(treebank).collect::<std::result::Result<Vec<_>, _>>().unwrap();
        let rules = EnglishHeadRules::from_str("").unwrap();
        (samples, HeadRulesArtifact::English(rules))
    }

    #[test]
    fn train_parser_chunker_rejects_invalid_beam_size_and_unsupported_trainer_type() {
        let (samples, _) = tiny_samples_and_rules();

        let mut bad_beam = TrainingParameters::new();
        bad_beam.put_str(BEAM_SIZE_KEY, "not-an-int");
        let err = train_parser_chunker_bytes("en", &samples, &bad_beam).unwrap_err();
        assert!(matches!(err, Error::Train(turbo_train::Error::ParameterFormat { .. })), "{err}");

        let mut seq_params = TrainingParameters::new();
        seq_params.put_str(Key::ALGORITHM, "PERCEPTRON_SEQUENCE");
        let err = train_parser_chunker_bytes("en", &samples, &seq_params).unwrap_err();
        assert!(matches!(err, Error::UnsupportedTrainerType(Some(TrainerType::EventModelSequence))), "{err}");
    }

    #[test]
    fn chunking_train_propagates_invalid_tagger_beam_size() {
        let (samples, rules) = tiny_samples_and_rules();
        let mut params = ParserTrainingParameters::with_iterations_and_cutoff(5, 0);
        params.tagger.put_str(BEAM_SIZE_KEY, "not-an-int");
        let err = chunking::train("en", &samples, rules, &params).unwrap_err();
        assert!(matches!(err, Error::Train(turbo_train::Error::ParameterFormat { .. })), "{err}");
    }
}
