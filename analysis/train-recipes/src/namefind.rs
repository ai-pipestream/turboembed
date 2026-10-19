//! English name finders trained on MASC's named entities, one per entity
//! type, and their gates against OpenNLP's English name finders of the
//! 1.5 models.
//!
//! Each recipe trains on the training documents of [`crate::masc::SPLIT`]
//! as `NameFinderME.train` does with what the trainer tool gives it when
//! no parameters file is named
//! ([`turbo_train_components::namefind`]): `TokenNameFinderFactory` with
//! the default feature generator descriptor, no resources, the BIO codec,
//! and the defaults `NameFinderME.train` puts into empty parameters (GIS,
//! 300 iterations, cutoff 0, one thread, beam size 3). Training uses no
//! randomness, so the recipe has no seed.
//!
//! # The samples
//!
//! A sample is a sentence of a document with a named entity file, its
//! Penn tokens, and the named entities of the recipe's type
//! ([`read_samples`]). The names are MASC's entities as the named entity
//! file links them to Penn tokens ([`MascSentence::entities`]), not the
//! names OpenNLP's `MascNamedEntitySampleStream` reads, which make a name
//! of one token at a sentence's start empty, mostly keep one name of a
//! sentence or none in its overlap removal, and lose the last token of a
//! name whose tokens the file lists in some orders (see
//! [`turbo_formats::masc`]). An entity is the span from its first token
//! to one past its last; one whose tokens have a gap is kept as that span
//! and counted ([`crate::card::NamesCard::gapped`]). Entities of other
//! types are left out, as OpenNLP's `NameSampleTypeFilter` leaves them out
//! of a sample. MASC labels organizations `org`; the recipe renames them
//! `organization`, the type the reference finds. Where two entities of
//! the type overlap, the first in `Span.compareTo` order (by start, the
//! longer first) is kept and the other left out and counted
//! ([`crate::card::NamesCard::overlapping`]), since a sample's names may
//! not overlap. A sentence of no tokens is skipped: it gives no event,
//! and as a held-out unit it would count nothing. An entity in no
//! sentence of the reader's ([`MascDocument::unplaced_entities`]: outside
//! every region of the sentence file, or in a region the reader drops) is
//! in no sample, and counted
//! ([`crate::card::NamesCard::outside_sentences`],
//! [`crate::card::NamesCard::in_dropped_sentences`]); its tokens are in no
//! sample either, so it is not a name the training samples mark as
//! other. A document's first sample clears the feature generators'
//! adaptive data, as an empty line does in OpenNLP's name format and in
//! what `TokenNameFinderTool` reads, so what the previous-map feature
//! generator learns of a document's names reaches only that document's
//! later sentences; `MascNamedEntitySampleStream` clears it on every
//! sentence. This was chosen before any result; the crate's README
//! records how the gates move when every sentence clears it instead.
//! Held-out documents are read the same way.
//!
//! # The gate
//!
//! The reference is OpenNLP 1.5's English name finder of the same type
//! ([`EN_NER_PERSON`], [`EN_NER_LOCATION`], [`EN_NER_ORGANIZATION`],
//! [`EN_NER_DATE`]); MASC's four entity types each have one, so no recipe
//! is ungated. Their manifests record OpenNLP 1.5.0, 100 iterations and
//! cutoff 5; they hold no feature generator descriptor, so they run with
//! the feature generators OpenNLP gives a model without one. Their
//! training data is not published, and the zoo does not list them: a run
//! reads each from its references directory, checks it against the
//! recorded size and SHA-256, and only scores it. They are never shipped
//! or bundled, and the trained models are trained on MASC alone.
//!
//! Both models are scored on the same held-out samples with OpenNLP's
//! `TokenNameFinderEvaluator`, one unit per sentence, over the names of
//! the recipe's type ([`SpanTypes`]), and gated by
//! [`turbo_eval::model_gate`], which resamples whole sentences and scores
//! each draw from its summed counts. A name is right only with the
//! held-out name's tokens and type.
//!
//! MASC's conventions are not those the references were trained on; the
//! crate's README lists the differences and which model each counts
//! against.

use turbo_analysis::Span;
use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, Task};
use turbo_formats::evaluate::NameFinderEvaluator;
use turbo_formats::masc::MascCorpus;
use turbo_formats::{Field, Name};
use turbo_maxent::ModelFile;
use turbo_sequence::PatternMode;
use turbo_sequence::namefind::{MODEL_ENTRY_NAME, NameCodec, NameFinder};
use turbo_train::{TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::namefind::{self as train_namefind, NameEventStream, TokenNameFinderFactory};
use turbo_zoo::Sha256;

use crate::card::{HeldOutCard, NamesCard};
use crate::error::{Error, LeftOut, Result};
use crate::segmentation::{LANGUAGE, TIMESTAMP_MS};
use crate::spans::{self, SpanTypes};
use crate::{GateRun, RecordedReference, Reference, Trained, check_skipped};

mod samples;

pub use self::samples::{NameSamples, read_samples};

/// OpenNLP 1.5's English person name finder, as published.
pub const EN_NER_PERSON: RecordedReference = RecordedReference {
    id: "opennlp-en-ner-person-1.5",
    file: "en-ner-person.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-ner-person.bin",
    size: 5_207_953,
    sha256: "687a9263d96b37fced707c9f2ac0560f9edaf54658856395555901924f64dbe4",
};

/// OpenNLP 1.5's English location name finder, as published.
pub const EN_NER_LOCATION: RecordedReference = RecordedReference {
    id: "opennlp-en-ner-location-1.5",
    file: "en-ner-location.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-ner-location.bin",
    size: 5_110_658,
    sha256: "8fe39e48633f4a86c4132d9c54b396a2d8e0460c1d71e3562dacf976984f447b",
};

/// OpenNLP 1.5's English organization name finder, as published.
pub const EN_NER_ORGANIZATION: RecordedReference = RecordedReference {
    id: "opennlp-en-ner-organization-1.5",
    file: "en-ner-organization.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-ner-organization.bin",
    size: 5_297_172,
    sha256: "0136c12afe1ac357142260c39bb879b7c9d121e41024114db5a6455b4fd5ba00",
};

/// OpenNLP 1.5's English date name finder, as published.
pub const EN_NER_DATE: RecordedReference = RecordedReference {
    id: "opennlp-en-ner-date-1.5",
    file: "en-ner-date.bin",
    url: "https://opennlp.sourceforge.net/models-1.5/en-ner-date.bin",
    size: 5_030_307,
    sha256: "1207030923852e1c244919d8f15d9e78c217323728fcf909029abd1703967855",
};

/// The codec the recipes train with: `TokenNameFinderFactory`'s default.
pub const CODEC: NameCodec = NameCodec::Bio;

/// A name finder a recipe trains, of one entity type, and the OpenNLP
/// model it is gated against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NameRecipe {
    /// The trained model's zoo id, also its file name without `.bin`.
    pub model_id: &'static str,
    /// The one type the model finds and is scored on, and the MASC label
    /// renamed to it.
    pub types: SpanTypes,
    pub reference: Reference,
}

pub const PERSON: NameRecipe = NameRecipe {
    model_id: "turbo-en-masc-3.0.0-ner-person-1",
    types: SpanTypes { types: &["person"], renamed: &[] },
    reference: Reference::Recorded(EN_NER_PERSON),
};

pub const LOCATION: NameRecipe = NameRecipe {
    model_id: "turbo-en-masc-3.0.0-ner-location-1",
    types: SpanTypes { types: &["location"], renamed: &[] },
    reference: Reference::Recorded(EN_NER_LOCATION),
};

pub const ORGANIZATION: NameRecipe = NameRecipe {
    model_id: "turbo-en-masc-3.0.0-ner-organization-1",
    types: SpanTypes { types: &["organization"], renamed: &[("org", "organization")] },
    reference: Reference::Recorded(EN_NER_ORGANIZATION),
};

pub const DATE: NameRecipe = NameRecipe {
    model_id: "turbo-en-masc-3.0.0-ner-date-1",
    types: SpanTypes { types: &["date"], renamed: &[] },
    reference: Reference::Recorded(EN_NER_DATE),
};

/// The name recipes, in the order the binary runs them.
pub const RECIPES: [NameRecipe; 4] = [PERSON, LOCATION, ORGANIZATION, DATE];

/// The factory of the recipes: the language, the default descriptor, no
/// resources and [`CODEC`], as the trainer tool builds it without a
/// parameters file.
fn factory() -> Result<TokenNameFinderFactory> {
    Ok(TokenNameFinderFactory::new(LANGUAGE, None, CODEC)?)
}

/// The parameters the recipes train with once `NameFinderME.train` has put
/// its defaults in: `Algorithm=MAXENT`, `Cutoff=0`, `Iterations=300`, and
/// the trainer's own default of one thread.
pub fn training_parameters() -> TrainingParameters {
    train_namefind::training_parameters(&TrainingParameters::new())
}

impl NameRecipe {
    /// The gate's task.
    pub fn task(&self) -> Task {
        Task::NameFinder
    }

    /// The type the model finds.
    pub fn kind(&self) -> &'static str {
        self.types.types[0]
    }

    /// Trains the name finder from the training documents.
    ///
    /// # Errors
    ///
    /// What [`read_samples`] fails on, no sample or no name
    /// ([`Error::EmptySplit`]), more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]),
    /// or what training fails on.
    pub fn train(&self, train: &MascCorpus) -> Result<Trained> {
        let factory = factory()?;
        let mut stream = NameEventStream::new(&factory, None)?;
        let mut events = Vec::new();
        let mut samples = 0;
        let (counts, skipped) = read_samples(train, &self.types, |sample| {
            stream.events(&sample, &mut events)?;
            samples += 1;
            Ok(())
        })?;
        check_training(samples, &counts, skipped)?;
        let params = training_parameters();
        let mut config = TrainingConfiguration::default();
        let t = train_namefind::train_name_model_from_events(
            &events,
            &factory,
            &params,
            &mut config,
            Stamp::at(TIMESTAMP_MS),
        )?;
        Ok(Trained {
            bytes: t.package.to_bytes(),
            samples,
            skipped_sentences: skipped,
            events: events.len(),
            event_hash: t.report.event_hash.expect("trained from events"),
            names: Some(counts),
            chunks: None,
        })
    }

    /// Reads the training samples as [`NameRecipe::train`] does, without
    /// making events, and checks them as it does.
    ///
    /// # Errors
    ///
    /// As [`NameRecipe::train`], but for training.
    pub fn check_training(&self, train: &MascCorpus) -> Result<()> {
        let mut samples = 0;
        let (counts, skipped) = read_samples(train, &self.types, |_| {
            samples += 1;
            Ok(())
        })?;
        check_training(samples, &counts, skipped)
    }

    /// The held-out samples, read as the training samples are.
    ///
    /// # Errors
    ///
    /// What [`read_samples`] fails on, no sample or no name
    /// ([`Error::EmptySplit`]), or more skipped sentences than
    /// [`crate::MAX_SKIPPED_SENTENCES`] allows ([`Error::TooManyLeftOut`]).
    pub fn held_out<'c>(&self, held_out: &'c MascCorpus) -> Result<NameSamples<'c>> {
        let mut samples = Vec::new();
        let (counts, skipped) = read_samples(held_out, &self.types, |s| {
            samples.push(s);
            Ok(())
        })?;
        if samples.is_empty() || counts.names == 0 {
            return Err(Error::EmptySplit { what: "held-out name" });
        }
        check_skipped(LeftOut::HeldOutSentences, skipped, samples.len())?;
        Ok(NameSamples { samples, counts, skipped })
    }

    /// Each held-out sample's counts for the name finder in `bytes`, as
    /// `TokenNameFinderEvaluator` counts them over the names of the
    /// recipe's type: the finder's adaptive data is cleared where a sample
    /// says so, and a found name of another type is left out.
    ///
    /// # Errors
    ///
    /// A model file that is not a name finder, or what the evaluator
    /// throws on.
    pub fn score(&self, bytes: &[u8], held_out: &NameSamples<'_>) -> Result<Vec<Counts>> {
        let model = |e: &dyn std::fmt::Display| Error::Model(e.to_string());
        let file = ModelFile::open(bytes.to_vec()).map_err(|e| model(&e))?;
        let maxent = file.maxent(MODEL_ENTRY_NAME).map_err(|e| model(&e))?;
        let mut finder = NameFinder::new(&file, &maxent, PatternMode::Utf16Units).map_err(|e| model(&e))?;
        let mut units = Vec::with_capacity(held_out.len());
        let mut tokens: Vec<&str> = Vec::new();
        for sample in held_out.samples() {
            if sample.clear_adaptive_data() {
                finder.clear_adaptive_data();
            }
            tokens.clear();
            tokens.extend(sample.tokens().iter().map(Field::as_str));
            let found = finder.find(&tokens);
            let predicted: Vec<Name<'_>> = found
                .iter()
                .filter(|f| f.span.kind.as_deref().is_some_and(|k| self.types.predicted(k)))
                .map(|f| Name {
                    start: f.span.start as u32,
                    end: f.span.end as u32,
                    kind: f.span.kind.as_deref(),
                    source: Span::default(),
                })
                .collect();
            let mut evaluator = NameFinderEvaluator::default();
            evaluator.evaluate(sample, &predicted)?;
            units.push(spans::counts(evaluator.fmeasure()));
        }
        Ok(units)
    }

    /// Scores the trained model and the reference on the same held-out
    /// samples and runs the gate. `reference_id` names the reference in
    /// the record ([`Reference::id`] of [`NameRecipe::reference`] for the
    /// published model), and `set_name` and `corpus_sha256` the held-out
    /// set.
    ///
    /// # Errors
    ///
    /// What [`NameRecipe::score`] fails on, or the gate's errors.
    pub fn gate(
        &self,
        trained: &[u8],
        reference_id: &str,
        reference: &[u8],
        held_out: &NameSamples<'_>,
        set_name: &str,
        corpus_sha256: &str,
    ) -> Result<GateRun> {
        let candidate = self.score(trained, held_out)?;
        let baseline = self.score(reference, held_out)?;
        let units = spans::pair(candidate, baseline);
        let counts = held_out.counts();
        let set = HeldOutSet {
            name: set_name.to_owned(),
            corpus_sha256: corpus_sha256.to_owned(),
            samples_sha256: held_out.sha256(&self.types),
            skipped_sentences: held_out.skipped() as u64,
            scoring: self.types.scoring(counts.names, counts.other_types, Vec::new()),
        };
        let id = |id: &str, bytes: &[u8]| ModelId { id: id.to_owned(), sha256: Sha256::of(bytes).to_string() };
        let record =
            ModelGateRecord::new(self.task(), set, id(self.model_id, trained), id(reference_id, reference), units)
                .map_err(Error::Gate)?;
        let held_out = HeldOutCard {
            samples: held_out.len(),
            skipped_sentences: held_out.skipped(),
            tokens: None,
            names: Some(counts.clone()),
            chunks: None,
        };
        Ok(GateRun { record, held_out })
    }
}

/// Checks the training side as [`NameRecipe::train`] does.
fn check_training(samples: usize, counts: &NamesCard, skipped: usize) -> Result<()> {
    if samples == 0 || counts.names == 0 {
        return Err(Error::EmptySplit { what: "training name" });
    }
    check_skipped(LeftOut::TrainingSentences, skipped, samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_recipe_finds_one_type_and_names_the_reference_of_that_type() {
        for r in RECIPES {
            assert_eq!(r.types.types.len(), 1, "{}", r.model_id);
            assert_eq!(r.model_id, format!("turbo-en-masc-3.0.0-ner-{}-1", r.kind()));
            let Reference::Recorded(rec) = r.reference else { panic!("{}: a recorded reference", r.model_id) };
            assert_eq!(rec.file, format!("en-ner-{}.bin", r.kind()));
            assert_eq!(rec.id, format!("opennlp-en-ner-{}-1.5", r.kind()));
            assert_eq!(rec.url, format!("https://opennlp.sourceforge.net/models-1.5/{}", rec.file));
            assert_eq!(r.task(), Task::NameFinder);
        }
        // MASC's `org` is the reference's `organization`.
        assert_eq!(ORGANIZATION.types.held_out("org"), Some("organization"));
        assert_eq!(PERSON.types.held_out("org"), None);
    }

    #[test]
    fn the_parameters_are_name_finder_me_train_s_defaults() {
        use turbo_train::Key;
        let p = training_parameters();
        assert_eq!(p.algorithm().unwrap(), Some("MAXENT"));
        assert_eq!(p.get_int(Key::ITERATIONS).unwrap(), Some(300));
        assert_eq!(p.get_int(Key::CUTOFF).unwrap(), Some(0));
        // No thread count: the trainer's default of one.
        assert_eq!(p.get(Key::THREADS), None);
        assert_eq!(p.len(), 3);
    }
}
