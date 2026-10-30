use std::collections::{HashMap, HashSet};

use turbo_eval::model_gate::{LabelMapping, ModelGateRecord};
use turbo_jvm::random::JavaRandom;
use turbo_train::{Algorithm, Key, TrainingConfiguration, TrainingParameters};
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageDetectorFactory, LanguageEventStream, train_language_model_from_events,
};
use turbo_zoo::Sha256;

use super::{
    ACCEPTED_LICENCES, ExcludedLanguage, LanguageCount, LanguageRecipe, Plan, Sample, TrainedLanguages, sentences_file,
};
use crate::error::{Error, Result};
use crate::segmentation::TIMESTAMP_MS;
use crate::tatoeba::{self, Export, Sentences};

impl LanguageRecipe {
    /// The training parameters: `MAXENT`, the recipe's iterations and
    /// cutoff; `train` adds the `OnePass` indexer.
    pub fn parameters(&self) -> TrainingParameters {
        let mut p = TrainingParameters::new();
        p.put(Key::ALGORITHM, Algorithm::Maxent.name());
        p.put(Key::ITERATIONS, self.iterations);
        p.put(Key::CUTOFF, self.cutoff);
        p
    }

    /// The pair of a Tatoeba code, by its index in the recipe's languages.
    pub(super) fn index(&self) -> HashMap<&'static str, usize> {
        self.languages.iter().enumerate().map(|(i, p)| (p.tatoeba, i)).collect()
    }

    /// Reads the sentences of the recipe's languages from `export`.
    ///
    /// # Errors
    ///
    /// What [`Export::sentences`] fails on.
    pub fn sentences<'a>(&self, export: &'a Export) -> Result<Sentences<'a>> {
        let index = self.index();
        Ok(export.sentences(|lang| index.contains_key(lang))?)
    }

    /// The licence check, de-duplication and split of `sentences`.
    ///
    /// # Errors
    ///
    /// [`Error::EmptySplit`] when no language has
    /// [`Split::min_sentences`](super::Split::min_sentences) sentences.
    pub fn plan<'a>(&self, sentences: Sentences<'a>) -> Result<Plan<'a>> {
        let index = self.index();
        let mut by_language: Vec<Vec<tatoeba::Sentence<'a>>> = vec![Vec::new(); self.languages.len()];
        let kept = sentences.kept.len();
        for s in sentences.kept {
            let i = *index.get(&*s.language).expect("the export kept the recipe's languages only");
            by_language[i].push(s);
        }
        let mut plan = Plan {
            training: Vec::new(),
            held_out: Vec::new(),
            languages: Vec::new(),
            excluded: Vec::new(),
            records: sentences.records,
            no_language: sentences.no_language,
            other_languages: sentences.records - sentences.no_language - kept,
            cc0_records: sentences.cc0_records,
            cc0_differing: sentences.cc0_differing,
            cc0_not_in_detailed: sentences.cc0_not_in_detailed,
        };
        for (language, mut all) in by_language.into_iter().enumerate() {
            let pair = self.languages[language];
            let total = all.len();
            all.retain(|s| ACCEPTED_LICENCES.contains(&s.licence));
            let licence_left_out = total - all.len();
            all.sort_by_key(|s| s.id);
            let mut texts = HashSet::new();
            let mut usable: Vec<tatoeba::Sentence<'a>> = Vec::with_capacity(all.len());
            for s in all {
                if texts.insert(s.text.clone()) {
                    usable.push(s);
                }
            }
            let duplicates = total - licence_left_out - usable.len();
            if usable.len() < self.split.min_sentences {
                plan.excluded.push(ExcludedLanguage {
                    tatoeba: pair.tatoeba.to_owned(),
                    sentences: total,
                    usable: usable.len(),
                });
                continue;
            }
            JavaRandom::new(self.split.seed).shuffle(&mut usable);
            let held_out = self.split.held_out.min(usable.len());
            let training = self.split.max_training.min(usable.len() - held_out);
            let mut it = usable.into_iter();
            for s in it.by_ref().take(held_out) {
                plan.held_out.push(Sample { language, id: s.id, text: s.text });
            }
            for s in it.take(training) {
                plan.training.push(Sample { language, id: s.id, text: s.text });
            }
            plan.languages.push(LanguageCount {
                tatoeba: pair.tatoeba.to_owned(),
                reference: pair.reference.iter().map(|c| (*c).to_owned()).collect(),
                sentences: total,
                licence_left_out,
                duplicates,
                training,
                held_out,
                candidate_correct: 0,
                reference_correct: 0,
            });
        }
        if plan.languages.is_empty() {
            return Err(Error::EmptySplit { what: "language with enough sentences" });
        }
        Ok(plan)
    }

    /// Trains the model on the plan's training sentences, one event each.
    ///
    /// # Errors
    ///
    /// What the trainer fails on.
    pub fn train(&self, plan: &Plan<'_>) -> Result<TrainedLanguages> {
        let factory = LanguageDetectorFactory::new();
        let mut stream = LanguageEventStream::new(&factory);
        let events: Vec<_> =
            plan.training.iter().map(|s| stream.event(self.languages[s.language].tatoeba, &s.text)).collect();
        let trained = train_language_model_from_events(
            &events,
            &factory,
            &self.parameters(),
            &mut TrainingConfiguration::default(),
            Stamp::at(TIMESTAMP_MS),
        )?;
        Ok(TrainedLanguages {
            bytes: trained.package.to_bytes(),
            samples: plan.training.len(),
            events: events.len(),
            event_hash: trained.report.event_hash.expect("trained from events"),
            inexact_events: stream.inexact_events(),
            sentences: sentences_file(plan),
        })
    }

    /// The reference's codes counted right for another held-out language,
    /// over the plan's languages.
    pub fn reference_mappings(&self, plan: &Plan<'_>) -> Vec<LabelMapping> {
        let mut out = Vec::new();
        for l in &plan.languages {
            for r in l.reference.iter().filter(|r| **r != l.tatoeba) {
                out.push(LabelMapping { held_out: l.tatoeba.clone(), reference: r.clone() });
            }
        }
        out
    }

    /// The SHA-256 of the held-out samples as scored: for each, its
    /// language, its length in bytes and its text, then each reference
    /// mapping, so two records scoring the same samples the same way have
    /// the same digest.
    pub fn samples_digest(&self, plan: &Plan<'_>) -> String {
        self.samples_digest_with(plan, &self.reference_mappings(plan))
    }

    /// The same digest with `mappings` in place of the reference's, as
    /// for a model scored in the reference's place that has Tatoeba's
    /// codes.
    pub fn samples_digest_with(&self, plan: &Plan<'_>, mappings: &[LabelMapping]) -> String {
        let mut text = String::new();
        for s in &plan.held_out {
            text.push_str(&format!("{}\t{}\t{}\n", self.languages[s.language].tatoeba, s.text.len(), s.text));
        }
        for m in mappings {
            text.push_str(&format!("mapping\t{}\t{}\n", m.held_out, m.reference));
        }
        Sha256::of(text.as_bytes()).to_string()
    }

    /// Classical held-out comparison is retired; native evaluation lives in `langid`.
    ///
    /// # Errors
    ///
    /// Always returns a retirement error.
    pub fn gate(
        &self,
        _trained: &[u8],
        _reference: &[u8],
        _plan: &mut Plan<'_>,
        _set_name: &str,
        _corpus_sha256: &str,
    ) -> Result<ModelGateRecord> {
        Err(Error::Model(
            "classical language detector gating is retired (#1648); use native n-gram evaluation (#1525)".into(),
        ))
    }

    /// The name of the held-out set in the gate record.
    pub fn held_out_name(&self, export_digest: &str) -> String {
        format!(
            "Tatoeba export {} held out ({} sentences per language, seed {})",
            &export_digest[..16.min(export_digest.len())],
            self.split.held_out,
            self.split.seed
        )
    }
}
