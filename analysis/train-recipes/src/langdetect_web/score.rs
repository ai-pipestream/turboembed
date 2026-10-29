use std::collections::{HashMap, HashSet};

use turbo_eval::model_gate::{
    Counts, HeldOutSet, LabelMapping, LabelScoring, ModelGateRecord, ModelId, PairedUnit, Scoring, Task,
};
use turbo_langdetect::{LanguageDetector, LanguageModel, Scratch};
use turbo_maxent::ModelFile;
use turbo_train::TrainingConfiguration;
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageDetectorFactory, LanguageEventStream, train_language_model_from_events,
};
use turbo_zoo::Sha256;

use super::{Scored, TrainedWeb, WebLanguageCount, WebLanguageRecipe};
use crate::error::{Error, Result};
use crate::fineweb::{self, WebFile, WebPlan, WebSample};
use crate::langdetect::{self, Plan};
use crate::segmentation::TIMESTAMP_MS;

fn open(what: &str, bytes: &[u8]) -> Result<LanguageDetector> {
    ModelFile::open(bytes.to_vec())
        .map_err(|e| e.to_string())
        .and_then(|f| LanguageModel::open(f).map_err(|e| e.to_string()))
        .map(LanguageDetector::new)
        .map_err(|e| Error::Model(format!("{what}: {e}")))
}

impl WebLanguageRecipe {
    /// Checks that the web recipe's languages match the Tatoeba recipe's
    /// languages in order.
    pub(super) fn check_languages(&self) -> Result<()> {
        let tatoeba: Vec<&str> = self.tatoeba.languages.iter().map(|p| p.tatoeba).collect();
        let web_codes: Vec<&str> = self.web.languages.iter().map(|l| l.tatoeba).collect();
        if tatoeba != web_codes {
            return Err(Error::Model(format!(
                "{}: the web languages are not the Tatoeba recipe's, in its order",
                self.model_id
            )));
        }
        Ok(())
    }

    /// Trains the model on the plan's training sentences and the web
    /// plan's training windows, one event each.
    ///
    /// # Errors
    ///
    /// What the trainer fails on.
    pub fn train(&self, plan: &Plan<'_>, web: &WebPlan, files: &[WebFile]) -> Result<TrainedWeb> {
        self.check_languages()?;
        let factory = LanguageDetectorFactory::new();
        let mut stream = LanguageEventStream::new(&factory);
        let languages = self.tatoeba.languages;
        let mut events = Vec::with_capacity(plan.training.len() + web.training.len());
        for s in &plan.training {
            events.push(stream.event(languages[s.language].tatoeba, &s.text));
        }
        for s in &web.training {
            events.push(stream.event(self.web.languages[s.language].tatoeba, &s.window));
        }
        let trained = train_language_model_from_events(
            &events,
            &factory,
            &self.tatoeba.parameters(),
            &mut TrainingConfiguration::default(),
            Stamp::at(TIMESTAMP_MS),
        )?;
        Ok(TrainedWeb {
            bytes: trained.package.to_bytes(),
            sentences: plan.training.len(),
            windows: web.training.len(),
            events: events.len(),
            event_hash: trained.report.event_hash.expect("trained from events"),
            inexact_events: stream.inexact_events(),
            sentences_file: langdetect::sentences_file(plan),
            documents_file: fineweb::documents_file(web, files),
        })
    }

    /// The reference's codes counted right for another language, over all
    /// of the recipe's languages: the web windows of every language are
    /// held out, whichever Tatoeba languages the plan keeps.
    pub fn web_mappings(&self) -> Vec<LabelMapping> {
        let mut out = Vec::new();
        for p in self.tatoeba.languages {
            for r in p.reference.iter().filter(|r| **r != p.tatoeba) {
                out.push(LabelMapping { held_out: p.tatoeba.to_owned(), reference: (*r).to_owned() });
            }
        }
        out
    }

    /// The SHA-256 of held-out windows as scored: for each, its
    /// language, its length in bytes and its text, then each of
    /// `mappings`.
    pub fn web_samples_digest<'s>(
        &self,
        samples: impl IntoIterator<Item = &'s WebSample>,
        mappings: &[LabelMapping],
    ) -> String {
        let mut text = String::new();
        for s in samples {
            text.push_str(&format!("{}\t{}\t{}\n", self.web.languages[s.language].tatoeba, s.window.len(), s.window));
        }
        for m in mappings {
            text.push_str(&format!("mapping\t{}\t{}\n", m.held_out, m.reference));
        }
        Sha256::of(text.as_bytes()).to_string()
    }

    /// The name of the held-out web set in the gate record.
    pub fn web_held_out_name(&self, lists_digest: &str) -> String {
        format!(
            "FineWeb-2 and FineWeb held out (at most {} windows of at most {} characters per language, lists {})",
            self.web.split.held_out,
            self.web.split.window,
            &lists_digest[..16.min(lists_digest.len())]
        )
    }

    /// Scores the trained model on both held-out sets against the
    /// reference and against [`COMPARED`](super::COMPARED), and fills in
    /// the plan's correct counts. `export_digest` and `lists_digest` name
    /// the sets.
    ///
    /// # Errors
    ///
    /// A model file that is not a language detector, or what the gate
    /// fails on.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        trained: &[u8],
        reference: &[u8],
        compared: &[u8],
        plan: &mut Plan<'_>,
        web: &WebPlan,
        export_digest: &str,
        lists_digest: &str,
    ) -> Result<Scored> {
        self.check_languages()?;
        let tatoeba_gate =
            self.tatoeba.gate(trained, reference, plan, &self.tatoeba.held_out_name(export_digest), export_digest)?;
        let candidate = open(self.model_id, trained)?;
        let reference_model = open(self.tatoeba.reference.id(), reference)?;
        let known: HashSet<&str> = reference_model.model().supported_languages().collect();
        for s in &web.held_out {
            let p = self.tatoeba.languages[s.language];
            if let Some(code) = p.reference.iter().find(|c| !known.contains(*c)) {
                return Err(Error::Model(format!(
                    "{}: no outcome {code}, which the recipe counts right for {}",
                    self.tatoeba.reference.id(),
                    p.tatoeba
                )));
            }
        }
        let compared_model = open(self.compared.id(), compared)?;
        let compared_known: HashSet<&str> = compared_model.model().supported_languages().collect();
        let mut scratch = Scratch::new();
        let mut right = |model: &LanguageDetector, text: &str, codes: &[&str]| {
            codes.contains(&model.predict_language(text, &mut scratch).code)
        };
        let candidate_id = ModelId { id: self.model_id.to_owned(), sha256: Sha256::of(trained).to_string() };
        let reference_id =
            ModelId { id: self.tatoeba.reference.id().to_owned(), sha256: self.tatoeba.reference.sha256()? };
        let compared_id = ModelId { id: self.compared.id().to_owned(), sha256: self.compared.sha256()? };
        let pair = |c: bool, r: bool| PairedUnit {
            candidate: Counts::accuracy(u64::from(c), 1),
            reference: Counts::accuracy(u64::from(r), 1),
        };

        // The held-out sentences against the compared model.
        let index: HashMap<&str, usize> =
            plan.languages.iter().enumerate().map(|(k, l)| (l.tatoeba.as_str(), k)).collect();
        let mut compared_sentences = vec![0; plan.languages.len()];
        let mut units = Vec::with_capacity(plan.held_out.len());
        for s in &plan.held_out {
            let code = self.tatoeba.languages[s.language].tatoeba;
            let c = right(&candidate, &s.text, &[code]);
            let o = right(&compared_model, &s.text, &[code]);
            compared_sentences[index[code]] += usize::from(o);
            units.push(pair(c, o));
        }
        // The compared model has Tatoeba's codes, so its set has no
        // mappings, and its samples' digest none either.
        let set = HeldOutSet {
            name: self.tatoeba.held_out_name(export_digest),
            corpus_sha256: export_digest.to_owned(),
            samples_sha256: self.tatoeba.samples_digest_with(plan, &[]),
            skipped_sentences: 0,
            scoring: Scoring::Labels(LabelScoring {
                samples: plan.held_out.len() as u64,
                reference_mappings: Vec::new(),
            }),
        };
        let compare_tatoeba =
            ModelGateRecord::new(Task::LanguageDetector, set, candidate_id.clone(), compared_id.clone(), units)?;

        // The held-out web windows against the reference and the
        // compared model.
        let mut web_counts: Vec<WebLanguageCount> = self
            .web
            .languages
            .iter()
            .map(|l| WebLanguageCount {
                tatoeba: l.tatoeba.to_owned(),
                training: 0,
                held_out: 0,
                candidate_correct: 0,
                reference_correct: 0,
                compared_correct: 0,
            })
            .collect();
        for s in &web.training {
            web_counts[s.language].training += 1;
        }
        let mut gate_units = Vec::with_capacity(web.held_out.len());
        let mut compare_units = Vec::with_capacity(web.held_out.len());
        let mut compared_windows = Vec::with_capacity(web.held_out.len());
        for s in &web.held_out {
            let p = self.tatoeba.languages[s.language];
            let c = right(&candidate, &s.window, &[p.tatoeba]);
            let r = right(&reference_model, &s.window, p.reference);
            let o = right(&compared_model, &s.window, &[p.tatoeba]);
            let count = &mut web_counts[s.language];
            count.held_out += 1;
            count.candidate_correct += usize::from(c);
            count.reference_correct += usize::from(r);
            count.compared_correct += usize::from(o);
            gate_units.push(pair(c, r));
            // The compared model cannot be right on a language it does
            // not have, so those windows say nothing of what the web text
            // changed.
            if compared_known.contains(p.tatoeba) {
                compare_units.push(pair(c, o));
                compared_windows.push(s);
            }
        }
        let mappings = self.web_mappings();
        let web_set = |name: String, samples: &[&WebSample], mappings: Vec<LabelMapping>| HeldOutSet {
            name,
            corpus_sha256: lists_digest.to_owned(),
            samples_sha256: self.web_samples_digest(samples.iter().copied(), &mappings),
            skipped_sentences: 0,
            scoring: Scoring::Labels(LabelScoring { samples: samples.len() as u64, reference_mappings: mappings }),
        };
        let all: Vec<&WebSample> = web.held_out.iter().collect();
        let web_gate = ModelGateRecord::new(
            Task::LanguageDetector,
            web_set(self.web_held_out_name(lists_digest), &all, mappings),
            candidate_id.clone(),
            reference_id,
            gate_units,
        )?;
        let compare_web = ModelGateRecord::new(
            Task::LanguageDetector,
            web_set(
                format!("{}, the languages of {}", self.web_held_out_name(lists_digest), self.compared.id()),
                &compared_windows,
                Vec::new(),
            ),
            candidate_id,
            compared_id,
            compare_units,
        )?;
        Ok(Scored { tatoeba_gate, web_gate, compare_tatoeba, compare_web, compared_sentences, web_counts })
    }
}
