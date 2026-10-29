use turbo_eval::model_gate::LabelMapping;
use turbo_train::TrainingConfiguration;
use turbo_train_components::Stamp;
use turbo_train_components::langdetect::{
    LanguageDetectorFactory, LanguageEventStream, train_language_model_from_events,
};
use turbo_zoo::Sha256;

use super::{Scored, TrainedWeb, WebLanguageRecipe};
use crate::error::{Error, Result};
use crate::fineweb::{self, WebFile, WebPlan, WebSample};
use crate::langdetect::{self, Plan};
use crate::segmentation::TIMESTAMP_MS;

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

    /// Classical held-out comparison is retired; native evaluation lives in `langid`.
    ///
    /// # Errors
    ///
    /// Returns a retirement error after validating the language lists.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        _trained: &[u8],
        _reference: &[u8],
        _compared: &[u8],
        _plan: &mut Plan<'_>,
        _web: &WebPlan,
        _export_digest: &str,
        _lists_digest: &str,
    ) -> Result<Scored> {
        self.check_languages()?;
        Err(Error::Model(
            "classical web language detector scoring is retired (#1648); use native n-gram evaluation (#1525)".into(),
        ))
    }
}
