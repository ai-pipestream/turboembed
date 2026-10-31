//! Language detection with the trained n-gram model (#1525).

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tokens};
use turbo_langdetect::NgramModel;
use turbo_unicode::{Script, category::general_category};

use crate::script_only::{append, letter_script};

/// A language detector backed by the trained n-gram model. Every letter
/// run carries the same whole-text prediction; runs are not classified
/// independently.
#[derive(Clone, Debug)]
pub struct NgramDetector {
    model: NgramModel,
}

impl NgramDetector {
    /// Creates a detector from a trained model.
    pub fn new(model: NgramModel) -> NgramDetector {
        NgramDetector { model }
    }
}

impl Annotator for NgramDetector {
    const ID: &'static str = "turbo-langspan:ngram";
    type Reads = (View, Option<Sentences>, Option<Tokens>);
    type Writes = (LanguageSpans, Language);

    fn model(&self) -> Option<ModelRef> {
        Some(self.model.model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        self.model.config_hash()
    }

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let invalid = |detail| Error::Annotator { annotator: Self::ID, detail };
        if u32::try_from(view.text().len()).is_err() {
            return Err(invalid("text exceeds the u32 offset range".into()));
        }
        if let Some(s) = sentences {
            s.check_contiguous_in(view).map_err(invalid)?;
        }
        if let Some(t) = tokens {
            t.check_contiguous_in(view).map_err(invalid)?;
        }
        let text = view.text();
        let mut spans = LanguageSpans::new(
            view.id(),
            TagSet::new("languages", &self.model.supported_languages().collect::<Vec<_>>()),
        );
        let mut language = Language { code: Strs::default(), score: Vec::new() };
        let Some((code, score)) = self.model.best(text) else {
            return Ok((spans, language));
        };
        let tag =
            self.model.supported_languages().position(|candidate| candidate == code).expect("best language") as u16;
        let mut runs = Vec::<(Span, Script)>::new();
        if let Some(tokens) = tokens {
            for (_, span) in tokens.spans.iter() {
                let mut counts = Vec::<(Script, usize)>::new();
                for c in text[span.range()].chars() {
                    if let Some(s) = letter_script(c) {
                        if let Some((_, count)) = counts.iter_mut().find(|(g, _)| *g == s) {
                            *count += 1;
                        } else {
                            counts.push((s, 1));
                        }
                    }
                }
                if let Some(&(mut best, mut count)) = counts.first() {
                    for &(s, n) in &counts[1..] {
                        if n > count {
                            best = s;
                            count = n;
                        }
                    }
                    append(&mut runs, span, best);
                }
            }
        } else {
            for (offset, c) in text.char_indices() {
                let end = (offset + c.len_utf8()) as u32;
                if let Some(s) = letter_script(c) {
                    append(&mut runs, Span::new(offset as u32, end), s);
                } else if general_category(c).is_mark()
                    && let Some((span, _)) = runs.last_mut()
                    && span.end == offset as u32
                {
                    span.end = end;
                }
            }
        }
        for (span, script) in runs {
            spans.push(span, Tag(tag), script, score.into());
        }
        if !spans.spans.is_empty() {
            language.code.push(code);
            language.score.push(score.into());
        }
        Ok((spans, language))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Analyzer, Document, StageCacheStatus};

    fn sample_model(bias_en: f32, bias_es: f32, rev: &str) -> NgramModel {
        NgramModel::new(
            vec!["eng".into(), "spa".into()],
            vec![0.0; 16],
            vec![bias_en, bias_es],
            (1, 3),
            8,
            ModelRef { id: "test-ngram".into(), revision: rev.into(), sha256: format!("sha-{rev}") },
        )
        .expect("valid test NgramModel")
    }

    #[test]
    fn ngram_detector_invalidates_incremental_cache_on_model_or_weights_change() {
        let model_en = sample_model(5.0, -5.0, "v1");
        let model_es_weights = sample_model(-5.0, 5.0, "v1");
        let model_es_rev = sample_model(-5.0, 5.0, "v2");

        let det_en = NgramDetector::new(model_en);
        let det_es_weights = NgramDetector::new(model_es_weights);
        let det_es_rev = NgramDetector::new(model_es_rev);

        assert_ne!(det_en.config_hash(), det_es_weights.config_hash());
        assert_ne!(det_es_weights.model(), det_es_rev.model());

        let p1 = Analyzer::builder().add(det_en).build().unwrap();
        let p2 = Analyzer::builder().add(det_es_weights).build().unwrap();
        let p3 = Analyzer::builder().add(det_es_rev).build().unwrap();

        let mut doc = Document::new("Hello world");
        let r1 = p1.analyze_incremental(&mut doc).unwrap();
        assert_eq!(r1.stages[0].cache_status, StageCacheStatus::Executed);
        assert_eq!(doc.get::<Language>().unwrap().code.get(0), "eng");

        // Re-running with identical detector skips in O(1).
        let r1_cached = p1.analyze_incremental(&mut doc).unwrap();
        assert_eq!(r1_cached.stages[0].cache_status, StageCacheStatus::Skipped);

        // Changing weights (same ModelRef, different config_hash) invalidates cache.
        let r2 = p2.analyze_incremental(&mut doc).unwrap();
        assert_eq!(r2.stages[0].cache_status, StageCacheStatus::Executed);
        assert_eq!(doc.get::<Language>().unwrap().code.get(0), "spa");

        // Changing ModelRef revision invalidates cache even with identical weights.
        let r3 = p3.analyze_incremental(&mut doc).unwrap();
        assert_eq!(r3.stages[0].cache_status, StageCacheStatus::Executed);
    }
}
