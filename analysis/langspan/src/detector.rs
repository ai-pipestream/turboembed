//! Language detection with the trained n-gram model (#1525).

use turbo_analysis::{Annotator, Error, Refs, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tokens};
use turbo_langdetect::NgramModel;

/// A language detector backed by the trained n-gram model.
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
        let scores = self.model.scores(text);
        let mut spans = LanguageSpans::new(
            view.id(),
            turbo_analysis::TagSet::new("languages", &self.model.supported_languages().collect::<Vec<_>>()),
        );
        let mut language = Language { code: Strs::default(), score: Vec::new() };
        if !text.is_empty() {
            let best = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some((i, score)) = best {
                let code = self.model.supported_languages().nth(i).expect("language index");
                language.code.push(code);
                language.score.push((*score).into());
                spans.push(
                    turbo_analysis::Span::new(0, text.len() as u32),
                    turbo_analysis::Tag(i as u16),
                    turbo_unicode::Script::Unknown,
                    (*score).into(),
                );
            }
        }
        Ok((spans, language))
    }
}
