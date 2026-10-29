//! Language detection with the trained n-gram model (#1525).

use turbo_analysis::{Annotator, Error, Refs, Span, Tag, TagSet, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tokens};
use turbo_langdetect::NgramModel;
use turbo_unicode::{Script, category::general_category};

use super::script_only::{append, letter_script};

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
