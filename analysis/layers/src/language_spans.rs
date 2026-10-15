//! Language by span: which language and script each stretch of a text
//! view is in, for documents that change language part way (a German
//! e-mail quoting English, code-switched chat, a bilingual notice).
//!
//! `opennlp:language` ([`Language`](crate::Language)) stays what it is in
//! OpenNLP, one ranking for the whole document, so a pipeline written
//! against OpenNLP reads the layer it expects. The spans are a layer of
//! their own, `turbo:language-spans`, because a document-anchored layer
//! has no spans to carry: changing the anchor of `opennlp:language`
//! would change what every reader of it gets. A step that writes both
//! derives the document ranking from the spans, so the two agree.

use std::sync::Arc;

use turbo_analysis::{Ix, Layer, Span, Spans, Tag, TagSet, ViewId};
use turbo_unicode::Script;

/// The code a span gets when the detector knows no language written in
/// its script (ISO 639-3 "undetermined").
pub const UNDETERMINED: &str = "und";

/// The language and script of each run of a text view, in text order and
/// not overlapping. Each span has the language code it was given (a tag
/// in `languages`, ISO 639-3 for the OpenNLP models), the script most of
/// its letters are in, and a confidence between 0 and 1.
///
/// The confidence ranks spans of one detector against each other; it is
/// not a calibrated probability, and readers should not threshold on it.
/// For `turbo-langspan` over OpenNLP's `langdetect-183.bin` it is the
/// model's probability of the span's language over the languages of its
/// script, and that model's probabilities are flat: a clear French
/// sentence gets about 0.07 among 65 Latin-script languages.
pub struct LanguageSpans {
    pub spans: Spans<LanguageSpans>,
    pub language: Vec<Tag>,
    /// The codes `language` indexes: the detector's languages and
    /// [`UNDETERMINED`].
    pub languages: Arc<TagSet>,
    pub script: Vec<Script>,
    /// Not calibrated; see the type's documentation.
    pub confidence: Vec<f64>,
}

/// One span with its columns, as [`LanguageSpans::get`] returns it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct LanguageSpan<'a> {
    pub span: Span,
    pub code: &'a str,
    pub script: Script,
    pub confidence: f64,
}

impl LanguageSpans {
    pub fn new(view: ViewId, languages: Arc<TagSet>) -> LanguageSpans {
        LanguageSpans {
            spans: Spans::new(view),
            language: Vec::new(),
            languages,
            script: Vec::new(),
            confidence: Vec::new(),
        }
    }

    pub fn push(&mut self, span: Span, language: Tag, script: Script, confidence: f64) -> Ix<LanguageSpans> {
        self.language.push(language);
        self.script.push(script);
        self.confidence.push(confidence);
        self.spans.push(span)
    }

    pub fn get(&self, i: Ix<LanguageSpans>) -> LanguageSpan<'_> {
        LanguageSpan {
            span: self.spans.get(i),
            code: self.languages.label(self.language[i.get()]),
            script: self.script[i.get()],
            confidence: self.confidence[i.get()],
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = LanguageSpan<'_>> {
        (0..self.spans.len() as u32).map(|i| self.get(Ix::new(i)))
    }
}

impl Layer for LanguageSpans {
    const ID: &'static str = "turbo:language-spans";
    fn len(&self) -> usize {
        self.spans.len()
    }
    fn text_spans(&self) -> Option<(ViewId, &[u32], &[u32])> {
        let (start, end) = self.spans.columns();
        Some((self.spans.view(), start, end))
    }
    fn check(&self) -> Result<(), String> {
        let n = self.spans.len();
        if self.language.len() != n || self.script.len() != n || self.confidence.len() != n {
            return Err("a language, a script and a confidence for every span".into());
        }
        if !self.spans.is_ordered() {
            return Err("spans in text order, not overlapping".into());
        }
        if self.spans.iter().any(|(_, s)| s.is_empty()) {
            return Err("no empty span".into());
        }
        if self.language.iter().any(|t| t.0 as usize >= self.languages.len()) {
            return Err(format!("a language outside {}", self.languages.name));
        }
        if self.confidence.iter().any(|c| !(0.0..=1.0).contains(c)) {
            return Err("a confidence between 0 and 1".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer() -> LanguageSpans {
        LanguageSpans::new(ViewId::SOURCE, TagSet::new("languages", &["deu", "eng", UNDETERMINED]))
    }

    #[test]
    fn spans_come_back_with_their_columns() {
        let mut l = layer();
        l.push(Span::new(0, 10), Tag(0), Script::Latin, 0.9);
        l.push(Span::new(10, 14), Tag(1), Script::Latin, 0.6);
        assert_eq!(l.check(), Ok(()));
        assert_eq!(l.len(), 2);
        let second = l.get(Ix::new(1));
        assert_eq!(
            (second.span, second.code, second.script, second.confidence),
            (Span::new(10, 14), "eng", Script::Latin, 0.6)
        );
        assert_eq!(l.iter().map(|s| s.code).collect::<Vec<_>>(), ["deu", "eng"]);
        assert_eq!(LanguageSpans::ID, "turbo:language-spans");
    }

    #[test]
    fn check_wants_ordered_nonempty_spans_and_known_tags() {
        let mut l = layer();
        l.push(Span::new(4, 8), Tag(0), Script::Latin, 0.5);
        l.push(Span::new(2, 3), Tag(0), Script::Latin, 0.5);
        assert!(l.check().unwrap_err().contains("text order"));

        let mut l = layer();
        l.push(Span::new(4, 4), Tag(0), Script::Latin, 0.5);
        assert_eq!(l.check().unwrap_err(), "no empty span");

        let mut l = layer();
        l.push(Span::new(0, 4), Tag(3), Script::Latin, 0.5);
        assert!(l.check().unwrap_err().contains("a language outside"));

        let mut l = layer();
        l.push(Span::new(0, 4), Tag(0), Script::Latin, 1.5);
        assert!(l.check().unwrap_err().contains("between 0 and 1"));

        let mut l = layer();
        l.push(Span::new(0, 4), Tag(0), Script::Latin, 0.5);
        l.script.clear();
        assert!(l.check().unwrap_err().contains("for every span"));
    }
}
