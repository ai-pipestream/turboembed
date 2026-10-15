//! The detector as an analysis step: it reads a text view and, when the
//! document has them, its sentences and tokens, and writes
//! `turbo:language-spans` and the document summary `opennlp:language`.

use std::cell::RefCell;

use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Tag, View};
use turbo_analysis_layers::{Language, LanguageSpans, Sentences, Strs, Tokens};
use turbo_langdetect::NO_EVIDENCE;

use crate::detector::{Run, SpanDetector, SpanScratch};

thread_local! {
    /// The step's buffers, one set per thread.
    static SCRATCH: RefCell<(SpanScratch, Vec<Run>)> = RefCell::new((SpanScratch::new(), Vec::new()));
}

/// The document's languages from its runs: every language of a run, by
/// its share of the letters in runs with a language, largest first, a
/// tie going to the language that comes first in the text. Runs whose
/// script the model has no language for (`und`) are left out, so the
/// shares add up to 1 when there is any language at all.
pub fn summary(detector: &SpanDetector, runs: &[Run]) -> Language {
    let mut letters: Vec<(u32, u64)> = Vec::new();
    for run in runs {
        let Some(outcome) = run.outcome else { continue };
        match letters.iter_mut().find(|(o, _)| *o == outcome) {
            Some((_, n)) => *n += u64::from(run.letters),
            None => letters.push((outcome, u64::from(run.letters))),
        }
    }
    // A stable sort keeps the text order among equal counts.
    letters.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
    let total: u64 = letters.iter().map(|(_, n)| n).sum();
    let mut code = Strs::default();
    let mut score = Vec::with_capacity(letters.len());
    for (outcome, n) in letters {
        code.push(detector.detector().model().code(outcome as usize));
        score.push(if total == 0 { 0.0 } else { n as f64 / total as f64 });
    }
    Language { code, score }
}

/// Reads a view, and its sentences and tokens when the document has them,
/// and writes the language of each run (`turbo:language-spans`) and the
/// document's languages derived from them (`opennlp:language`, each
/// language with its share of the letters).
///
/// A text with no word the detector would read (empty, or only digits,
/// URLs, emoji, punctuation) is declined with `Error::Skipped("no language
/// evidence")`, as OpenNLP's detector is. A text whose words are all in
/// scripts the model has no language for gets its spans, each `und`, and
/// an empty `opennlp:language`.
impl Annotator for SpanDetector {
    const ID: &'static str = "turbo-langspan:script-ngram";
    type Reads = (View, Option<Sentences>, Option<Tokens>);
    type Writes = (LanguageSpans, Language);

    fn annotate(&self, (view, sentences, tokens): Refs<'_, Self>) -> Result<(LanguageSpans, Language), Error> {
        let other_view =
            |what: &str| Error::Annotator { annotator: Self::ID, detail: format!("the {what} are in another view") };
        if sentences.is_some_and(|s| s.spans.view() != view.id()) {
            return Err(other_view("sentences"));
        }
        if tokens.is_some_and(|t| t.spans.view() != view.id()) {
            return Err(other_view("tokens"));
        }
        if tokens.is_some_and(|t| !t.spans.is_ordered()) {
            return Err(Error::Annotator { annotator: Self::ID, detail: "the tokens are not in text order".into() });
        }
        let sentence_starts = sentences.map(|s| s.spans.columns().0);
        let token_columns = tokens.map(|t| t.spans.columns());
        SCRATCH.with_borrow_mut(|(scratch, runs)| {
            self.detect(view.text(), sentence_starts, token_columns, scratch, runs);
            if runs.is_empty() {
                return Err(Error::Skipped(NO_EVIDENCE.into()));
            }
            let mut layer = LanguageSpans::new(view.id(), self.languages().clone());
            for run in runs.iter() {
                let tag = Tag(run.outcome.map_or(self.undetermined(), |o| o as u16));
                layer.push(Span::new(run.span.start, run.span.end), tag, run.script, run.confidence);
            }
            Ok((layer, summary(self, runs)))
        })
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.detector().model().model_ref().clone())
    }

    fn config_hash(&self) -> u64 {
        let s = self.settings();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for word in [
            s.switch.space.to_bits(),
            s.switch.punctuation.to_bits(),
            s.switch.sentence.to_bits(),
            u64::from(s.clause_letters),
        ] {
            h = (h ^ word).wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }
}
