//! The maxent tokenizer: OpenNLP's `TokenizerME.tokenizePos`.

use turbo_analysis::Span;
use turbo_unicode::Utf16Cursor;

use crate::TokenizerOptions;

use super::context::{Features, UnitFacts, token_context};
use super::model::TokenizerModel;
use super::units::WhitespaceUnits;

/// The buffers one document's tokenization reuses: the features of the
/// current position, the model's output and the dictionary probe. Make
/// one per document (or per thread) and pass it to every call.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    features: Features,
    probs: Vec<f64>,
    lowered: String,
}

impl Scratch {
    pub fn new() -> Scratch {
        Scratch::default()
    }
}

/// TokenizerME over a loaded model.
#[derive(Debug, Clone)]
pub struct TokenizerMe {
    pub(crate) model: TokenizerModel,
    pub(crate) options: TokenizerOptions,
}

impl TokenizerMe {
    pub fn new(model: TokenizerModel, options: TokenizerOptions) -> TokenizerMe {
        TokenizerMe { model, options }
    }

    /// The loaded model. (`Annotator::model` gives its `ModelRef`.)
    pub fn tokenizer_model(&self) -> &TokenizerModel {
        &self.model
    }

    pub fn options(&self) -> TokenizerOptions {
        self.options
    }

    /// Appends the tokens of `text` to `out`, each a byte span of `text`
    /// with its probability, in the order `tokenizePos` returns them and
    /// with the values `probs()` gives. Allocates only what `out` and a
    /// fresh [`Scratch`] need; use [`tokenize_with`](Self::tokenize_with)
    /// to reuse the buffers.
    pub fn tokenize(&self, text: &str, out: &mut Vec<(Span, f64)>) {
        self.tokenize_with(text, &mut Scratch::new(), out);
    }

    /// [`tokenize`](Self::tokenize) with the caller's buffers.
    ///
    /// Each whitespace unit of one code unit, or one the alphanumeric
    /// check accepts, is a token with probability 1. In any other unit the
    /// model decides, at every UTF-16 position from the second, whether a
    /// token boundary falls there; a token's probability is the product
    /// of the best outcome's probability at every position inside it,
    /// its closing split included. When the model splits a unit that the
    /// abbreviation dictionary lists, the Java emits the unit from the
    /// current token start to its end and jumps forward by the number of
    /// dots in the unit; that is reproduced, including the spans that
    /// overlap.
    ///
    /// The one difference from the Java: a split between the two halves
    /// of a character outside the Basic Multilingual Plane is not a UTF-8
    /// boundary, so it is not taken. The model is still evaluated there,
    /// and the token simply goes on, its probability multiplied by that
    /// decision's as at any other position: the token Java would have cut
    /// in two comes out whole, with the product of the two halves'
    /// probabilities. Likewise a token start that the abbreviation jump
    /// would put between two halves moves to the end of the character.
    pub fn tokenize_with(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<(Span, f64)>) {
        let maxent = self.model.maxent();
        scratch.probs.resize(maxent.num_outcomes(), 0.0);
        let units = WhitespaceUnits::new(text, self.options.whitespace, self.options.keep_newlines);
        for (start, end) in units {
            let unit = &text[start..end];
            if fewer_than_two_units(unit) || self.model.alphanumeric().is_some_and(|a| a.test(unit)) {
                out.push((span(start, end), 1.0));
            } else {
                self.split_unit(unit, start, scratch, out);
            }
        }
    }

    /// The model's loop over one unit at byte `base` of the text.
    fn split_unit(&self, unit: &str, base: usize, scratch: &mut Scratch, out: &mut Vec<(Span, f64)>) {
        let maxent = self.model.maxent();
        let Scratch { features, probs, lowered } = scratch;
        // The dictionary is asked about the unit as a whole, so once.
        let abbreviation = self.model.abbreviations().is_some_and(|d| d.contains_with(unit, lowered));
        let facts = UnitFacts::of(unit, abbreviation);
        let end = unit.len();
        // The current token's start, a byte offset in the unit; `end` when
        // the abbreviation jump went past the unit's end.
        let mut start = 0;
        let mut prob = 1.0;
        let mut at = Utf16Cursor::new(unit);
        at.step_forward();
        while !at.at_end() {
            token_context(&at, facts, self.options.whitespace, features);
            maxent.eval_features(features.iter(), probs);
            let outcome = self.model.outcome(maxent.best_outcome(probs));
            prob *= probs[outcome.index];
            if outcome.split {
                if abbreviation {
                    out.push((span(base + start, base + end), prob));
                    let dots = unit.bytes().filter(|&b| b == b'.').count();
                    for _ in 0..dots {
                        at.step_forward();
                    }
                    let mut next = at.clone();
                    next.step_forward();
                    start = if next.char_start() { next.byte_offset() } else { next.byte_offset() + 4 };
                } else if at.char_start() {
                    out.push((span(base + start, base + at.byte_offset()), prob));
                    start = at.byte_offset();
                } else {
                    // Between the halves of a pair: not taken.
                    at.step_forward();
                    continue;
                }
                prob = 1.0;
            }
            at.step_forward();
        }
        if start < end {
            out.push((span(base + start, base + end), prob));
        }
    }
}

/// `tok.length() < 2`: one character of at most three UTF-8 bytes, which
/// is one UTF-16 code unit.
#[inline]
fn fewer_than_two_units(unit: &str) -> bool {
    match unit.len() {
        1 => true,
        2 | 3 => unit.chars().nth(1).is_none(),
        _ => false,
    }
}

#[inline]
fn span(start: usize, end: usize) -> Span {
    Span::new(start as u32, end as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    use turbo_unicode::WhitespaceMode;

    fn english() -> TokenizerMe {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/models/opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin");
        TokenizerMe::new(TokenizerModel::read(path).unwrap(), TokenizerOptions::default())
    }

    fn tokens(t: &TokenizerMe, text: &str) -> Vec<(String, f64)> {
        let mut out = Vec::new();
        t.tokenize(text, &mut out);
        out.iter().map(|(s, p)| (text[s.range()].to_string(), *p)).collect()
    }

    #[test]
    fn short_units() {
        assert!(fewer_than_two_units("a"));
        assert!(fewer_than_two_units("\u{e9}"));
        assert!(fewer_than_two_units("\u{20ac}"));
        assert!(!fewer_than_two_units("\u{1f600}"), "a pair is two code units");
        assert!(!fewer_than_two_units("ab"));
        assert!(!fewer_than_two_units("a\u{e9}"));
    }

    #[test]
    fn splits_and_probabilities() {
        // Values from the golden file english-abbreviations for the
        // English model are checked in tests/me_golden.rs; here a sentence
        // checked against TokenizerME directly.
        let t = english();
        let got = tokens(&t, "Hello, world!");
        let words: Vec<&str> = got.iter().map(|(w, _)| w.as_str()).collect();
        assert_eq!(words, ["Hello", ",", "world", "!"]);
        assert!(got.iter().all(|&(_, p)| p > 0.0 && p <= 1.0));
    }

    #[test]
    fn options_change_the_units() {
        let model = english().model;
        let legacy = TokenizerMe::new(
            model.clone(),
            TokenizerOptions { whitespace: WhitespaceMode::Legacy, ..Default::default() },
        );
        let unicode = TokenizerMe::new(model.clone(), TokenizerOptions::default());
        // U+001F separates only in the legacy mode.
        assert_eq!(tokens(&legacy, "a\u{1f}b").len(), 2);
        assert_eq!(tokens(&unicode, "a\u{1f}b").len(), 1);
        let keep = TokenizerMe::new(model, TokenizerOptions { keep_newlines: true, ..Default::default() });
        assert_eq!(tokens(&keep, "a\nb"), [("a".into(), 1.0), ("\n".into(), 1.0), ("b".into(), 1.0)]);
        assert_ne!(TokenizerOptions::default().config_hash(), keep.options().config_hash());
    }

    #[test]
    fn the_alphanumeric_optimization_skips_the_model() {
        // With the optimization on, "ab12" is a token of probability 1
        // without consulting the model; with it off the model decides.
        let t = english();
        let off = tokens(&t, "ab12");
        assert_eq!(off.len(), 1);
        assert!(off[0].1 < 1.0);
        let on = TokenizerMe::new(
            t.model.clone().with_alphanumeric_optimization(Some("^[a-z0-9]+$")).unwrap(),
            TokenizerOptions::default(),
        );
        assert_eq!(tokens(&on, "ab12"), [("ab12".into(), 1.0)]);
        // A unit outside the pattern still goes to the model, which splits
        // it as it does without the optimization.
        assert_eq!(tokens(&on, "Ab12"), tokens(&t, "Ab12"));
        assert!(tokens(&on, "Ab12").iter().all(|&(_, p)| p < 1.0));
        let off_again = on.model.clone().with_alphanumeric_optimization(None).unwrap();
        assert!(off_again.alphanumeric().is_none());
    }
}
