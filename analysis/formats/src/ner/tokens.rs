//! The tokenizers and sentence detector the MUC and brat readers split
//! text with.

use turbo_analysis::Span;
use turbo_tokenize::{TokenizerOptions, WhitespaceMode, simple_spans, whitespace_spans};

/// A tokenizer, as the MUC and brat streams use OpenNLP's `Tokenizer`:
/// the byte spans of the tokens of a text (`tokenizePos`). OpenNLP's
/// factories take a maxent tokenizer model, or for brat a rule-based
/// tokenizer by name; [`RuleTokenizer`] is the rule-based ones, and a
/// maxent tokenizer implements this trait to stand in for a model.
pub trait TokenSpans {
    /// Appends the spans of the tokens of `text` to `out`, in order.
    fn token_spans(&self, text: &str, out: &mut Vec<Span>);
}

impl<F: Fn(&str, &mut Vec<Span>)> TokenSpans for F {
    fn token_spans(&self, text: &str, out: &mut Vec<Span>) {
        self(text, out)
    }
}

/// OpenNLP's rule-based tokenizers, `WhitespaceTokenizer.INSTANCE` and
/// `SimpleTokenizer.INSTANCE`, with the options of the process.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RuleTokenizer {
    /// Runs of characters that are not whitespace.
    Whitespace(TokenizerOptions),
    /// Runs of letters, of digits, of whitespace and of one other
    /// character repeated.
    Simple(TokenizerOptions),
}

impl RuleTokenizer {
    /// The tokenizer `BratNameSampleStreamFactory` takes for its
    /// `ruleBasedTokenizer` parameter: `simple` or `whitespace`.
    pub fn from_name(name: &str, options: TokenizerOptions) -> Option<RuleTokenizer> {
        match name {
            "simple" => Some(RuleTokenizer::Simple(options)),
            "whitespace" => Some(RuleTokenizer::Whitespace(options)),
            _ => None,
        }
    }
}

impl Default for RuleTokenizer {
    /// The whitespace tokenizer, which the brat factory uses when given
    /// no tokenizer.
    fn default() -> RuleTokenizer {
        RuleTokenizer::Whitespace(TokenizerOptions::default())
    }
}

impl TokenSpans for RuleTokenizer {
    fn token_spans(&self, text: &str, out: &mut Vec<Span>) {
        match *self {
            RuleTokenizer::Whitespace(options) => whitespace_spans(text, options, out),
            RuleTokenizer::Simple(options) => simple_spans(text, options, out),
        }
    }
}

/// A sentence detector, as the brat stream uses OpenNLP's
/// `SentenceDetector`: the byte spans of the sentences of a text
/// (`sentPosDetect`). [`NewlineSentences`] is the one the brat factory
/// uses when it is given no sentence model.
pub trait SentenceSpans {
    /// Appends the spans of the sentences of `text` to `out`, in order.
    fn sentence_spans(&self, text: &str, out: &mut Vec<Span>);
}

/// `NewlineSentenceDetector`: a sentence per line, a line ending at `\n`
/// or `\r`, trimmed of whitespace (`StringUtil.isWhitespace` in `mode`),
/// and lines left empty by the trim dropped.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NewlineSentences(pub WhitespaceMode);

impl SentenceSpans for NewlineSentences {
    fn sentence_spans(&self, text: &str, out: &mut Vec<Span>) {
        let mode = self.0;
        let mut push = |start: usize, end: usize| {
            let line = &text[start..end];
            let ws = |c: char| mode.is_whitespace(c as u32);
            let from = start + (line.len() - line.trim_start_matches(ws).len());
            let to = start + line.trim_end_matches(ws).len();
            if from < to {
                out.push(Span::new(from as u32, to as u32));
            }
        };
        let mut start = 0;
        for (i, c) in text.char_indices() {
            // the detector moves past a line end only after a line that is
            // not empty, so an empty line's end starts the next line, which
            // the trim takes off again
            if (c == '\n' || c == '\r') && i > start {
                push(start, i);
                start = i + 1;
            }
        }
        if text.len() > start {
            push(start, text.len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tokens(t: &impl TokenSpans, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        t.token_spans(text, &mut out);
        out.iter().map(|s| text[s.range()].to_owned()).collect()
    }

    #[test]
    fn rule_tokenizers_by_name() {
        let o = TokenizerOptions::default();
        assert_eq!(tokens(&RuleTokenizer::from_name("simple", o).unwrap(), "a-b c"), ["a", "-", "b", "c"]);
        assert_eq!(tokens(&RuleTokenizer::from_name("whitespace", o).unwrap(), "a-b c"), ["a-b", "c"]);
        assert_eq!(RuleTokenizer::from_name("maxent", o), None);
        assert_eq!(RuleTokenizer::default(), RuleTokenizer::Whitespace(o));
    }

    #[test]
    fn a_sentence_per_line() {
        let text = "\n a b \r\n\n\u{85}c\n \n d";
        let mut out = Vec::new();
        NewlineSentences(WhitespaceMode::Unicode).sentence_spans(text, &mut out);
        assert_eq!(out.iter().map(|s| &text[s.range()]).collect::<Vec<_>>(), ["a b", "c", "d"]);
        out.clear();
        NewlineSentences(WhitespaceMode::Legacy).sentence_spans(text, &mut out);
        assert_eq!(out.iter().map(|s| &text[s.range()]).collect::<Vec<_>>(), ["a b", "\u{85}c", "d"]);
    }

    #[test]
    fn a_closure_is_a_tokenizer() {
        let whole = |text: &str, out: &mut Vec<Span>| out.push(Span::new(0, text.len() as u32));
        assert_eq!(tokens(&whole, "a b"), ["a b"]);
    }
}
