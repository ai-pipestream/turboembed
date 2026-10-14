//! The word and tag format: `WordTagSampleStream` and `POSSample.parse`.
//!
//! One sentence per line, tokens separated by whitespace, each token
//! `word_tag`:
//!
//! ```text
//! The_DT driver_NN stopped_VBD ._.
//! ```
//!
//! The tag is everything after the token's last `_`, so a word may hold
//! `_` and a tag may be empty (`word_`). Every line is a sample, an empty
//! one included (a sample with no tokens). A line with a token that has
//! no `_` is not refused: OpenNLP logs it and returns a sample with no
//! tokens in its place, and so does this reader, which keeps a
//! [`Warning::NoTagSeparator`] for it.

use turbo_unicode::WhitespaceMode;

use crate::error::{FormatError, Warning};
use crate::sample::pos::PosSample;
use crate::source::{Line, Lines, whitespace_tokens};

/// Reads part-of-speech samples from the word and tag format.
#[derive(Clone, Debug)]
pub struct WordTagReader<'a> {
    lines: Lines<'a>,
    mode: WhitespaceMode,
    warnings: Vec<Warning>,
}

impl<'a> WordTagReader<'a> {
    /// A reader of `source`, splitting lines on whitespace in `mode`.
    pub fn new(source: &'a str, mode: WhitespaceMode) -> Result<WordTagReader<'a>, FormatError> {
        Ok(WordTagReader { lines: Lines::new(source)?, mode, warnings: Vec::new() })
    }

    /// The lines read so far that were replaced by an empty sample.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }
}

impl<'a> Iterator for WordTagReader<'a> {
    type Item = Result<PosSample<'a>, FormatError>;

    fn next(&mut self) -> Option<Self::Item> {
        let line = self.lines.next()?;
        Some(Ok(match parse_line(line, self.mode) {
            Ok(sample) => sample,
            Err(warning) => {
                self.warnings.push(warning);
                PosSample::empty(line.span())
            }
        }))
    }
}

/// `POSSample.parse` of one line; the warning when a token has no `_`.
pub fn parse_line(line: Line<'_>, mode: WhitespaceMode) -> Result<PosSample<'_>, Warning> {
    let mut tokens = Vec::new();
    let mut tags = Vec::new();
    for part in whitespace_tokens(line.field, mode) {
        let Some(split) = part.text.rfind('_') else {
            return Err(Warning::NoTagSeparator { line: line.number, token: part.span });
        };
        tokens.push(part.slice(0, split));
        tags.push(part.slice(split + 1, part.text.len()));
    }
    Ok(PosSample::new(tokens, tags, line.span()).expect("one tag per token"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_at_the_last_underscore() {
        let src = "a_b_C d_ _x\n\nno tag_T\n";
        let mut r = WordTagReader::new(src, WhitespaceMode::Unicode).unwrap();
        let samples: Vec<PosSample<'_>> = r.by_ref().map(Result::unwrap).collect();
        assert_eq!(samples.len(), 3);
        let pairs: Vec<(&str, &str)> =
            samples[0].tokens().iter().zip(samples[0].tags()).map(|(w, t)| (w.text, t.text)).collect();
        assert_eq!(pairs, [("a_b", "C"), ("d", ""), ("", "x")]);
        assert_eq!(samples[0].to_string(), "a_b_C d_ _x");
        assert!(samples[1].is_empty());
        assert!(samples[2].is_empty());
        assert_eq!(samples[2].source(), turbo_analysis::Span::new(13, 21));
        assert_eq!(r.warnings(), [Warning::NoTagSeparator { line: 3, token: turbo_analysis::Span::new(13, 15) }]);
        for t in samples[0].tokens().iter().chain(samples[0].tags()) {
            assert_eq!(&src[t.span.range()], t.text);
        }
    }
}
