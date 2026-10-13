//! Windows of a fixed number of the model's tokens.

use turbo_analysis::{Document, Ix, Span as TextSpan};
use turbo_analysis_layers::Subwords;

use super::{Chunker, chunk_span, need, setting};
use crate::Result;
use crate::embedder::Embedder;
use crate::record::Setting;
use crate::spans::Span;

/// Windows of `tokens` subword pieces, each starting `tokens - overlap`
/// pieces after the one before, the first at the first piece. The last
/// window ends at the last piece and may be shorter; no window lies
/// wholly inside the one before. A window spans from its first piece's
/// start to its last piece's end, so it can end inside a word, as such
/// baselines do.
///
/// `tokens` counts the text's own pieces; the row the model sees also
/// holds the template's special tokens and any prompt prefix.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct FixedTokens {
    tokens: u32,
    overlap: u32,
}

impl FixedTokens {
    /// Refused: no tokens, and an overlap of the whole window or more.
    pub fn new(tokens: u32, overlap: u32) -> Result<FixedTokens> {
        if tokens == 0 {
            return Err("a fixed window of 0 tokens".into());
        }
        if overlap >= tokens {
            return Err(format!("an overlap of {overlap} tokens in windows of {tokens}"));
        }
        Ok(FixedTokens { tokens, overlap })
    }

    pub fn tokens(&self) -> u32 {
        self.tokens
    }
}

impl Chunker for FixedTokens {
    fn name(&self) -> &'static str {
        "fixed-tokens"
    }

    fn config(&self) -> Vec<Setting> {
        vec![setting("tokens", self.tokens), setting("overlap", self.overlap)]
    }

    fn chunk(&self, doc: &Document, _: &mut dyn Embedder) -> Result<Vec<Span>> {
        let pieces = &need::<Subwords>(doc, self.name())?.spans;
        let n = pieces.len();
        let (size, step) = (self.tokens as usize, (self.tokens - self.overlap) as usize);
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let end = (start + size).min(n);
            let (first, last) = (pieces.get(Ix::new(start as u32)), pieces.get(Ix::new(end as u32 - 1)));
            if let Some(s) = chunk_span(TextSpan::new(first.start, last.end.max(first.start))) {
                out.push(s);
            }
            if end == n {
                break;
            }
            start += step;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunkers::tests::{analyzed, texts};
    use crate::embedder::stub::Stub;

    const TEXT: &str = "one two three four five six seven";

    fn cut(tokens: u32, overlap: u32, text: &str) -> Vec<String> {
        let mut e = Stub::new(&["one"]);
        let doc = analyzed(text, &e, None);
        let spans = FixedTokens::new(tokens, overlap).unwrap().chunk(&doc, &mut e).unwrap();
        texts(text, &spans).into_iter().map(String::from).collect()
    }

    #[test]
    fn windows_by_hand() {
        // Seven pieces, windows of three, one shared: starts 0, 2, 4; the
        // window at 4 reaches the last piece, so it is the last.
        assert_eq!(cut(3, 1, TEXT), ["one two three", "three four five", "five six seven"]);
        // No overlap: starts 0, 3, 6; the last holds one piece.
        assert_eq!(cut(3, 0, TEXT), ["one two three", "four five six", "seven"]);
        // A window as long as the text is the text.
        assert_eq!(cut(7, 2, TEXT), [TEXT]);
        assert_eq!(cut(100, 0, TEXT), [TEXT]);
        // Pieces across lines: the window holds the line break between.
        assert_eq!(cut(2, 0, "a b\nc d"), ["a b", "c d"]);
        assert!(cut(3, 0, "").is_empty());
    }

    #[test]
    fn bad_settings_are_refused() {
        assert!(FixedTokens::new(0, 0).is_err());
        assert!(FixedTokens::new(4, 4).is_err());
        let c = FixedTokens::new(256, 32).unwrap();
        assert_eq!(c.config(), [setting("tokens", 256), setting("overlap", 32)]);
    }
}
