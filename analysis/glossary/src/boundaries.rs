//! The offsets a character match may start and end at.

use turbo_analysis::Span;
use turbo_analysis_layers::Tokens;

/// A set of byte offsets of one text, `0..=len`, as a bit per offset. A
/// character match is kept only when both its ends are in the set: the
/// word boundaries of UAX #29 for OpenNLP's matcher, or the edges of a
/// document's tokens for the step.
#[derive(Clone, Debug)]
pub struct Boundaries {
    bits: Vec<u64>,
    len: u32,
}

impl Boundaries {
    /// No offset of a text of `len` bytes.
    pub fn none(len: u32) -> Boundaries {
        Boundaries { bits: vec![0; (len as usize / 64) + 1], len }
    }

    /// The given offsets of a text of `len` bytes. Offsets past `len` are
    /// ignored.
    pub fn from_offsets(len: u32, offsets: impl IntoIterator<Item = u32>) -> Boundaries {
        let mut b = Boundaries::none(len);
        for x in offsets {
            b.insert(x);
        }
        b
    }

    /// The start and end of the text, and the start and end of every
    /// token. With the tokens of the UAX #29 word tokenizer these are the
    /// word boundaries around every word, but not those inside a run of
    /// spaces, punctuation or symbols, whose segments are not tokens.
    pub fn of_tokens(len: u32, tokens: &Tokens) -> Boundaries {
        let mut b = Boundaries::none(len);
        b.insert(0);
        b.insert(len);
        let (starts, ends) = tokens.spans.columns();
        for (&s, &e) in starts.iter().zip(ends) {
            b.insert(s);
            b.insert(e);
        }
        b
    }

    pub fn insert(&mut self, x: u32) {
        if x <= self.len {
            self.bits[x as usize / 64] |= 1 << (x % 64);
        }
    }

    /// Whether `x` is in the set; an offset past the text never is.
    #[inline]
    pub fn contains(&self, x: u32) -> bool {
        x <= self.len && self.bits[x as usize / 64] & (1 << (x % 64)) != 0
    }

    /// Whether both ends of `s` are in the set.
    #[inline]
    pub fn around(&self, s: Span) -> bool {
        self.contains(s.start) && self.contains(s.end)
    }

    /// The length of the text the set is for.
    pub fn text_len(&self) -> u32 {
        self.len
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Spans, ViewId};

    #[test]
    fn token_edges_and_text_ends() {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(Span::new(1, 4));
        spans.push(Span::new(5, 8));
        let b = Boundaries::of_tokens(70, &Tokens { spans, sentence: None });
        let set: Vec<u32> = (0..=72).filter(|&x| b.contains(x)).collect();
        assert_eq!(set, [0, 1, 4, 5, 8, 70]);
        assert!(b.around(Span::new(1, 8)));
        assert!(!b.around(Span::new(2, 8)));
        assert!(!b.around(Span::new(70, 71)));
    }
}
