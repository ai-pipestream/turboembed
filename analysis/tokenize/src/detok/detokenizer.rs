//! The detokenizer: resolves each token's operation and writes the
//! rebuilt text.
//!
//! The rules are those of `DictionaryDetokenizer`. A token the dictionary
//! does not hold gets [`NoOperation`](Operation::NoOperation). A
//! `RightLeftMatching` token resolves to `MergeToRight` the first time
//! that token is seen and to `MergeToLeft` the next time, and so on,
//! with one flag per distinct token: two different quote characters
//! alternate independently, and an odd count leaves the last one
//! attached to the right.
//!
//! Between a token and the next, the text gets one space unless the next
//! token attaches to the left (`MergeToLeft`, `MergeBoth`) or this token
//! attaches to the right (`MergeToRight`, `MergeBoth`). Where no space is
//! written, the split marker is, when there is one. Nothing is written
//! after the last token.
//!
//! The operations are resolved as the text is written, one token ahead.
//! Besides growing the output, a call allocates only a flag vector of one
//! `u64` per 64 matching tokens beyond the first 64, and nothing when the
//! dictionary has at most 64 matching tokens.

use turbo_analysis::Span;

use super::dictionary::DetokenizerDictionary;
use super::{DictionaryOperation, Operation};

/// Rebuilds text from tokens with a [`DetokenizerDictionary`].
#[derive(Copy, Clone, Debug)]
pub struct Detokenizer<'d> {
    dictionary: &'d DetokenizerDictionary,
}

impl<'d> Detokenizer<'d> {
    pub fn new(dictionary: &'d DetokenizerDictionary) -> Detokenizer<'d> {
        Detokenizer { dictionary }
    }

    pub fn dictionary(&self) -> &'d DetokenizerDictionary {
        self.dictionary
    }

    /// Appends the resolved operation of every token to `out`, one per
    /// token, as `DictionaryDetokenizer.detokenize(String[])` returns
    /// them.
    pub fn operations(&self, tokens: &[&str], out: &mut Vec<Operation>) {
        out.reserve(tokens.len());
        let mut resolver = Resolver::new(self.dictionary);
        out.extend(tokens.iter().map(|t| resolver.next(t)));
    }

    /// Appends the text rebuilt from `tokens` to `out`, as
    /// `DictionaryDetokenizer.detokenize(String[], String)` returns it.
    pub fn detokenize(&self, tokens: &[&str], split_marker: Option<&str>, out: &mut String) {
        self.write(tokens.len(), |i| tokens[i], split_marker, out, None);
    }

    /// Appends the text rebuilt from the tokens at `tokens` in `text` to
    /// `out`, and appends to `token_offsets`, for every token, the byte
    /// offset in `out` where that token starts. The token ends at its
    /// start plus its length, so `out[start..start + len]` is the token.
    ///
    /// # Panics
    ///
    /// When a span is not a range of character boundaries in `text`, or
    /// when a token starts at or beyond `u32::MAX` bytes into `out`.
    pub fn detokenize_spans(
        &self,
        text: &str,
        tokens: &[Span],
        split_marker: Option<&str>,
        out: &mut String,
        token_offsets: &mut Vec<u32>,
    ) {
        token_offsets.reserve(tokens.len());
        self.write(tokens.len(), |i| &text[tokens[i].range()], split_marker, out, Some(token_offsets));
    }

    /// The one implementation behind both forms: `n` tokens, the `i`th
    /// given by `token(i)`.
    fn write<'t>(
        &self,
        n: usize,
        token: impl Fn(usize) -> &'t str,
        split_marker: Option<&str>,
        out: &mut String,
        mut starts: Option<&mut Vec<u32>>,
    ) {
        if n == 0 {
            return;
        }
        // Each gap between two tokens is a space or the marker.
        let gap = split_marker.map_or(1, |m| m.len().max(1));
        out.reserve((0..n).map(|i| token(i).len()).sum::<usize>() + (n - 1) * gap);
        let mut resolver = Resolver::new(self.dictionary);
        let mut current = resolver.next(token(0));
        let mut i = 0;
        loop {
            if let Some(starts) = starts.as_deref_mut() {
                let start = u32::try_from(out.len()).expect("the detokenized text is longer than u32::MAX bytes");
                starts.push(start);
            }
            out.push_str(token(i));
            i += 1;
            if i == n {
                return;
            }
            let next = resolver.next(token(i));
            if next.merges_left() || current.merges_right() {
                if let Some(marker) = split_marker {
                    out.push_str(marker);
                }
            } else {
                out.push(' ');
            }
            current = next;
        }
    }
}

/// Resolves the operation of each token in order, keeping one flag per
/// matching token: set while the token is open (seen an odd number of
/// times).
struct Resolver<'d> {
    dictionary: &'d DetokenizerDictionary,
    /// The flags of slots 0 to 63.
    low: u64,
    /// The flags of slots 64 and up, 64 per word; empty, and never
    /// allocated, when the dictionary has no more than 64 matching tokens.
    high: Vec<u64>,
}

impl<'d> Resolver<'d> {
    fn new(dictionary: &'d DetokenizerDictionary) -> Resolver<'d> {
        let slots = dictionary.slots() as usize;
        let high = if slots > 64 { vec![0; (slots - 64).div_ceil(64)] } else { Vec::new() };
        Resolver { dictionary, low: 0, high }
    }

    fn next(&mut self, token: &str) -> Operation {
        let Some(rule) = self.dictionary.rule(token) else {
            return Operation::NoOperation;
        };
        match rule.operation {
            DictionaryOperation::MoveLeft => return Operation::MergeToLeft,
            DictionaryOperation::MoveRight => return Operation::MergeToRight,
            DictionaryOperation::MoveBoth => return Operation::MergeBoth,
            DictionaryOperation::RightLeftMatching => {}
        }
        let slot = rule.slot as usize;
        let word = if slot < 64 { &mut self.low } else { &mut self.high[(slot - 64) / 64] };
        let bit = 1u64 << (slot % 64);
        let open = *word & bit != 0;
        *word ^= bit;
        if open { Operation::MergeToLeft } else { Operation::MergeToRight }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_than_64_matching_tokens() {
        let names: Vec<String> = (0..200).map(|i| format!("q{i}")).collect();
        let mut d = DetokenizerDictionary::new();
        for name in &names {
            d.insert(name, DictionaryOperation::RightLeftMatching);
        }
        let mut r = Resolver::new(&d);
        assert_eq!(r.high.len(), 3);
        // Open every token, then close them in reverse: each flag is its
        // own, in the low word and in the spilled words alike.
        for name in &names {
            assert_eq!(r.next(name), Operation::MergeToRight, "{name}");
        }
        for name in names.iter().rev() {
            assert_eq!(r.next(name), Operation::MergeToLeft, "{name}");
        }
        assert_eq!(r.low, 0);
        assert!(r.high.iter().all(|&w| w == 0));
        assert_eq!(r.next("q199"), Operation::MergeToRight);
        assert_eq!(r.next("q5"), Operation::MergeToRight);
        assert_eq!(r.next("q5"), Operation::MergeToLeft);
        assert_eq!(r.next("q199"), Operation::MergeToLeft);
    }

    #[test]
    fn the_output_is_reserved_once() {
        let mut d = DetokenizerDictionary::new();
        d.insert(".", DictionaryOperation::MoveLeft);
        let detokenizer = Detokenizer::new(&d);
        // A marker in every gap: 50 tokens and 49 markers of 7 bytes. An
        // empty string grows to exactly what is reserved, so a capacity
        // of exactly that length means the text never outgrew the one
        // reservation.
        let tokens: Vec<&str> = std::iter::repeat_n(".", 50).collect();
        let mut out = String::new();
        detokenizer.detokenize(&tokens, Some("<SPLIT>"), &mut out);
        assert_eq!(out.len(), 50 + 49 * 7);
        assert_eq!(out.capacity(), out.len());
        // Without a marker every gap may be a space.
        let tokens: Vec<&str> = std::iter::repeat_n("ab", 50).collect();
        let mut out = String::new();
        detokenizer.detokenize(&tokens, None, &mut out);
        assert_eq!(out.capacity(), out.len());
        assert_eq!(out.len(), 100 + 49);
    }

    #[test]
    fn few_matching_tokens_allocate_no_flags() {
        let mut d = DetokenizerDictionary::new();
        for i in 0..64 {
            d.insert(&i.to_string(), DictionaryOperation::RightLeftMatching);
        }
        assert_eq!(Resolver::new(&d).high.capacity(), 0);
        d.insert("64", DictionaryOperation::RightLeftMatching);
        assert_eq!(Resolver::new(&d).high.len(), 1);
    }
}
