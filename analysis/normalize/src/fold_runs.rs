//! The pass shared by the folds that decompose text, change some of the
//! decomposed characters and normalize the result again: accent folding
//! (NFD, drop marks, NFC) and the confusable skeleton (NFD, map, NFD).
//!
//! Applied to a whole text, such a fold is not a function of each
//! character on its own: normalization puts combining marks in canonical
//! order across character boundaries and composes a base with a mark that
//! was a separate character. The text is therefore cut into runs that
//! cannot interact. A run starts at a boundary character, one that the
//! fold guarantees nothing before it can reorder or combine with, and goes
//! on up to the next one. Each run is folded on its own, which gives the
//! same text as folding the whole text at once.
//!
//! Inside a run the fold is attributed to single characters when it can
//! be: each character is folded on its own as well, and when the pieces
//! put together equal the folded run, every character that changed is
//! its own edit. Otherwise (marks reordered or composed across characters)
//! the whole run is one edit. Either way every output byte maps back to
//! the source characters that produced it.

use std::ops::Range;

use turbo_analysis::ViewBuilder;
use turbo_unicode::normalize::canonical_decomposition;

/// A fold applied run by run. `State` is what folding one character
/// passes on to the next inside a run; every run starts from the default.
pub(crate) trait RunFold {
    type State: Default;

    /// Whether `c` starts a run: nothing before `c` can reorder past it or
    /// combine with it or with anything after it, and the fold of the
    /// text from `c` on does not depend on what came before.
    fn is_boundary(&self, c: char) -> bool;

    /// Whether `c` is a boundary character that folds to itself when it
    /// is a run on its own. Runs made of one such character are skipped.
    fn is_inert(&self, c: char) -> bool;

    /// Appends the fold of a whole run to `out`.
    fn fold_run(&self, run: &str, out: &mut String, scratch: &mut Scratch);

    /// Appends the fold of one character of a run to `out`, given and
    /// updating the state the characters before it in the run left.
    fn fold_char(&self, c: char, state: &mut Self::State, out: &mut String, scratch: &mut Scratch);
}

/// Buffers reused across runs, so folding allocates only while they grow.
#[derive(Default)]
pub(crate) struct Scratch {
    pub(crate) decomposed: String,
    pub(crate) mapped: String,
    piece: String,
    whole: String,
    pieces: String,
    /// For each character of a run: its end in the run and the end of its
    /// piece in `pieces`.
    ends: Vec<(usize, usize)>,
}

/// Copies `text` into `out` with `fold` applied run by run.
pub(crate) fn fold_runs<F: RunFold>(fold: &F, text: &str, out: &mut ViewBuilder<'_>) {
    let mut scratch = Scratch::default();
    let bytes = text.as_bytes();
    let mut run_start = 0;
    let mut needs_work = false;
    let mut i = 0;
    while i < bytes.len() {
        let c = if bytes[i] < 0x80 {
            char::from(bytes[i])
        } else {
            text[i..].chars().next().expect("i is at a character boundary")
        };
        if fold.is_boundary(c) {
            if needs_work {
                fold_one_run(fold, text, run_start..i, out, &mut scratch);
            }
            run_start = i;
            needs_work = !fold.is_inert(c);
        } else {
            needs_work = true;
        }
        i += c.len_utf8();
    }
    if needs_work {
        fold_one_run(fold, text, run_start..text.len(), out, &mut scratch);
    }
}

/// Folds one run and writes it: nothing when it folds to itself, one edit
/// per changed character when the per-character folds add up to the fold
/// of the run, one edit for the run otherwise.
fn fold_one_run<F: RunFold>(fold: &F, text: &str, run: Range<usize>, out: &mut ViewBuilder<'_>, s: &mut Scratch) {
    let source = &text[run.clone()];
    let mut whole = std::mem::take(&mut s.whole);
    whole.clear();
    fold.fold_run(source, &mut whole, s);
    if whole != source {
        if per_char_matches(fold, source, &whole, s) {
            write_per_char(source, run.start, out, s);
        } else {
            out.keep(run.start);
            out.replace(run.end, &whole);
        }
    }
    s.whole = whole;
}

/// Folds each character of `source` on its own into `s.pieces`, recording
/// the ends in `s.ends`, and says whether the pieces equal `whole`.
fn per_char_matches<F: RunFold>(fold: &F, source: &str, whole: &str, s: &mut Scratch) -> bool {
    let mut pieces = std::mem::take(&mut s.pieces);
    let mut piece = std::mem::take(&mut s.piece);
    pieces.clear();
    s.ends.clear();
    let mut state = F::State::default();
    for (at, c) in source.char_indices() {
        piece.clear();
        fold.fold_char(c, &mut state, &mut piece, s);
        pieces.push_str(&piece);
        s.ends.push((at + c.len_utf8(), pieces.len()));
    }
    let matches = pieces == whole;
    s.pieces = pieces;
    s.piece = piece;
    matches
}

/// Writes the pieces in `s` for the run that starts at `offset`: a
/// replace for each character whose piece differs from it.
fn write_per_char(source: &str, offset: usize, out: &mut ViewBuilder<'_>, s: &Scratch) {
    let (mut from, mut piece_from) = (0, 0);
    for &(to, piece_to) in &s.ends {
        let piece = &s.pieces[piece_from..piece_to];
        if piece != &source[from..to] {
            out.keep(offset + from);
            out.replace(offset + to, piece);
        }
        (from, piece_from) = (to, piece_to);
    }
}

/// The first character of the canonical decomposition (NFD) of `c`:
/// `c` itself when it does not decompose, the leading jamo of a Hangul
/// syllable, otherwise the first character of its mapping, followed down.
pub(crate) fn nfd_first(mut c: char) -> char {
    const HANGUL_FIRST: u32 = 0xAC00;
    const HANGUL_COUNT: u32 = 11172;
    const LEADING_FIRST: u32 = 0x1100;
    const PER_LEADING: u32 = 588;
    loop {
        let offset = (c as u32).wrapping_sub(HANGUL_FIRST);
        if offset < HANGUL_COUNT {
            return char::from_u32(LEADING_FIRST + offset / PER_LEADING).expect("a leading jamo");
        }
        match canonical_decomposition(c) {
            Some(mapping) => c = mapping[0],
            None => return c,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_character_of_the_decomposition() {
        assert_eq!(nfd_first('a'), 'a');
        assert_eq!(nfd_first('\u{E9}'), 'e');
        // U+1EC7 decomposes to U+1EB9 U+0302, and U+1EB9 to e U+0323.
        assert_eq!(nfd_first('\u{1EC7}'), 'e');
        // U+0344 COMBINING GREEK DIALYTIKA TONOS decomposes to U+0308 U+0301.
        assert_eq!(nfd_first('\u{344}'), '\u{308}');
        assert_eq!(nfd_first('\u{AC00}'), '\u{1100}');
        assert_eq!(nfd_first('\u{D7A3}'), '\u{1112}');
    }
}
