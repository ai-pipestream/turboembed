//! Cutting a Spanish compound number word into the words it is made of:
//! `tresmillonesimo` is `tres` and `millonesimo`, `doscientosavos` is
//! `doscientos` and `avos`, `treintaicincoavos` is `treinta`, `i`, `cinco`
//! and `avos`.
//!
//! A word the lexicon does not hold is cut when it ends in a scale or
//! other ordinal, or in `avo`, and everything before is number words: the
//! cut with the fewest pieces wins, then the one with the longest first
//! piece. `i` joins only tens and a unit, as in `treintaicinco`.

use crate::engine::Lexicon;

use super::{AVO, HUNDREDS, ORD_HUND, ORD_SCALE, ORD_TENS, ORD_WORD, SCALE, TEEN, TENS, UNIT};

/// The most pieces a word is cut into.
const MAX_PIECES: usize = 6;

/// What a piece of a compound is.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Piece {
    /// A number word that can open or continue a compound.
    Number { tens: bool, unit: bool },
    /// An ordinal word that can continue one: `vigesimo` in
    /// `vigesimoprimero`.
    OrdinalHead,
    /// What ends one: a scale or other ordinal, or `avo`.
    End,
    /// The `i` of `treintaicinco`.
    Join,
}

fn classify(lexicon: &Lexicon, word: &str, last: bool) -> Option<Piece> {
    if word == "i" {
        return (!last).then_some(Piece::Join);
    }
    let mut found = None;
    for e in lexicon.lookup(word) {
        let piece = match e.dim {
            UNIT | TEEN | TENS | HUNDREDS | SCALE if !last => {
                Piece::Number { tens: e.dim == TENS, unit: e.dim == UNIT }
            }
            ORD_SCALE | ORD_WORD | ORD_TENS | ORD_HUND | AVO if last => Piece::End,
            ORD_TENS | ORD_HUND if !last => Piece::OrdinalHead,
            _ => continue,
        };
        found = Some(piece);
        if piece != Piece::OrdinalHead {
            break;
        }
    }
    found
}

/// Whether `next` may follow `prev` in a compound.
fn follows(prev: Option<Piece>, next: Piece) -> bool {
    match (prev, next) {
        (None, Piece::Join) => false,
        (Some(Piece::Join), Piece::Number { unit, .. }) => unit,
        (Some(Piece::Join), _) => false,
        (Some(Piece::Number { tens, .. }), Piece::Join) => tens,
        (Some(Piece::Number { .. }), Piece::OrdinalHead) => false,
        (Some(Piece::OrdinalHead), Piece::Number { .. } | Piece::Join) => false,
        _ => true,
    }
}

/// Where `word` is cut, appended to `cuts`; nothing when it is a word of
/// the lexicon or no cut fits.
pub(crate) fn split(lexicon: &Lexicon, word: &str, cuts: &mut Vec<u32>) {
    if word.len() < 4 || !lexicon.lookup(word).is_empty() || !word.bytes().all(|b| b.is_ascii_lowercase()) {
        return;
    }
    // The last piece first, then the shortest cut of what is before it.
    let n = word.len();
    let mut best: Option<([u32; MAX_PIECES], usize)> = None;
    for start in 1..n {
        let Some(end) = classify(lexicon, &word[start..], true) else { continue };
        let mut path = [0u32; MAX_PIECES];
        if let Some(k) = cut_prefix(lexicon, &word[..start], None, end, &mut path, 0)
            && best.is_none_or(|(_, b)| k < b)
        {
            path[k - 1] = start as u32;
            best = Some((path, k));
        }
    }
    if let Some((path, k)) = best {
        cuts.extend_from_slice(&path[..k]);
    }
}

/// Cuts `prefix` into pieces that may come before `end`, the fewest
/// first and then the longest first piece. Writes the cut offsets (after
/// the first piece) into `path` from `depth` on and returns the count of
/// offsets the whole word needs before the last piece.
fn cut_prefix(
    lexicon: &Lexicon,
    prefix: &str,
    prev: Option<Piece>,
    end: Piece,
    path: &mut [u32; MAX_PIECES],
    depth: usize,
) -> Option<usize> {
    if depth + 1 >= MAX_PIECES {
        return None;
    }
    let base = if depth == 0 { 0 } else { path[depth - 1] as usize };
    let mut best: Option<([u32; MAX_PIECES], usize)> = None;
    for len in (1..=prefix.len()).rev() {
        let Some(piece) = classify(lexicon, &prefix[..len], false) else { continue };
        if !follows(prev, piece) {
            continue;
        }
        let mut trial = *path;
        let rest = &prefix[len..];
        let count = if rest.is_empty() {
            follows(Some(piece), end).then_some(depth + 1)
        } else {
            trial[depth] = (base + len) as u32;
            cut_prefix(lexicon, rest, Some(piece), end, &mut trial, depth + 1)
        };
        if let Some(k) = count
            && best.is_none_or(|(_, b)| k < b)
        {
            best = Some((trial, k));
        }
    }
    let (found, k) = best?;
    *path = found;
    Some(k)
}
