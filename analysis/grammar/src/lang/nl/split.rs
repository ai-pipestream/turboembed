//! Cutting a Dutch compound number word into the words it is made of:
//! `tweeentwintig` (folded from `tweeëntwintig`) is `twee`, `en` and
//! `twintig`; `driehonderdeenentachtigduizend` is `drie`, `honderd`,
//! `een`, `en`, `tachtig` and `duizend`; `honderdvijfentwintigste` ends in
//! the ordinal `twintigste`.
//!
//! A word the lexicon does not hold is cut when every piece is a number
//! word, `en` stands only between two of them, and an ordinal or a
//! fraction word comes last. The cut with the fewest pieces wins, then the
//! one found first, which has the shorter early pieces.

use crate::engine::Lexicon;
use crate::value::flag::PLURAL;

use super::{DOZEN, FRAC_WORD, HUNDRED, ORD_SCALE, ORD_TENS, ORD_WORD, SCALE, TEEN, TENS, UNIT};

/// The most pieces a word is cut into.
const MAX_PIECES: u8 = 12;
/// The longest word that is tried.
const MAX_LEN: usize = 96;

/// What a piece is, as far as the order of pieces goes.
#[derive(Copy, Clone, PartialEq, Eq)]
enum Piece {
    /// A cardinal that can go on: `twee`, `honderd`, `duizend`.
    Number,
    /// `vijv`, which only stands before `en`.
    Vijv,
    /// The joining `en`.
    Join,
    /// An ordinal or a fraction word, which ends the compound.
    Last,
}

fn classify(lexicon: &Lexicon, piece: &str) -> Option<Piece> {
    if piece == "en" {
        return Some(Piece::Join);
    }
    if piece == "vijv" {
        return Some(Piece::Vijv);
    }
    let mut found = None;
    for e in lexicon.lookup(piece) {
        match e.dim {
            // `honderden` is no piece: `honderdenzes` is `honderd en zes`.
            UNIT | TEEN | TENS | HUNDRED | SCALE | DOZEN if e.flags & PLURAL == 0 => return Some(Piece::Number),
            ORD_WORD | ORD_TENS | ORD_SCALE | FRAC_WORD => found = Some(Piece::Last),
            _ => {}
        }
    }
    found
}

/// The state after a piece: what may come next.
#[derive(Copy, Clone, PartialEq, Eq)]
enum State {
    /// The start, or after `en`: a number must come.
    Open,
    AfterNumber,
    AfterVijv,
    Done,
}

fn next(state: State, piece: Piece) -> Option<State> {
    match (state, piece) {
        (State::Done, _) => None,
        (State::AfterVijv, Piece::Join) => Some(State::Open),
        (State::AfterVijv, _) => None,
        (State::AfterNumber, Piece::Join) => Some(State::Open),
        (State::Open, Piece::Join) => None,
        (_, Piece::Number) => Some(State::AfterNumber),
        (_, Piece::Vijv) => Some(State::AfterVijv),
        (_, Piece::Last) => Some(State::Done),
    }
}

/// The index of a state in a row of the table.
fn ix(s: State) -> usize {
    s as usize
}

/// Where `word` is cut, appended to `cuts`; nothing when it is a word of
/// the lexicon or no cut fits.
pub(crate) fn split(lexicon: &Lexicon, word: &str, cuts: &mut Vec<u32>) {
    let n = word.len();
    if !(4..=MAX_LEN).contains(&n) || !word.bytes().all(|b| b.is_ascii_lowercase()) || !lexicon.lookup(word).is_empty()
    {
        return;
    }
    // best[pos][state]: the fewest pieces that reach `pos` in `state`, and
    // where the last piece started with which state before it.
    const NONE: (u8, u8, u8) = (u8::MAX, 0, 0);
    let mut best = [[NONE; 4]; MAX_LEN + 1];
    best[0][ix(State::Open)] = (0, 0, 0);
    for i in 0..n {
        for s in [State::Open, State::AfterNumber, State::AfterVijv] {
            let (count, _, _) = best[i][ix(s)];
            if count >= MAX_PIECES {
                continue;
            }
            for j in i + 1..=n {
                let piece = &word[i..j];
                if !lexicon.has_prefix(piece) && !"vijv".starts_with(piece) && !"en".starts_with(piece) {
                    break;
                }
                let Some(p) = classify(lexicon, piece) else { continue };
                let Some(t) = next(s, p) else { continue };
                if t == State::Done && j != n {
                    continue;
                }
                let cell = &mut best[j][ix(t)];
                if count + 1 < cell.0 {
                    *cell = (count + 1, i as u8, ix(s) as u8);
                }
            }
        }
    }
    let end = [State::AfterNumber, State::Done]
        .into_iter()
        .map(|s| (best[n][ix(s)], s))
        .filter(|((c, _, _), _)| *c != u8::MAX)
        .min_by_key(|((c, _, _), s)| (*c, ix(*s)));
    let Some(((count, _, _), state)) = end else { return };
    if count < 2 {
        return;
    }
    let from = cuts.len();
    let (mut pos, mut s) = (n, ix(state));
    while pos > 0 {
        let (_, start, prev) = best[pos][s];
        if start > 0 {
            cuts.push(u32::from(start));
        }
        pos = usize::from(start);
        s = usize::from(prev);
    }
    cuts[from..].reverse();
}

#[cfg(test)]
mod tests {
    use super::super::DUTCH;
    use super::*;
    use crate::engine::Grammar;

    fn pieces(word: &str) -> Vec<&str> {
        let mut cuts = Vec::new();
        split(DUTCH.lexicon(), word, &mut cuts);
        let mut out = Vec::new();
        let mut at = 0;
        for &c in &cuts {
            out.push(&word[at..c as usize]);
            at = c as usize;
        }
        out.push(&word[at..]);
        out
    }

    #[test]
    fn compounds_are_cut_into_number_words() {
        assert_eq!(pieces("tweeentwintig"), ["twee", "en", "twintig"]);
        assert_eq!(pieces("driehonderdeenentachtigduizend"), ["drie", "honderd", "een", "en", "tachtig", "duizend"]);
        assert_eq!(pieces("vijfentwintighonderdenzevenenzeventig").len(), 8);
        assert_eq!(pieces("honderdvijfentwintigste"), ["honderd", "vijf", "en", "twintigste"]);
        assert_eq!(pieces("vijventwintigste"), ["vijv", "en", "twintigste"]);
        assert_eq!(pieces("achttienhonderd"), ["achttien", "honderd"], "the fewest pieces");
        assert_eq!(pieces("tweeenhalf"), ["twee", "en", "half"]);
        assert_eq!(pieces("driekwart"), ["drie", "kwart"]);
    }

    #[test]
    fn other_words_stay_whole() {
        for w in ["zestien", "tweede", "achten", "tienen", "eenheid", "vijvers", "enkel", "honderden", "kwartaal"] {
            assert_eq!(pieces(w), [w], "{w}");
        }
        assert_eq!(pieces("vijvtien"), ["vijvtien"], "vijv only before en");
        assert_eq!(pieces("derdetwee"), ["derdetwee"], "an ordinal ends a compound");
        assert_eq!(pieces("eentwee√"), ["eentwee√"]);
    }
}
