//! SentencePiece's BPE trainer (`bpe_model_trainer.cc` of the
//! sentencepiece library, version 0.2.2) with the `identity`
//! normalization, written as a `.model` file that the core's
//! SentencePiece BPE (`turbo::sentencepiece`) and the library both load.
//!
//! Each line is a sentence. A line that is empty, longer than 4192 bytes
//! or holds U+2585 (`▅`, the library's mark for an unknown character) is
//! left out, as the library leaves it out. A sentence is normalized as
//! the identity rule does: spaces at its ends dropped, each run of spaces
//! one space, each space U+2581 (`▁`), a `▁` put in front, and every `▁`
//! at the end taken off. The meta pieces (`<unk>`, `<s>`, `</s>`) written
//! in it become a boundary no piece crosses, as does a tab. The
//! characters are counted; the most frequent are kept until
//! `character_coverage` of all characters is covered, and the rest are
//! unknown. The sentence is cut into words, each starting at a `▁`.
//!
//! The most frequent pair of adjacent pieces is merged until the
//! vocabulary is full: of equal counts the one whose piece is shorter in
//! characters, then the one whose piece comes first in code point order.
//! A pair is never merged when its piece would be unknown, hold `▁`
//! anywhere but at its start, mix two Unicode scripts (Hiragana and
//! Katakana count as Han, a combining mark as the script before it, and
//! a character outside every script as Common), or be longer than 16
//! characters. A pair whose piece an earlier merge already made is set
//! aside, not merged. When no pair occurs any more and the vocabulary is
//! not yet full, the library goes on with the pairs that occurred at some
//! point and no longer do, in the same order at count 0, each a piece no
//! word holds; so does this. The model's pieces are `<unk>`, `<s>` and `</s>`,
//! then each merged piece with the score minus its rank, then the kept
//! characters, most frequent first, with the scores that follow.
//!
//! Where this differs from the library:
//!
//! - Only the identity normalization is run: no character map
//!   (`nmt_nfkc`, the library's default, or a rule file). A corpus to be
//!   normalized otherwise must be normalized first.
//! - The other trainer options are the library's defaults and cannot be
//!   set: no user-defined or control symbols, no byte fallback, no digit
//!   splitting, whitespace as a prefix, no sentence sampling.
//! - Scripts are Unicode's current ones (`turbo-unicode`); the library's
//!   table is of Unicode 9, so a character assigned since then is Common
//!   there (`scripts_are_current_unicodes` in `tests/sentencepiece.rs`).
//! - Two different pairs that make the same piece with the same count are
//!   the one tie the library leaves to the layout of its heap; here the
//!   pair with the smaller ids merges, and the other is set aside as
//!   above. On the corpora of `tests/sentencepiece.rs` the pieces are the
//!   library's.

use turbo_unicode::Script;

use crate::TrainError;
use crate::merger::{Merger, Order};
use crate::pieces::{FxMap, Pair, Pieces, SymbolId};
use crate::proto::Message;

/// `▁`, which stands for a space.
pub const SPACE: char = '\u{2581}';
/// What an unknown character becomes in the corpus (`kUNKChar`).
const UNKNOWN: char = '\u{2585}';
/// A boundary no piece crosses (`kUPPBoundaryChar`).
const BOUNDARY: char = '\t';
/// The library's `max_sentence_length`, in bytes.
const MAX_SENTENCE_BYTES: usize = 4192;
/// The library's `max_sentencepiece_length`, in characters.
const MAX_PIECE_CHARS: usize = 16;
/// The meta pieces and their ids, as the library's defaults make them.
const META: [(&str, PieceKind); 3] =
    [("<unk>", PieceKind::Unknown), ("<s>", PieceKind::Control), ("</s>", PieceKind::Control)];

#[derive(Clone, Debug, PartialEq)]
pub struct SentencePieceConfig {
    /// The number of pieces, the three meta pieces and the characters
    /// included. The library refuses a size it cannot reach; so does this.
    pub vocab_size: usize,
    /// The share of the corpus's characters the kept characters cover,
    /// from 0 to 1. The library's default is 0.9995.
    pub character_coverage: f32,
}

/// A piece's type, as `ModelProto.SentencePiece.Type` numbers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PieceKind {
    Normal = 1,
    Unknown = 2,
    Control = 3,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TrainedPiece {
    pub piece: String,
    pub score: f32,
    pub kind: PieceKind,
}

/// A trained SentencePiece BPE.
#[derive(Clone, Debug)]
pub struct SentencePieceBpe {
    config: SentencePieceConfig,
    pieces: Vec<TrainedPiece>,
}

/// A sentence as the identity rule normalizes it, with spaces escaped,
/// a `▁` in front and none at the end; empty when it holds nothing but
/// spaces and `▁`.
pub fn normalize(sentence: &str) -> String {
    let mut out = String::with_capacity(sentence.len() + 3);
    let mut space = true;
    for c in sentence.trim_start_matches(' ').chars() {
        if c == ' ' {
            space = true;
            continue;
        }
        if space {
            out.push(SPACE);
            space = false;
        }
        out.push(c);
    }
    // The library takes off every `▁` at the end, one written in the
    // text included.
    while out.ends_with(SPACE) {
        out.pop();
    }
    out
}

/// `text` with each meta piece written in it replaced by the boundary,
/// the longest first where two start at one place.
fn mark_meta(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    'outer: while let Some(c) = rest.chars().next() {
        let mut longest: Option<usize> = None;
        for (m, _) in META {
            if rest.starts_with(m) && longest.is_none_or(|l| m.len() > l) {
                longest = Some(m.len());
            }
        }
        if let Some(len) = longest {
            out.push(BOUNDARY);
            rest = &rest[len..];
            continue 'outer;
        }
        out.push(c);
        rest = &rest[c.len_utf8()..];
    }
    out
}

/// The script a piece may not mix with another, as the library sees it.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PieceScript {
    /// Not yet known: the start of the piece, or a combining mark there.
    Any,
    Is(Script),
}

fn piece_script(c: char, before: PieceScript) -> PieceScript {
    match turbo_unicode::script::script(c) {
        Script::Hiragana | Script::Katakana | Script::Han => PieceScript::Is(Script::Han),
        _ if c == '\u{30fc}' => PieceScript::Is(Script::Han),
        Script::Inherited => before,
        Script::Unknown => PieceScript::Is(Script::Common),
        s => PieceScript::Is(s),
    }
}

/// `IsValidSentencePiece` with the library's default options.
pub fn is_valid_piece(piece: &str) -> bool {
    let mut n = 0;
    let mut before = PieceScript::Any;
    for (i, c) in piece.chars().enumerate() {
        n += 1;
        if n > MAX_PIECE_CHARS || matches!(c, UNKNOWN | '\0' | BOUNDARY | ' ') {
            return false;
        }
        if c == SPACE {
            if i > 0 {
                return false;
            }
            continue;
        }
        let s = piece_script(c, before);
        if let (PieceScript::Is(a), PieceScript::Is(b)) = (before, s)
            && a != b
        {
            return false;
        }
        before = s;
    }
    n > 0
}

/// The library's order: shorter pieces first, then code point order,
/// which is the order of their UTF-8 bytes.
struct ByPiece;

impl Order for ByPiece {
    type Key = (usize, String);

    fn key(&self, pieces: &Pieces, pair: Pair) -> (usize, String) {
        let s = format!("{}{}", pieces.text(pair.left), pieces.text(pair.right));
        (s.chars().count(), s)
    }

    fn allows(&self, pieces: &Pieces, pair: Pair) -> bool {
        is_valid_piece(&format!("{}{}", pieces.text(pair.left), pieces.text(pair.right)))
    }
}

/// The sentences the library trains on, normalized and marked, each
/// with how often it occurs (always 1 from text).
fn sentences<'a>(lines: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut out = Vec::new();
    for line in lines {
        if line.is_empty() || line.len() > MAX_SENTENCE_BYTES || line.contains(UNKNOWN) {
            continue;
        }
        let n = mark_meta(&normalize(line));
        if !n.is_empty() {
            out.push(n);
        }
    }
    out
}

/// The characters kept, with their counts, most frequent first and in
/// code point order among equal counts.
fn kept_chars(sentences: &[String], coverage: f32) -> Vec<(char, u64)> {
    let mut counts: FxMap<char, u64> = FxMap::default();
    let mut all = 0u64;
    for s in sentences {
        for c in s.chars().filter(|&c| c != '\0') {
            *counts.entry(c).or_default() += 1;
            all += 1;
        }
    }
    let mut sorted: Vec<(char, u64)> = counts.into_iter().collect();
    sorted.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut kept = Vec::new();
    let mut covered = 0u64;
    for (c, n) in sorted {
        // The library works the share out as a double, keeps it as a
        // float and compares it with the float option.
        if (covered as f64 / all as f64) as f32 >= coverage {
            break;
        }
        covered += n;
        if c != BOUNDARY {
            kept.push((c, n));
        }
    }
    kept
}

/// The words of the sentences, each starting at a `▁` or at the start,
/// with how often each occurs, unknown characters made `UNKNOWN`.
fn words(sentences: &[String], kept: &FxMap<char, u64>) -> Vec<(String, u64)> {
    let mut at: FxMap<String, usize> = FxMap::default();
    let mut out: Vec<(String, u64)> = Vec::new();
    for s in sentences {
        let s: String = s.chars().map(|c| if kept.contains_key(&c) { c } else { UNKNOWN }).collect();
        let mut start = 0;
        let starts = s.char_indices().filter(|&(i, c)| i > 0 && c == SPACE).map(|(i, _)| i).chain([s.len()]);
        for end in starts {
            let w = &s[start..end];
            start = end;
            match at.get(w) {
                Some(&i) => out[i].1 += 1,
                None => {
                    at.insert(w.to_owned(), out.len());
                    out.push((w.to_owned(), 1));
                }
            }
        }
    }
    out
}

impl SentencePieceBpe {
    /// Trains on `lines`, one sentence each.
    pub fn train<'a>(
        lines: impl IntoIterator<Item = &'a str>,
        config: SentencePieceConfig,
    ) -> Result<SentencePieceBpe, TrainError> {
        if !(config.character_coverage > 0.0 && config.character_coverage <= 1.0) {
            return Err(TrainError::Coverage(config.character_coverage));
        }
        let sentences = sentences(lines);
        if sentences.is_empty() {
            return Err(TrainError::EmptyCorpus);
        }
        let kept = kept_chars(&sentences, config.character_coverage);
        let fixed = META.len() + kept.len();
        let Some(merges) = config.vocab_size.checked_sub(fixed) else {
            return Err(TrainError::VocabTooSmall { asked: config.vocab_size, least: fixed });
        };
        let kept_map: FxMap<char, u64> = kept.iter().copied().collect();
        let words = words(&sentences, &kept_map);

        let mut chars: Vec<char> = words.iter().flat_map(|(w, _)| w.chars()).collect();
        chars.sort_unstable();
        chars.dedup();
        let mut pieces = Pieces::default();
        for c in chars {
            pieces.intern(c.encode_utf8(&mut [0; 4]));
        }
        let words: Vec<(Vec<SymbolId>, u64)> = words
            .iter()
            .map(|(w, n)| (w.chars().map(|c| pieces.get(c.encode_utf8(&mut [0; 4])).expect("interned")).collect(), *n))
            .collect();

        let mut m = Merger::new(ByPiece, pieces, words);
        let mut made: Vec<String> = Vec::with_capacity(merges);
        let mut seen: FxMap<String, ()> = FxMap::default();
        while made.len() < merges {
            let Some(c) = m.best() else { break };
            let p = m.pieces();
            let piece = format!("{}{}", p.text(c.pair.left), p.text(c.pair.right));
            if seen.contains_key(&piece) {
                m.discard(c.pair);
                continue;
            }
            seen.insert(piece.clone(), ());
            made.push(piece);
            m.merge(c.pair);
        }
        // No pair occurs any more. The library goes on: every pair it
        // has seen stays in its queue at its last count, so the pairs that
        // occurred once and no longer do come out at count 0, in its
        // order, each a piece no word holds.
        for pair in m.spent() {
            if made.len() == merges {
                break;
            }
            let p = m.pieces();
            let piece = format!("{}{}", p.text(pair.left), p.text(pair.right));
            if seen.insert(piece.clone(), ()).is_none() {
                made.push(piece);
            }
        }
        if made.len() < merges {
            return Err(TrainError::VocabTooLarge { asked: config.vocab_size, most: fixed + made.len() });
        }

        let mut out: Vec<TrainedPiece> =
            META.iter().map(|&(p, kind)| TrainedPiece { piece: p.into(), score: 0.0, kind }).collect();
        let scored = made.into_iter().chain(kept.iter().map(|(c, _)| c.to_string()));
        for (rank, piece) in scored.enumerate() {
            out.push(TrainedPiece { piece, score: -(rank as f32), kind: PieceKind::Normal });
        }
        Ok(SentencePieceBpe { config, pieces: out })
    }

    /// Every piece, by id.
    pub fn pieces(&self) -> &[TrainedPiece] {
        &self.pieces
    }

    /// The `.model` file: the pieces, a `TrainerSpec` naming BPE, the
    /// vocabulary size and the coverage, and the identity
    /// `NormalizerSpec`. A normal piece's type is left out, as the
    /// library leaves it out.
    pub fn to_model_proto(&self) -> Vec<u8> {
        let mut m = Message::default();
        for p in &self.pieces {
            let mut piece = Message::default();
            piece.bytes(1, p.piece.as_bytes());
            piece.float(2, p.score);
            if p.kind != PieceKind::Normal {
                piece.int(3, p.kind as i64);
            }
            m.message(1, piece);
        }
        let mut trainer = Message::default();
        // model_type BPE, vocab_size, character_coverage.
        trainer.int(3, 2);
        trainer.int(4, self.pieces.len() as i64);
        trainer.float(10, self.config.character_coverage);
        m.message(2, trainer);
        let mut normalizer = Message::default();
        normalizer.bytes(1, b"identity");
        normalizer.bytes(2, b"");
        normalizer.int(3, 1);
        normalizer.int(4, 1);
        normalizer.int(5, 1);
        m.message(3, normalizer);
        m.into_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spaces_are_dropped_at_the_ends_and_runs_are_one() {
        assert_eq!(normalize("  Hello   world  "), "▁Hello▁world");
        assert_eq!(normalize("a"), "▁a");
        assert_eq!(normalize("   "), "");
        // Only U+0020 is a space to the identity rule.
        assert_eq!(normalize("a\u{3000}b\tc"), "▁a\u{3000}b\tc");
        // A `▁` in the text is kept, but not at the end.
        assert_eq!(normalize("a ▁b▁ ▁"), "▁a▁▁b");
        assert_eq!(normalize("▁"), "");
    }

    /// The library leaves out a line holding `▅`, its mark for an unknown
    /// character, and keeps one holding `⁇`, the surface it decodes the
    /// unknown piece to.
    #[test]
    fn lines_left_out() {
        let long = "a".repeat(MAX_SENTENCE_BYTES + 1);
        let lines = ["a ▅ b", "", "   ", "a ⁇ b", &long, &long[1..]];
        assert_eq!(sentences(lines), ["▁a▁⁇▁b".to_owned(), format!("▁{}", &long[1..])]);
    }

    #[test]
    fn a_meta_piece_in_the_text_is_a_boundary() {
        assert_eq!(mark_meta("▁a<s>b</s><unk>"), "▁a\tb\t\t");
        assert_eq!(mark_meta("▁<s"), "▁<s");
    }

    #[test]
    fn a_piece_keeps_to_one_script_and_a_leading_space() {
        assert!(is_valid_piece("▁ab"));
        assert!(!is_valid_piece("a▁b"));
        assert!(!is_valid_piece("ab▁"));
        assert!(!is_valid_piece("a,"));
        assert!(!is_valid_piece("a1"));
        assert!(is_valid_piece("12"));
        assert!(is_valid_piece("▁,."));
        // Hiragana, Katakana and the long vowel mark go with Han.
        assert!(is_valid_piece("東京タワーです"));
        assert!(!is_valid_piece("aб"));
        // A combining mark takes the script before it.
        assert!(is_valid_piece("e\u{301}t"));
        assert!(!is_valid_piece("1\u{301}a"));
        assert!(!is_valid_piece("a\u{2585}"));
        assert!(is_valid_piece("\u{2047}\u{2047}"));
        assert!(!is_valid_piece(&"a".repeat(17)));
        assert!(is_valid_piece(&"a".repeat(16)));
        assert!(!is_valid_piece(""));
    }

    #[test]
    fn coverage_keeps_the_most_frequent_characters() {
        let s = vec!["▁aaaab".to_owned()];
        assert_eq!(kept_chars(&s, 1.0), [('a', 4), ('b', 1), ('▁', 1)]);
        // a covers 4 of 6: two thirds is not yet 0.7, so b is kept too,
        // and is 0.6 already.
        assert_eq!(kept_chars(&s, 0.7), [('a', 4), ('b', 1)]);
        assert_eq!(kept_chars(&s, 0.6), [('a', 4)]);
    }

    #[test]
    fn the_vocabulary_must_hold_the_meta_pieces_and_the_characters() {
        let c = SentencePieceConfig { vocab_size: 5, character_coverage: 1.0 };
        assert!(matches!(SentencePieceBpe::train(["abc"], c), Err(TrainError::VocabTooSmall { asked: 5, least: 7 })));
        let c = SentencePieceConfig { vocab_size: 100, character_coverage: 1.0 };
        assert!(matches!(SentencePieceBpe::train(["abc"], c), Err(TrainError::VocabTooLarge { asked: 100, most: 13 })));
    }

    /// After the three merges "▁abc" has, the pairs it had on the way
    /// come out at count 0, shortest and then first in code point order:
    /// the pieces the library 0.2.2 gives for this corpus and size.
    #[test]
    fn pairs_that_occur_no_longer_follow_the_merges() {
        let c = SentencePieceConfig { vocab_size: 13, character_coverage: 1.0 };
        let m = SentencePieceBpe::train(["abc"], c).unwrap();
        let pieces: Vec<&str> = m.pieces().iter().map(|p| p.piece.as_str()).collect();
        let want = ["<unk>", "<s>", "</s>", "ab", "abc", "▁abc", "bc", "▁a", "▁ab", "a", "b", "c", "▁"];
        assert_eq!(pieces, want);
        let scores: Vec<f32> = m.pieces().iter().skip(3).map(|p| p.score).collect();
        assert_eq!(scores, (0..10).map(|i| -(i as f32)).collect::<Vec<_>>());
    }
}
