//! SentencePiece's own tokenizer, read from its `.model` file (a
//! `ModelProto`), as Google's sentencepiece library runs it: the model's
//! normalizer with its precompiled character map, then the unigram
//! search or BPE, unknown characters as the unknown piece or, with byte
//! fallback, as their bytes' pieces.
//!
//! [`SentencePiece::encode`] gives each piece's id and its offsets in the
//! input exactly as the library's `SentencePieceText` does, and
//! [`SentencePiece::normalize`] the normalized text with the library's
//! `norm_to_orig`. Through a bundle the same model is a [`Tokenizer`]
//! kind, whose spans follow the rules every kind's spans follow
//! (docs/tokenizer.md).
//!
//! The library is the reference; where this differs from it, the
//! difference is listed in docs/tokenizer.md and tested in
//! core/tests/sentencepiece.rs.
//!
//! [`Tokenizer`]: crate::tokenizer::Tokenizer

mod bpe;
mod normalizer;
mod proto;
mod unigram;

use std::cell::Cell;
use std::collections::BinaryHeap;

use crate::status::{Result, invalid};
use crate::tokenizer::{Sink, trim_span};
use crate::unigram::Trie;

use normalizer::Normalizer;
pub use proto::PieceType;
use proto::{ModelProto, ModelType};

/// No position: a best path not found yet.
const NONE: u32 = u32::MAX;

/// The library's penalty under the lowest score for a character no
/// piece covers (`kUnkPenalty`).
const UNK_PENALTY: f32 = 10.0;

/// The library refuses a piece this long or longer (`kMaxPieceSize`).
const MAX_PIECE_LEN: usize = 8000;

/// The segmentation a model was trained for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    Unigram,
    Bpe,
}

/// One piece of an encoding: its id and the bytes `[begin, end)` of the
/// input it came from, as the library's `SentencePieceText` gives them.
/// A byte piece other than the last of its character is empty at the
/// character's start, and a piece made of characters written for one
/// character of the input, other than the last, is empty there.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    pub id: u32,
    pub begin: u32,
    pub end: u32,
}

/// A normalized text and where each of its bytes came from.
#[derive(Debug, Default, Clone)]
pub struct Normalized {
    /// The normalized text. A model's character map can write bytes that
    /// are not UTF-8, so it is bytes.
    pub text: Vec<u8>,
    /// For each byte of `text` and one past its end, the byte of the
    /// input it came from: the start of the input's characters that were
    /// replaced together. Never decreasing.
    pub norm_to_orig: Vec<u32>,
}

impl Normalized {
    /// The other way: for each byte offset `o` of an input of `len`
    /// bytes, and `len` itself, the first byte of `text` that came from
    /// `o` or after it, `text.len()` when none did. So
    /// `norm_to_orig[orig_to_norm[o]] >= o`, and a byte of the input the
    /// normalizer dropped maps to the next byte it wrote.
    pub fn orig_to_norm(&self, len: usize, out: &mut Vec<u32>) {
        out.clear();
        out.reserve(len + 1);
        // norm_to_orig never decreases, so one pass of both.
        let mut i = 0usize;
        for o in 0..=len as u32 {
            while i < self.text.len() && self.norm_to_orig[i] < o {
                i += 1;
            }
            out.push(i as u32);
        }
    }
}

/// A piece of the result in normalized bytes, before unknown pieces are
/// joined or written as bytes.
#[derive(Clone, Copy)]
pub(crate) struct Segment {
    start: u32,
    end: u32,
    id: u32,
}

/// Room an encoding works in, kept between calls. What it needs depends
/// on the text, not only its length (BPE's queue of pairs, its UNUSED
/// splits), so encoding allocates nothing once the scratch has grown to
/// what the texts it is given need: after one pass over them, a second
/// allocates nothing.
#[derive(Default)]
pub struct Scratch {
    normalized: Normalized,
    best: Vec<unigram::Best>,
    symbols: Vec<bpe::Symbol>,
    agenda: BinaryHeap<bpe::Pair>,
    rev_merge: bpe::RevMerge,
    segments: Vec<Segment>,
}

/// What one emitted piece is, for its offsets.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// The piece's own bytes, or the whole of an unknown run or character.
    Whole,
    /// A byte piece before the last of its character: empty at its start.
    Inner,
    /// A control piece: empty at its start.
    Control,
}

pub struct SentencePiece {
    algorithm: Algorithm,
    /// Each piece, by id.
    pieces: Vec<Box<str>>,
    scores: Vec<f32>,
    kinds: Vec<PieceType>,
    /// The pieces the search and BPE look up: for a unigram model those
    /// of type NORMAL, USER_DEFINED and UNUSED (the library's `pieces_`),
    /// for BPE every piece.
    trie: Trie,
    /// A unigram model's other pieces (the library's `reserved_id_map_`),
    /// which `piece_to_id` finds first.
    reserved: Option<Trie>,
    /// The user-defined pieces, which BPE keeps whole.
    user: Option<Trie>,
    unk_id: u32,
    unk_score: f32,
    byte_fallback: bool,
    /// The id of each byte's piece, with byte fallback.
    byte_ids: Box<[u32; 256]>,
    normalizer: Normalizer,
    denormalizer: Option<Normalizer>,
    /// What a space is in the normalized text: U+2581, or a space when
    /// the model does not escape whitespace.
    space: &'static [u8],
    /// BPE may merge word by word (`bpe`).
    split_words: bool,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    unk_surface: Box<str>,
}

thread_local! {
    /// The tokenizer's scratch, per thread, as its other kinds keep their
    /// buffers.
    static SCRATCH: Cell<Option<Box<Scratch>>> = const { Cell::new(None) };
}

/// The largest normalized text whose scratch a thread keeps.
const KEPT_SCRATCH: usize = 64 << 10;

impl SentencePiece {
    /// The model in a `.model` file's bytes, checked as the library
    /// checks it on load, its self-test samples included.
    pub fn parse(bytes: &[u8]) -> Result<SentencePiece> {
        let m = ModelProto::parse(bytes)?;
        let bad = |why: String| invalid(format!("SentencePiece model: {why}"));
        let algorithm = match m.trainer.model_type {
            ModelType::Unigram => Algorithm::Unigram,
            ModelType::Bpe => Algorithm::Bpe,
            other => return Err(bad(format!("model type {other:?}: only unigram and BPE models are read"))),
        };
        let unigram = algorithm == Algorithm::Unigram;
        let reserved_kind = |k: PieceType| !matches!(k, PieceType::Normal | PieceType::UserDefined | PieceType::Unused);

        let n = m.pieces.len();
        let mut pieces = Vec::with_capacity(n);
        let mut scores = Vec::with_capacity(n);
        let mut kinds = Vec::with_capacity(n);
        let mut main: Vec<(&[u8], u32)> = Vec::with_capacity(n);
        let mut reserved: Vec<(&[u8], u32)> = Vec::new();
        let mut user: Vec<&[u8]> = Vec::new();
        let mut unk_id = None;
        let mut byte_found = [false; 256];
        for (i, p) in m.pieces.iter().enumerate() {
            if p.piece.len() >= MAX_PIECE_LEN {
                return Err(bad(format!("piece {i} is {} bytes long", p.piece.len())));
            }
            let text = std::str::from_utf8(&p.piece)
                .map_err(|_| bad(format!("piece {i} is not UTF-8, which the core's pieces must be")))?;
            pieces.push(Box::<str>::from(text));
            scores.push(p.score);
            kinds.push(p.kind);
        }
        for (i, p) in m.pieces.iter().enumerate() {
            if p.piece.is_empty() {
                return Err(bad(format!("piece {i} is empty")));
            }
            if p.piece.contains(&0) {
                return Err(bad(format!("piece {i} holds a NUL")));
            }
            if unigram && reserved_kind(p.kind) {
                reserved.push((&p.piece, i as u32));
            } else {
                main.push((&p.piece, i as u32));
            }
            match p.kind {
                PieceType::UserDefined => user.push(&p.piece),
                PieceType::Unknown if unk_id.is_some() => return Err(bad("two pieces are of type UNKNOWN".into())),
                PieceType::Unknown => unk_id = Some(i as u32),
                PieceType::Byte => {
                    if !m.trainer.byte_fallback {
                        return Err(bad(format!("piece {i} is a byte piece, and byte_fallback is off")));
                    }
                    match byte_of(&p.piece) {
                        Some(b) => byte_found[b as usize] = true,
                        None => return Err(bad(format!("piece {i}, {:?}, is not a byte piece", pieces[i]))),
                    }
                }
                _ => {}
            }
        }
        let Some(unk_id) = unk_id else { return Err(bad("no piece is of type UNKNOWN".into())) };
        if m.trainer.byte_fallback && byte_found.contains(&false) {
            return Err(bad("byte_fallback is on, and the 256 byte pieces are not all there".into()));
        }
        let twice = |p: String| bad(format!("{p:?} is defined twice"));
        if main.is_empty() {
            return Err(bad("no pieces are loaded".into()));
        }
        let trie = Trie::of(main).map_err(twice)?;
        let reserved = if unigram { Some(Trie::of(reserved).map_err(twice)?) } else { None };

        let mut min_score = f32::MAX;
        if unigram {
            for (i, (&s, &k)) in scores.iter().zip(&kinds).enumerate() {
                if !s.is_finite() {
                    return Err(bad(format!("piece {i}'s score is NaN or infinite")));
                }
                if k == PieceType::Normal {
                    min_score = min_score.min(s);
                }
            }
        }
        let normalizer = Normalizer::new(&m.normalizer, m.trainer.treat_whitespace_as_suffix, &user)?;
        let denormalizer = match &m.denormalizer {
            Some(d) if !d.precompiled_charsmap.is_empty() => Some(Normalizer::new(d, false, &[])?),
            _ => None,
        };
        let user =
            if user.is_empty() { None } else { Some(Trie::of(user.iter().map(|&p| (p, 0)).collect()).map_err(twice)?) };
        let space: &'static [u8] = if m.normalizer.escape_whitespaces { normalizer::SPACE_SYMBOL } else { b" " };
        // Each piece BPE may merge into holds spaces only in the run it
        // starts with, and none is UNUSED (whose splits the library keeps
        // by text across the whole input).
        let split_words = !unigram
            && !m.trainer.treat_whitespace_as_suffix
            && m.pieces.iter().all(|p| match p.kind {
                PieceType::Normal | PieceType::UserDefined => {
                    let mut rest: &[u8] = &p.piece;
                    while let Some(r) = rest.strip_prefix(space) {
                        rest = r;
                    }
                    !rest.windows(space.len()).any(|w| w == space)
                }
                PieceType::Unused => false,
                _ => true,
            });
        let mut sp = SentencePiece {
            algorithm,
            pieces,
            scores,
            kinds,
            trie,
            reserved,
            user,
            unk_id,
            unk_score: min_score - UNK_PENALTY,
            byte_fallback: m.trainer.byte_fallback,
            byte_ids: Box::new([unk_id; 256]),
            normalizer,
            denormalizer,
            space,
            split_words,
            add_dummy_prefix: m.normalizer.add_dummy_prefix,
            remove_extra_whitespaces: m.normalizer.remove_extra_whitespaces,
            unk_surface: String::from_utf8_lossy(&m.trainer.unk_surface).into(),
        };
        if sp.byte_fallback {
            for b in 0..=255u8 {
                sp.byte_ids[b as usize] = sp.piece_to_id(byte_piece(b).as_bytes());
            }
        }
        sp.self_test(&m)?;
        Ok(sp)
    }

    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// How many pieces the model has: one past the highest id.
    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    /// Each piece, by id.
    pub fn pieces(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().map(|p| &**p)
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.pieces.get(id as usize).map(|p| &**p)
    }

    pub fn kind(&self, id: u32) -> Option<PieceType> {
        self.kinds.get(id as usize).copied()
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// The id of the piece that is exactly `piece`, if one is.
    pub fn id(&self, piece: &str) -> Option<u32> {
        let p = piece.as_bytes();
        self.reserved.as_ref().and_then(|r| r.get(p)).or_else(|| self.trie.get(p))
    }

    /// The library's `PieceToId`: the unknown piece's id for a text that
    /// is no piece.
    fn piece_to_id(&self, piece: &[u8]) -> u32 {
        self.reserved.as_ref().and_then(|r| r.get(piece)).or_else(|| self.trie.get(piece)).unwrap_or(self.unk_id)
    }

    /// `text` as the model's normalizer writes it, with where each byte
    /// came from.
    pub fn normalize(&self, text: &str, out: &mut Normalized) {
        self.normalizer.normalize(text, &mut out.text, Some(&mut out.norm_to_orig));
        if out.norm_to_orig.is_empty() {
            // The library's norm_to_orig of an empty result.
            out.norm_to_orig.push(0);
        }
    }

    /// The pieces of `text` with their offsets, as the library's
    /// `Encode` into a `SentencePieceText` gives them, appended to `out`.
    pub fn encode(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<Piece>) {
        self.segment(text, scratch, true);
        let n2o = &scratch.normalized.norm_to_orig;
        self.emit(&scratch.normalized.text, &scratch.segments, |id, start, end, part| {
            let begin = n2o[start as usize];
            let end = if part == Part::Whole { n2o[end as usize] } else { begin };
            out.push(Piece { id, begin, end });
        });
    }

    /// The ids of `text`, as the library's `Encode` into ids gives them,
    /// appended to `out`. The normalizer records no offsets.
    pub fn encode_ids(&self, text: &str, scratch: &mut Scratch, out: &mut Vec<u32>) {
        self.segment(text, scratch, false);
        self.emit(&scratch.normalized.text, &scratch.segments, |id, _, _, _| out.push(id));
    }

    /// The ids of `piece`, the bytes of `text` from `base` on, into a
    /// tokenizer's row, each with its span by the rules of every kind's
    /// spans (docs/tokenizer.md) rather than the library's offsets:
    ///
    /// - A piece runs from the input byte its first byte came from to
    ///   where the input it came from ends, so a piece written for part
    ///   of one input character (a byte piece, or "f" of the "fi" a
    ///   ligature becomes) spans that whole character where the library
    ///   gives an empty offset.
    /// - Whitespace at either end is left out unless it is all there is,
    ///   where the library's offsets of "\u{2581}word" include the space.
    /// - A control piece is empty where it stands.
    pub(crate) fn encode_into<S: Sink>(&self, text: &str, base: usize, piece: &str, out: &mut S) {
        let mut scratch = SCRATCH.take().unwrap_or_default();
        self.segment(piece, &mut scratch, S::SPANS);
        let n2o = &scratch.normalized.norm_to_orig;
        let at = |o: u32| (base as u32) + o;
        self.emit(&scratch.normalized.text, &scratch.segments, |id, start, end, part| {
            out.push(id as i32, || {
                let s = n2o[start as usize];
                if part == Part::Control {
                    return [at(s), at(s)];
                }
                let mut e = n2o[end as usize];
                if e == s {
                    e = n2o[end as usize + 1..].iter().copied().find(|&o| o > s).unwrap_or(s);
                }
                trim_span(text, [at(s), at(e)])
            });
        });
        if scratch.normalized.text.capacity() <= KEPT_SCRATCH {
            SCRATCH.set(Some(scratch));
        }
    }

    /// The text `ids` stand for, as the library's `Decode` writes it:
    /// control pieces left out, the unknown piece as the model's unknown
    /// surface, runs of byte pieces read as UTF-8 (a byte that is not
    /// being U+FFFD), each U+2581 a space, and the one the first piece
    /// starts with dropped when the model puts one in front.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out: Vec<u8> = Vec::new();
        let mut bytes: Vec<u8> = Vec::new();
        let flush = |out: &mut Vec<u8>, bytes: &mut Vec<u8>| {
            for chunk in bytes.utf8_chunks() {
                out.extend_from_slice(chunk.valid().as_bytes());
                for _ in chunk.invalid() {
                    out.extend_from_slice("\u{fffd}".as_bytes());
                }
            }
            bytes.clear();
        };
        let mut is_bos_ws = true;
        for (i, &id) in ids.iter().enumerate() {
            let Some(piece) = self.piece(id) else {
                return Err(crate::status::Error::new(
                    crate::status::INVALID_ARGUMENT,
                    format!("ids[{i}]: {id} is not an id under {}", self.len()),
                ));
            };
            let kind = self.kinds[id as usize];
            if kind == PieceType::Byte {
                bytes.push(byte_of(piece.as_bytes()).expect("checked on load"));
                continue;
            }
            flush(&mut out, &mut bytes);
            if !out.is_empty() {
                is_bos_ws = false;
            }
            if kind == PieceType::Control {
                continue;
            }
            let mut p = piece;
            let mut has_bos_ws = false;
            if is_bos_ws && (self.add_dummy_prefix || self.remove_extra_whitespaces) {
                if let Some(rest) = p.strip_prefix('\u{2581}') {
                    p = rest;
                    has_bos_ws = true;
                }
                if self.remove_extra_whitespaces {
                    has_bos_ws = false;
                }
            }
            if kind == PieceType::Unknown {
                out.extend_from_slice(self.unk_surface.as_bytes());
            } else {
                for c in p.chars() {
                    let mut buf = [0u8; 4];
                    out.extend_from_slice(if c == '\u{2581}' { b" " } else { c.encode_utf8(&mut buf).as_bytes() });
                }
            }
            if has_bos_ws || !out.is_empty() {
                is_bos_ws = false;
            }
        }
        flush(&mut out, &mut bytes);
        if let Some(d) = &self.denormalizer {
            let text = String::from_utf8_lossy(&out).into_owned();
            d.normalize(&text, &mut out, None);
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// The normalized text into scratch.normalized (with its offsets when
    /// `offsets`), and its pieces into scratch.segments.
    fn segment(&self, text: &str, scratch: &mut Scratch, offsets: bool) {
        let Scratch { normalized, best, symbols, agenda, rev_merge, segments } = scratch;
        let n2o = &mut normalized.norm_to_orig;
        self.normalizer.normalize(text, &mut normalized.text, if offsets { Some(n2o) } else { None });
        if offsets && normalized.norm_to_orig.is_empty() {
            normalized.norm_to_orig.push(0);
        }
        match self.algorithm {
            Algorithm::Unigram => self.unigram(&normalized.text, best, segments),
            Algorithm::Bpe => self.bpe(&normalized.text, symbols, agenda, rev_merge, segments),
        }
    }

    /// The pieces of the result, as the library writes them out of the
    /// model's: a run of unknown pieces joined into one, or with byte
    /// fallback each unknown piece written as its bytes' pieces.
    /// `f(id, start, end, part)` for each, start and end in normalized
    /// bytes: for a byte piece those of its character.
    fn emit(&self, normalized: &[u8], segments: &[Segment], mut f: impl FnMut(u32, u32, u32, Part)) {
        // An open run of unknown pieces: where it starts and ends.
        let mut unknown: Option<(u32, u32)> = None;
        for s in segments {
            let kind = self.kinds[s.id as usize];
            if kind == PieceType::Unknown && !self.byte_fallback {
                unknown = Some(match unknown {
                    Some((start, _)) => (start, s.end),
                    None => (s.start, s.end),
                });
                continue;
            }
            if let Some((start, end)) = unknown.take() {
                f(self.unk_id, start, end, Part::Whole);
            }
            match kind {
                PieceType::Control => f(s.id, s.start, s.end, Part::Control),
                PieceType::Unknown => {
                    // Byte fallback: the character's bytes, the last of
                    // them holding its offsets.
                    let bytes = &normalized[s.start as usize..s.end as usize];
                    for (k, &b) in bytes.iter().enumerate() {
                        let part = if k + 1 == bytes.len() { Part::Whole } else { Part::Inner };
                        f(self.byte_ids[b as usize], s.start, s.end, part);
                    }
                }
                _ => f(s.id, s.start, s.end, Part::Whole),
            }
        }
        if let Some((start, end)) = unknown {
            f(self.unk_id, start, end, Part::Whole);
        }
    }

    /// The library's load-time self-test: each sample's pieces, written
    /// with a space between them, are what the model file expects.
    fn self_test(&self, m: &ModelProto) -> Result<()> {
        if m.self_test.is_empty() {
            return Ok(());
        }
        let mut scratch = Scratch::default();
        let mut failed = 0;
        for s in &m.self_test {
            let Ok(input) = std::str::from_utf8(&s.input) else {
                failed += 1;
                continue;
            };
            self.segment(input, &mut scratch, false);
            let mut got: Vec<u8> = Vec::new();
            let text = &scratch.normalized.text;
            self.emit(&scratch.normalized.text, &scratch.segments, |id, start, end, part| {
                if !got.is_empty() {
                    got.push(b' ');
                }
                match (self.kinds[id as usize], part) {
                    (PieceType::Byte, _) => got.extend_from_slice(self.pieces[id as usize].as_bytes()),
                    _ => got.extend_from_slice(&text[start as usize..end as usize]),
                }
            });
            let same = match self.algorithm {
                Algorithm::Unigram => self.unigram_equivalent(&s.expected, &got),
                Algorithm::Bpe => s.expected == got,
            };
            if !same {
                failed += 1;
            }
        }
        if failed > 0 {
            return Err(invalid(format!(
                "SentencePiece model: {failed} of its {} self-test samples do not come out as it expects",
                m.self_test.len()
            )));
        }
        Ok(())
    }
}

/// The piece of byte `b` under byte fallback: `<0xAB>`.
fn byte_piece(b: u8) -> String {
    format!("<0x{b:02X}>")
}

/// The byte a byte piece stands for: `<0x00>` to `<0xFF>`, upper-case
/// hexadecimal, as the library writes them.
fn byte_of(piece: &[u8]) -> Option<u8> {
    let hex = piece.strip_prefix(b"<0x")?.strip_suffix(b">")?;
    let digit = |c: u8| match c {
        b'0'..=b'9' => Some(c - b'0'),
        b'A'..=b'F' => Some(c - b'A' + 10),
        _ => None,
    };
    match hex {
        [h, l] => Some(digit(*h)? << 4 | digit(*l)?),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(name: &str) -> SentencePiece {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece").join(name);
        SentencePiece::parse(&std::fs::read(path).unwrap()).unwrap()
    }

    /// BPE word by word gives what BPE over the whole text gives, on the
    /// model that allows it, for every golden text and their join.
    #[test]
    fn bpe_word_by_word_is_bpe_over_the_whole_text() {
        let words = model("bpe-bytes.model");
        assert!(words.split_words);
        let mut whole = model("bpe-bytes.model");
        whole.split_words = false;
        // A model with UNUSED pieces merges the whole text at once.
        assert!(!model("bpe-identity.model").split_words);
        let file =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece/bpe-bytes.goldens.jsonl");
        let file = std::fs::read_to_string(file).unwrap();
        let mut texts: Vec<String> = file
            .lines()
            .skip(1)
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap()["text"].as_str().map(str::to_owned))
            .collect();
        texts.push(texts.join("\n"));
        let mut scratch = Scratch::default();
        let (mut a, mut b) = (Vec::new(), Vec::new());
        for t in &texts {
            a.clear();
            b.clear();
            words.encode(t, &mut scratch, &mut a);
            whole.encode(t, &mut scratch, &mut b);
            assert_eq!(a, b, "{t:?}");
        }
    }
}
