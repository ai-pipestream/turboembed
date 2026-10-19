//! Loading a model: the `.model` file's pieces checked as the library
//! checks them, and its self-test samples run.

use super::normalizer::{self, Normalizer};
use super::proto::{ModelProto, ModelType, PieceType};
use super::{Algorithm, MAX_PIECE_LEN, Scratch, SentencePiece, UNK_PENALTY, byte_of, byte_piece};
use crate::status::{Result, invalid};
use crate::unigram::Trie;

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
