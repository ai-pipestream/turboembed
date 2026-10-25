//! Learned terms added to a static model's vocabulary, as Model2Vec's
//! `distill` takes a vocabulary of its own: each term becomes one more
//! entry of the WordPiece vocabulary, after the base's ids, and its row is
//! the base encoder's mean-pooled vector of the term as text, reduced and
//! weighted with the base entries.
//!
//! A term is added only when it is one word to the base tokenizer: its
//! WordPiece pieces are a word-initial piece followed only by continuing
//! pieces, with no unknown token. Its entry is the pieces joined without
//! the continuing prefix, so it is already in the tokenizer's normalized
//! form (lowercased, accents stripped, as the base's normalizer says). A
//! term of one piece is already an entry. WordPiece matches the longest
//! entry from the start of a word, so a new entry also changes how longer
//! words that begin with it are split, as it does in Model2Vec.

use std::collections::HashSet;

use serde_json::Value;
use turbo::manifest::{Manifest, PromptRole, SpecialRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo::{TURBO_NORMALIZE_NONE, TURBO_POOLING_MEAN};

use crate::Result;
use crate::api;

/// The most pieces a term may have.
const MAX_PIECES: usize = 32;

/// Why a learned term was left out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Nothing is left of it after normalization.
    Empty,
    /// It holds a character the base tokenizer gives the unknown token.
    Unknown,
    /// It is already one entry of the vocabulary.
    Present,
    /// It is more than one word to the base tokenizer.
    Words,
    /// It splits into more than `MAX_PIECES` pieces.
    Long,
    /// An earlier term has the same entry.
    Duplicate,
}

/// The terms to add, in the order they get ids, with each one's base
/// pieces, and the terms left out.
#[derive(Debug, Default)]
pub struct Added {
    pub entries: Vec<String>,
    pub pieces: Vec<Vec<i32>>,
    pub refused: Vec<(String, Refusal)>,
}

/// Sorts `terms` into those added and those refused.
pub(super) fn plan(tok: &Tokenizer, bm: &Manifest, terms: &[String]) -> Result<Added> {
    let mut out = Added::default();
    if terms.is_empty() {
        return Ok(out);
    }
    let prefix =
        &bm.tokenizer.wordpiece.as_ref().ok_or("learned terms need a WordPiece base tokenizer")?.continuing_prefix;
    let e = Encode {
        add_special_tokens: false,
        truncation: Truncation::Right,
        max_tokens: MAX_PIECES as u32 + 1,
        prompt: PromptRole::None,
    };
    let mut seen = HashSet::new();
    for t in terms {
        let ids = tok.encode(t, e).map_err(|e| e.message)?;
        let pieces: Vec<&str> = ids.iter().map(|&id| tok.piece(id).unwrap_or("")).collect();
        let refusal = if ids.is_empty() {
            Some(Refusal::Empty)
        } else if ids.contains(&tok.unk_id) {
            Some(Refusal::Unknown)
        } else if ids.len() == 1 {
            Some(Refusal::Present)
        } else if ids.len() > MAX_PIECES {
            Some(Refusal::Long)
        } else if pieces[0].starts_with(prefix.as_str()) || !pieces[1..].iter().all(|p| p.starts_with(prefix.as_str()))
        {
            Some(Refusal::Words)
        } else {
            None
        };
        if let Some(r) = refusal {
            out.refused.push((t.clone(), r));
            continue;
        }
        let entry: String = std::iter::once(pieces[0]).chain(pieces[1..].iter().map(|p| &p[prefix.len()..])).collect();
        if !seen.insert(entry.clone()) {
            out.refused.push((t.clone(), Refusal::Duplicate));
            continue;
        }
        out.entries.push(entry);
        out.pieces.push(ids);
    }
    Ok(out)
}

/// The base's tokenizer file with `entries` added to its WordPiece
/// vocabulary as ids `vocab_size..`.
pub(super) fn extend(tokenizer_json: &[u8], vocab_size: usize, entries: &[String]) -> Result<Vec<u8>> {
    let mut v: Value = serde_json::from_slice(tokenizer_json).map_err(|e| format!("tokenizer file: {e}"))?;
    let vocab = v["model"]["vocab"].as_object_mut().ok_or("tokenizer file: model.vocab is not a map")?;
    if vocab.values().any(|id| id.as_u64().is_none_or(|id| id >= vocab_size as u64)) {
        return Err(format!("tokenizer file: an id is not under the base's vocab_size {vocab_size}"));
    }
    for (i, e) in entries.iter().enumerate() {
        if vocab.insert(e.clone(), Value::from(vocab_size + i)).is_some() {
            return Err(format!("tokenizer file: {e:?} is already an entry"));
        }
    }
    Ok(serde_json::to_vec(&v).unwrap())
}

/// Each term's row as the base encoder gives it: its pieces between the
/// template's special tokens, mean-pooled, not normalized. [terms, h].
pub(super) fn embed(
    model: &api::Model,
    info: &turbo::turbo_model_info,
    bm: &Manifest,
    pieces: &[Vec<i32>],
    h: usize,
    mut done: impl FnMut(usize),
) -> Result<Vec<f64>> {
    if pieces.is_empty() {
        return Ok(Vec::new());
    }
    let special = |s: &str| bm.tokenizer.special_tokens.iter().find(|t| t.content == s).map(|t| t.id as i32);
    let template: Vec<Option<i32>> = bm
        .tokenizer
        .template
        .iter()
        .map(|s| if s == "$TEXT" { Ok(None) } else { special(s).map(Some).ok_or(format!("template {s:?}")) })
        .collect::<Result<_>>()?;
    let pad = bm.tokenizer.special_tokens.iter().find(|t| t.role == SpecialRole::Pad).map_or(0, |t| t.id as i32);
    let longest = pieces.iter().map(Vec::len).max().unwrap_or(0);
    let seq = template.len() - 1 + longest;
    let batch = info.max_batch as usize;
    let session = model.session(info.max_batch, seq as u32)?;
    let o = api::options(TURBO_POOLING_MEAN, TURBO_NORMALIZE_NONE);
    let mut out = Vec::with_capacity(pieces.len() * h);
    let mut vecs = vec![0f32; batch * h];
    for chunk in pieces.chunks(batch) {
        let (mut ids, mut mask) = (Vec::with_capacity(chunk.len() * seq), Vec::with_capacity(chunk.len() * seq));
        for p in chunk {
            let row: Vec<i32> = template.iter().flat_map(|t| t.map_or_else(|| p.clone(), |id| vec![id])).collect();
            mask.extend(std::iter::repeat_n(1, row.len()).chain(std::iter::repeat_n(0, seq - row.len())));
            ids.extend(row.iter().copied().chain(std::iter::repeat_n(pad, seq - row.len())));
        }
        let n = chunk.len();
        session.tokens(n as u32, seq as u32, &ids, &mask, &o, &mut vecs[..n * h])?;
        out.extend(vecs[..n * h].iter().map(|&v| v as f64));
        done(out.len() / h);
    }
    Ok(out)
}
