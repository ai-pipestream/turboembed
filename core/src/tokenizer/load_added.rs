//! The manifest's special and added tokens against the tokenizer file's
//! `added_tokens`, and the matchers made from them.

use super::*;
use crate::manifest;

/// What the tokenizer keeps of its special and added tokens.
pub(super) struct Matched {
    pub(super) raw: Added,
    pub(super) normalized: Added,
    /// The vocabulary's entries and the added tokens past them.
    pub(super) vocab_size: u32,
    /// The ids a decode that skips special tokens leaves out.
    pub(super) special_ids: Vec<i32>,
    /// Each token's piece for decode, as upstream writes it back: its
    /// content, normalized when it is matched normalized.
    pub(super) pieces: Vec<(u32, String)>,
}

/// A special or added token of the manifest.
struct Token<'a> {
    content: &'a str,
    id: u32,
    special: bool,
    normalized: bool,
    lstrip: bool,
    rstrip: bool,
    single_word: bool,
}

pub(super) fn matched(file: &str, json: &Value, t: &manifest::Tokenizer, kind: &Kind) -> Result<Matched> {
    let specials = t.special_tokens.iter().map(|s| Token {
        content: &s.content,
        id: s.id,
        special: true,
        normalized: s.normalized,
        lstrip: s.lstrip,
        rstrip: s.rstrip,
        single_word: s.single_word,
    });
    let added = t.added_tokens.iter().map(|a| Token {
        content: &a.content,
        id: a.id,
        special: a.special,
        normalized: a.normalized,
        lstrip: a.lstrip,
        rstrip: a.rstrip,
        single_word: a.single_word,
    });
    let tokens: Vec<Token> = specials.chain(added).collect();
    let special_ids = tokens.iter().filter(|s| s.special).map(|s| s.id as i32).collect();
    let model_size = kind.vocab_size();

    // The library matches no special token in the text: a control piece
    // written there is its characters. Each is a piece of the model.
    if matches!(kind, Kind::SentencePiece(_)) {
        if let Some(s) = tokens.iter().find(|s| kind.id(s.content) != Some(s.id)) {
            return Err(invalid(format!("{file}: model.vocab has no {:?} with id {}", s.content, s.id)));
        }
        let none = || Added::new(Vec::new()).expect("no token");
        return Ok(Matched {
            raw: none(),
            normalized: none(),
            vocab_size: model_size,
            special_ids,
            pieces: Vec::new(),
        });
    }

    // The file's added tokens are exactly the manifest's, field by field,
    // each under the id upstream gives it: its entry in the vocabulary,
    // or else the next id past the vocabulary, in the file's order.
    let added = json["added_tokens"].as_array().map(Vec::as_slice).unwrap_or_default();
    let mut seen = std::collections::HashSet::with_capacity(added.len());
    let mut next = model_size;
    for (i, a) in added.iter().enumerate() {
        let content = a["content"].as_str().unwrap_or_default();
        if !seen.insert(content) {
            return Err(invalid(format!("{file}: added_tokens[{i}] duplicates {content:?}")));
        }
        let Some(s) = tokens.iter().find(|s| s.content == content) else {
            return Err(invalid(format!(
                "{file}: added_tokens[{i}] {content:?} is not a special or added token in manifest.json"
            )));
        };
        if a["special"] != s.special
            || a["normalized"] != s.normalized
            || a["lstrip"] != s.lstrip
            || a["rstrip"] != s.rstrip
            || a["single_word"] != s.single_word
        {
            return Err(invalid(format!(
                "{file}: added_tokens[{i}] {content:?} is matched differently from what manifest.json says"
            )));
        }
        let id = kind.id(content).unwrap_or_else(|| {
            next += 1;
            next - 1
        });
        if a["id"] != id || s.id != id {
            return Err(invalid(format!(
                "{file}: added_tokens[{i}] {content:?} has id {} in the file and {} in manifest.json; upstream gives it {id}",
                a["id"], s.id
            )));
        }
    }
    if added.len() != tokens.len() {
        return Err(invalid(format!(
            "{file}: {} added_tokens, manifest.json has {} special and added tokens",
            added.len(),
            tokens.len()
        )));
    }
    let vocab_size = tokens.iter().map(|s| s.id + 1).fold(model_size, u32::max);

    // A normalized token is matched with its content normalized as the
    // text is: a BPE without NFC has no normalizer, so it is its content.
    let normalize = |s: &str| match kind {
        Kind::WordPiece { normalizer, ascii, .. } => normalized(normalizer, ascii, s),
        Kind::Bpe(b) => b.normalize(s),
        Kind::Unigram(u) => u.normalize(s),
        Kind::SentencePiece(_) => unreachable!("matches nothing"),
    };
    let (mut raw, mut norm) = (Vec::new(), Vec::new());
    let mut pieces = Vec::with_capacity(tokens.len());
    for s in &tokens {
        let content = if s.normalized { normalize(s.content) } else { s.content.to_owned() };
        pieces.push((s.id, content.clone()));
        // A token the normalizer empties matches nothing.
        if content.is_empty() {
            continue;
        }
        let m = Special { content, id: s.id as i32, lstrip: s.lstrip, rstrip: s.rstrip, single_word: s.single_word };
        if s.normalized { norm.push(m) } else { raw.push(m) }
    }
    let twice = |e: String| invalid(format!("{file}: added_tokens: {e} after normalization"));
    Ok(Matched {
        raw: Added::new(raw).map_err(twice)?,
        normalized: Added::new(norm).map_err(twice)?,
        vocab_size,
        special_ids,
        pieces,
    })
}
