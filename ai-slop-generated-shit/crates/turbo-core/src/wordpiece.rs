//! A native WordPiece tokenizer for `tokenizer.json` files of the BERT
//! family: the `BertNormalizer`, the `BertPreTokenizer`, the `WordPiece`
//! model, a `TemplateProcessing` or `BertProcessing` post-processor and the
//! `WordPiece` decoder, with no dependency beyond `serde_json` and the
//! generated Unicode tables in `unicode_data.rs`.
//!
//! It reproduces the Hugging Face `tokenizers` crate's ids and byte offsets
//! for these files (`bpe.rs` does the same for byte-level BPE files; the
//! `hf-tokenizers` feature keeps that crate available for Unigram and
//! other files and for the parity tests). Anything the file
//! declares that this implementation does not do is an error naming it,
//! never a silent approximation.

use std::collections::HashMap;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::unicode_data::{COMBINING_CLASS, CONTROL, MARK_NONSPACING, NFD, PUNCTUATION, SPACE_SEPARATOR};

/// One token of an encoding: its id and byte offsets into the input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    /// Vocabulary id.
    pub id: i32,
    /// First byte of the source text this token covers.
    pub start: u32,
    /// One past the last byte this token covers.
    pub end: u32,
}

/// One piece of the single-sequence template.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Piece {
    Special { id: i32, type_id: i32 },
    Sequence { type_id: i32 },
}

/// The single-sequence template of a `TemplateProcessing` post-processor
/// node: its `single` pieces, with each special token's id from the
/// node's `special_tokens` or the vocabulary.
pub(crate) fn template_from_json(p: &Value, vocab: &HashMap<Box<str>, i32>, what: &str) -> Result<Vec<Piece>> {
    let single = p
        .get("single")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::bundle_invalid(format!("{what}: TemplateProcessing has no `single`")))?;
    let specials = p.get("special_tokens").and_then(Value::as_object);
    let mut pieces = Vec::new();
    for item in single {
        if let Some(s) = item.get("SpecialToken") {
            let name = s.get("id").and_then(Value::as_str).unwrap_or("");
            let type_id = s.get("type_id").and_then(Value::as_i64).unwrap_or(0) as i32;
            let id = specials
                .and_then(|m| m.get(name))
                .and_then(|e| e.get("ids"))
                .and_then(Value::as_array)
                .and_then(|ids| ids.first())
                .and_then(Value::as_i64)
                .map(|i| i as i32)
                .or_else(|| vocab.get(name).copied())
                .ok_or_else(|| Error::bundle_invalid(format!("{what}: template special token `{name}` has no id")))?;
            pieces.push(Piece::Special { id, type_id });
        } else if let Some(s) = item.get("Sequence") {
            let which = s.get("id").and_then(Value::as_str).unwrap_or("A");
            if which != "A" {
                return Err(Error::unsupported(format!("{what}: single-sequence template names sequence `{which}`")));
            }
            pieces.push(Piece::Sequence { type_id: s.get("type_id").and_then(Value::as_i64).unwrap_or(0) as i32 });
        } else {
            return Err(Error::bundle_invalid(format!(
                "{what}: template piece {item} is neither SpecialToken nor Sequence"
            )));
        }
    }
    Ok(pieces)
}

/// Wrap content tokens in a template: `(ids, type_ids, offsets)`; the
/// template's specials have offsets `(0, 0)`.
pub(crate) fn apply_template(
    template: &[Piece],
    content: &[Token],
    add_special_tokens: bool,
) -> (Vec<i32>, Vec<i32>, Vec<(u32, u32)>) {
    let mut ids = Vec::with_capacity(content.len() + 2);
    let mut type_ids = Vec::with_capacity(content.len() + 2);
    let mut offsets = Vec::with_capacity(content.len() + 2);
    for piece in template {
        match piece {
            Piece::Special { id, type_id } => {
                if add_special_tokens {
                    ids.push(*id);
                    type_ids.push(*type_id);
                    offsets.push((0, 0));
                }
            }
            Piece::Sequence { type_id } => {
                for t in content {
                    ids.push(t.id);
                    type_ids.push(*type_id);
                    offsets.push((t.start, t.end));
                }
            }
        }
    }
    (ids, type_ids, offsets)
}

/// A loaded WordPiece tokenizer.
#[derive(Debug)]
pub struct WordPiece {
    vocab: HashMap<Box<str>, i32>,
    tokens: Vec<Option<Box<str>>>,
    unk_id: i32,
    prefix: Box<str>,
    max_chars: usize,
    lowercase: bool,
    strip_accents: bool,
    clean_text: bool,
    handle_chinese_chars: bool,
    /// Added tokens matched verbatim before normalization, longest first.
    added: Vec<(Box<str>, i32, bool)>,
    special_ids: Vec<i32>,
    template: Vec<Piece>,
    decoder_cleanup: bool,
}

fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    let i = ranges.partition_point(|&(first, _)| first <= cp);
    i > 0 && cp <= ranges[i - 1].1
}

fn combining_class(cp: u32) -> u8 {
    match COMBINING_CLASS.binary_search_by_key(&cp, |&(c, _)| c) {
        Ok(i) => COMBINING_CLASS[i].1,
        Err(_) => 0,
    }
}

/// The full NFD sequence of `cp`, or `None` when it decomposes to itself.
fn nfd(cp: u32) -> Option<&'static [u32]> {
    NFD.binary_search_by_key(&cp, |&(c, _)| c).ok().map(|i| NFD[i].1)
}

fn is_whitespace(cp: u32) -> bool {
    matches!(cp, 0x20 | 0x09 | 0x0A | 0x0D) || in_ranges(cp, SPACE_SEPARATOR)
}

fn is_control(cp: u32) -> bool {
    !matches!(cp, 0x09 | 0x0A | 0x0D) && in_ranges(cp, CONTROL)
}

fn is_punctuation(cp: u32) -> bool {
    matches!(cp, 33..=47 | 58..=64 | 91..=96 | 123..=126) || in_ranges(cp, PUNCTUATION)
}

fn is_chinese_char(cp: u32) -> bool {
    matches!(
        cp,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B820..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

/// A normalized character with the byte range of the source character it
/// came from.
#[derive(Clone, Copy)]
struct NChar {
    ch: char,
    start: u32,
    end: u32,
    ccc: u8,
}

impl WordPiece {
    /// Load a `tokenizer.json`.
    pub fn from_json(text: &str, what: &str) -> Result<Self> {
        let j: Value = serde_json::from_str(text).map_err(|e| Error::bundle_invalid(format!("{what}: {e}")))?;
        let model = j.get("model").ok_or_else(|| Error::bundle_invalid(format!("{what}: no `model`")))?;
        let model_type = model.get("type").and_then(Value::as_str).unwrap_or("");
        if model_type != "WordPiece" {
            return Err(Error::unsupported(format!(
                "{what}: model type `{model_type}` is not served by the native WordPiece tokenizer"
            )));
        }
        let vocab_json = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| Error::bundle_invalid(format!("{what}: model.vocab is not an object")))?;
        let mut vocab: HashMap<Box<str>, i32> = HashMap::with_capacity(vocab_json.len());
        let mut max_id = -1i64;
        for (tok, id) in vocab_json {
            let id = id
                .as_i64()
                .ok_or_else(|| Error::bundle_invalid(format!("{what}: vocab entry `{tok}` has no integer id")))?;
            if id < 0 || id > i32::MAX as i64 {
                return Err(Error::bundle_invalid(format!("{what}: vocab id {id} of `{tok}` is outside i32")));
            }
            max_id = max_id.max(id);
            vocab.insert(tok.as_str().into(), id as i32);
        }
        let mut added: Vec<(Box<str>, i32, bool)> = Vec::new();
        let mut special_ids: Vec<i32> = Vec::new();
        if let Some(list) = j.get("added_tokens").and_then(Value::as_array) {
            for a in list {
                let content = a
                    .get("content")
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::bundle_invalid(format!("{what}: an added token has no content")))?;
                let id = a
                    .get("id")
                    .and_then(Value::as_i64)
                    .ok_or_else(|| Error::bundle_invalid(format!("{what}: added token `{content}` has no id")))?;
                if id < 0 || id > i32::MAX as i64 {
                    return Err(Error::bundle_invalid(format!("{what}: added token id {id} is outside i32")));
                }
                let special = a.get("special").and_then(Value::as_bool).unwrap_or(false);
                max_id = max_id.max(id);
                vocab.entry(content.into()).or_insert(id as i32);
                added.push((Box::from(content), id as i32, special));
                if special {
                    special_ids.push(id as i32);
                }
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        let mut tokens: Vec<Option<Box<str>>> = vec![None; (max_id + 1) as usize];
        for (tok, &id) in &vocab {
            let slot = &mut tokens[id as usize];
            if slot.is_none() {
                *slot = Some(tok.clone());
            }
        }
        let unk_token = model
            .get("unk_token")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::bundle_invalid(format!("{what}: model.unk_token is missing")))?;
        let unk_id = *vocab.get(unk_token).ok_or_else(|| {
            Error::bundle_invalid(format!("{what}: unk_token `{unk_token}` is not in the vocabulary"))
        })?;
        let prefix = model.get("continuing_subword_prefix").and_then(Value::as_str).unwrap_or("##");
        let max_chars = model.get("max_input_chars_per_word").and_then(Value::as_u64).unwrap_or(100) as usize;

        let (lowercase, strip_accents, clean_text, handle_chinese_chars) = match j.get("normalizer") {
            None | Some(Value::Null) => (false, false, false, false),
            Some(n) => {
                let t = n.get("type").and_then(Value::as_str).unwrap_or("");
                if t != "BertNormalizer" {
                    return Err(Error::unsupported(format!(
                        "{what}: normalizer `{t}` is not served by the native tokenizer (BertNormalizer)"
                    )));
                }
                let lowercase = n.get("lowercase").and_then(Value::as_bool).unwrap_or(true);
                // `strip_accents: null` means "as lowercase", the BERT rule.
                let strip = n.get("strip_accents").and_then(Value::as_bool).unwrap_or(lowercase);
                (
                    lowercase,
                    strip,
                    n.get("clean_text").and_then(Value::as_bool).unwrap_or(true),
                    n.get("handle_chinese_chars").and_then(Value::as_bool).unwrap_or(true),
                )
            }
        };
        match j.get("pre_tokenizer") {
            Some(p) if !p.is_null() => {
                let t = p.get("type").and_then(Value::as_str).unwrap_or("");
                if t != "BertPreTokenizer" {
                    return Err(Error::unsupported(format!(
                        "{what}: pre_tokenizer `{t}` is not served by the native tokenizer (BertPreTokenizer)"
                    )));
                }
            }
            _ => {
                return Err(Error::unsupported(format!(
                    "{what}: a WordPiece file without a BertPreTokenizer is not served by the native tokenizer"
                )))
            }
        }
        let template = match j.get("post_processor") {
            None | Some(Value::Null) => vec![Piece::Sequence { type_id: 0 }],
            Some(p) => {
                let t = p.get("type").and_then(Value::as_str).unwrap_or("");
                match t {
                    "TemplateProcessing" => template_from_json(p, &vocab, what)?,
                    "BertProcessing" => {
                        let id_of = |key: &str| -> Result<i32> {
                            p.get(key)
                                .and_then(Value::as_array)
                                .and_then(|pair| pair.get(1))
                                .and_then(Value::as_i64)
                                .map(|i| i as i32)
                                .ok_or_else(|| Error::bundle_invalid(format!("{what}: BertProcessing.{key} is not [token, id]")))
                        };
                        vec![
                            Piece::Special { id: id_of("cls")?, type_id: 0 },
                            Piece::Sequence { type_id: 0 },
                            Piece::Special { id: id_of("sep")?, type_id: 0 },
                        ]
                    }
                    other => return Err(Error::unsupported(format!("{what}: post_processor `{other}` is not served by the native tokenizer (TemplateProcessing, BertProcessing)"))),
                }
            }
        };
        let decoder_cleanup = match j.get("decoder") {
            None | Some(Value::Null) => true,
            Some(d) => {
                let t = d.get("type").and_then(Value::as_str).unwrap_or("");
                if t != "WordPiece" {
                    return Err(Error::unsupported(format!(
                        "{what}: decoder `{t}` is not served by the native tokenizer (WordPiece)"
                    )));
                }
                if let Some(p) = d.get("prefix").and_then(Value::as_str) {
                    if p != prefix {
                        return Err(Error::bundle_invalid(format!(
                            "{what}: decoder prefix `{p}` differs from the model's `{prefix}`"
                        )));
                    }
                }
                d.get("cleanup").and_then(Value::as_bool).unwrap_or(true)
            }
        };
        Ok(Self {
            vocab,
            tokens,
            unk_id,
            prefix: prefix.into(),
            max_chars,
            lowercase,
            strip_accents,
            clean_text,
            handle_chinese_chars,
            added,
            special_ids,
            template,
            decoder_cleanup,
        })
    }

    /// Vocabulary size including added tokens (one past the largest id).
    pub fn vocab_size(&self) -> u32 {
        self.tokens.len() as u32
    }

    /// The id of a token string, if any.
    pub fn token_to_id(&self, token: &str) -> Option<i32> {
        self.vocab.get(token).copied()
    }

    /// The number of special tokens the template adds around one sequence.
    pub fn specials_per_sequence(&self) -> u32 {
        self.template.iter().filter(|p| matches!(p, Piece::Special { .. })).count() as u32
    }

    /// Whether `id` is a special (added, `special: true`) token.
    pub fn is_special(&self, id: i32) -> bool {
        self.special_ids.contains(&id)
    }

    /// The content tokens of `text` (no template), with byte offsets.
    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut out = Vec::new();
        let mut word: Vec<NChar> = Vec::new();
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // Added tokens (specials among them) match verbatim, longest first.
            if let Some((tok, id, _)) = self.added.iter().find(|(tok, _, _)| bytes[i..].starts_with(tok.as_bytes())) {
                self.flush_word(&mut word, &mut out);
                out.push(Token { id: *id, start: i as u32, end: (i + tok.len()) as u32 });
                i += tok.len();
                continue;
            }
            let ch = text[i..].chars().next().expect("i is a char boundary");
            let start = i as u32;
            i += ch.len_utf8();
            let end = i as u32;
            let cp = ch as u32;
            if self.clean_text && (cp == 0 || cp == 0xFFFD || is_control(cp)) {
                continue;
            }
            if is_whitespace(cp) {
                self.flush_word(&mut word, &mut out);
                continue;
            }
            let chinese = self.handle_chinese_chars && is_chinese_char(cp);
            if chinese {
                self.flush_word(&mut word, &mut out);
            }
            let one = [cp];
            let decomposed: &[u32] = if self.strip_accents { nfd(cp).unwrap_or(&one) } else { &one };
            for &d in decomposed {
                if self.strip_accents && in_ranges(d, MARK_NONSPACING) {
                    continue;
                }
                let dch = char::from_u32(d).expect("NFD yields scalar values");
                let lowered: Vec<char> = if self.lowercase { dch.to_lowercase().collect() } else { vec![dch] };
                for lc in lowered {
                    let lcp = lc as u32;
                    if is_whitespace(lcp) {
                        self.flush_word(&mut word, &mut out);
                    } else if is_punctuation(lcp) {
                        self.flush_word(&mut word, &mut out);
                        let mut single = vec![NChar { ch: lc, start, end, ccc: 0 }];
                        self.flush_word(&mut single, &mut out);
                    } else {
                        // Canonical ordering of the marks that remain: a mark
                        // moves before higher-class marks preceding it.
                        let ccc = combining_class(lcp);
                        let mut at = word.len();
                        while at > 0 && ccc != 0 && word[at - 1].ccc > ccc {
                            at -= 1;
                        }
                        word.insert(at, NChar { ch: lc, start, end, ccc });
                    }
                }
            }
            if chinese {
                self.flush_word(&mut word, &mut out);
            }
        }
        self.flush_word(&mut word, &mut out);
        out
    }

    /// WordPiece one word (greedy longest match, `##` continuation) into `out`.
    fn flush_word(&self, word: &mut Vec<NChar>, out: &mut Vec<Token>) {
        if word.is_empty() {
            return;
        }
        let start = word[0].start;
        let end = word[word.len() - 1].end;
        if word.len() > self.max_chars {
            out.push(Token { id: self.unk_id, start, end });
            word.clear();
            return;
        }
        let text: String = word.iter().map(|c| c.ch).collect();
        // Char index -> byte index in `text`, so substrings are sliced by chars.
        let mut byte_at: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
        byte_at.push(text.len());
        let mut pieces = Vec::new();
        let mut buf = String::new();
        let mut s = 0usize;
        let n = word.len();
        while s < n {
            let mut e = n;
            let mut found: Option<i32> = None;
            while s < e {
                buf.clear();
                if s > 0 {
                    buf.push_str(&self.prefix);
                }
                buf.push_str(&text[byte_at[s]..byte_at[e]]);
                if let Some(&id) = self.vocab.get(buf.as_str()) {
                    found = Some(id);
                    break;
                }
                e -= 1;
            }
            match found {
                None => {
                    // One unmatched piece makes the whole word unknown.
                    pieces.clear();
                    pieces.push(Token { id: self.unk_id, start, end });
                    break;
                }
                Some(id) => {
                    pieces.push(Token { id, start: word[s].start, end: word[e - 1].end });
                    s = e;
                }
            }
        }
        out.extend(pieces);
        word.clear();
    }

    /// Wrap content tokens in the template: `(ids, type_ids, offsets)`; the
    /// template's specials have offsets `(0, 0)`.
    pub fn apply_template(&self, content: &[Token], add_special_tokens: bool) -> (Vec<i32>, Vec<i32>, Vec<(u32, u32)>) {
        apply_template(&self.template, content, add_special_tokens)
    }

    /// Decode ids to text with the WordPiece decoder's joining and, when
    /// the file asks for it, its cleanup of spacing around punctuation and
    /// contractions.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let mut words: Vec<&str> = Vec::with_capacity(ids.len());
        for &id in ids {
            let tok = self.tokens.get(id as usize).and_then(|t| t.as_deref()).ok_or_else(|| {
                Error::invalid_argument(format!("token id {id} is outside the vocabulary of {}", self.tokens.len()))
            })?;
            if skip_special_tokens && self.is_special(id) {
                continue;
            }
            words.push(tok);
        }
        // The Hugging Face WordPiece decoder joins with a space, strips the
        // continuation prefix, and cleans each token by itself (not the
        // joined text): a token that is exactly " ." or " 's" loses its
        // space; split punctuation ("'" then "s") stays split.
        let mut text = String::new();
        for (i, w) in words.iter().enumerate() {
            let mut piece = match w.strip_prefix(&*self.prefix) {
                Some(rest) if i > 0 => rest.to_string(),
                Some(rest) => rest.to_string(),
                None if i > 0 => format!(" {w}"),
                None => w.to_string(),
            };
            if self.decoder_cleanup {
                for (from, to) in [
                    (" .", "."),
                    (" ?", "?"),
                    (" !", "!"),
                    (" ,", ","),
                    (" ' ", "'"),
                    (" n't", "n't"),
                    (" 'm", "'m"),
                    (" do not", " don't"),
                    (" 's", "'s"),
                    (" 've", "'ve"),
                    (" 're", "'re"),
                ] {
                    piece = piece.replace(from, to);
                }
            }
            text.push_str(&piece);
        }
        Ok(text)
    }
}
