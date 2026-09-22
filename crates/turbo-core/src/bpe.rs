//! A native byte-level BPE tokenizer for `tokenizer.json` files of the
//! GPT-2 family (Qwen, GPT-2, RoBERTa, Llama 3): an optional `NFC`
//! normalizer, the `ByteLevel` pre-tokenizer with its GPT-2 pattern or a
//! `Split` on the Qwen or cl100k pattern ahead of it, the `BPE` model
//! (vocabulary and ranked merges, no byte fallback, no subword prefix), a
//! `ByteLevel`, `TemplateProcessing` or `RobertaProcessing` post-processor
//! and the `ByteLevel` decoder, with no dependency beyond `serde_json` and
//! the generated Unicode tables in `unicode_data.rs`.
//!
//! It reproduces the Hugging Face `tokenizers` crate's ids and byte offsets
//! for these files. Anything the file declares that this implementation
//! does not do is an error naming it, never a silent approximation; the
//! `hf-tokenizers` feature keeps that crate available for those files and
//! for the parity test.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;

use crate::error::{Error, Result};
use crate::unicode_data::{COMBINING_CLASS, COMPOSE, LETTER, NFD, NUMBER, WHITE_SPACE};
use crate::wordpiece::{template_from_json, Piece, Token};

/// The pre-tokenizer pattern, matched by hand (the alternation is ordered,
/// as in the regex the file names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pattern {
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    Gpt2,
    /// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
    Qwen2,
    /// The Qwen pattern with `\p{N}{1,3}` (GPT-4, Llama 3).
    Cl100k,
}

const GPT2_PATTERN: &str = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
const QWEN2_PATTERN: &str =
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const CL100K_PATTERN: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// Cached encodings of pre-tokens; cleared when full, as the Hugging Face
/// crate's cache is.
const CACHE_CAPACITY: usize = 10_000;

/// A pre-token's UTF-8 bytes to its ids.
type Cache = HashMap<Box<[u8]>, Box<[i32]>>;

/// A loaded byte-level BPE tokenizer.
pub struct Bpe {
    vocab: HashMap<Box<str>, i32>,
    tokens: Vec<Option<Box<str>>>,
    /// `(left id, right id)` to `(rank, merged id)`.
    ranks: HashMap<(i32, i32), (u32, i32)>,
    /// The id of each byte's single-character token.
    byte_ids: [i32; 256],
    /// Added tokens matched verbatim, longest first: content, id, special.
    added: Vec<(Box<str>, i32, bool)>,
    special_ids: Vec<i32>,
    template: Vec<Piece>,
    nfc: bool,
    pattern: Pattern,
    add_prefix_space: bool,
    trim_offsets: bool,
    ignore_merges: bool,
    cache: Mutex<Cache>,
}

impl std::fmt::Debug for Bpe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bpe")
            .field("vocab", &self.tokens.len())
            .field("merges", &self.ranks.len())
            .field("pattern", &self.pattern)
            .field("nfc", &self.nfc)
            .finish()
    }
}

fn in_ranges(cp: u32, ranges: &[(u32, u32)]) -> bool {
    let i = ranges.partition_point(|&(first, _)| first <= cp);
    i > 0 && cp <= ranges[i - 1].1
}

fn is_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || (c as u32 > 0x7F && in_ranges(c as u32, LETTER))
}

fn is_number(c: char) -> bool {
    c.is_ascii_digit() || (c as u32 > 0x7F && in_ranges(c as u32, NUMBER))
}

fn is_space(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r' | '\x0B' | '\x0C') || (c as u32 > 0x7F && in_ranges(c as u32, WHITE_SPACE))
}

fn combining_class(cp: u32) -> u8 {
    match COMBINING_CLASS.binary_search_by_key(&cp, |&(c, _)| c) {
        Ok(i) => COMBINING_CLASS[i].1,
        Err(_) => 0,
    }
}

fn nfd(cp: u32) -> Option<&'static [u32]> {
    NFD.binary_search_by_key(&cp, |&(c, _)| c).ok().map(|i| NFD[i].1)
}

/// The canonical composite of `a` followed by `b`, if NFC composes them
/// (the table's primary composites, plus the Hangul algorithm).
fn compose(a: u32, b: u32) -> Option<u32> {
    const S_BASE: u32 = 0xAC00;
    const L_BASE: u32 = 0x1100;
    const V_BASE: u32 = 0x1161;
    const T_BASE: u32 = 0x11A7;
    const L_COUNT: u32 = 19;
    const V_COUNT: u32 = 21;
    const T_COUNT: u32 = 28;
    const N_COUNT: u32 = V_COUNT * T_COUNT;
    const S_COUNT: u32 = L_COUNT * N_COUNT;
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT);
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE) % T_COUNT == 0
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return Some(a + (b - T_BASE));
    }
    COMPOSE.binary_search_by(|&(x, y, _)| (x, y).cmp(&(a, b))).ok().map(|i| COMPOSE[i].2)
}

/// A character of the normalized text with the byte range of the source
/// text it came from.
#[derive(Clone, Copy)]
struct NChar {
    ch: char,
    start: u32,
    end: u32,
}

/// NFC of `text`, each character with its source byte range. The text is
/// returned as it is (one range per character) when no character has a
/// decomposition or a non-zero combining class, which is the common case.
fn nfc(text: &str) -> Vec<NChar> {
    let mut out: Vec<NChar> = Vec::with_capacity(text.len());
    let mut needs_work = false;
    for (i, ch) in text.char_indices() {
        let cp = ch as u32;
        if cp > 0x7F && (nfd(cp).is_some() || combining_class(cp) != 0) {
            needs_work = true;
        }
        out.push(NChar { ch, start: i as u32, end: (i + ch.len_utf8()) as u32 });
    }
    if !needs_work {
        return out;
    }
    // Full decomposition with canonical ordering of the marks.
    let mut decomposed: Vec<(NChar, u8)> = Vec::with_capacity(out.len() + 8);
    for c in out {
        let one = [c.ch as u32];
        let seq = nfd(c.ch as u32).unwrap_or(&one);
        for &d in seq {
            let ccc = combining_class(d);
            let dc = NChar { ch: char::from_u32(d).expect("NFD yields scalar values"), start: c.start, end: c.end };
            let mut at = decomposed.len();
            while at > 0 && ccc != 0 && decomposed[at - 1].1 > ccc {
                at -= 1;
            }
            decomposed.insert(at, (dc, ccc));
        }
    }
    // Canonical composition: a starter takes the following character when
    // no character of a class at least as high stands between them.
    let mut result: Vec<(NChar, u8)> = Vec::with_capacity(decomposed.len());
    let mut starter: Option<usize> = None;
    let mut last_ccc: u8 = 0;
    for (c, ccc) in decomposed {
        if let Some(s) = starter {
            let blocked = result.len() > s + 1 && (last_ccc == 0 || last_ccc >= ccc);
            if !blocked {
                if let Some(m) = compose(result[s].0.ch as u32, c.ch as u32) {
                    // The composite keeps the starter's source range alone;
                    // the mark's bytes belong to no token, as the Hugging
                    // Face crate's alignment reports it.
                    result[s].0.ch = char::from_u32(m).expect("compositions are scalar values");
                    continue;
                }
            }
        }
        if ccc == 0 {
            starter = Some(result.len());
        }
        last_ccc = ccc;
        result.push((c, ccc));
    }
    result.into_iter().map(|(c, _)| c).collect()
}

/// The GPT-2 byte-to-character table: printable bytes map to themselves,
/// the rest to characters from U+0100 up.
fn byte_chars() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut next = 256u32;
    for b in 0..=255u32 {
        let printable = (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).expect("printable byte")
        } else {
            let c = char::from_u32(next).expect("byte character");
            next += 1;
            c
        };
    }
    table
}

fn pattern_of(pattern: &str, what: &str) -> Result<Pattern> {
    match pattern {
        GPT2_PATTERN => Ok(Pattern::Gpt2),
        QWEN2_PATTERN => Ok(Pattern::Qwen2),
        CL100K_PATTERN => Ok(Pattern::Cl100k),
        other => Err(Error::unsupported(format!(
            "{what}: split pattern `{other}` is not one the native tokenizer matches (the GPT-2, Qwen2 and cl100k patterns)"
        ))),
    }
}

impl Bpe {
    /// Load a `tokenizer.json`.
    pub fn from_json(text: &str, what: &str) -> Result<Self> {
        let j: Value = serde_json::from_str(text).map_err(|e| Error::bundle_invalid(format!("{what}: {e}")))?;
        let model = j.get("model").ok_or_else(|| Error::bundle_invalid(format!("{what}: no `model`")))?;
        let model_type = model.get("type").and_then(Value::as_str).unwrap_or("");
        if model_type != "BPE" {
            return Err(Error::unsupported(format!(
                "{what}: model type `{model_type}` is not served by the native tokenizer (WordPiece, byte-level BPE)"
            )));
        }
        if model.get("byte_fallback").and_then(Value::as_bool).unwrap_or(false) {
            return Err(Error::unsupported(format!(
                "{what}: BPE with byte_fallback is not served by the native tokenizer"
            )));
        }
        for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
            if model.get(key).and_then(Value::as_str).is_some_and(|s| !s.is_empty()) {
                return Err(Error::unsupported(format!(
                    "{what}: BPE with a {key} is not served by the native tokenizer"
                )));
            }
        }
        if let Some(d) = model.get("dropout") {
            if !d.is_null() && d.as_f64().is_some_and(|x| x != 0.0) {
                return Err(Error::unsupported(format!("{what}: BPE dropout is not served by the native tokenizer")));
            }
        }
        let ignore_merges = model.get("ignore_merges").and_then(Value::as_bool).unwrap_or(false);
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
        let merges_json = model
            .get("merges")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::bundle_invalid(format!("{what}: model.merges is not an array")))?;
        let mut ranks: HashMap<(i32, i32), (u32, i32)> = HashMap::with_capacity(merges_json.len());
        let mut pair = String::new();
        for (rank, m) in merges_json.iter().enumerate() {
            // A merge is "left right" (one string) or ["left", "right"].
            let (left, right) = match m {
                Value::String(s) => s
                    .split_once(' ')
                    .ok_or_else(|| Error::bundle_invalid(format!("{what}: merge {rank} `{s}` has no space")))?,
                Value::Array(a) if a.len() == 2 => match (a[0].as_str(), a[1].as_str()) {
                    (Some(l), Some(r)) => (l, r),
                    _ => return Err(Error::bundle_invalid(format!("{what}: merge {rank} is not a pair of strings"))),
                },
                _ => return Err(Error::bundle_invalid(format!("{what}: merge {rank} is neither a string nor a pair"))),
            };
            let l = *vocab.get(left).ok_or_else(|| {
                Error::bundle_invalid(format!("{what}: merge {rank} left `{left}` is not in the vocabulary"))
            })?;
            let r = *vocab.get(right).ok_or_else(|| {
                Error::bundle_invalid(format!("{what}: merge {rank} right `{right}` is not in the vocabulary"))
            })?;
            pair.clear();
            pair.push_str(left);
            pair.push_str(right);
            let merged = *vocab.get(pair.as_str()).ok_or_else(|| {
                Error::bundle_invalid(format!("{what}: merge {rank} result `{pair}` is not in the vocabulary"))
            })?;
            ranks.entry((l, r)).or_insert((rank as u32, merged));
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
                for flag in ["single_word", "lstrip", "rstrip"] {
                    if a.get(flag).and_then(Value::as_bool).unwrap_or(false) {
                        return Err(Error::unsupported(format!(
                            "{what}: added token `{content}` sets {flag}, which the native tokenizer does not do"
                        )));
                    }
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
        let chars = byte_chars();
        let mut byte_ids = [-1i32; 256];
        let mut buf = String::new();
        for b in 0..256usize {
            buf.clear();
            buf.push(chars[b]);
            byte_ids[b] = *vocab.get(buf.as_str()).ok_or_else(|| {
                Error::bundle_invalid(format!(
                    "{what}: the vocabulary has no token for byte {b} (`{buf}`); not a byte-level BPE"
                ))
            })?;
        }

        let nfc = match j.get("normalizer") {
            None | Some(Value::Null) => false,
            Some(n) => {
                let t = n.get("type").and_then(Value::as_str).unwrap_or("");
                match t {
                    "NFC" => true,
                    other => {
                        return Err(Error::unsupported(format!(
                            "{what}: normalizer `{other}` is not served by the native tokenizer (NFC or none)"
                        )))
                    }
                }
            }
        };

        // The pre-tokenizer: ByteLevel alone (its own GPT-2 pattern when
        // use_regex), or a Split on a known pattern followed by ByteLevel.
        let pre = j.get("pre_tokenizer").filter(|p| !p.is_null()).ok_or_else(|| {
            Error::unsupported(format!(
                "{what}: a BPE file without a pre_tokenizer is not served by the native tokenizer"
            ))
        })?;
        let (pattern, add_prefix_space) = Self::pre_tokenizer_of(pre, what)?;

        // The post-processor: ByteLevel (offset trimming), TemplateProcessing,
        // RobertaProcessing, or a Sequence of those.
        let mut template = vec![Piece::Sequence { type_id: 0 }];
        let mut trim_offsets = false;
        if let Some(p) = j.get("post_processor").filter(|p| !p.is_null()) {
            let nodes: Vec<&Value> = match p.get("type").and_then(Value::as_str) {
                Some("Sequence") => p
                    .get("processors")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::bundle_invalid(format!("{what}: post_processor Sequence has no processors")))?
                    .iter()
                    .collect(),
                _ => vec![p],
            };
            for node in nodes {
                match node.get("type").and_then(Value::as_str).unwrap_or("") {
                    "ByteLevel" => trim_offsets = node.get("trim_offsets").and_then(Value::as_bool).unwrap_or(true),
                    "TemplateProcessing" => template = template_from_json(node, &vocab, what)?,
                    "RobertaProcessing" => {
                        trim_offsets = node.get("trim_offsets").and_then(Value::as_bool).unwrap_or(true);
                        let id_of = |key: &str| -> Result<i32> {
                            node.get(key)
                                .and_then(Value::as_array)
                                .and_then(|pair| pair.get(1))
                                .and_then(Value::as_i64)
                                .map(|i| i as i32)
                                .ok_or_else(|| Error::bundle_invalid(format!("{what}: RobertaProcessing.{key} is not [token, id]")))
                        };
                        template = vec![
                            Piece::Special { id: id_of("cls")?, type_id: 0 },
                            Piece::Sequence { type_id: 0 },
                            Piece::Special { id: id_of("sep")?, type_id: 0 },
                        ];
                    }
                    other => {
                        return Err(Error::unsupported(format!(
                            "{what}: post_processor `{other}` is not served by the native tokenizer (ByteLevel, TemplateProcessing, RobertaProcessing)"
                        )))
                    }
                }
            }
        }
        match j.get("decoder") {
            None | Some(Value::Null) => {}
            Some(d) => {
                let t = d.get("type").and_then(Value::as_str).unwrap_or("");
                if t != "ByteLevel" {
                    return Err(Error::unsupported(format!(
                        "{what}: decoder `{t}` is not served by the native tokenizer (ByteLevel)"
                    )));
                }
            }
        }
        Ok(Self {
            vocab,
            tokens,
            ranks,
            byte_ids,
            added,
            special_ids,
            template,
            nfc,
            pattern,
            add_prefix_space,
            trim_offsets,
            ignore_merges,
            cache: Mutex::new(HashMap::new()),
        })
    }

    fn pre_tokenizer_of(pre: &Value, what: &str) -> Result<(Pattern, bool)> {
        let byte_level = |node: &Value| -> Result<(bool, bool)> {
            let add_prefix_space = node.get("add_prefix_space").and_then(Value::as_bool).unwrap_or(true);
            let use_regex = node.get("use_regex").and_then(Value::as_bool).unwrap_or(true);
            Ok((add_prefix_space, use_regex))
        };
        match pre.get("type").and_then(Value::as_str).unwrap_or("") {
            "ByteLevel" => {
                let (add_prefix_space, use_regex) = byte_level(pre)?;
                if !use_regex {
                    return Err(Error::unsupported(format!("{what}: a ByteLevel pre_tokenizer without its pattern and without a Split is not served by the native tokenizer")));
                }
                Ok((Pattern::Gpt2, add_prefix_space))
            }
            "Sequence" => {
                let parts = pre
                    .get("pretokenizers")
                    .and_then(Value::as_array)
                    .ok_or_else(|| Error::bundle_invalid(format!("{what}: pre_tokenizer Sequence has no pretokenizers")))?;
                if parts.len() != 2 {
                    return Err(Error::unsupported(format!("{what}: a pre_tokenizer Sequence of {} steps is not served by the native tokenizer (Split then ByteLevel)", parts.len())));
                }
                let split = &parts[0];
                if split.get("type").and_then(Value::as_str) != Some("Split") {
                    return Err(Error::unsupported(format!("{what}: pre_tokenizer Sequence must start with a Split to be served by the native tokenizer")));
                }
                if split.get("behavior").and_then(Value::as_str) != Some("Isolated") || split.get("invert").and_then(Value::as_bool).unwrap_or(false) {
                    return Err(Error::unsupported(format!("{what}: Split behavior other than Isolated is not served by the native tokenizer")));
                }
                let pattern = split
                    .get("pattern")
                    .and_then(|p| p.get("Regex"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| Error::unsupported(format!("{what}: Split without a Regex pattern is not served by the native tokenizer")))?;
                let pattern = pattern_of(pattern, what)?;
                let bl = &parts[1];
                if bl.get("type").and_then(Value::as_str) != Some("ByteLevel") {
                    return Err(Error::unsupported(format!("{what}: pre_tokenizer Sequence must end with ByteLevel to be served by the native tokenizer")));
                }
                let (add_prefix_space, use_regex) = byte_level(bl)?;
                if use_regex {
                    return Err(Error::unsupported(format!("{what}: a Split followed by a ByteLevel that also splits is not served by the native tokenizer")));
                }
                Ok((pattern, add_prefix_space))
            }
            other => Err(Error::unsupported(format!("{what}: pre_tokenizer `{other}` is not served by the native tokenizer (ByteLevel, Split then ByteLevel)"))),
        }
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
        let bytes = text.as_bytes();
        let mut i = 0usize;
        let mut segment_start = 0usize;
        while i < bytes.len() {
            let hit = self.added.iter().find(|(tok, _, _)| bytes[i..].starts_with(tok.as_bytes()));
            if let Some((tok, id, _)) = hit {
                self.tokenize_segment(&text[segment_start..i], segment_start as u32, &mut out);
                out.push(Token { id: *id, start: i as u32, end: (i + tok.len()) as u32 });
                i += tok.len();
                segment_start = i;
            } else {
                // Skip to the next character; added tokens start on a char boundary.
                i += text[i..].chars().next().map_or(1, char::len_utf8);
            }
        }
        self.tokenize_segment(&text[segment_start..], segment_start as u32, &mut out);
        out
    }

    /// One segment between added tokens: normalize, split on the pattern,
    /// encode every piece.
    fn tokenize_segment(&self, segment: &str, base: u32, out: &mut Vec<Token>) {
        if segment.is_empty() {
            return;
        }
        let mut chars = if self.nfc {
            nfc(segment)
        } else {
            segment
                .char_indices()
                .map(|(i, ch)| NChar { ch, start: i as u32, end: (i + ch.len_utf8()) as u32 })
                .collect()
        };
        for c in &mut chars {
            c.start += base;
            c.end += base;
        }
        if self.add_prefix_space && chars.first().is_some_and(|c| c.ch != ' ') {
            // The prefix space is not in the source; it takes the first
            // character's start as an empty range, as the crate reports it.
            let start = chars[0].start;
            chars.insert(0, NChar { ch: ' ', start, end: start });
        }
        let mut pos = 0usize;
        let mut piece_bytes: Vec<u8> = Vec::with_capacity(64);
        let mut piece_ranges: Vec<(u32, u32)> = Vec::with_capacity(64);
        while pos < chars.len() {
            let len = self.next_piece(&chars, pos);
            debug_assert!(len > 0);
            piece_bytes.clear();
            piece_ranges.clear();
            for c in &chars[pos..pos + len] {
                let mut buf = [0u8; 4];
                let enc = c.ch.encode_utf8(&mut buf);
                for _ in enc.bytes() {
                    piece_ranges.push((c.start, c.end));
                }
                piece_bytes.extend_from_slice(enc.as_bytes());
            }
            self.encode_piece(&piece_bytes, &piece_ranges, out);
            pos += len;
        }
    }

    /// The length in characters of the pattern's match at `pos`.
    fn next_piece(&self, chars: &[NChar], pos: usize) -> usize {
        let at = |k: usize| chars.get(pos + k).map(|c| c.ch);
        let n = chars.len() - pos;
        // The contraction alternatives, first in every pattern.
        if at(0) == Some('\'') {
            let fold = |c: Option<char>| -> Option<char> {
                match c? {
                    'ſ' => Some('s'),
                    x => Some(x.to_ascii_lowercase()),
                }
            };
            let (c1, c2) = if self.pattern == Pattern::Gpt2 { (at(1), at(2)) } else { (fold(at(1)), fold(at(2))) };
            match (c1, c2) {
                (Some('s' | 't' | 'm' | 'd'), _) => return 2,
                (Some('r'), Some('e')) | (Some('v'), Some('e')) | (Some('l'), Some('l')) => return 3,
                _ => {}
            }
        }
        let run = |from: usize, pred: &dyn Fn(char) -> bool| -> usize {
            let mut k = from;
            while k < n && pred(chars[pos + k].ch) {
                k += 1;
            }
            k - from
        };
        let is_crlf = |c: char| c == '\r' || c == '\n';
        match self.pattern {
            Pattern::Gpt2 => {
                let lead = usize::from(at(0) == Some(' '));
                let letters = run(lead, &is_letter);
                if letters > 0 {
                    return lead + letters;
                }
                let numbers = run(lead, &is_number);
                if numbers > 0 {
                    return lead + numbers;
                }
                let other = run(lead, &|c| !is_space(c) && !is_letter(c) && !is_number(c));
                if other > 0 {
                    return lead + other;
                }
            }
            Pattern::Qwen2 | Pattern::Cl100k => {
                // [^\r\n\p{L}\p{N}]?\p{L}+
                let c0 = at(0).expect("pos is inside the text");
                let lead = usize::from(!is_crlf(c0) && !is_letter(c0) && !is_number(c0));
                let letters = run(lead, &is_letter);
                if letters > 0 {
                    return lead + letters;
                }
                // \p{N} or \p{N}{1,3}
                if is_number(c0) {
                    return if self.pattern == Pattern::Qwen2 { 1 } else { run(0, &is_number).min(3) };
                }
                // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
                let lead = usize::from(c0 == ' ');
                let other = run(lead, &|c| !is_space(c) && !is_letter(c) && !is_number(c));
                if other > 0 {
                    return lead + other + run(lead + other, &is_crlf);
                }
                // \s*[\r\n]+ : the whitespace run up to its last \r or \n
                let spaces = run(0, &is_space);
                if let Some(last) = (0..spaces).rev().find(|&k| is_crlf(chars[pos + k].ch)) {
                    return last + 1;
                }
            }
        }
        // \s+(?!\S) then \s+
        let spaces = run(0, &is_space);
        if spaces > 0 {
            if pos + spaces == chars.len() {
                return spaces;
            }
            if spaces > 1 {
                return spaces - 1;
            }
            return 1;
        }
        // A character no alternative takes (a lone apostrophe in the GPT-2
        // pattern is `[^\s\p{L}\p{N}]+`; here nothing is left over, but a
        // single character is the safe fallback).
        1
    }

    /// BPE over one pre-token (its UTF-8 bytes) into `out`, one source
    /// range per byte.
    fn encode_piece(&self, bytes: &[u8], ranges: &[(u32, u32)], out: &mut Vec<Token>) {
        let ids: Box<[i32]> = {
            let cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            cache.get(bytes).cloned()
        }
        .unwrap_or_else(|| {
            let ids = self.merge(bytes);
            let mut cache = self.cache.lock().unwrap_or_else(|p| p.into_inner());
            if cache.len() >= CACHE_CAPACITY {
                cache.clear();
            }
            cache.insert(bytes.into(), ids.clone());
            ids
        });
        // Each token covers a run of bytes; its length in bytes is the
        // byte-level string's character count.
        let mut b = 0usize;
        for &id in ids.iter() {
            let len = self.tokens[id as usize].as_deref().map_or(1, |t| t.chars().count());
            let (mut start, mut end) = (ranges[b].0, ranges[(b + len - 1).min(ranges.len() - 1)].1);
            if self.trim_offsets {
                if let Some(t) = self.tokens[id as usize].as_deref() {
                    let leading = t.chars().take_while(|&c| c == 'Ġ').count();
                    let trailing = t.chars().rev().take_while(|&c| c == 'Ġ').count();
                    if leading + trailing < len {
                        start = ranges[b + leading].0;
                        end = ranges[b + len - 1 - trailing].1;
                    }
                }
            }
            out.push(Token { id, start, end });
            b += len;
        }
    }

    /// The merge loop: the lowest-ranked adjacent pair merges first, the
    /// leftmost on a tie.
    fn merge(&self, bytes: &[u8]) -> Box<[i32]> {
        if self.ignore_merges {
            let word: String = bytes.iter().map(|&b| byte_chars()[b as usize]).collect();
            if let Some(&id) = self.vocab.get(word.as_str()) {
                return Box::from([id]);
            }
        }
        let mut parts: Vec<i32> = bytes.iter().map(|&b| self.byte_ids[b as usize]).collect();
        loop {
            let mut best: Option<(u32, usize, i32)> = None;
            for i in 0..parts.len().saturating_sub(1) {
                if let Some(&(rank, merged)) = self.ranks.get(&(parts[i], parts[i + 1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i, merged)) => {
                    parts[i] = merged;
                    parts.remove(i + 1);
                }
            }
        }
        parts.into_boxed_slice()
    }

    /// Wrap content tokens in the template: `(ids, type_ids, offsets)`; the
    /// template's specials have offsets `(0, 0)`.
    pub fn apply_template(&self, content: &[Token], add_special_tokens: bool) -> (Vec<i32>, Vec<i32>, Vec<(u32, u32)>) {
        crate::wordpiece::apply_template(&self.template, content, add_special_tokens)
    }

    /// Decode ids to text through the byte-level table.
    pub fn decode(&self, ids: &[i32], skip_special_tokens: bool) -> Result<String> {
        let chars = byte_chars();
        let mut back: HashMap<char, u8> = HashMap::with_capacity(256);
        for (b, &c) in chars.iter().enumerate() {
            back.insert(c, b as u8);
        }
        let mut bytes: Vec<u8> = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            let tok = self.tokens.get(id as usize).and_then(|t| t.as_deref()).ok_or_else(|| {
                Error::invalid_argument(format!("token id {id} is outside the vocabulary of {}", self.tokens.len()))
            })?;
            if skip_special_tokens && self.is_special(id) {
                continue;
            }
            if self.added.iter().any(|(t, i, _)| *i == id && &**t == tok) {
                // An added token is stored as text, not byte-level characters.
                bytes.extend_from_slice(tok.as_bytes());
                continue;
            }
            for c in tok.chars() {
                match back.get(&c) {
                    Some(&b) => bytes.push(b),
                    None => bytes.extend_from_slice(c.to_string().as_bytes()),
                }
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }
}
