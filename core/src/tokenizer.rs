//! The one tokenizer (README rule 6): WordPiece as the bundle describes it,
//! built from the upstream tokenizer.json and checked against the
//! reference's ids on every load (docs/bundle.md loader rule 5).

use std::collections::HashMap;

use serde_json::Value;
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::bundle::Bundle;
use crate::manifest::{Normalizer, PromptRole, SpecialRole, Truncation};
use crate::safetensors::{self, Dtype};
use crate::status::{CAPACITY, Error, INVALID_ARGUMENT, Result, invalid};

pub struct Tokenizer {
    vocab: HashMap<String, u32>,
    normalizer: Normalizer,
    continuing_prefix: String,
    max_chars_per_word: usize,
    /// Special tokens as they are matched in raw text: longest first.
    specials: Vec<(String, i32)>,
    /// The row layout: Some(id) for a special token, None for the text.
    template: Vec<Option<i32>>,
    truncation: Truncation,
    pub max_seq: u32,
    prefix_query: String,
    prefix_document: String,
    pub pad_id: i32,
    pub bos_id: i32,
    pub eos_id: i32,
    pub unk_id: i32,
    /// SHA-256 of the tokenizer file.
    pub sha256: String,
    /// SHA-256 of the manifest the tokenizer was made from.
    pub manifest_sha256: String,
}

/// How to lay out one row. `max_tokens` includes the special tokens when
/// they are added.
#[derive(Debug, Clone, Copy)]
pub struct Encode {
    pub add_special_tokens: bool,
    pub truncation: Truncation,
    pub max_tokens: u32,
    pub prompt: PromptRole,
}

impl Tokenizer {
    /// Build the tokenizer the bundle names and check it against the
    /// reference ids. This is where `turbo_tokenizer_create` stops.
    pub fn load(bundle: &Bundle) -> Result<Tokenizer> {
        let tok = Self::from_bundle(bundle)?;
        tok.check_reference(bundle)?;
        Ok(tok)
    }

    fn from_bundle(bundle: &Bundle) -> Result<Tokenizer> {
        let m = &bundle.manifest;
        let t = &m.tokenizer;
        let file = t.file.as_str();
        let bytes = bundle.read_verified(file)?;
        let json: Value = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("{file}: {e}")))?;
        let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));

        let model = &json["model"];
        if model["type"] != "WordPiece" {
            return Err(invalid(format!("{file}: model.type is {}, not WordPiece", model["type"])));
        }
        let unk = t.special_tokens.iter().find(|s| s.role == SpecialRole::Unk).expect("validated");
        if model["unk_token"] != unk.content.as_str() {
            return Err(disagree("model.unk_token"));
        }
        if model["continuing_subword_prefix"] != t.wordpiece.continuing_prefix.as_str() {
            return Err(disagree("model.continuing_subword_prefix"));
        }
        if model["max_input_chars_per_word"] != t.wordpiece.max_chars_per_word {
            return Err(disagree("model.max_input_chars_per_word"));
        }

        let n = &json["normalizer"];
        let lowercase = n["lowercase"].as_bool();
        let strip = if n["strip_accents"].is_null() { lowercase } else { n["strip_accents"].as_bool() };
        let norm = &t.normalizer;
        if n["type"] != "BertNormalizer"
            || n["clean_text"].as_bool() != Some(norm.clean_text)
            || n["handle_chinese_chars"].as_bool() != Some(norm.split_cjk)
            || lowercase != Some(norm.lowercase)
            || strip != Some(norm.strip_accents)
        {
            return Err(disagree("normalizer"));
        }
        if json["pre_tokenizer"]["type"] != "BertPreTokenizer" {
            return Err(invalid(format!(
                "{file}: pre_tokenizer is {}, not BertPreTokenizer",
                json["pre_tokenizer"]["type"]
            )));
        }

        let raw = model["vocab"].as_object().ok_or_else(|| invalid(format!("{file}: model.vocab is not an object")))?;
        let mut vocab = HashMap::with_capacity(raw.len());
        let mut seen = vec![false; raw.len()];
        for (piece, id) in raw {
            let id = id.as_u64().filter(|&i| (i as usize) < raw.len());
            let Some(id) = id else {
                return Err(invalid(format!("{file}: model.vocab[{piece:?}] is not an id under {}", raw.len())));
            };
            if std::mem::replace(&mut seen[id as usize], true) {
                return Err(invalid(format!("{file}: model.vocab: id {id} is used twice")));
            }
            vocab.insert(piece.clone(), id as u32);
        }
        if let Some(a) = &m.architecture
            && vocab.len() as u64 > a.vocab_size as u64
        {
            return Err(invalid(format!(
                "{file}: {} vocabulary entries, architecture.vocab_size is {}",
                vocab.len(),
                a.vocab_size
            )));
        }

        // Every special token is in the vocabulary under its id, and the
        // added tokens upstream matches in raw text are exactly these.
        for s in &t.special_tokens {
            if vocab.get(&s.content) != Some(&s.id) {
                return Err(invalid(format!("{file}: model.vocab has no {:?} with id {}", s.content, s.id)));
            }
        }
        let added = json["added_tokens"].as_array().map(Vec::as_slice).unwrap_or_default();
        for (i, a) in added.iter().enumerate() {
            let content = a["content"].as_str().unwrap_or_default();
            let Some(s) = t.special_tokens.iter().find(|s| s.content == content) else {
                return Err(invalid(format!(
                    "{file}: added_tokens[{i}] {content:?} is not a special token in manifest.json"
                )));
            };
            if a["id"] != s.id
                || a["normalized"] != false
                || a["lstrip"] != false
                || a["rstrip"] != false
                || a["single_word"] != false
            {
                return Err(invalid(format!(
                    "{file}: added_tokens[{i}] {content:?} is matched differently from a plain special token"
                )));
            }
        }
        if added.len() != t.special_tokens.len() {
            return Err(invalid(format!(
                "{file}: {} added_tokens, manifest.json has {} special tokens",
                added.len(),
                t.special_tokens.len()
            )));
        }

        let id_of = |content: &str| t.special_tokens.iter().find(|s| s.content == content).map(|s| s.id as i32);
        let role = |r: SpecialRole| t.special_tokens.iter().find(|s| s.role == r).map_or(-1, |s| s.id as i32);
        let mut specials: Vec<(String, i32)> =
            t.special_tokens.iter().map(|s| (s.content.clone(), s.id as i32)).collect();
        specials.sort_by(|a, b| b.0.len().cmp(&a.0.len()));
        let e = m.embed();
        Ok(Tokenizer {
            vocab,
            normalizer: norm.clone(),
            continuing_prefix: t.wordpiece.continuing_prefix.clone(),
            max_chars_per_word: t.wordpiece.max_chars_per_word as usize,
            specials,
            template: t.template.iter().map(|s| if s == "$TEXT" { None } else { id_of(s) }).collect(),
            truncation: t.truncation,
            max_seq: e.max_seq,
            prefix_query: e.prefix_query.clone(),
            prefix_document: e.prefix_document.clone(),
            pad_id: role(SpecialRole::Pad),
            bos_id: role(SpecialRole::Bos),
            eos_id: role(SpecialRole::Eos),
            unk_id: role(SpecialRole::Unk),
            sha256: crate::bundle::sha256_hex(&bytes),
            manifest_sha256: bundle.manifest_sha256.clone(),
        })
    }

    /// Loader rule 5: encode every reference case and compare the ids
    /// exactly with the reference file's.
    fn check_reference(&self, bundle: &Bundle) -> Result<()> {
        let m = &bundle.manifest;
        let file = m.reference.file.as_str();
        let bytes = bundle.read_verified(file)?;
        let st = safetensors::File::parse(file, &bytes)?;
        let cases = &m.reference.cases;
        let n = cases.len() as u64;
        let ids = st.get("ids", Dtype::I32, 2)?;
        let lengths = st.get("lengths", Dtype::I32, 1)?;
        let emb = st.get("embeddings", Dtype::F32, 2)?;
        let dim = m.embed().dim as u64;
        if ids.shape[0] != n || lengths.shape[0] != n || emb.shape != [n, dim] {
            return Err(invalid(format!(
                "{file}: ids {:?}, lengths {:?} and embeddings {:?} are not [{n}, L], [{n}] and [{n}, {dim}]",
                ids.shape, lengths.shape, emb.shape
            )));
        }
        let width = ids.shape[1] as usize;
        let ids = ids.i32s();
        let lengths = lengths.i32s();

        let mut truncated = false;
        let opts = Encode {
            add_special_tokens: true,
            truncation: self.truncation,
            max_tokens: self.max_seq,
            prompt: PromptRole::None,
        };
        for (i, case) in cases.iter().enumerate() {
            let at = |why: String| invalid(format!("reference case {i} ({}): {why}", preview(&case.text)));
            let got = self.encode(&case.text, Encode { prompt: case.prompt_role, ..opts })?;
            truncated |= self.count(&case.text, case.prompt_role) > self.max_seq as usize;
            let row = &ids[i * width..(i + 1) * width];
            let len = lengths[i];
            if len < 0 || len as usize > width {
                return Err(at(format!("length {len} is outside the row width {width}")));
            }
            let want = &row[..len as usize];
            if let Some(p) = (0..got.len().min(want.len())).find(|&p| got[p] != want[p]) {
                return Err(at(format!("position {p}: the core gives {}, the reference has {}", got[p], want[p])));
            }
            if got.len() != want.len() {
                return Err(at(format!(
                    "position {}: the core gives {} tokens, the reference has {}",
                    got.len().min(want.len()),
                    got.len(),
                    want.len()
                )));
            }
            if let Some(p) = (want.len()..width).find(|&p| row[p] != self.pad_id) {
                return Err(at(format!("position {p}: padding is {}, not pad id {}", row[p], self.pad_id)));
            }
        }
        if !truncated {
            return Err(invalid(format!(
                "reference.cases: none is longer than embed.max_seq {}, so truncation is not checked",
                self.max_seq
            )));
        }
        Ok(())
    }

    /// What TURBO_TRUNCATE_MODEL means for this bundle.
    pub fn truncation(&self) -> Truncation {
        self.truncation
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab.len() as u32
    }

    pub fn specials_per_sequence(&self) -> u32 {
        self.template.iter().filter(|p| p.is_some()).count() as u32
    }

    /// One row of ids. Too long for `max_tokens` is cut as `truncation`
    /// says, or CAPACITY for TRUNCATE_NONE.
    pub fn encode(&self, text: &str, opts: Encode) -> Result<Vec<i32>> {
        let specials = if opts.add_special_tokens { self.specials_per_sequence() } else { 0 };
        if opts.max_tokens < specials {
            return Err(Error::field(
                INVALID_ARGUMENT,
                3,
                format!("max_tokens {} leaves no room beside {specials} special tokens", opts.max_tokens),
            ));
        }
        let budget = (opts.max_tokens - specials) as usize;
        let mut body = self.text_ids(&self.prompted(text, opts.prompt));
        if body.len() > budget {
            match opts.truncation {
                Truncation::None => {
                    return Err(Error::new(
                        CAPACITY,
                        format!(
                            "{} tokens is over max_tokens {} and truncation is TRUNCATE_NONE",
                            body.len() + specials as usize,
                            opts.max_tokens
                        ),
                    ));
                }
                Truncation::Right => body.truncate(budget),
                Truncation::Left => {
                    body.drain(..body.len() - budget);
                }
            }
        }
        if !opts.add_special_tokens {
            return Ok(body);
        }
        let mut row = Vec::with_capacity(body.len() + specials as usize);
        for piece in &self.template {
            match piece {
                Some(id) => row.push(*id),
                None => row.extend_from_slice(&body),
            }
        }
        Ok(row)
    }

    /// Tokens `text` produces, with special tokens and no truncation.
    pub fn count(&self, text: &str, prompt: PromptRole) -> usize {
        self.text_ids(&self.prompted(text, prompt)).len() + self.specials_per_sequence() as usize
    }

    fn prompted(&self, text: &str, prompt: PromptRole) -> String {
        let prefix = match prompt {
            PromptRole::None => "",
            PromptRole::Query => &self.prefix_query,
            PromptRole::Document => &self.prefix_document,
        };
        format!("{prefix}{text}")
    }

    /// Special tokens are matched in the raw text first, as upstream does
    /// for its added tokens; the text between them is normalized, split
    /// into words and cut into word pieces.
    fn text_ids(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        let mut rest = text;
        let mut plain = 0;
        while plain < rest.len() {
            let hit = self.specials.iter().find(|(s, _)| rest[plain..].starts_with(s.as_str()));
            match hit {
                Some((s, id)) => {
                    self.plain_ids(&rest[..plain], &mut out);
                    out.push(*id);
                    rest = &rest[plain + s.len()..];
                    plain = 0;
                }
                None => plain += rest[plain..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.plain_ids(rest, &mut out);
        out
    }

    fn plain_ids(&self, text: &str, out: &mut Vec<i32>) {
        let normalized = self.normalize(text);
        for word in pre_tokenize(&normalized) {
            self.word_pieces(word, out);
        }
    }

    /// BertNormalizer, in upstream's order: clean, split CJK, strip
    /// accents, lowercase.
    fn normalize(&self, text: &str) -> String {
        let n = &self.normalizer;
        let mut s: String = if n.clean_text {
            text.chars()
                .filter(|&c| !(c == '\0' || c == '\u{fffd}' || is_control(c)))
                .map(|c| if is_whitespace(c) { ' ' } else { c })
                .collect()
        } else {
            text.to_owned()
        };
        if n.split_cjk {
            let mut t = String::with_capacity(s.len());
            for c in s.chars() {
                if is_cjk(c) {
                    t.push(' ');
                    t.push(c);
                    t.push(' ');
                } else {
                    t.push(c);
                }
            }
            s = t;
        }
        if n.strip_accents {
            s = s.nfd().map(|(c, _)| c).filter(|c| !c.is_mark_nonspacing()).collect();
        }
        if n.lowercase {
            s = s.chars().flat_map(char::to_lowercase).collect();
        }
        s
    }

    /// Greedy longest-match-first; a word with any piece missing, or longer
    /// than max_chars_per_word, is one unknown token.
    fn word_pieces(&self, word: &str, out: &mut Vec<i32>) {
        if word.chars().count() > self.max_chars_per_word {
            out.push(self.unk_id);
            return;
        }
        let first = out.len();
        let mut start = 0;
        let mut piece = String::new();
        while start < word.len() {
            let mut end = word.len();
            let mut found = None;
            while start < end {
                piece.clear();
                if start > 0 {
                    piece.push_str(&self.continuing_prefix);
                }
                piece.push_str(&word[start..end]);
                if let Some(&id) = self.vocab.get(&piece) {
                    found = Some(id);
                    break;
                }
                end -= word[..end].chars().next_back().map_or(1, char::len_utf8);
            }
            match found {
                Some(id) => out.push(id as i32),
                None => {
                    out.truncate(first);
                    out.push(self.unk_id);
                    return;
                }
            }
            start = end;
        }
    }
}

/// BertPreTokenizer: split on whitespace, and each punctuation character
/// is a word of its own.
fn pre_tokenize(s: &str) -> Vec<&str> {
    let mut words = Vec::new();
    for chunk in s.split(char::is_whitespace) {
        let mut begin = 0;
        for (i, c) in chunk.char_indices() {
            if c.is_ascii_punctuation() || c.is_punctuation() {
                if begin < i {
                    words.push(&chunk[begin..i]);
                }
                words.push(&chunk[i..i + c.len_utf8()]);
                begin = i + c.len_utf8();
            }
        }
        if begin < chunk.len() {
            words.push(&chunk[begin..]);
        }
    }
    words
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && c.is_other()
}

fn is_cjk(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF
            | 0x3400..=0x4DBF
            | 0x20000..=0x2A6DF
            | 0x2A700..=0x2B73F
            | 0x2B740..=0x2B81F
            | 0x2B920..=0x2CEAF
            | 0xF900..=0xFAFF
            | 0x2F800..=0x2FA1F
    )
}

fn preview(text: &str) -> String {
    let short: String = text.chars().take(32).collect();
    if short.len() < text.len() { format!("{short:?}...") } else { format!("{short:?}") }
}
