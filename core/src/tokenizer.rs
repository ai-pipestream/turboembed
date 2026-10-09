//! The tokenizer (README rule 6): WordPiece or SentencePiece's Unigram
//! as the bundle describes it, built from the upstream tokenizer.json and
//! checked against the reference's ids on every load (docs/bundle.md
//! loader rule 5).

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

use serde_json::Value;
use unicode_categories::UnicodeCategories;
use unicode_normalization_alignments::UnicodeNormalization;

use crate::bundle::Bundle;
use crate::manifest::{Normalizer, PromptRole, SpecialRole, Truncation};
use crate::safetensors::{self, Dtype};
use crate::status::{CAPACITY, Error, INVALID_ARGUMENT, Result, invalid};
use crate::unigram::Unigram;

/// FxHash, as rustc uses it: a multiply and a rotate per word. The vocab
/// is fixed and comes from the bundle, so the keys need no defence against
/// collisions; SipHash, the std default, was most of a short text's time.
#[derive(Default)]
struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &w in words {
            self.add(u64::from_le_bytes(w));
        }
        let mut tail = 0u64;
        for (i, &b) in rest.iter().enumerate() {
            tail |= (b as u64) << (8 * i);
        }
        self.add(tail);
    }

    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }

    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

impl Fx {
    fn add(&mut self, w: u64) {
        self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

/// How a static model turns a text into ids, beside the encoding itself
/// (docs/static.md): when truncating at the right, the text is cut to
/// max_tokens times `median_chars` characters and its encoding to
/// max_tokens; then the unknown token is dropped. The tokenizer file's
/// own truncation and padding are never used, so a text's ids do not
/// depend on its batch.
struct StaticRules {
    /// The median length of the vocabulary's entries in characters,
    /// rounded down.
    median_chars: usize,
}

type Vocab = HashMap<String, u32, BuildHasherDefault<Fx>>;

/// The two models the core runs.
enum Kind {
    WordPiece {
        vocab: Vocab,
        /// The entries that start with the continuing prefix, under the
        /// rest of their text, so a piece inside a word is looked up as
        /// it stands in the text.
        continuing: Vocab,
        /// The longest entry in bytes, and the longest in `continuing`.
        longest: usize,
        longest_continuing: usize,
        normalizer: Normalizer,
        /// What the normalizer makes of each ASCII character.
        ascii: [u8; 128],
        continuing_prefix: String,
        max_chars_per_word: usize,
    },
    Unigram(Unigram),
}

/// A special token as it is matched in raw text, or in normalized text.
struct Special {
    content: String,
    id: i32,
    /// The whitespace before it is part of the match.
    lstrip: bool,
    /// Matched only with no word character on either side.
    single_word: bool,
}

pub struct Tokenizer {
    kind: Kind,
    /// Special tokens as they are matched in raw text: longest first.
    specials: Vec<Special>,
    /// Special tokens matched in each piece of normalized text, their
    /// content normalized: longest first.
    normalized_specials: Vec<Special>,
    /// The bytes a special token starts with, so text_ids looks for a
    /// match only where one can begin.
    special_starts: [bool; 256],
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
    /// A static model's rules (manifest static_embedding), None for an
    /// encoder.
    stat: Option<StaticRules>,
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

    /// Build the tokenizer the bundle names without checking it against
    /// the reference ids: for the bundle tool, which writes those ids.
    pub fn unchecked(bundle: &Bundle) -> Result<Tokenizer> {
        Self::from_bundle(bundle)
    }

    fn from_bundle(bundle: &Bundle) -> Result<Tokenizer> {
        let m = &bundle.manifest;
        let t = &m.tokenizer;
        let file = t.file.as_str();
        let bytes = bundle.read_verified(file)?;
        let json: Value = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("{file}: {e}")))?;
        let unk = t.special_tokens.iter().find(|s| s.role == SpecialRole::Unk).expect("validated");

        let kind = match (&t.wordpiece, &t.unigram) {
            (Some(w), _) => {
                let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));
                let model = &json["model"];
                if model["type"] != "WordPiece" {
                    return Err(invalid(format!("{file}: model.type is {}, not WordPiece", model["type"])));
                }
                if model["unk_token"] != unk.content.as_str() {
                    return Err(disagree("model.unk_token"));
                }
                if model["continuing_subword_prefix"] != w.continuing_prefix.as_str() {
                    return Err(disagree("model.continuing_subword_prefix"));
                }
                if model["max_input_chars_per_word"] != w.max_chars_per_word {
                    return Err(disagree("model.max_input_chars_per_word"));
                }

                let n = &json["normalizer"];
                let lowercase = n["lowercase"].as_bool();
                let strip = if n["strip_accents"].is_null() { lowercase } else { n["strip_accents"].as_bool() };
                let norm = t.normalizer.as_ref().expect("validated");
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

                let raw = model["vocab"]
                    .as_object()
                    .ok_or_else(|| invalid(format!("{file}: model.vocab is not an object")))?;
                let mut vocab = Vocab::with_capacity_and_hasher(raw.len(), Default::default());
                let mut seen = vec![false; raw.len()];
                for (piece, id) in raw {
                    let id = id.as_u64().filter(|&i| (i as usize) < raw.len());
                    let Some(id) = id else {
                        return Err(invalid(format!(
                            "{file}: model.vocab[{piece:?}] is not an id under {}",
                            raw.len()
                        )));
                    };
                    if std::mem::replace(&mut seen[id as usize], true) {
                        return Err(invalid(format!("{file}: model.vocab: id {id} is used twice")));
                    }
                    vocab.insert(piece.clone(), id as u32);
                }
                // The category table is made here, at load, rather than
                // by the first text that is not ASCII.
                category('\0', OTHER);
                let mut continuing = Vocab::default();
                if !w.continuing_prefix.is_empty() {
                    for (piece, &id) in &vocab {
                        if let Some(rest) = piece.strip_prefix(w.continuing_prefix.as_str()) {
                            continuing.insert(rest.to_owned(), id);
                        }
                    }
                }
                Kind::WordPiece {
                    longest: vocab.keys().map(String::len).max().unwrap_or(0),
                    longest_continuing: continuing.keys().map(String::len).max().unwrap_or(0),
                    continuing,
                    vocab,
                    ascii: ascii_map(norm),
                    normalizer: norm.clone(),
                    continuing_prefix: w.continuing_prefix.clone(),
                    max_chars_per_word: w.max_chars_per_word as usize,
                }
            }
            (None, Some(u)) => Kind::Unigram(Unigram::parse(file, &json, u, unk.id)?),
            (None, None) => unreachable!("validated"),
        };
        let vocab_size = kind.vocab_size();
        let table = m
            .architecture
            .as_ref()
            .map(|a| ("architecture", a.vocab_size))
            .or(m.static_embedding.as_ref().map(|st| ("static_embedding", st.vocab_size)));
        if let Some((block, rows)) = table
            && vocab_size as u64 > rows as u64
        {
            return Err(invalid(format!("{file}: {vocab_size} vocabulary entries, {block}.vocab_size is {rows}")));
        }

        // Every special token is in the vocabulary under its id, and the
        // added tokens upstream matches in raw text are exactly these.
        for s in &t.special_tokens {
            if kind.id(&s.content) != Some(s.id) {
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
                || a["normalized"] != s.normalized
                || a["lstrip"] != s.lstrip
                || a["rstrip"] != s.rstrip
                || a["single_word"] != s.single_word
            {
                return Err(invalid(format!(
                    "{file}: added_tokens[{i}] {content:?} is matched differently from what manifest.json says"
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
        let mut specials: Vec<Special> = t
            .special_tokens
            .iter()
            .filter(|s| !s.normalized)
            .map(|s| Special { content: s.content.clone(), id: s.id as i32, lstrip: s.lstrip, single_word: false })
            .collect();
        specials.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        // Whitespace around a normalized special token is dropped by the
        // pre-tokenizer either way, so lstrip and rstrip change no id.
        let mut normalized_specials: Vec<Special> = match &kind {
            Kind::WordPiece { normalizer, ascii, .. } => t
                .special_tokens
                .iter()
                .filter(|s| s.normalized)
                .map(|s| Special {
                    content: normalize(normalizer, ascii, &s.content),
                    id: s.id as i32,
                    lstrip: s.lstrip,
                    single_word: s.single_word,
                })
                .filter(|s| !s.content.is_empty())
                .collect(),
            Kind::Unigram(_) => Vec::new(),
        };
        normalized_specials.sort_by_key(|a| std::cmp::Reverse(a.content.len()));
        let mut special_starts = [false; 256];
        for s in &specials {
            if let Some(&b) = s.content.as_bytes().first() {
                special_starts[b as usize] = true;
            }
        }
        let stat = m.static_embedding.as_ref().map(|_| StaticRules { median_chars: kind.median_chars() });
        let e = m.embed();
        Ok(Tokenizer {
            kind,
            specials,
            normalized_specials,
            special_starts,
            template: t.template.iter().map(|s| if s == "$TEXT" { None } else { id_of(s) }).collect(),
            truncation: t.truncation,
            max_seq: m.static_embedding.as_ref().map_or(e.max_seq, |st| st.max_length),
            prefix_query: e.prefix_query.clone(),
            prefix_document: e.prefix_document.clone(),
            pad_id: role(SpecialRole::Pad),
            bos_id: role(SpecialRole::Bos),
            eos_id: role(SpecialRole::Eos),
            unk_id: role(SpecialRole::Unk),
            stat,
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
        self.kind.vocab_size()
    }

    /// `turbo_tokenizer_info.kind`.
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            Kind::WordPiece { .. } => "wordpiece",
            Kind::Unigram(_) => "unigram",
        }
    }

    /// The id under mask 0 in rows the session lays out: the pad token,
    /// or the unk token (which every bundle has) when there is no pad
    /// token, so every id a backend reads is in the vocabulary.
    pub fn fill_id(&self) -> i32 {
        if self.pad_id < 0 { self.unk_id } else { self.pad_id }
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
        let text = self.prompted(text, opts.prompt);
        let mut body = match &self.stat {
            None => self.text_ids(&text),
            Some(st) => {
                let mut ids = match opts.truncation {
                    Truncation::Right => {
                        let mut ids = self.text_ids(char_prefix(&text, budget.saturating_mul(st.median_chars)));
                        ids.truncate(budget);
                        ids
                    }
                    Truncation::Left => {
                        let mut ids = self.text_ids(&text);
                        ids.drain(..ids.len().saturating_sub(budget));
                        ids
                    }
                    Truncation::None => self.text_ids(&text),
                };
                let unk = self.unk_id;
                ids.retain(|&id| id != unk);
                ids
            }
        };
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
        let mut ids = self.text_ids(&self.prompted(text, prompt));
        if self.stat.is_some() {
            let unk = self.unk_id;
            ids.retain(|&id| id != unk);
        }
        ids.len() + self.specials_per_sequence() as usize
    }

    fn prompted<'a>(&self, text: &'a str, prompt: PromptRole) -> Cow<'a, str> {
        let prefix = match prompt {
            PromptRole::None => "",
            PromptRole::Query => &self.prefix_query,
            PromptRole::Document => &self.prefix_document,
        };
        if prefix.is_empty() { Cow::Borrowed(text) } else { Cow::Owned(format!("{prefix}{text}")) }
    }

    /// Special tokens are matched in the raw text first, as upstream does
    /// for its added tokens, one that takes the whitespace before it
    /// taking it; each piece of text between them is encoded on its own.
    fn text_ids(&self, text: &str) -> Vec<i32> {
        let mut out = Vec::new();
        let mut rest = text;
        let mut plain = 0;
        while plain < rest.len() {
            // A byte a special token starts with is never inside a UTF-8
            // sequence, so `plain` is on a character boundary wherever
            // this looks.
            let b = rest.as_bytes()[plain];
            let hit = if self.special_starts[b as usize] {
                self.specials.iter().find(|s| rest[plain..].starts_with(s.content.as_str()))
            } else {
                None
            };
            match hit {
                Some(s) => {
                    let before = if s.lstrip { rest[..plain].trim_end() } else { &rest[..plain] };
                    self.plain_ids(before, &mut out);
                    out.push(s.id);
                    rest = &rest[plain + s.content.len()..];
                    plain = 0;
                }
                None if b < 0x80 => plain += 1,
                None => plain += rest[plain..].chars().next().map_or(1, char::len_utf8),
            }
        }
        self.plain_ids(rest, &mut out);
        out
    }

    fn plain_ids(&self, text: &str, out: &mut Vec<i32>) {
        match &self.kind {
            Kind::WordPiece { normalizer, ascii, .. } => {
                let normalized = normalize(normalizer, ascii, text);
                let mut rest = normalized.as_str();
                // Upstream's leftmost-longest matches, each searched for
                // after the last; one that is not a whole word when it must
                // be is passed over, its text left to the words around it.
                let mut from = 0;
                while !self.normalized_specials.is_empty() && from < rest.len() {
                    let at = &rest[from..];
                    let hit = self.normalized_specials.iter().find(|s| at.starts_with(s.content.as_str()));
                    let Some(s) = hit else {
                        from += at.chars().next().map_or(1, char::len_utf8);
                        continue;
                    };
                    let end = from + s.content.len();
                    let whole = !s.single_word
                        || (!rest[..from].chars().next_back().is_some_and(is_word)
                            && !rest[end..].chars().next().is_some_and(is_word));
                    if whole {
                        pre_tokenize(&rest[..from], |word| self.word_pieces(word, out));
                        out.push(s.id);
                        rest = &rest[end..];
                        from = 0;
                    } else {
                        from = end;
                    }
                }
                pre_tokenize(rest, |word| self.word_pieces(word, out));
            }
            Kind::Unigram(u) => {
                if !text.is_empty() {
                    u.encode(text, out);
                }
            }
        }
    }

    /// Greedy longest-match-first; a word with any piece missing, or longer
    /// than max_chars_per_word, is one unknown token. A piece is never
    /// looked for longer than the longest entry.
    fn word_pieces(&self, word: &str, out: &mut Vec<i32>) {
        let Kind::WordPiece {
            vocab,
            continuing,
            longest,
            longest_continuing,
            continuing_prefix,
            max_chars_per_word,
            ..
        } = &self.kind
        else {
            unreachable!("WordPiece only");
        };
        // A character is at least one byte.
        if word.len() > *max_chars_per_word && word.chars().count() > *max_chars_per_word {
            out.push(self.unk_id);
            return;
        }
        let first = out.len();
        let mut start = 0;
        while start < word.len() {
            // With no continuing prefix, a piece inside a word is looked
            // up in the whole vocabulary.
            let (table, most) = if start == 0 || continuing_prefix.is_empty() {
                (vocab, *longest)
            } else {
                (continuing, *longest_continuing)
            };
            let mut end = word.len().min(start + most);
            while !word.is_char_boundary(end) {
                end -= 1;
            }
            let mut found = None;
            while start < end {
                if let Some(&id) = table.get(&word[start..end]) {
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

impl Kind {
    fn vocab_size(&self) -> u32 {
        match self {
            Kind::WordPiece { vocab, .. } => vocab.len() as u32,
            Kind::Unigram(u) => u.vocab_size(),
        }
    }

    fn id(&self, piece: &str) -> Option<u32> {
        match self {
            Kind::WordPiece { vocab, .. } => vocab.get(piece).copied(),
            Kind::Unigram(u) => u.id(piece),
        }
    }

    /// The median length of the vocabulary's entries in characters,
    /// rounded down: numpy's median of the lengths, then int().
    fn median_chars(&self) -> usize {
        let mut lens: Vec<usize> = match self {
            Kind::WordPiece { vocab, .. } => vocab.keys().map(|p| p.chars().count()).collect(),
            Kind::Unigram(u) => u.pieces().map(|p| p.chars().count()).collect(),
        };
        lens.sort_unstable();
        let n = lens.len();
        match n {
            0 => 0,
            _ if n % 2 == 1 => lens[n / 2],
            _ => (lens[n / 2 - 1] + lens[n / 2]) / 2,
        }
    }
}

/// The first `n` characters of `text`, all of it when it has fewer.
fn char_prefix(text: &str, n: usize) -> &str {
    if text.len() <= n {
        return text;
    }
    text.char_indices().nth(n).map_or(text, |(at, _)| &text[..at])
}

/// BertNormalizer, in upstream's order: clean, split CJK, strip accents,
/// lowercase. Every step works on one character at a time except
/// stripping accents, whose NFD reorders a run of combining marks; an
/// ASCII character is never one, so each run of non-ASCII text goes
/// through the steps on its own and ASCII text through a table.
fn normalize(n: &Normalizer, ascii: &[u8; 128], text: &str) -> String {
    let mut s = String::with_capacity(text.len());
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let b = bytes[i];
        if b < 0x80 {
            let m = ascii[b as usize];
            if m != DROP {
                s.push(m as char);
            }
            i += 1;
            continue;
        }
        // An ASCII byte is never inside a UTF-8 sequence, so the run ends
        // on a character boundary.
        let start = i;
        while i < bytes.len() && bytes[i] >= 0x80 {
            i += 1;
        }
        normalize_run(n, &text[start..i], &mut s);
    }
    s
}

/// What normalize() makes of each ASCII character: no ASCII character is
/// CJK or decomposes, its lowercase is its ASCII lowercase, and its
/// control characters are those under 0x20 and 0x7f.
const DROP: u8 = 0xff;

fn ascii_map(n: &Normalizer) -> [u8; 128] {
    let mut m = [0u8; 128];
    for (b, out) in m.iter_mut().enumerate() {
        let b = b as u8;
        *out = if n.clean_text && matches!(b, b'\t' | b'\n' | b'\r') {
            b' '
        } else if n.clean_text && (b < 0x20 || b == 0x7f) {
            DROP
        } else if n.lowercase {
            b.to_ascii_lowercase()
        } else {
            b
        };
    }
    m
}

/// The normalizer's steps on text, one character at a time, appended to
/// `out`.
fn normalize_run(n: &Normalizer, text: &str, out: &mut String) {
    let clean = n.clean_text;
    let cleaned = text
        .chars()
        .filter(move |&c| !(clean && (c == '\0' || c == '\u{fffd}' || is_control(c))))
        .map(move |c| if clean && is_whitespace(c) { ' ' } else { c });
    let split_cjk = n.split_cjk;
    let split = cleaned.flat_map(move |c| {
        let cjk = split_cjk && is_cjk(c);
        [cjk.then_some(' '), Some(c), cjk.then_some(' ')].into_iter().flatten()
    });
    if n.strip_accents {
        lowercased(n, split.nfd().map(|(c, _)| c).filter(|&c| !category(c, MARK_NONSPACING)), out);
    } else {
        lowercased(n, split, out);
    }
}

fn lowercased(n: &Normalizer, chars: impl Iterator<Item = char>, out: &mut String) {
    if n.lowercase {
        out.extend(chars.flat_map(char::to_lowercase));
    } else {
        out.extend(chars);
    }
}

/// BertPreTokenizer: split on whitespace, and each punctuation character
/// is a word of its own.
fn pre_tokenize<'a>(s: &'a str, mut word: impl FnMut(&'a str)) {
    for chunk in s.split(char::is_whitespace) {
        let mut begin = 0;
        for (i, c) in chunk.char_indices() {
            if c.is_ascii_punctuation() || (!c.is_ascii() && category(c, PUNCTUATION)) {
                if begin < i {
                    word(&chunk[begin..i]);
                }
                word(&chunk[i..i + c.len_utf8()]);
                begin = i + c.len_utf8();
            }
        }
        if begin < chunk.len() {
            word(&chunk[begin..]);
        }
    }
}

/// A character of regex's Unicode \\w, which upstream's single_word
/// reads: alphabetic, a mark, a decimal digit, a connector, or a joiner.
fn is_word(c: char) -> bool {
    c.is_alphabetic()
        || c.is_mark()
        || c.is_number_decimal_digit()
        || c.is_punctuation_connector()
        || matches!(c, '\u{200c}' | '\u{200d}')
}

fn is_whitespace(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r') || c.is_whitespace()
}

fn is_control(c: char) -> bool {
    !matches!(c, '\t' | '\n' | '\r') && category(c, OTHER)
}

const OTHER: u8 = 1;
const MARK_NONSPACING: u8 = 2;
const PUNCTUATION: u8 = 4;

/// Whether `c` is in the categories `bits` names (Unicode's C, Mn and P):
/// from a table of the Basic Multilingual Plane, made once (about 7 ms),
/// rather than a binary search per character.
fn category(c: char, bits: u8) -> bool {
    static BMP: std::sync::OnceLock<Box<[u8; 0x10000]>> = std::sync::OnceLock::new();
    let Ok(at) = u16::try_from(c as u32) else {
        return (bits & OTHER != 0 && c.is_other())
            || (bits & MARK_NONSPACING != 0 && c.is_mark_nonspacing())
            || (bits & PUNCTUATION != 0 && c.is_punctuation());
    };
    let table = BMP.get_or_init(|| {
        let mut t = Box::new([0u8; 0x10000]);
        for (i, f) in t.iter_mut().enumerate() {
            // A surrogate is no char, and in category C.
            let Some(c) = char::from_u32(i as u32) else {
                *f = OTHER;
                continue;
            };
            *f = (c.is_other() as u8) | ((c.is_mark_nonspacing() as u8) << 1) | ((c.is_punctuation() as u8) << 2);
        }
        t
    });
    table[at as usize] & bits != 0
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The normalizer's steps, each on the whole text in turn.
    fn steps(n: &Normalizer, text: &str) -> String {
        let mut s: String = if n.clean_text {
            text.chars()
                .filter(|&c| !(c == '\0' || c == '\u{fffd}' || (!matches!(c, '\t' | '\n' | '\r') && c.is_other())))
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

    /// The table gives each character's categories as unicode_categories
    /// does.
    #[test]
    fn the_category_table_is_the_unicode_categories() {
        for c in (0..=0x10ffffu32).filter(|&u| u < 0x10000 || u % 7 == 0).filter_map(char::from_u32) {
            assert_eq!(category(c, OTHER), c.is_other(), "{c:?}");
            assert_eq!(category(c, MARK_NONSPACING), c.is_mark_nonspacing(), "{c:?}");
            assert_eq!(category(c, PUNCTUATION), c.is_punctuation(), "{c:?}");
        }
    }

    /// WordPiece as it reads with no second table and no bound on a
    /// piece: every piece looked up with its prefix in the one vocabulary.
    fn pieces(vocab: &Vocab, prefix: &str, max_chars: usize, unk: i32, word: &str) -> Vec<i32> {
        if word.chars().count() > max_chars {
            return vec![unk];
        }
        let (mut out, mut start) = (Vec::new(), 0);
        while start < word.len() {
            let mut end = word.len();
            let found = loop {
                if start == end {
                    break None;
                }
                let key = if start == 0 { word[..end].to_owned() } else { format!("{prefix}{}", &word[start..end]) };
                if let Some(&id) = vocab.get(&key) {
                    break Some(id as i32);
                }
                end -= word[..end].chars().next_back().unwrap().len_utf8();
            };
            let Some(id) = found else { return vec![unk] };
            out.push(id);
            start = end;
        }
        out
    }

    /// The test bundle's WordPiece splits words as the plain algorithm
    /// does: words in the vocabulary, words made of pieces, words with a
    /// piece missing, and words at and over max_chars_per_word.
    #[test]
    fn word_pieces_are_the_plain_algorithm_s() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
        let tok = Tokenizer::unchecked(&Bundle::open(&dir).unwrap()).unwrap();
        let Kind::WordPiece { vocab, continuing_prefix, max_chars_per_word, .. } = &tok.kind else {
            panic!("the test bundle is WordPiece");
        };
        let mut words: Vec<String> = vocab.keys().cloned().collect();
        let parts: Vec<&str> =
            vocab.keys().map(|k| k.trim_start_matches(continuing_prefix.as_str())).take(500).collect();
        let mut x: u64 = 0x9e37_79b9_7f4a_7c15;
        for _ in 0..5000 {
            let mut w = String::new();
            for _ in 0..(x >> 62) + 1 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.push_str(parts[(x % parts.len() as u64) as usize]);
            }
            words.push(w);
        }
        // The longest pieces inside a word, after a few first pieces.
        let mut rests: Vec<&str> = vocab.keys().filter_map(|k| k.strip_prefix(continuing_prefix.as_str())).collect();
        rests.sort_by_key(|r| std::cmp::Reverse(r.len()));
        for r in rests.iter().take(30) {
            words.extend(["a", "the", "un", "x", "q"].map(|first| format!("{first}{r}")));
        }
        for n in [*max_chars_per_word - 1, *max_chars_per_word, *max_chars_per_word + 1] {
            words.push("a".repeat(n));
            words.push("é".repeat(n));
        }
        words.extend(["東京", "😀x", "qqqzzz", "ab\u{301}c"].map(String::from));
        for w in words.iter().filter(|w| !w.is_empty()) {
            let mut got = Vec::new();
            tok.word_pieces(w, &mut got);
            assert_eq!(got, pieces(vocab, continuing_prefix, *max_chars_per_word, tok.unk_id, w), "{w:?}");
        }
    }

    /// normalize() gives what its steps give, each on the whole text, for every
    /// combination of options: ASCII control and whitespace characters,
    /// combining marks after ASCII letters and in runs that NFD reorders,
    /// CJK, characters whose lowercase is longer, and random mixes of them.
    #[test]
    fn normalizing_by_runs_is_normalizing_the_whole_text() {
        let pool: Vec<char> = "aZ09 .,!\t\n\r\x00\x01\x0b\x0c\x1f\x7f\u{85}\u{a0}\u{3000}\u{fffd}\u{200b}\
             \u{301}\u{300}\u{327}\u{316}\u{93f}\u{94d}\u{903}\u{20dd}éÅñçİΣσςß東京한국\u{1100}\u{1161}ＡＢ１😀\u{e000}\u{feff}"
            .chars()
            .collect();
        let mut texts: Vec<String> = [
            "",
            "plain ascii, UPPER and lower!",
            "e\u{301}\u{316}\u{327}x",
            "a\u{316}\u{301}\u{903}\u{327}b",
            "Ｃａｆé 東京 İstanbul ΣΑΣ\u{85}end",
            "\x00\x7f\t\x0b\x0c\r\n",
        ]
        .iter()
        .map(|t| t.to_string())
        .collect();
        let mut x: u64 = 0x2545_f491_4f6c_dd1d;
        for _ in 0..2000 {
            let len = (x >> 59) as usize + 1;
            let mut t = String::new();
            for _ in 0..len {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                t.push(pool[(x % pool.len() as u64) as usize]);
            }
            texts.push(t);
        }
        for bits in 0..16u8 {
            let n = Normalizer {
                clean_text: bits & 1 != 0,
                split_cjk: bits & 2 != 0,
                strip_accents: bits & 4 != 0,
                lowercase: bits & 8 != 0,
                unicode_form: crate::manifest::UnicodeForm::None,
            };
            for t in &texts {
                assert_eq!(normalize(&n, &ascii_map(&n), t), steps(&n, t), "{t:?} with {n:?}");
            }
        }
    }
}
