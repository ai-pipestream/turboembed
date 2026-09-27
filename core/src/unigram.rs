//! SentencePiece's Unigram model as upstream `tokenizers` runs it, from
//! the upstream tokenizer.json: the precompiled character map it carries
//! (SentencePiece's own normalizer, `nmt_nfkc` for XLM-RoBERTa), the
//! collapse of repeated spaces, the metaspace in place of each space and
//! in front of each piece of text, the cut into words at each metaspace,
//! and each word's segmentation into the pieces of highest total score.
//! A character no piece covers is the unknown token, and a run of them
//! is one unknown token, as upstream fuses them.

use std::collections::HashMap;

use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::manifest;
use crate::status::{Result, invalid};

/// Upstream's penalty under the lowest score for a character no piece
/// covers.
const UNK_PENALTY: f64 = 10.0;

pub struct Unigram {
    ids: HashMap<String, u32>,
    /// Each piece's log probability, by id.
    scores: Vec<f64>,
    /// The longest piece, in characters: no lookup goes past it.
    max_piece_chars: usize,
    unk_score: f64,
    unk_id: u32,
    charsmap: Option<Charsmap>,
    collapse_spaces: bool,
    metaspace: char,
    add_prefix_space: bool,
}

impl Unigram {
    /// The model from the tokenizer file, checked against what the
    /// manifest says of it. `unk_id` is the manifest's SPECIAL_UNK id.
    pub fn parse(file: &str, json: &Value, m: &manifest::Unigram, unk_id: u32) -> Result<Unigram> {
        let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));
        let model = &json["model"];
        if model["type"] != "Unigram" {
            return Err(invalid(format!("{file}: model.type is {}, not Unigram", model["type"])));
        }
        if model["unk_id"] != unk_id {
            return Err(disagree("model.unk_id"));
        }
        if model["byte_fallback"].as_bool().unwrap_or(false) {
            return Err(invalid(format!("{file}: model.byte_fallback: the core has no byte fallback")));
        }
        let raw = model["vocab"].as_array().ok_or_else(|| invalid(format!("{file}: model.vocab is not an array")))?;
        let mut ids = HashMap::with_capacity(raw.len());
        let mut scores = Vec::with_capacity(raw.len());
        let mut max_piece_chars = 0;
        for (id, entry) in raw.iter().enumerate() {
            let (piece, score) = match entry.as_array().map(Vec::as_slice) {
                Some([p, s]) => (p.as_str(), s.as_f64()),
                _ => (None, None),
            };
            let (Some(piece), Some(score)) = (piece, score) else {
                return Err(invalid(format!("{file}: model.vocab[{id}] is not [piece, score]")));
            };
            if piece.is_empty() {
                return Err(invalid(format!("{file}: model.vocab[{id}] is an empty piece")));
            }
            if ids.insert(piece.to_owned(), id as u32).is_some() {
                return Err(invalid(format!("{file}: model.vocab: piece {piece:?} is listed twice")));
            }
            scores.push(score);
            max_piece_chars = max_piece_chars.max(piece.chars().count());
        }
        if (unk_id as usize) >= scores.len() {
            return Err(invalid(format!("{file}: model.unk_id {unk_id} is not under {} pieces", scores.len())));
        }
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);

        // The normalizer: the precompiled map, then the collapse of
        // spaces, each present exactly when the manifest says so, and
        // nothing else.
        let mut steps: Vec<&Value> = Vec::new();
        match json["normalizer"]["type"].as_str() {
            Some("Sequence") => {
                let list = json["normalizer"]["normalizers"].as_array();
                steps.extend(list.map(Vec::as_slice).unwrap_or_default());
            }
            Some(_) => steps.push(&json["normalizer"]),
            None => {}
        }
        let mut charsmap = None;
        let mut collapse_spaces = false;
        for (i, n) in steps.iter().enumerate() {
            match n["type"].as_str() {
                Some("Precompiled") if i == 0 => {
                    let b64 = n["precompiled_charsmap"].as_str().unwrap_or_default();
                    charsmap = Some(Charsmap::parse(file, b64)?);
                }
                Some("Replace") if n["pattern"]["Regex"] == " {2,}" && n["content"] == " " => collapse_spaces = true,
                _ => {
                    return Err(invalid(format!(
                        "{file}: normalizer {}: the core runs the precompiled map and the collapse of spaces",
                        n["type"]
                    )));
                }
            }
        }
        if charsmap.is_some() != m.precompiled_charsmap || collapse_spaces != m.collapse_spaces {
            return Err(disagree("normalizer"));
        }

        // The pre-tokenizer: Metaspace, prepending to every piece of text
        // when the manifest says so, and splitting at the metaspace.
        let pre = &json["pre_tokenizer"];
        if pre["type"] != "Metaspace" {
            return Err(invalid(format!("{file}: pre_tokenizer is {}, not Metaspace", pre["type"])));
        }
        if pre["replacement"] != m.metaspace.as_str() {
            return Err(disagree("pre_tokenizer.replacement"));
        }
        let prepend = match (pre.get("prepend_scheme").and_then(Value::as_str), pre.get("add_prefix_space")) {
            (Some("always"), _) => true,
            (Some("never"), _) => false,
            (Some(other), _) => {
                return Err(invalid(format!(
                    "{file}: pre_tokenizer.prepend_scheme {other:?}: the core runs always or never"
                )));
            }
            (None, Some(v)) => v.as_bool().unwrap_or(true),
            (None, None) => true,
        };
        if prepend != m.add_prefix_space {
            return Err(disagree("pre_tokenizer.add_prefix_space"));
        }
        if pre.get("split").is_some_and(|s| s == false) {
            return Err(invalid(format!("{file}: pre_tokenizer.split is false; the core cuts at the metaspace")));
        }

        Ok(Unigram {
            ids,
            scores,
            max_piece_chars,
            unk_score: min_score - UNK_PENALTY,
            unk_id,
            charsmap,
            collapse_spaces,
            metaspace: m.metaspace.chars().next().expect("validated"),
            add_prefix_space: m.add_prefix_space,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.scores.len() as u32
    }

    /// The id of `piece`, if it is one.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.ids.get(piece).copied()
    }

    /// The ids of one piece of text between special tokens.
    pub fn encode(&self, text: &str, out: &mut Vec<i32>) {
        let normalized = self.normalize(text);
        // The metaspace in place of each space, and in front.
        let mut s = String::with_capacity(normalized.len() + 3);
        if self.add_prefix_space && !normalized.starts_with(' ') && !normalized.starts_with(self.metaspace) {
            s.push(self.metaspace);
        }
        for c in normalized.chars() {
            s.push(if c == ' ' { self.metaspace } else { c });
        }
        // Cut at each metaspace, which starts the word after it.
        let mut start = 0;
        for (i, c) in s.char_indices() {
            if c == self.metaspace && i > start {
                self.segment(&s[start..i], out);
                start = i;
            }
        }
        if start < s.len() {
            self.segment(&s[start..], out);
        }
    }

    /// The precompiled map, then runs of spaces collapsed to one.
    fn normalize(&self, text: &str) -> String {
        let mapped = match &self.charsmap {
            Some(c) => c.normalize(text),
            None => text.to_owned(),
        };
        if !self.collapse_spaces {
            return mapped;
        }
        let mut out = String::with_capacity(mapped.len());
        let mut last_space = false;
        for c in mapped.chars() {
            if c == ' ' && last_space {
                continue;
            }
            last_space = c == ' ';
            out.push(c);
        }
        out
    }

    /// One word's pieces of highest total score: the best path through
    /// the lattice of every piece at every position, as upstream's
    /// optimized encode walks it, with the unknown token for a character
    /// no piece starts at, and runs of unknown characters fused into one.
    fn segment(&self, word: &str, out: &mut Vec<i32>) {
        #[derive(Clone, Copy)]
        struct Best {
            score: f64,
            start: usize,
            id: u32,
            set: bool,
        }
        let n = word.len();
        let mut best = vec![Best { score: 0.0, start: 0, id: 0, set: false }; n + 1];
        let mut start = 0;
        while start < n {
            let base = best[start].score;
            let mblen = word[start..].chars().next().map_or(1, char::len_utf8);
            let mut has_single = false;
            let mut update = |end: usize, score: f64, id: u32| {
                let b = &mut best[end];
                if !b.set || score > b.score {
                    *b = Best { score, start, id, set: true };
                }
            };
            for (chars, (end, c)) in word[start..].char_indices().enumerate() {
                if chars >= self.max_piece_chars {
                    break;
                }
                let end = start + end + c.len_utf8();
                if let Some(&id) = self.ids.get(&word[start..end]) {
                    update(end, base + self.scores[id as usize], id);
                    if end - start == mblen {
                        has_single = true;
                    }
                }
            }
            if !has_single {
                update(start + mblen, base + self.unk_score, self.unk_id);
            }
            start += mblen;
        }
        let first = out.len();
        let mut end = n;
        let mut unk_run = false;
        while end > 0 {
            let b = best[end];
            if b.id == self.unk_id {
                if !unk_run {
                    out.push(self.unk_id as i32);
                }
                unk_run = true;
            } else {
                out.push(b.id as i32);
                unk_run = false;
            }
            end = b.start;
        }
        out[first..].reverse();
    }
}

/// SentencePiece's precompiled character map: a double-array trie over
/// UTF-8 bytes whose leaves index a string of replacements, each ended
/// by a NUL. Applied as upstream does: to each grapheme cluster under six
/// bytes as a whole, else to its characters one by one.
struct Charsmap {
    trie: Vec<u32>,
    normalized: Vec<u8>,
}

impl Charsmap {
    fn parse(file: &str, b64: &str) -> Result<Charsmap> {
        let bad = |why: &str| invalid(format!("{file}: normalizer.precompiled_charsmap: {why}"));
        let bytes = base64(b64).ok_or_else(|| bad("not base64"))?;
        if bytes.len() < 4 {
            return Err(bad("shorter than its length field"));
        }
        let trie_bytes = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
        if !trie_bytes.is_multiple_of(4) || 4 + trie_bytes > bytes.len() {
            return Err(bad("trie length is not inside the map"));
        }
        let trie: Vec<u32> =
            bytes[4..4 + trie_bytes].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
        if trie.is_empty() {
            return Err(bad("empty trie"));
        }
        let normalized = bytes[4 + trie_bytes..].to_vec();
        Ok(Charsmap { trie, normalized })
    }

    /// The replacement for the longest prefix of `key` in the map: the
    /// first leaf met on the way, as upstream takes it.
    fn transform(&self, key: &[u8]) -> Option<&[u8]> {
        let unit = |pos: usize| self.trie.get(pos).copied();
        let offset = |u: u32| ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize;
        let mut pos = 0usize;
        let mut u = unit(pos)?;
        pos ^= offset(u);
        for &c in key {
            if c == 0 {
                break;
            }
            pos ^= c as usize;
            u = unit(pos)?;
            if u & ((1 << 31) | 0xFF) != c as u32 {
                return None;
            }
            pos ^= offset(u);
            if (u >> 8) & 1 == 1 {
                let at = (unit(pos)? & ((1 << 31) - 1)) as usize;
                let rest = self.normalized.get(at..)?;
                let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                return Some(&rest[..end]);
            }
        }
        None
    }

    fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let push = |out: &mut String, bytes: &[u8]| out.push_str(&String::from_utf8_lossy(bytes));
        for grapheme in text.graphemes(true) {
            if grapheme.len() < 6
                && let Some(n) = self.transform(grapheme.as_bytes())
            {
                push(&mut out, n);
                continue;
            }
            for c in grapheme.chars() {
                let mut buf = [0u8; 4];
                match self.transform(c.encode_utf8(&mut buf).as_bytes()) {
                    Some(n) => push(&mut out, n),
                    None => out.push(c),
                }
            }
        }
        out
    }
}

/// Standard base64 with padding, as the tokenizer file carries the map.
fn base64(s: &str) -> Option<Vec<u8>> {
    let value = |c: u8| -> Option<u32> {
        Some(match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a') as u32 + 26,
            b'0'..=b'9' => (c - b'0') as u32 + 52,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        })
    };
    let bytes: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3);
    for chunk in bytes.chunks(4) {
        let pad = chunk.iter().filter(|&&b| b == b'=').count();
        if chunk.len() != 4 || pad > 2 || chunk[..4 - pad].contains(&b'=') {
            return None;
        }
        let mut v = 0u32;
        for &b in &chunk[..4 - pad] {
            v = (v << 6) | value(b)?;
        }
        v <<= 6 * pad as u32;
        out.push((v >> 16) as u8);
        if pad < 2 {
            out.push((v >> 8) as u8);
        }
        if pad < 1 {
            out.push(v as u8);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_with_and_without_padding_bytes() {
        assert_eq!(base64("").unwrap(), b"");
        assert_eq!(base64("YQ==").unwrap(), b"a");
        assert_eq!(base64("YWI=").unwrap(), b"ab");
        assert_eq!(base64("YWJj").unwrap(), b"abc");
        assert_eq!(base64("YWJjZA==").unwrap(), b"abcd");
        assert!(base64("YQ=").is_none());
        assert!(base64("Y!==").is_none());
    }
}
