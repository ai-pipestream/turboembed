//! SentencePiece's Unigram model as upstream `tokenizers` runs it, from
//! the upstream tokenizer.json: the precompiled character map it carries
//! (SentencePiece's own normalizer, `nmt_nfkc` for XLM-RoBERTa), the
//! collapse of repeated spaces, the metaspace in place of each space and
//! in front of each piece of text, the cut into words at each metaspace,
//! and each word's segmentation into the pieces of highest total score.
//! A character no piece covers is the unknown token, and a run of them
//! is one unknown token, as upstream fuses them.

use serde_json::Value;
use unicode_segmentation::UnicodeSegmentation;

use crate::manifest;
use crate::status::{Result, invalid};

/// Upstream's penalty under the lowest score for a character no piece
/// covers.
const UNK_PENALTY: f64 = 10.0;

pub struct Unigram {
    /// Each piece, by id.
    pieces: Vec<Box<str>>,
    /// Every piece, for the lattice's walk from each position.
    trie: Trie,
    /// Each piece's log probability, by id.
    scores: Vec<f64>,
    unk_score: f64,
    unk_id: u32,
    charsmap: Option<Charsmap>,
    collapse_spaces: bool,
    space_punctuation: bool,
    collapse_whitespace: bool,
    strip: bool,
    whole_text: bool,
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
        let mut pieces = Vec::with_capacity(raw.len());
        let mut scores = Vec::with_capacity(raw.len());
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
            pieces.push(Box::<str>::from(piece));
            scores.push(score);
        }
        if (unk_id as usize) >= scores.len() {
            return Err(invalid(format!("{file}: model.unk_id {unk_id} is not under {} pieces", scores.len())));
        }
        let min_score = scores.iter().copied().fold(f64::INFINITY, f64::min);

        // The normalizer, its sequences flattened: the precompiled map,
        // the collapse of spaces, the spacing of punctuation, the collapse
        // of whitespace and the strip, each in that order exactly when the
        // manifest says so, and nothing else.
        fn flatten<'a>(n: &'a Value, out: &mut Vec<&'a Value>) {
            match n["type"].as_str() {
                Some("Sequence") => {
                    for step in n["normalizers"].as_array().map(Vec::as_slice).unwrap_or_default() {
                        flatten(step, out);
                    }
                }
                Some(_) => out.push(n),
                None => {}
            }
        }
        let mut steps: Vec<&Value> = Vec::new();
        flatten(&json["normalizer"], &mut steps);
        let mut charsmap = None;
        let (mut collapse_spaces, mut collapse_whitespace, mut strip) = (false, false, false);
        let mut punctuation = String::new();
        // Each step's place in the order; a step may not come before one
        // already seen.
        let mut stage = 0;
        for (i, n) in steps.iter().enumerate() {
            let replace = |pattern: &str, regex: bool| {
                n["type"] == "Replace" && n["pattern"][if regex { "Regex" } else { "String" }] == pattern
            };
            let at = match n["type"].as_str() {
                Some("Precompiled") if i == 0 => {
                    let b64 = n["precompiled_charsmap"].as_str().unwrap_or_default();
                    charsmap = Some(Charsmap::parse(file, b64)?);
                    0
                }
                Some("Replace") if replace(" {2,}", true) && n["content"] == " " && !collapse_spaces => {
                    collapse_spaces = true;
                    1
                }
                Some("Replace") if punctuation_step(n).is_some_and(|c| !punctuation.contains(c)) => {
                    punctuation.push(punctuation_step(n).expect("matched"));
                    2
                }
                Some("Replace") if replace("\\s+", true) && n["content"] == " " && !collapse_whitespace => {
                    collapse_whitespace = true;
                    3
                }
                Some("Strip") if n["strip_left"] == true && n["strip_right"] == true && !strip => {
                    strip = true;
                    4
                }
                _ => {
                    return Err(invalid(format!(
                        "{file}: normalizer step {i}, {}: the core runs the precompiled map, the collapse of \
                         spaces, the spacing of punctuation, the collapse of whitespace and the strip, in that order",
                        n["type"]
                    )));
                }
            };
            if at < stage {
                return Err(invalid(format!("{file}: normalizer step {i} is out of the core's order")));
            }
            stage = at;
        }
        let space_punctuation = !punctuation.is_empty();
        if space_punctuation && punctuation.len() != PUNCTUATION.len() {
            return Err(invalid(format!(
                "{file}: normalizer: punctuation spaced is {punctuation:?}, not every ASCII punctuation character"
            )));
        }
        if charsmap.is_some() != m.precompiled_charsmap
            || collapse_spaces != m.collapse_spaces
            || space_punctuation != m.space_punctuation
            || collapse_whitespace != m.collapse_whitespace
            || strip != m.strip
        {
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
        if pre.get("split").is_some_and(|s| s == false) != m.whole_text {
            return Err(disagree("pre_tokenizer.split"));
        }

        let trie =
            Trie::new(&pieces).map_err(|p| invalid(format!("{file}: model.vocab: piece {p:?} is listed twice")))?;
        Ok(Unigram {
            pieces,
            trie,
            scores,
            unk_score: min_score - UNK_PENALTY,
            unk_id,
            charsmap,
            collapse_spaces,
            space_punctuation,
            collapse_whitespace,
            strip,
            whole_text: m.whole_text,
            metaspace: m.metaspace.chars().next().expect("validated"),
            add_prefix_space: m.add_prefix_space,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.scores.len() as u32
    }

    /// Every piece of the vocabulary.
    pub fn pieces(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().map(|p| &**p)
    }

    /// The id of `piece`, if it is one.
    pub fn id(&self, piece: &str) -> Option<u32> {
        self.trie.get(piece.as_bytes())
    }

    /// The ids of one piece of text between special tokens.
    pub fn encode(&self, text: &str, out: &mut Vec<i32>) {
        let normalized = self.normalize(text);
        if normalized.is_empty() {
            // Upstream prepends nothing to an empty text.
            return;
        }
        // The metaspace in place of each space, and in front.
        let mut s = String::with_capacity(normalized.len() + 3);
        if self.add_prefix_space && !normalized.starts_with(' ') && !normalized.starts_with(self.metaspace) {
            s.push(self.metaspace);
        }
        for c in normalized.chars() {
            s.push(if c == ' ' { self.metaspace } else { c });
        }
        let mut best = Vec::new();
        if self.whole_text {
            self.segment(&s, &mut best, out);
            return;
        }
        // Cut at each metaspace, which starts the word after it.
        let mut start = 0;
        for (i, c) in s.char_indices() {
            if c == self.metaspace && i > start {
                self.segment(&s[start..i], &mut best, out);
                start = i;
            }
        }
        if start < s.len() {
            self.segment(&s[start..], &mut best, out);
        }
    }

    /// The precompiled map, then runs of spaces collapsed to one.
    fn normalize(&self, text: &str) -> String {
        let mapped = match &self.charsmap {
            Some(c) => c.normalize(text),
            None => text.to_owned(),
        };
        let mut out = if self.collapse_spaces {
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
        } else {
            mapped
        };
        if self.space_punctuation {
            let mut spaced = String::with_capacity(out.len() + out.len() / 4);
            for c in out.chars() {
                if c.is_ascii_punctuation() {
                    spaced.push(' ');
                    spaced.push(c);
                    spaced.push(' ');
                } else {
                    spaced.push(c);
                }
            }
            out = spaced;
        }
        if self.collapse_whitespace {
            // Regex \s+ as upstream's Oniguruma reads it in UTF-8: the
            // Unicode White_Space characters, which char::is_whitespace is.
            let mut folded = String::with_capacity(out.len());
            let mut in_run = false;
            for c in out.chars() {
                if c.is_whitespace() {
                    if !in_run {
                        folded.push(' ');
                    }
                    in_run = true;
                } else {
                    folded.push(c);
                    in_run = false;
                }
            }
            out = folded;
        }
        if self.strip {
            let t = out.trim();
            if t.len() != out.len() {
                out = t.to_owned();
            }
        }
        out
    }

    /// One word's pieces of highest total score: the best path through
    /// the lattice of every piece at every position, as upstream's
    /// optimized encode walks it, with the unknown token for a character
    /// no piece starts at, and runs of unknown characters fused into one.
    /// `best` is room for the lattice, kept between words.
    fn segment(&self, word: &str, best: &mut Vec<Best>, out: &mut Vec<i32>) {
        let n = word.len();
        best.clear();
        best.resize(n + 1, Best { score: 0.0, start: 0, id: 0, set: false });
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
            // Every piece the word continues with here, shortest first:
            // one walk down the trie, which ends where no piece goes on.
            self.trie.prefixes(&word.as_bytes()[start..], |len, id| {
                update(start + len, base + self.scores[id as usize], id);
                if len == mblen {
                    has_single = true;
                }
            });
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

/// The best way found to a byte of a word.
#[derive(Clone, Copy)]
struct Best {
    score: f64,
    start: usize,
    id: u32,
    set: bool,
}

/// The pieces as a trie over their bytes: each node's children are one
/// run of `nodes`, in byte order; a walk scans a node's few children, or
/// reads a table of its many.
struct Trie {
    nodes: Vec<Node>,
    /// Each node's `byte`, apart, so a search through a node's children
    /// reads one byte each.
    bytes: Vec<u8>,
    /// For a node of many children, each byte's child as its place among
    /// them, or u16::MAX: no search there.
    wide: Vec<[u16; 256]>,
    /// The root's child for each first byte, or NONE: no search there,
    /// where every walk starts.
    root: Box<[u32; 256]>,
}

#[derive(Clone, Copy)]
struct Node {
    /// The byte on the edge into this node.
    byte: u8,
    /// The piece that ends here, or NONE.
    id: u32,
    /// The children: nodes[first..first + count].
    first: u32,
    count: u32,
    /// The node's table in `wide`, or NONE.
    wide: u32,
}

impl Trie {
    const NONE: u32 = u32::MAX;

    /// The trie of `pieces`, each the id of its place; Err is a piece
    /// listed twice.
    fn new(pieces: &[Box<str>]) -> std::result::Result<Trie, String> {
        let mut pieces: Vec<(&[u8], u32)> =
            pieces.iter().enumerate().map(|(id, p)| (p.as_bytes(), id as u32)).collect();
        pieces.sort_unstable();
        if let Some(w) = pieces.windows(2).find(|w| w[0].0 == w[1].0) {
            return Err(String::from_utf8_lossy(w[0].0).into_owned());
        }
        let mut nodes = vec![Node { byte: 0, id: Self::NONE, first: 0, count: 0, wide: Self::NONE }];
        // Depth first, a node's children appended together when it is
        // reached, so that a walk down the trie stays near where it
        // started: (node, the pieces under it, its depth).
        let mut stack = vec![(0usize, 0usize, pieces.len(), 0usize)];
        while let Some((node, mut lo, hi, depth)) = stack.pop() {
            if lo < hi && pieces[lo].0.len() == depth {
                nodes[node].id = pieces[lo].1;
                lo += 1;
            }
            nodes[node].first = nodes.len() as u32;
            let pushed = stack.len();
            while lo < hi {
                let byte = pieces[lo].0[depth];
                let end = lo + pieces[lo..hi].partition_point(|p| p.0[depth] == byte);
                stack.push((nodes.len(), lo, end, depth + 1));
                nodes.push(Node { byte, id: Self::NONE, first: 0, count: 0, wide: Self::NONE });
                lo = end;
            }
            nodes[node].count = nodes.len() as u32 - nodes[node].first;
            // The first child next.
            stack[pushed..].reverse();
        }
        let mut root = Box::new([Self::NONE; 256]);
        let r = nodes[0];
        for k in r.first..r.first + r.count {
            root[nodes[k as usize].byte as usize] = k;
        }
        let bytes = nodes.iter().map(|n| n.byte).collect();
        let mut wide = Vec::new();
        for i in 0..nodes.len() {
            let n = nodes[i];
            if n.count as usize > Self::NARROW {
                let mut t = [u16::MAX; 256];
                for k in 0..n.count {
                    t[nodes[(n.first + k) as usize].byte as usize] = k as u16;
                }
                nodes[i].wide = wide.len() as u32;
                wide.push(t);
            }
        }
        Ok(Trie { nodes, bytes, wide, root })
    }

    /// The id of the piece that is exactly `piece`.
    fn get(&self, piece: &[u8]) -> Option<u32> {
        let (&b0, rest) = piece.split_first()?;
        let mut node = *self.nodes.get(self.root[b0 as usize] as usize)?;
        for &b in rest {
            node = self.nodes[self.child(node, b)?];
        }
        (node.id != Self::NONE).then_some(node.id)
    }

    /// `found(len, id)` for every piece that `text` starts with, shortest
    /// first.
    #[inline]
    fn prefixes(&self, text: &[u8], mut found: impl FnMut(usize, u32)) {
        let Some(&b0) = text.first() else { return };
        let Some(&first) = self.nodes.get(self.root[b0 as usize] as usize) else { return };
        if first.id != Self::NONE {
            found(1, first.id);
        }
        let mut node = first;
        for (i, &b) in text.iter().enumerate().skip(1) {
            let Some(k) = self.child(node, b) else { return };
            node = self.nodes[k];
            if node.id != Self::NONE {
                found(i + 1, node.id);
            }
        }
    }

    /// The most children a node has with no table of them.
    const NARROW: usize = 8;

    /// The index of `node`'s child on byte `b`: a scan of a few children,
    /// a table of many.
    #[inline]
    fn child(&self, node: Node, b: u8) -> Option<usize> {
        if node.wide != Self::NONE {
            let k = self.wide[node.wide as usize][b as usize];
            return (k != u16::MAX).then_some(node.first as usize + k as usize);
        }
        let first = node.first as usize;
        let kids = &self.bytes[first..first + node.count as usize];
        kids.iter().position(|&k| k == b).map(|k| first + k)
    }
}

/// The ASCII punctuation characters, which `space_punctuation` spaces.
const PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// The character a Replace step spaces, when it is one of PUNCTUATION
/// replaced by itself between two spaces.
fn punctuation_step(n: &Value) -> Option<char> {
    let p = n["pattern"]["String"].as_str()?;
    let mut chars = p.chars();
    let c = chars.next().filter(|c| c.is_ascii_punctuation() && chars.next().is_none())?;
    (n["content"] == format!(" {c} ")).then_some(c)
}

/// SentencePiece's precompiled character map: a double-array trie over
/// UTF-8 bytes whose leaves index a string of replacements, each ended
/// by a NUL. Applied as upstream does: to each grapheme cluster under six
/// bytes as a whole, else to its characters one by one.
struct Charsmap {
    trie: Vec<u32>,
    normalized: Vec<u8>,
    /// What the map makes of each ASCII character alone.
    ascii: Vec<Option<Box<str>>>,
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
        let mut map = Charsmap { trie, normalized, ascii: Vec::new() };
        map.ascii = (0..128u8)
            .map(|b| map.transform(&[b]).map(|n| String::from_utf8_lossy(n).into_owned().into_boxed_str()))
            .collect();
        Ok(map)
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
        let b = text.as_bytes();
        let mut i = 0;
        while i < b.len() {
            // An ASCII character followed by another (CR LF aside) is a
            // grapheme cluster of its own: the table's.
            if b[i].is_ascii() && b.get(i + 1).is_none_or(|n| n.is_ascii() && !(b[i] == b'\r' && *n == b'\n')) {
                match &self.ascii[b[i] as usize] {
                    Some(n) => out.push_str(n),
                    None => out.push(b[i] as char),
                }
                i += 1;
                continue;
            }
            // Otherwise, up to and including the next ASCII character that
            // another ASCII character follows (a cluster boundary), as
            // clusters.
            let mut end = i + 1;
            while end < b.len()
                && !(b[end - 1].is_ascii() && b[end].is_ascii() && !(b[end - 1] == b'\r' && b[end] == b'\n'))
            {
                end += 1;
            }
            self.clusters(&text[i..end], &mut out);
            i = end;
        }
        out
    }

    /// `text`'s grapheme clusters through the map.
    fn clusters(&self, text: &str, out: &mut String) {
        let push = |out: &mut String, bytes: &[u8]| out.push_str(&String::from_utf8_lossy(bytes));
        for grapheme in text.graphemes(true) {
            if grapheme.len() < 6
                && let Some(n) = self.transform(grapheme.as_bytes())
            {
                push(out, n);
                continue;
            }
            for c in grapheme.chars() {
                let mut buf = [0u8; 4];
                match self.transform(c.encode_utf8(&mut buf).as_bytes()) {
                    Some(n) => push(out, n),
                    None => out.push(c),
                }
            }
        }
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

    /// The trie finds every piece of XLM-RoBERTa's vocabulary (bge-m3)
    /// by its bytes, and every piece a text starts with, shortest first,
    /// as a search of the whole list does: at nodes with a table of their
    /// children and nodes without.
    #[test]
    fn the_trie_finds_what_the_list_holds() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
        let json: Value = serde_json::from_slice(&std::fs::read(root.join("bge-m3/tokenizer.json")).unwrap()).unwrap();
        let pieces: Vec<Box<str>> =
            json["model"]["vocab"].as_array().unwrap().iter().map(|p| p[0].as_str().unwrap().into()).collect();
        let trie = Trie::new(&pieces).unwrap();
        assert!(trie.wide.len() > 100 && trie.nodes.iter().any(|n| n.count > 1 && n.wide == Trie::NONE));
        for (id, p) in pieces.iter().enumerate() {
            assert_eq!(trie.get(p.as_bytes()), Some(id as u32), "{p:?}");
        }
        let file = std::fs::read_to_string(root.join("tokenizer-texts.jsonl")).unwrap();
        for line in file.lines() {
            let text = serde_json::from_str::<Value>(line).unwrap()["text"].as_str().unwrap().replace(' ', "\u{2581}");
            for (at, _) in text.char_indices() {
                let rest = &text.as_bytes()[at..];
                let mut got = Vec::new();
                trie.prefixes(rest, |len, id| got.push((len, id)));
                let mut want: Vec<(usize, u32)> = pieces
                    .iter()
                    .enumerate()
                    .filter(|(_, p)| rest.starts_with(p.as_bytes()))
                    .map(|(id, p)| (p.len(), id as u32))
                    .collect();
                want.sort();
                assert_eq!(got, want, "{:?}", &text[at..]);
            }
        }
    }

    /// The ASCII shortcut gives what the map gives cluster by cluster:
    /// on XLM-RoBERTa's map (bge-m3), for the tokenizer texts and for
    /// ASCII next to what joins a cluster.
    #[test]
    fn the_ascii_shortcut_is_the_cluster_by_cluster_map() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata");
        let json: Value = serde_json::from_slice(&std::fs::read(root.join("bge-m3/tokenizer.json")).unwrap()).unwrap();
        let mut norms = Vec::new();
        fn find<'a>(n: &'a Value, out: &mut Vec<&'a Value>) {
            if n["type"] == "Precompiled" {
                out.push(n);
            }
            for c in n["normalizers"].as_array().into_iter().flatten() {
                find(c, out);
            }
        }
        find(&json["normalizer"], &mut norms);
        let map = Charsmap::parse("bge-m3", norms[0]["precompiled_charsmap"].as_str().unwrap()).unwrap();
        let file = std::fs::read_to_string(root.join("tokenizer-texts.jsonl")).unwrap();
        let mut texts: Vec<String> = file
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["text"].as_str().unwrap().to_owned())
            .collect();
        texts.extend(
            [
                "a\r\nb",
                "\r\n",
                "x\r",
                "e\u{301}a",
                "ae\u{301}",
                "\u{600}1a",
                "a\u{600}1",
                "\u{0}\u{1}a\u{7f}",
                "ＡＢＣ abc",
                "ﬁ ligature ㎏",
                "a\u{200d}b",
                "🇫🇷x",
                "x🇫🇷",
                "e\u{301}\u{302}\u{303}\u{304}xyz",
            ]
            .map(String::from),
        );
        for t in &texts {
            let mut want = String::new();
            map.clusters(t, &mut want);
            assert_eq!(map.normalize(t), want, "{t:?}");
        }
    }
}
