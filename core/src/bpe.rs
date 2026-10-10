//! Byte-level BPE as upstream `tokenizers` runs it for GPT-2 and
//! RoBERTa, from the upstream tokenizer.json: the text is cut into words
//! by GPT-2's pattern (a contraction, or an optional space and a run of
//! letters, of numbers or of other characters, or whitespace), each
//! word's bytes are written as GPT-2's printable characters, and each
//! word is merged pair by pair, the pair of lowest rank first and the
//! leftmost of equal ranks, until no listed pair is left.

use std::collections::{BinaryHeap, HashMap};
use std::hash::BuildHasherDefault;

use serde_json::Value;

use crate::manifest;
use crate::status::{Result, invalid};
use crate::tokenizer::{Fx, LETTER, NUMBER, Sink, Span, category, trim_span};

type Merges = HashMap<u64, (u32, u32), BuildHasherDefault<Fx>>;

pub struct Bpe {
    /// Each piece, by id.
    pieces: Vec<Box<str>>,
    vocab: HashMap<Box<str>, u32, BuildHasherDefault<Fx>>,
    /// (left id, right id) to (rank, merged id).
    merges: Merges,
    /// The id of each byte's character, or NONE when the vocabulary has
    /// none.
    byte_ids: Box<[u32; 256]>,
    unk_id: u32,
    add_prefix_space: bool,
    ignore_merges: bool,
}

const NONE: u32 = u32::MAX;

impl Bpe {
    /// The model from the tokenizer file, checked against what the
    /// manifest says of it. `unk_id` is the manifest's SPECIAL_UNK id.
    pub fn parse(file: &str, json: &Value, m: &manifest::Bpe, unk_id: u32) -> Result<Bpe> {
        let disagree = |what: &str| invalid(format!("{file}: {what} is not what manifest.json says"));
        let model = &json["model"];
        if model["type"] != "BPE" {
            return Err(invalid(format!("{file}: model.type is {}, not BPE", model["type"])));
        }
        for (key, why) in [
            ("dropout", "the core runs no dropout"),
            ("continuing_subword_prefix", "byte-level BPE has no continuing prefix"),
            ("end_of_word_suffix", "byte-level BPE has no end-of-word suffix"),
        ] {
            let v = &model[key];
            if !(v.is_null() || v == "") {
                return Err(invalid(format!("{file}: model.{key} is {v}: {why}")));
            }
        }
        for key in ["byte_fallback", "fuse_unk"] {
            if model[key].as_bool().unwrap_or(false) {
                return Err(invalid(format!("{file}: model.{key}: the core runs byte-level BPE without it")));
            }
        }
        if model["ignore_merges"].as_bool().unwrap_or(false) != m.ignore_merges {
            return Err(disagree("model.ignore_merges"));
        }
        if !json["normalizer"].is_null() {
            return Err(invalid(format!(
                "{file}: normalizer is {}: the core runs byte-level BPE on the text as it is",
                json["normalizer"]["type"]
            )));
        }
        let pre = &json["pre_tokenizer"];
        if pre["type"] != "ByteLevel" || !pre["use_regex"].as_bool().unwrap_or(true) {
            return Err(invalid(format!(
                "{file}: pre_tokenizer is {}: the core runs ByteLevel with GPT-2's pattern",
                pre["type"]
            )));
        }
        if pre["add_prefix_space"].as_bool().unwrap_or(true) != m.add_prefix_space {
            return Err(disagree("pre_tokenizer.add_prefix_space"));
        }

        let raw = model["vocab"].as_object().ok_or_else(|| invalid(format!("{file}: model.vocab is not an object")))?;
        let mut pieces = vec![None; raw.len()];
        let mut vocab = HashMap::with_capacity_and_hasher(raw.len(), Default::default());
        for (piece, id) in raw {
            let Some(id) = id.as_u64().filter(|&i| (i as usize) < raw.len()) else {
                return Err(invalid(format!("{file}: model.vocab[{piece:?}] is not an id under {}", raw.len())));
            };
            if pieces[id as usize].replace(Box::<str>::from(piece.as_str())).is_some() {
                return Err(invalid(format!("{file}: model.vocab: id {id} is used twice")));
            }
            vocab.insert(Box::<str>::from(piece.as_str()), id as u32);
        }
        let pieces: Vec<Box<str>> = pieces.into_iter().map(|p| p.expect("every id under the count, once")).collect();

        let list =
            model["merges"].as_array().ok_or_else(|| invalid(format!("{file}: model.merges is not an array")))?;
        let mut merges = Merges::with_capacity_and_hasher(list.len(), Default::default());
        for (rank, merge) in list.iter().enumerate() {
            // "left right", or ["left", "right"].
            let pair = match merge {
                Value::String(s) => s.split_once(' '),
                Value::Array(a) => match a.as_slice() {
                    [Value::String(l), Value::String(r)] => Some((l.as_str(), r.as_str())),
                    _ => None,
                },
                _ => None,
            };
            let Some((l, r)) = pair else {
                return Err(invalid(format!("{file}: model.merges[{rank}] is not a pair of pieces")));
            };
            let id = |p: &str| {
                vocab.get(p).copied().ok_or_else(|| invalid(format!("{file}: model.merges[{rank}]: no piece {p:?}")))
            };
            let merged = id(&format!("{l}{r}"))?;
            merges.entry(key(id(l)?, id(r)?)).or_insert((rank as u32, merged));
        }

        let chars = byte_chars();
        let mut byte_ids = Box::new([NONE; 256]);
        for (b, id) in byte_ids.iter_mut().enumerate() {
            let mut buf = [0u8; 4];
            *id = vocab.get(&*chars[b].encode_utf8(&mut buf)).copied().unwrap_or(NONE);
        }
        Ok(Bpe {
            pieces,
            vocab,
            merges,
            byte_ids,
            unk_id,
            add_prefix_space: m.add_prefix_space,
            ignore_merges: m.ignore_merges,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.pieces.len() as u32
    }

    /// Every piece of the vocabulary, by id.
    pub fn pieces(&self) -> impl Iterator<Item = &str> {
        self.pieces.iter().map(|p| &**p)
    }

    pub fn id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied()
    }

    /// The ids of `piece`, the bytes of `text` from `base` on with no
    /// special token in them.
    pub(crate) fn encode<S: Sink>(&self, text: &str, base: usize, piece: &str, out: &mut S) {
        // The space put in front comes from the character it is put
        // before; each byte from the character it is part of.
        let prefixed = self.add_prefix_space && !piece.starts_with(' ');
        let span = |from: usize, to: usize| {
            let (mut a, mut b) = (from.saturating_sub(prefixed as usize), to - prefixed as usize);
            if prefixed && to == 1 {
                b = piece.chars().next().map_or(0, char::len_utf8);
            }
            while !piece.is_char_boundary(a) {
                a -= 1;
            }
            while !piece.is_char_boundary(b) {
                b += 1;
            }
            trim_span(text, [(base + a) as u32, (base + b) as u32])
        };
        let mut word = Word::default();
        let mut each = |from: usize, bytes: &[u8]| self.word(bytes, from, &span, &mut word, out);
        if prefixed {
            let mut s = Vec::with_capacity(piece.len() + 1);
            s.push(b' ');
            s.extend_from_slice(piece.as_bytes());
            // A space then the text's first character: the pattern reads
            // the two as it would in the text.
            let with = String::from_utf8(s).expect("a space and UTF-8");
            split(&with, |a, b| each(a, &with.as_bytes()[a..b]));
        } else {
            split(piece, |a, b| each(a, &piece.as_bytes()[a..b]));
        }
    }

    /// One word's pieces: the whole word when `ignore_merges` and it is a
    /// piece, else its bytes merged. `from` is where the word starts in
    /// what `span` reads.
    fn word<S: Sink>(
        &self,
        bytes: &[u8],
        from: usize,
        span: &impl Fn(usize, usize) -> Span,
        w: &mut Word,
        out: &mut S,
    ) {
        if self.ignore_merges {
            let chars = byte_chars();
            let mapped: String = bytes.iter().map(|&b| chars[b as usize]).collect();
            if let Some(&id) = self.vocab.get(mapped.as_str()) {
                out.push(id as i32, || span(from, from + bytes.len()));
                return;
            }
        }
        w.symbols.clear();
        for (i, &b) in bytes.iter().enumerate() {
            let id = self.byte_ids[b as usize];
            w.symbols.push(Symbol {
                id: if id == NONE { self.unk_id } else { id },
                start: i as u32,
                end: i as u32 + 1,
            });
        }
        w.merge(&self.merges);
        for s in &w.symbols {
            out.push(s.id as i32, || span(from + s.start as usize, from + s.end as usize));
        }
    }
}

fn key(l: u32, r: u32) -> u64 {
    ((l as u64) << 32) | r as u64
}

/// GPT-2's characters for bytes: the printable bytes as themselves, the
/// others as the characters from U+0100 on, in byte order.
pub(crate) fn byte_chars() -> &'static [char; 256] {
    static CHARS: std::sync::OnceLock<[char; 256]> = std::sync::OnceLock::new();
    CHARS.get_or_init(|| {
        let mut chars = ['\0'; 256];
        let mut n = 0;
        for (b, c) in chars.iter_mut().enumerate() {
            let printable = matches!(b, 0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff);
            *c = if printable {
                char::from(b as u8)
            } else {
                n += 1;
                char::from_u32(0xff + n).expect("under U+0200")
            };
        }
        chars
    })
}

/// The byte a GPT-2 character stands for, if it is one.
pub(crate) fn char_byte(c: char) -> Option<u8> {
    let u = c as u32;
    match u {
        0x21..=0x7e | 0xa1..=0xac | 0xae..=0xff => Some(u as u8),
        0x100..=0x143 => {
            // The bytes that are not printable, in byte order.
            static OTHERS: std::sync::OnceLock<[u8; 68]> = std::sync::OnceLock::new();
            let others = OTHERS.get_or_init(|| {
                let mut o = [0u8; 68];
                for (b, &c) in byte_chars().iter().enumerate() {
                    if let Some(at) = (c as u32).checked_sub(0x100) {
                        o[at as usize] = b as u8;
                    }
                }
                o
            });
            Some(others[(u - 0x100) as usize])
        }
        _ => None,
    }
}

/// A word being merged: its symbols in order, each its id and the bytes
/// of the word it covers.
#[derive(Default)]
struct Word {
    symbols: Vec<Symbol>,
    heap: BinaryHeap<Merge>,
    next: Vec<u32>,
    prev: Vec<u32>,
}

#[derive(Clone, Copy)]
struct Symbol {
    id: u32,
    start: u32,
    end: u32,
}

/// A pair to merge, at the left symbol's first place: popped lowest rank
/// first, then leftmost.
#[derive(PartialEq, Eq)]
struct Merge {
    rank: u32,
    at: u32,
    id: u32,
}

impl Ord for Merge {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.rank.cmp(&self.rank).then(o.at.cmp(&self.at))
    }
}

impl PartialOrd for Merge {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

/// The most symbols a word is merged by scanning its pairs; a longer one
/// keeps its pairs in a heap.
const SCANNED: usize = 48;

impl Word {
    /// Merge until no pair is listed: each time the listed pair of lowest
    /// rank, the leftmost of equal ranks.
    fn merge(&mut self, merges: &Merges) {
        if self.symbols.len() <= SCANNED {
            self.merge_scanning(merges);
        } else {
            self.merge_heap(merges);
        }
    }

    fn merge_scanning(&mut self, merges: &Merges) {
        let s = &mut self.symbols;
        while s.len() > 1 {
            let mut best: Option<(u32, usize, u32)> = None;
            for i in 0..s.len() - 1 {
                if let Some(&(rank, id)) = merges.get(&key(s[i].id, s[i + 1].id))
                    && best.is_none_or(|b| rank < b.0)
                {
                    best = Some((rank, i, id));
                }
            }
            let Some((_, i, id)) = best else { break };
            s[i] = Symbol { id, start: s[i].start, end: s[i + 1].end };
            s.remove(i + 1);
        }
    }

    /// As upstream merges: the symbols as a list, the pairs in a heap,
    /// a popped pair skipped when either side has changed since.
    fn merge_heap(&mut self, merges: &Merges) {
        let n = self.symbols.len();
        let none = u32::MAX;
        self.next.clear();
        self.prev.clear();
        self.next.extend((1..=n as u32).map(|i| if i as usize == n { none } else { i }));
        self.prev.extend((0..n as u32).map(|i| if i == 0 { none } else { i - 1 }));
        self.heap.clear();
        for i in 0..n - 1 {
            if let Some(&(rank, id)) = merges.get(&key(self.symbols[i].id, self.symbols[i + 1].id)) {
                self.heap.push(Merge { rank, at: i as u32, id });
            }
        }
        // A removed symbol's end is 0.
        while let Some(m) = self.heap.pop() {
            let at = m.at as usize;
            let next = self.next[at];
            if self.symbols[at].end == 0 || next == none {
                continue;
            }
            let right = self.symbols[next as usize];
            if merges.get(&key(self.symbols[at].id, right.id)).is_none_or(|&(_, id)| id != m.id) {
                continue;
            }
            self.symbols[at] = Symbol { id: m.id, start: self.symbols[at].start, end: right.end };
            self.symbols[next as usize].end = 0;
            let after = self.next[next as usize];
            self.next[at] = after;
            if after != none {
                self.prev[after as usize] = at as u32;
            }
            let p = self.prev[at];
            if p != none
                && let Some(&(rank, id)) = merges.get(&key(self.symbols[p as usize].id, m.id))
            {
                self.heap.push(Merge { rank, at: p, id });
            }
            if after != none
                && let Some(&(rank, id)) = merges.get(&key(m.id, self.symbols[after as usize].id))
            {
                self.heap.push(Merge { rank, at: at as u32, id });
            }
        }
        self.symbols.retain(|s| s.end != 0);
    }
}

/// What the pattern sees a character as.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Space,
    Letter,
    Number,
    Other,
}

fn class(c: char) -> Class {
    if c.is_ascii() {
        return match c {
            'a'..='z' | 'A'..='Z' => Class::Letter,
            '0'..='9' => Class::Number,
            '\t' | '\n' | '\x0b' | '\x0c' | '\r' | ' ' => Class::Space,
            _ => Class::Other,
        };
    }
    if c.is_whitespace() {
        Class::Space
    } else if category(c, LETTER) {
        Class::Letter
    } else if category(c, NUMBER) {
        Class::Number
    } else {
        Class::Other
    }
}

/// GPT-2's pattern, `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|
/// ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`, as upstream matches it: `word(start,
/// end)` for each match, which together cover the text.
pub(crate) fn split(s: &str, mut word: impl FnMut(usize, usize)) {
    let b = s.as_bytes();
    let at = |i: usize| s[i..].chars().next().expect("inside the text");
    // The end of the run of `class` from i.
    let run = |mut i: usize, k: Class| {
        while i < b.len() {
            let c = at(i);
            if class(c) != k {
                break;
            }
            i += c.len_utf8();
        }
        i
    };
    let mut i = 0;
    while i < b.len() {
        let c = at(i);
        // A contraction.
        if c == '\'' {
            let rest = &b[i + 1..];
            let len = if rest.starts_with(b"re") || rest.starts_with(b"ve") || rest.starts_with(b"ll") {
                2
            } else if matches!(rest.first(), Some(b's' | b't' | b'm' | b'd')) {
                1
            } else {
                0
            };
            if len > 0 {
                word(i, i + 1 + len);
                i += 1 + len;
                continue;
            }
        }
        // An optional space, then a run of letters, numbers or others.
        let j = if c == ' ' { i + 1 } else { i };
        if j < b.len() {
            let k = class(at(j));
            if k != Class::Space {
                let end = run(j, k);
                word(i, end);
                i = end;
                continue;
            }
        }
        // Whitespace: up to the last before a character that is not,
        // unless that leaves none.
        let end = run(i, Class::Space);
        if end < b.len() {
            let last = s[..end].chars().next_back().expect("a run of one at least").len_utf8();
            if end - last > i {
                word(i, end - last);
                i = end - last;
                continue;
            }
        }
        word(i, end);
        i = end;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pattern's classes are the regular expressions': \\s, \\p{L}
    /// and \\p{N} for every character.
    #[test]
    fn classes_are_the_regular_expressions() {
        let space = regex::Regex::new(r"^\s$").unwrap();
        let letter = regex::Regex::new(r"^\p{L}$").unwrap();
        let number = regex::Regex::new(r"^\p{N}$").unwrap();
        let mut wrong = Vec::new();
        for c in (0..=0x10ffffu32).filter_map(char::from_u32) {
            let mut buf = [0u8; 4];
            let s = &*c.encode_utf8(&mut buf);
            let want = if space.is_match(s) {
                Class::Space
            } else if letter.is_match(s) {
                Class::Letter
            } else if number.is_match(s) {
                Class::Number
            } else {
                Class::Other
            };
            if class(c) != want {
                wrong.push(c);
            }
        }
        assert!(wrong.is_empty(), "{} characters: {:?}", wrong.len(), &wrong[..wrong.len().min(40)]);
    }

    /// Writes core/src/classes.rs from the regex crate's tables.
    #[test]
    #[ignore = "writes the tables; run by hand when Unicode moves"]
    fn write_classes() {
        let ranges = |re: &str| {
            let re = regex::Regex::new(re).unwrap();
            let mut out: Vec<(u32, u32)> = Vec::new();
            for c in (0..=0x10ffffu32).filter_map(char::from_u32) {
                let mut buf = [0u8; 4];
                if re.is_match(c.encode_utf8(&mut buf)) {
                    match out.last_mut() {
                        Some(r) if r.1 + 1 == c as u32 => r.1 = c as u32,
                        _ => out.push((c as u32, c as u32)),
                    }
                }
            }
            out
        };
        let table = |name: &str, what: &str, r: &[(u32, u32)]| {
            let mut s = format!("/// {what}: first and last of each run.\npub const {name}: &[(u32, u32)] = &[\n");
            for chunk in r.chunks(6) {
                s.push_str("    ");
                s.push_str(&chunk.iter().map(|(a, b)| format!("(0x{a:x}, 0x{b:x})")).collect::<Vec<_>>().join(", "));
                s.push_str(",\n");
            }
            s.push_str("];\n");
            s
        };
        let file = format!(
            "//! Unicode's L and N, as the regular expressions of upstream's\n\
             //! pre-tokenizers read them. Written by bpe::tests::write_classes.\n\n{}\n{}",
            table("LETTERS", "\\p{L}", &ranges(r"^\p{L}$")),
            table("NUMBERS", "\\p{N}", &ranges(r"^\p{N}$"))
        );
        std::fs::write(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/classes.rs"), file).unwrap();
    }

    /// Each byte has a character and each character its byte back.
    #[test]
    fn byte_chars_go_both_ways() {
        let chars = byte_chars();
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            assert!(seen.insert(chars[b as usize]));
            assert_eq!(char_byte(chars[b as usize]), Some(b));
        }
        assert_eq!(chars[b' ' as usize], '\u{120}');
        assert_eq!(chars[b'\n' as usize], '\u{10a}');
        assert_eq!(char_byte('\u{144}'), None);
    }

    /// Merging by scanning and by the heap give the same symbols, on
    /// random merge lists and words.
    #[test]
    fn scanning_and_the_heap_merge_alike() {
        let mut x: u64 = 0x1234_5678_9abc_def1;
        let mut next = |n: u64| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x % n
        };
        for _ in 0..300 {
            let mut merges = Merges::default();
            let count = next(40) as u32;
            for (rank, id) in (0..count).zip(8u32..) {
                let (l, r) = (next(id as u64) as u32, next(id as u64) as u32);
                merges.entry(key(l, r)).or_insert((rank, id));
            }
            let len = 1 + next(120) as usize;
            let symbols: Vec<Symbol> =
                (0..len).map(|i| Symbol { id: next(8) as u32, start: i as u32, end: i as u32 + 1 }).collect();
            let mut a = Word { symbols: symbols.clone(), ..Default::default() };
            let mut b = Word { symbols, ..Default::default() };
            a.merge_scanning(&merges);
            b.merge_heap(&merges);
            let ids = |w: &Word| w.symbols.iter().map(|s| (s.id, s.start, s.end)).collect::<Vec<_>>();
            assert_eq!(ids(&a), ids(&b));
        }
    }
}
