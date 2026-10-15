//! The word dictionaries and ICU4X's dictionary segmenter.
//!
//! A dictionary is a set of words over an alphabet of at most 128
//! characters. The words are stored in a [`zerotrie`](super::zerotrie)
//! trie whose bytes are the positions of the characters in the alphabet,
//! as ICU4X's `DictionaryBreakData::ZeroTrie` stores them. A character
//! outside the alphabet ends every word that reaches it.
//!
//! The file format (`.dict`), little-endian:
//!
//! | Bytes | Contents |
//! |---|---|
//! | 4 | the magic `TZD1` |
//! | 4 | the number of alphabet ranges `R`, at most 128 |
//! | `8 R` | each range: its first code point and the code point after its last |
//! | rest | the trie |
//!
//! The ranges are sorted and disjoint and hold at most 128 code points in
//! all; a character's position is the number of alphabet characters below
//! it. The trie is borrowed from the file's bytes as it is.
//!
//! [`Dictionary::segment`] is ICU4X's `DictionaryBreakIterator` (ICU4X,
//! `components/segmenter/src/complex/dictionary.rs`, Unicode-3.0): a
//! greedy longest match that only accepts a word ending inside the text
//! where a grapheme cluster ends.

use std::fmt;

use turbo_unicode::grapheme::{GraphemeBoundaries, grapheme_boundaries};

use super::zerotrie::{self, Cursor};

/// The first four bytes of a dictionary file.
pub const MAGIC: [u8; 4] = *b"TZD1";

/// The most characters an alphabet can hold: positions are ASCII bytes.
const ALPHABET_MAX: u32 = 128;

/// The byte stepped for a character outside the alphabet. It is not an
/// ASCII byte, so no edge of the trie matches it.
const OUTSIDE: u8 = 0xFF;

/// Why a dictionary file was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DictionaryError {
    /// The file does not start with [`MAGIC`].
    Magic,
    /// The file ends inside its header.
    Truncated,
    /// The alphabet has more than 128 ranges or characters, or its ranges
    /// are empty, unsorted, overlapping or outside the code space.
    Alphabet,
}

impl fmt::Display for DictionaryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            DictionaryError::Magic => "not a dictionary file: the magic TZD1 is missing",
            DictionaryError::Truncated => "the dictionary file ends inside its header",
            DictionaryError::Alphabet => "the dictionary alphabet is malformed",
        })
    }
}

impl std::error::Error for DictionaryError {}

/// A dictionary read from a `.dict` file, borrowing the file's bytes.
#[derive(Clone, Debug)]
pub struct Dictionary<'a> {
    /// The alphabet ranges, as (first, after last, position of first).
    ranges: Vec<(u32, u32, u8)>,
    trie: &'a [u8],
}

/// The ranges of a range table, and the rest of the bytes.
fn read_u32(bytes: &[u8], at: usize) -> Result<u32, DictionaryError> {
    let word = bytes.get(at..at + 4).ok_or(DictionaryError::Truncated)?;
    Ok(u32::from_le_bytes([word[0], word[1], word[2], word[3]]))
}

impl<'a> Dictionary<'a> {
    /// Reads a dictionary from the bytes of a `.dict` file. The header is
    /// checked; the trie is not walked, and a malformed one ends words
    /// early rather than failing.
    pub fn from_bytes(bytes: &'a [u8]) -> Result<Dictionary<'a>, DictionaryError> {
        if bytes.get(..4) != Some(&MAGIC[..]) {
            return Err(DictionaryError::Magic);
        }
        let count = read_u32(bytes, 4)?;
        if count > ALPHABET_MAX {
            return Err(DictionaryError::Alphabet);
        }
        let mut ranges = Vec::with_capacity(count as usize);
        let mut size = 0u32;
        let mut previous_end = 0u32;
        for i in 0..count as usize {
            let first = read_u32(bytes, 8 + 8 * i)?;
            let end = read_u32(bytes, 12 + 8 * i)?;
            if first >= end || end > 0x11_0000 || (i > 0 && first < previous_end) {
                return Err(DictionaryError::Alphabet);
            }
            if size + (end - first) > ALPHABET_MAX {
                return Err(DictionaryError::Alphabet);
            }
            ranges.push((first, end, size as u8));
            size += end - first;
            previous_end = end;
        }
        let trie = &bytes[8 + 8 * count as usize..];
        Ok(Dictionary { ranges, trie })
    }

    /// The number of characters in the alphabet.
    pub fn alphabet_len(&self) -> usize {
        self.ranges.iter().map(|&(first, end, _)| (end - first) as usize).sum()
    }

    /// The byte the trie stores for `c`: its position in the alphabet.
    #[inline]
    fn symbol(&self, c: char) -> u8 {
        let cp = c as u32;
        for &(first, end, base) in &self.ranges {
            if cp < first {
                break;
            }
            if cp < end {
                return base + (cp - first) as u8;
            }
        }
        OUTSIDE
    }

    /// Whether `word` is in the dictionary.
    pub fn contains(&self, word: &str) -> bool {
        let mut cursor = Cursor::new(self.trie);
        for c in word.chars() {
            cursor.step(self.symbol(c));
        }
        !word.is_empty() && cursor.take_value()
    }

    /// Calls `visit` with every word of the dictionary, in the order the
    /// trie stores them (by alphabet position).
    pub fn for_each_word(&self, mut visit: impl FnMut(&str)) {
        let mut alphabet = Vec::with_capacity(self.alphabet_len());
        for &(first, end, _) in &self.ranges {
            alphabet.extend((first..end).filter_map(char::from_u32));
        }
        let mut word = String::new();
        zerotrie::for_each_string(self.trie, &mut |symbols, _| {
            word.clear();
            for &s in symbols {
                if let Some(&c) = alphabet.get(usize::from(s)) {
                    word.push(c);
                }
            }
            visit(&word);
        });
    }

    /// The word boundaries of `text` that the dictionary finds, as byte
    /// offsets after each word, in increasing order; the last is the end
    /// of the text. Nothing is reported for an empty text.
    ///
    /// From each position, the trie is walked one character at a time.
    /// A word that no longer word continues is taken at once. Otherwise
    /// the longest word seen whose end is a grapheme cluster boundary is
    /// taken when the walk can go no further. Where no word starts, the
    /// piece read until the walk failed is taken, or the rest of the text
    /// when it ran out first.
    ///
    /// Deviation from ICU4X: ICU4X takes the piece read when no word
    /// starts, and a word that no longer word continues, without looking at
    /// grapheme clusters, so a boundary can fall between a letter and its
    /// combining marks, or between two marks. Here such a boundary moves
    /// forward to the end of its grapheme cluster, and the next word is
    /// looked for from there. Every boundary is a grapheme cluster
    /// boundary; where ICU4X's are, the two agree.
    pub fn segment(&self, text: &str, mut boundary: impl FnMut(usize)) {
        let mut at = 0usize;
        let mut graphemes = grapheme_boundaries(text);
        let mut clusters = grapheme_boundaries(text);
        let mut cluster_end = 0usize;
        while let Some(end) = self.next_boundary(text, &mut at, &mut graphemes) {
            while cluster_end < end {
                cluster_end = clusters.next().unwrap_or(text.len());
            }
            if cluster_end != end {
                at = cluster_end;
                graphemes = clusters;
            }
            boundary(cluster_end);
        }
    }

    /// One step of ICU4X's `DictionaryBreakIterator::next`: the boundary
    /// after the word that starts at `*at`, with `*at` and `graphemes`
    /// moved to where the next word starts. `None` at the end of the text.
    fn next_boundary(&self, text: &str, at: &mut usize, graphemes: &mut GraphemeBoundaries<'_>) -> Option<usize> {
        let mut cursor = Cursor::new(self.trie);
        let mut longest = 0usize;
        let mut prefix_read = false;
        let mut rewind: Option<(usize, GraphemeBoundaries<'_>)> = None;
        let mut grapheme_end = 0usize;
        let rest = &text[*at..];
        for (offset, c) in rest.char_indices() {
            let end = *at + offset + c.len_utf8();
            cursor.step(self.symbol(c));
            let word = cursor.take_value();
            let last = cursor.is_empty();
            match (word, last) {
                (true, true) => {
                    // A word that no longer word continues.
                    *at = end;
                    return Some(end);
                }
                (true, false) => {
                    // A word that a longer one may continue: kept only
                    // where a grapheme cluster ends.
                    while grapheme_end < end {
                        match graphemes.next() {
                            Some(g) => grapheme_end = g,
                            None => {
                                grapheme_end = text.len();
                                break;
                            }
                        }
                    }
                    if grapheme_end != end {
                        continue;
                    }
                    longest = end;
                    rewind = Some((end, *graphemes));
                }
                (false, true) => {
                    if longest > 0 {
                        if let Some((position, saved)) = rewind {
                            *at = position;
                            *graphemes = saved;
                        }
                        return Some(longest);
                    }
                    // No word starts here.
                    *at = end;
                    return Some(end);
                }
                (false, false) => prefix_read = true,
            }
        }
        *at = text.len();
        if longest > 0 {
            if let Some((position, saved)) = rewind {
                *at = position;
                *graphemes = saved;
            }
            Some(longest)
        } else if prefix_read {
            Some(text.len())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dictionary file over the alphabet `a` to `e` holding `trie`.
    fn file(trie: &[u8]) -> Vec<u8> {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&u32::from('a').to_le_bytes());
        bytes.extend_from_slice(&u32::from('f').to_le_bytes());
        bytes.extend_from_slice(trie);
        bytes
    }

    #[test]
    fn header_is_checked() {
        assert_eq!(Dictionary::from_bytes(b"XZD1").unwrap_err(), DictionaryError::Magic);
        assert_eq!(Dictionary::from_bytes(b"TZD1\x01\0\0\0").unwrap_err(), DictionaryError::Truncated);
        let mut two = MAGIC.to_vec();
        for v in [2u32, 10, 20, 15, 30] {
            two.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Dictionary::from_bytes(&two).unwrap_err(), DictionaryError::Alphabet);
        let mut wide = MAGIC.to_vec();
        for v in [1u32, 0, 129] {
            wide.extend_from_slice(&v.to_le_bytes());
        }
        assert_eq!(Dictionary::from_bytes(&wide).unwrap_err(), DictionaryError::Alphabet);
        let empty = file(&[]);
        let dict = Dictionary::from_bytes(&empty).unwrap();
        assert_eq!(dict.alphabet_len(), 5);
        assert_eq!(dict.symbol('a'), 0);
        assert_eq!(dict.symbol('e'), 4);
        assert_eq!(dict.symbol('f'), OUTSIDE);
        assert_eq!(dict.symbol('`'), OUTSIDE);
    }

    /// The words "ab" (position 0 1), "abc" and "d", as a trie over
    /// alphabet positions: `0 1 v(0) branch(2: 2, 3? )`.
    fn small() -> Vec<u8> {
        // Root branch of two edges, 0 ('a') and 3 ('d').
        //   'a' subtrie: 1 ('b'), value 0, 2 ('c'), value 0
        //   'd' subtrie: value 0
        let a = [0x01, 0b1000_0000, 0x02, 0b1000_0000];
        let mut trie = vec![0b1100_0010, 0x00, 0x03, a.len() as u8];
        trie.extend_from_slice(&a);
        trie.push(0b1000_0000);
        file(&trie)
    }

    fn segments(dict: &Dictionary<'_>, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut start = 0;
        dict.segment(text, |end| {
            out.push(text[start..end].to_owned());
            start = end;
        });
        out
    }

    #[test]
    fn longest_match() {
        let bytes = small();
        let dict = Dictionary::from_bytes(&bytes).unwrap();
        let mut words = Vec::new();
        dict.for_each_word(|w| words.push(w.to_owned()));
        assert_eq!(words, ["ab", "abc", "d"]);
        assert!(dict.contains("ab") && dict.contains("abc") && dict.contains("d"));
        assert!(!dict.contains("a") && !dict.contains("") && !dict.contains("abcd"));
        assert_eq!(segments(&dict, "abcd"), ["abc", "d"]);
        assert_eq!(segments(&dict, "abd"), ["ab", "d"]);
        // No word starts with "b": one character.
        assert_eq!(segments(&dict, "bd"), ["b", "d"]);
        // "a" starts words, but "ae" none: the piece read is taken.
        assert_eq!(segments(&dict, "aed"), ["ae", "d"]);
        // The text ends inside a word: the rest is one piece.
        assert_eq!(segments(&dict, "da"), ["d", "a"]);
        // A character outside the alphabet.
        assert_eq!(segments(&dict, "ab\u{0E01}d"), ["ab", "\u{0E01}", "d"]);
        assert_eq!(segments(&dict, ""), Vec::<String>::new());
    }

    #[test]
    fn a_boundary_never_splits_a_grapheme_cluster() {
        let thai =
            Dictionary::from_bytes(crate::complex::bundled_dictionary(crate::complex::ComplexScript::Thai)).unwrap();
        // No Thai word starts with U+0E03: ICU4X takes the one character
        // and breaks between it and its vowel sign (ฃ | ◌ุ); here the
        // boundary moves to the end of the cluster.
        assert_eq!(segments(&thai, "\u{0E03}\u{0E38}"), ["\u{0E03}\u{0E38}"]);
        assert_eq!(segments(&thai, "\u{0E03}\u{0E38}ภาษา"), ["\u{0E03}\u{0E38}", "ภาษา"]);
    }

    #[test]
    fn a_word_inside_a_grapheme_cluster_is_not_taken() {
        let bytes = small();
        let dict = Dictionary::from_bytes(&bytes).unwrap();
        // "ab" is a word that "abc" continues; with a combining mark after
        // "b" it does not end a cluster, so it is not taken, and the walk
        // fails at the mark with nothing kept.
        assert_eq!(segments(&dict, "ab\u{0301}d"), ["ab\u{0301}", "d"]);
    }
}
