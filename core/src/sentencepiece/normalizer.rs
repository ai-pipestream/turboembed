//! SentencePiece's normalizer (normalizer.cc): the precompiled character
//! map of the model's `NormalizerSpec`, the user-defined pieces kept as
//! they are, and the handling of whitespace (the dummy prefix, extra
//! whitespace removed, U+2581 for each space). It records, for every
//! byte it writes, the byte of the input it came from, as the library's
//! `norm_to_orig` does.

use crate::status::{Result, invalid};
use crate::unigram::Trie;

use super::proto::NormalizerSpec;

/// U+2581, the metaspace: what a space becomes with `escape_whitespaces`.
pub const SPACE_SYMBOL: &[u8] = "\u{2581}".as_bytes();

/// The most matches the library reads from one search of the character
/// map (`kMaxTrieResultsSize`) and of the user-defined pieces.
const MAX_MAP_RESULTS: usize = 32;
const MAX_USER_RESULTS: usize = 64;

/// The length of the UTF-8 sequence a byte starts, as the library's
/// `OneCharLen` reads it: a byte inside a sequence counts as one.
#[inline]
pub(crate) fn one_char_len(b: u8) -> usize {
    const LEN: [u8; 16] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4];
    LEN[(b >> 4) as usize] as usize
}

/// A darts-clone double array, as the precompiled map stores it: one
/// 32-bit unit per node.
pub(crate) struct DoubleArray {
    units: Vec<u32>,
}

impl DoubleArray {
    #[inline]
    fn has_leaf(u: u32) -> bool {
        (u >> 8) & 1 == 1
    }

    #[inline]
    fn value(u: u32) -> u32 {
        u & ((1 << 31) - 1)
    }

    #[inline]
    fn label(u: u32) -> u32 {
        u & ((1 << 31) | 0xff)
    }

    #[inline]
    fn offset(u: u32) -> usize {
        ((u >> 10) << ((u & (1 << 9)) >> 6)) as usize
    }

    /// darts-clone's `validate`, no stricter, so that a map the library
    /// takes is taken: every unit's children and leaf lie inside the
    /// array, and every value of a unit with no label is under
    /// `max_value`. After it, a search reads no unit outside the array.
    /// The unit a leaf flag points at may still be one with a label,
    /// whose value is unchecked; `Normalizer::prefix` passes over a value
    /// past the replacement strings, as the library does.
    fn validate(&self, max_value: usize) -> bool {
        let n = self.units.len();
        let Some(&root) = self.units.first() else { return false };
        if Self::label(root) != 0 || Self::has_leaf(root) || Self::offset(root) == 0 || (Self::offset(root) | 0xff) >= n
        {
            return false;
        }
        for (i, &u) in self.units.iter().enumerate().skip(1) {
            if Self::label(u) <= 0xff {
                if ((i ^ Self::offset(u)) | 0xff) >= n {
                    return false;
                }
            } else if Self::value(u) as usize >= max_value {
                return false;
            }
        }
        true
    }

    /// `found(length, value)` for each key that `key` starts with,
    /// shortest first, as `commonPrefixSearch` gives them.
    #[inline]
    fn common_prefix_search(&self, key: &[u8], mut found: impl FnMut(usize, u32)) {
        let mut pos = Self::offset(self.units[0]);
        for (i, &c) in key.iter().enumerate() {
            pos ^= c as usize;
            let u = self.units[pos];
            if Self::label(u) != c as u32 {
                return;
            }
            pos ^= Self::offset(u);
            if Self::has_leaf(u) {
                found(i + 1, Self::value(self.units[pos]));
            }
        }
    }
}

/// The normalizer of one `NormalizerSpec`.
pub(crate) struct Normalizer {
    /// The precompiled map: its trie and the replacement strings, each
    /// ended by a NUL.
    map: Option<(DoubleArray, Vec<u8>)>,
    /// The user-defined pieces, written as they are (`SetPrefixMatcher`).
    user: Option<Trie>,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
    treat_whitespace_as_suffix: bool,
}

impl Normalizer {
    /// `user` are the model's user-defined pieces, for the normalizer of
    /// the text; the denormalizer has none.
    pub(crate) fn new(spec: &NormalizerSpec, treat_whitespace_as_suffix: bool, user: &[&[u8]]) -> Result<Normalizer> {
        let map = if spec.precompiled_charsmap.is_empty() { None } else { Some(charsmap(&spec.precompiled_charsmap)?) };
        let user = if user.is_empty() {
            None
        } else {
            let pairs = user.iter().map(|&p| (p, 0)).collect();
            Some(Trie::of(pairs).map_err(|p| invalid(format!("SentencePiece model: {p:?} is defined twice")))?)
        };
        Ok(Normalizer {
            map,
            user,
            add_dummy_prefix: spec.add_dummy_prefix,
            remove_extra_whitespaces: spec.remove_extra_whitespaces,
            escape_whitespaces: spec.escape_whitespaces,
            treat_whitespace_as_suffix,
        })
    }

    /// The longest user-defined piece `input` starts with, as the
    /// library's `PrefixMatcher::PrefixMatch` finds it.
    #[inline]
    pub(crate) fn user_match(user: &Trie, input: &[u8]) -> Option<usize> {
        let mut seen = 0;
        let mut longest = 0;
        user.prefixes(input, |len, _| {
            if seen < MAX_USER_RESULTS {
                longest = longest.max(len);
            }
            seen += 1;
        });
        (longest > 0).then_some(longest)
    }

    /// What the start of `input` becomes, and how many of its bytes that
    /// takes (`NormalizePrefix`).
    #[inline]
    fn prefix<'a>(&'a self, input: &'a [u8]) -> (&'a [u8], usize) {
        if let Some(user) = &self.user
            && let Some(len) = Self::user_match(user, input)
        {
            return (&input[..len], len);
        }
        if let Some((trie, normalized)) = &self.map {
            // The longest rule among the first MAX_MAP_RESULTS matches.
            let (mut longest, mut value, mut seen) = (0usize, 0usize, 0usize);
            trie.common_prefix_search(input, |len, v| {
                if seen < MAX_MAP_RESULTS && (longest == 0 || len > longest) {
                    longest = len;
                    value = v as usize;
                }
                seen += 1;
            });
            if longest > 0 && value < normalized.len() {
                let rest = &normalized[value..];
                let end = rest.iter().position(|&b| b == 0).unwrap_or(rest.len());
                return (&rest[..end], longest);
            }
        }
        // The input is UTF-8, so its first character is valid.
        let len = one_char_len(input[0]).min(input.len());
        (&input[..len], len)
    }

    /// `input` normalized into `out`, and when `norm_to_orig` is given,
    /// the source of each byte written and one past the end, offsets in
    /// `input`, as `Normalizer::Normalize` gives them. `out` and
    /// `norm_to_orig` are cleared first.
    pub(crate) fn normalize(&self, input: &str, out: &mut Vec<u8>, mut norm_to_orig: Option<&mut Vec<u32>>) {
        out.clear();
        if let Some(n) = norm_to_orig.as_deref_mut() {
            n.clear();
        }
        let mut input = input.as_bytes();
        if input.is_empty() {
            return;
        }
        let mut consumed = 0usize;
        if self.remove_extra_whitespaces {
            while !input.is_empty() {
                let (p, len) = self.prefix(input);
                if p != b" " {
                    break;
                }
                input = &input[len..];
                consumed += len;
            }
        }
        if input.is_empty() {
            return;
        }
        out.reserve(input.len() + input.len() / 2);
        if let Some(n) = norm_to_orig.as_deref_mut() {
            n.reserve(input.len() + input.len() / 2);
        }
        let space: &[u8] = if self.escape_whitespaces { SPACE_SYMBOL } else { b" " };
        let add_ws = |out: &mut Vec<u8>, n: &mut Option<&mut Vec<u32>>, consumed: usize| {
            out.extend_from_slice(space);
            if let Some(n) = n.as_deref_mut() {
                n.extend(std::iter::repeat_n(consumed as u32, space.len()));
            }
        };
        if !self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            add_ws(out, &mut norm_to_orig, consumed);
        }
        let mut is_prev_space = self.remove_extra_whitespaces;
        while !input.is_empty() {
            let (mut sp, len) = self.prefix(input);
            // Spaces at the start of what a prefix becomes are dropped
            // after one that ended with a space.
            while is_prev_space && sp.first() == Some(&b' ') {
                sp = &sp[1..];
            }
            if !sp.is_empty() {
                for &b in sp {
                    if b == b' ' {
                        add_ws(out, &mut norm_to_orig, consumed);
                    } else {
                        out.push(b);
                        if let Some(n) = norm_to_orig.as_deref_mut() {
                            n.push(consumed as u32);
                        }
                    }
                }
                is_prev_space = sp.last() == Some(&b' ');
            }
            consumed += len;
            input = &input[len..];
            if !self.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }
        if self.remove_extra_whitespaces {
            while out.ends_with(space) {
                let length = out.len() - space.len();
                out.truncate(length);
                if let Some(n) = norm_to_orig.as_deref_mut() {
                    consumed = n[length] as usize;
                    n.truncate(length);
                }
            }
        }
        if self.treat_whitespace_as_suffix && self.add_dummy_prefix {
            add_ws(out, &mut norm_to_orig, consumed);
        }
        if let Some(n) = norm_to_orig {
            n.push(consumed as u32);
        }
    }
}

/// The precompiled map: a 32-bit little-endian length, the trie's units,
/// then the replacement strings, as `DecodePrecompiledCharsMap` reads it.
fn charsmap(blob: &[u8]) -> Result<(DoubleArray, Vec<u8>)> {
    let bad = |why: &str| invalid(format!("SentencePiece model: normalizer_spec.precompiled_charsmap: {why}"));
    if blob.len() <= 4 {
        return Err(bad("shorter than its length field"));
    }
    let trie_bytes = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
    if trie_bytes >= blob.len() - 4 {
        return Err(bad("the trie is longer than the map"));
    }
    if trie_bytes < 1024 || trie_bytes & 0x3ff != 0 {
        return Err(bad("the trie is not a whole number of 1024-byte blocks"));
    }
    let units: Vec<u32> = blob[4..4 + trie_bytes].as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c)).collect();
    let normalized = blob[4 + trie_bytes..].to_vec();
    if normalized.last() != Some(&0) {
        return Err(bad("the replacement strings do not end with a NUL"));
    }
    let trie = DoubleArray { units };
    if !trie.validate(normalized.len()) {
        return Err(bad("the trie points outside itself or past the replacement strings"));
    }
    Ok((trie, normalized))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_char_len_reads_the_lead_byte_only() {
        assert_eq!(one_char_len(b'a'), 1);
        assert_eq!(one_char_len(0x80), 1);
        assert_eq!(one_char_len(0xc3), 2);
        assert_eq!(one_char_len(0xe2), 3);
        assert_eq!(one_char_len(0xf0), 4);
    }

    #[test]
    fn a_broken_map_is_refused() {
        assert!(charsmap(&[0, 0, 0, 0]).is_err());
        // A trie length that is not a whole block.
        let mut blob = 8u32.to_le_bytes().to_vec();
        blob.extend_from_slice(&[0; 9]);
        assert!(charsmap(&blob).is_err());
        // A trie of one block whose root points outside it.
        let mut blob = 1024u32.to_le_bytes().to_vec();
        let mut units = vec![0u32; 256];
        units[0] = 300 << 10;
        for u in units {
            blob.extend_from_slice(&u.to_le_bytes());
        }
        blob.extend_from_slice(b"x\0");
        assert!(charsmap(&blob).is_err());
    }

    /// A map of one block that rewrites "a" as "x", its leaf unit `leaf`.
    fn a_to_x(leaf: u32) -> Vec<u8> {
        let mut units = vec![0u32; 256];
        // The root's children start at 1; "a" lands on 1 ^ 0x61 = 0x60.
        units[0] = 1 << 10;
        // Label "a", a leaf, children at 0x60 ^ 0x70 = 0x10.
        units[0x60] = 0x61 | (1 << 8) | (0x70 << 10);
        units[0x10] = leaf;
        let mut blob = 1024u32.to_le_bytes().to_vec();
        for u in units {
            blob.extend_from_slice(&u.to_le_bytes());
        }
        blob.extend_from_slice(b"x\0");
        blob
    }

    fn normalized(blob: Vec<u8>, text: &str) -> String {
        let spec = NormalizerSpec { precompiled_charsmap: blob, ..NormalizerSpec::default() };
        let n = Normalizer::new(&spec, false, &[]).unwrap();
        let (mut out, mut n2o) = (Vec::new(), Vec::new());
        n.normalize(text, &mut out, Some(&mut n2o));
        assert_eq!(n2o.len(), out.len() + 1);
        String::from_utf8(out).unwrap()
    }

    /// darts-clone's validate does not check what a leaf flag points at,
    /// and the library takes such a map: here the unit holds a label and
    /// value 0, which the library reads as the first replacement.
    /// sentencepiece 0.2.2 normalizes "abc a" to "▁xbc▁x" with either
    /// map, and to "▁abc▁a" when that unit holds value 2.
    #[test]
    fn a_map_the_library_takes_is_taken() {
        assert_eq!(normalized(a_to_x(1 << 31), "abc a"), "\u{2581}xbc\u{2581}x");
        assert_eq!(normalized(a_to_x(0), "abc a"), "\u{2581}xbc\u{2581}x");
        // There a value past the replacement strings is no rule.
        assert_eq!(normalized(a_to_x(2), "abc a"), "\u{2581}abc\u{2581}a");
    }

    /// A small xorshift, for the mutated maps.
    fn next(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// Maps with bytes changed at random, cut short or with a length that
    /// lies are refused or normalize text, and never read outside the map.
    #[test]
    fn a_mutated_map_never_panics() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/sentencepiece/t5.model");
        let model = super::super::proto::ModelProto::parse(&std::fs::read(path).unwrap()).unwrap();
        let original = model.normalizer.precompiled_charsmap;
        let texts = ["", " ", "abc ABC", "\u{fb01}x \u{2460}\u{ff76}\u{ff9e}", "e\u{301} \u{3000}\u{200b}\u{1f642}"];
        let mut state = 193u64;
        let (mut taken, mut refused) = (0, 0);
        for round in 0..300 {
            let mut blob = original.clone();
            match round % 3 {
                0 => {
                    for _ in 0..1 + next(&mut state) % 64 {
                        let at = next(&mut state) as usize % blob.len();
                        blob[at] = next(&mut state) as u8;
                    }
                }
                1 => blob.truncate(next(&mut state) as usize % blob.len()),
                _ => {
                    let len = (next(&mut state) as u32) & !0x3ff;
                    blob[..4].copy_from_slice(&len.to_le_bytes());
                }
            }
            let spec = NormalizerSpec { precompiled_charsmap: blob, ..NormalizerSpec::default() };
            let Ok(n) = Normalizer::new(&spec, false, &[]) else {
                refused += 1;
                continue;
            };
            taken += 1;
            let (mut out, mut n2o) = (Vec::new(), Vec::new());
            for t in texts {
                n.normalize(t, &mut out, Some(&mut n2o));
                assert!(n2o.windows(2).all(|w| w[0] <= w[1]), "{t:?}");
            }
        }
        assert!(taken > 0 && refused > 0, "{taken} taken, {refused} refused");
    }
}
