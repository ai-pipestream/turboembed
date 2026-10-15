//! A reader for ICU4X's `ZeroTrieSimpleAscii`, the byte trie the ICU4X
//! segmenter data stores its dictionaries in.
//!
//! This is a port of the stepping functions of the `zerotrie` crate
//! (ICU4X, `utils/zerotrie/src/reader.rs` and `varint.rs`, Unicode-3.0;
//! the notice is in `analysis/tokenize/data/icu4x-segmenter/LICENSE`),
//! limited to what a `ZeroTrieSimpleAscii` holds: ASCII nodes, value
//! nodes and binary search branch nodes. The trie is read where it lies;
//! nothing is copied or decoded ahead.
//!
//! The layout, from the `zerotrie` documentation: a trie is a sequence of
//! nodes in a byte slice.
//!
//! - `0xxxxxxx`: an ASCII node, which matches that byte.
//! - `100xxxxx`: a value node, whose varint (three bits of metadata) is
//!   the value of the string read so far.
//! - `11xxxxxx`: a branch node, whose varint (two bits of metadata) holds
//!   the number of edges `N` in its low eight bits (0 meaning 256) and the
//!   width `W` of its offset table above them. `N` sorted bytes follow,
//!   then `W + 1` tables of `N - 1` offset bytes (high bytes first), then
//!   the `N` subtries one after another. The subtrie of edge `i` spans
//!   from offset `s_i` to `s_(i+1)` of what follows the tables, with
//!   `s_0 = 0` and `s_N` the end of the slice.
//!
//! A malformed trie never makes the reader panic or read outside the
//! slice: a node that runs past the end reads as the end of the trie.

/// The kind of a node, from the top bits of its first byte.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Node {
    Ascii,
    Value,
    Branch,
    /// A span node (`101xxxxx`), which a `ZeroTrieSimpleAscii` never holds.
    Span,
}

#[inline]
fn node(b: u8) -> Node {
    match b & 0b1110_0000 {
        0b1000_0000 => Node::Value,
        0b1010_0000 => Node::Span,
        0b1100_0000 | 0b1110_0000 => Node::Branch,
        _ => Node::Ascii,
    }
}

/// Reads a varint whose lead byte `lead` keeps `meta` bits of metadata
/// (2 or 3), with the bytes after it in `rest`. Each extension byte adds
/// the latent value of the shorter forms, so every value has one
/// encoding. Returns the value and what follows the varint.
#[inline]
fn varint(lead: u8, mut rest: &[u8], meta: u32) -> (usize, &[u8]) {
    let value_bits = 7 - meta;
    let mut value = usize::from(lead & ((1 << value_bits) - 1) as u8);
    if lead & (1 << value_bits) != 0 {
        let latent = 1usize << value_bits;
        loop {
            let Some((&next, tail)) = rest.split_first() else { break };
            rest = tail;
            value = (value << 7).wrapping_add(usize::from(next & 0x7F)).wrapping_add(latent);
            if next & 0x80 == 0 {
                break;
            }
        }
    }
    (value, rest)
}

/// The subtrie of edge `i` of a branch with `n` edges and `w + 1` offset
/// tables, given the slice that starts at the offset tables.
#[inline]
fn branch(trie: &[u8], i: usize, n: usize, w: usize) -> &[u8] {
    let tables = (n - 1) * (w + 1);
    let Some(subtries) = trie.get(tables..) else { return &[] };
    let mut start = 0usize;
    let mut end = 0usize;
    for table in 0..=w {
        let offsets = &trie[table * (n - 1)..(table + 1) * (n - 1)];
        start = (start << 8) + if i == 0 { 0 } else { usize::from(offsets[i - 1]) };
        end = (end << 8) + offsets.get(i).map_or(0, |&b| usize::from(b));
    }
    if i + 1 == n {
        end = subtries.len();
    }
    subtries.get(start..end).unwrap_or(&[])
}

/// A position in a trie: the subtrie of the strings that begin with the
/// bytes stepped so far.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Cursor<'a> {
    trie: &'a [u8],
}

impl<'a> Cursor<'a> {
    /// A cursor at the root of `trie`.
    pub(crate) fn new(trie: &'a [u8]) -> Cursor<'a> {
        Cursor { trie }
    }

    /// Steps over the byte `c`. When no string continues with it, the
    /// cursor becomes empty.
    #[inline]
    pub(crate) fn step(&mut self, c: u8) {
        loop {
            let Some((&b, rest)) = self.trie.split_first() else { return };
            match node(b) {
                Node::Ascii => {
                    self.trie = if b == c { rest } else { &[] };
                    return;
                }
                Node::Value => self.trie = varint(b, rest, 3).1,
                Node::Span => {
                    self.trie = &[];
                    return;
                }
                Node::Branch => {
                    let (x, rest) = varint(b, rest, 2);
                    let (n, w) = if x >= 256 { (x & 0xFF, (x >> 8) & 0x3) } else { (x, 0) };
                    let n = if n == 0 { 256 } else { n };
                    let Some((keys, rest)) = rest.split_at_checked(n) else {
                        self.trie = &[];
                        return;
                    };
                    self.trie = match keys.binary_search(&c) {
                        Ok(i) => branch(rest, i, n, w),
                        Err(_) => &[],
                    };
                    return;
                }
            }
        }
    }

    /// Whether the bytes stepped so far are a string of the trie. The value
    /// node is consumed, as ICU4X's `take_value` does, so a second call
    /// says no.
    #[inline]
    pub(crate) fn take_value(&mut self) -> bool {
        match self.trie.split_first() {
            Some((&b, rest)) if node(b) == Node::Value => {
                self.trie = varint(b, rest, 3).1;
                true
            }
            _ => false,
        }
    }

    /// Whether no string continues past the bytes stepped so far.
    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.trie.is_empty()
    }
}

/// Calls `visit` with every string of `trie` and its value, in the order
/// the trie stores them. For tests and for checking a dictionary file.
pub(crate) fn for_each_string(trie: &[u8], visit: &mut impl FnMut(&[u8], usize)) {
    fn walk(mut trie: &[u8], prefix: &mut Vec<u8>, visit: &mut impl FnMut(&[u8], usize)) {
        let depth = prefix.len();
        while let Some((&b, rest)) = trie.split_first() {
            match node(b) {
                Node::Ascii => {
                    prefix.push(b);
                    trie = rest;
                }
                Node::Value => {
                    let (value, rest) = varint(b, rest, 3);
                    visit(prefix, value);
                    trie = rest;
                }
                Node::Span => break,
                Node::Branch => {
                    let (x, rest) = varint(b, rest, 2);
                    let (n, w) = if x >= 256 { (x & 0xFF, (x >> 8) & 0x3) } else { (x, 0) };
                    let n = if n == 0 { 256 } else { n };
                    let Some((keys, rest)) = rest.split_at_checked(n) else { break };
                    for (i, &key) in keys.iter().enumerate() {
                        prefix.push(key);
                        walk(branch(rest, i, n, w), prefix, visit);
                        prefix.pop();
                    }
                    break;
                }
            }
        }
        prefix.truncate(depth);
    }
    walk(trie, &mut Vec::new(), visit);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The example of the `zerotrie` documentation: "" 0, "axb" 100,
    /// "ayc" 2, "azd" 3, "bxe" 4, "bxefg" 500, "bxefh" 6, "bxei" 7,
    /// "bxeikl" 8.
    const EXAMPLE: [u8; 36] = [
        0b1000_0000,
        0b1100_0010,
        b'a',
        b'b',
        13,
        0b1100_0011,
        b'x',
        b'y',
        b'z',
        3,
        5,
        b'b',
        0b1001_0000,
        0x54,
        b'c',
        0b1000_0010,
        b'd',
        0b1000_0011,
        b'x',
        b'e',
        0b1000_0100,
        0b1100_0010,
        b'f',
        b'i',
        7,
        0b1100_0010,
        b'g',
        b'h',
        2,
        0b1001_0011,
        0x64,
        0b1000_0110,
        0b1000_0111,
        b'k',
        b'l',
        0b1000_1000,
    ];

    fn lookup(trie: &[u8], key: &[u8]) -> (bool, bool) {
        let mut cursor = Cursor::new(trie);
        for &b in key {
            cursor.step(b);
        }
        let value = cursor.take_value();
        (value, cursor.is_empty())
    }

    #[test]
    fn steps_through_the_documented_example() {
        assert_eq!(lookup(&EXAMPLE, b"axb"), (true, true));
        assert_eq!(lookup(&EXAMPLE, b"bxe"), (true, false));
        assert_eq!(lookup(&EXAMPLE, b"bxef"), (false, false));
        assert_eq!(lookup(&EXAMPLE, b"bxeikl"), (true, true));
        assert_eq!(lookup(&EXAMPLE, b"bx"), (false, false));
        assert_eq!(lookup(&EXAMPLE, b"bxq"), (false, true));
        assert_eq!(lookup(&EXAMPLE, b"c"), (false, true));
        assert_eq!(lookup(&EXAMPLE, b"axbz"), (false, true));
        assert_eq!(lookup(&EXAMPLE, &[0xFF]), (false, true));
    }

    #[test]
    fn enumerates_the_documented_example() {
        let mut all = Vec::new();
        for_each_string(&EXAMPLE, &mut |s, v| all.push((String::from_utf8(s.to_vec()).unwrap(), v)));
        let expected = [
            ("", 0),
            ("axb", 100),
            ("ayc", 2),
            ("azd", 3),
            ("bxe", 4),
            ("bxefg", 500),
            ("bxefh", 6),
            ("bxei", 7),
            ("bxeikl", 8),
        ];
        let expected: Vec<(String, usize)> = expected.iter().map(|&(s, v)| (s.to_owned(), v)).collect();
        assert_eq!(all, expected);
    }

    #[test]
    fn varints_follow_the_documented_examples() {
        // Three bits of metadata.
        assert_eq!(varint(0b000_0_1010, &[], 3).0, 10);
        assert_eq!(varint(0b000_0_1111, &[], 3).0, 15);
        assert_eq!(varint(0b000_1_0000, &[0b0000_0000], 3).0, 16);
        assert_eq!(varint(0b000_1_0000, &[0b0000_0001], 3).0, 17);
        assert_eq!(varint(0b000_1_1111, &[0b0111_1111], 3).0, 2063);
        assert_eq!(varint(0b000_1_0000, &[0b1000_0000, 0], 3).0, 2064);
        // Two bits: the latent values are 32 and 4128.
        assert_eq!(varint(0b11_1_00000, &[0], 2).0, 32);
        assert_eq!(varint(0b11_1_00000, &[0x80, 0], 2).0, 4128);
        // An extension byte missing at the end of the slice ends the varint.
        assert_eq!(varint(0b000_1_0000, &[], 3), (0, &[][..]));
    }

    #[test]
    fn malformed_tries_do_not_panic() {
        for len in 0..EXAMPLE.len() {
            let cut = &EXAMPLE[..len];
            for key in [&b"axb"[..], b"bxeikl", b"bxefg", b"zz"] {
                let _ = lookup(cut, key);
            }
            for_each_string(cut, &mut |_, _| {});
        }
        // A branch whose offset table points past the end.
        let bad = [0b1100_0010, b'a', b'b', 200, b'x'];
        assert_eq!(lookup(&bad, b"b"), (false, true));
        // A span node is not part of an ASCII trie.
        assert_eq!(lookup(&[0b1010_0001, 0x80], b"a"), (false, true));
    }
}
