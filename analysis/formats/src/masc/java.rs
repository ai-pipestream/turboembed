//! The parts of the JDK whose behaviour shows in what the MASC readers
//! read: the UTF-8 decoder, `Integer.parseInt`, `String.equalsIgnoreCase`,
//! the split functions, `String.hashCode` and the iteration order of a
//! `java.util.HashMap`.

const REPLACEMENT: char = '\u{FFFD}';

/// The text an `InputStreamReader` with JDK 21's UTF-8 decoder reads
/// from `bytes`: each malformed sequence is replaced with one U+FFFD.
///
/// Where a sequence ends differs from `String::from_utf8_lossy` (which
/// follows the Unicode recommendation of maximal subparts), and every
/// replacement shifts the UTF-16 offsets the annotation counts in. The
/// JDK takes three bytes for an encoded surrogate (`ED A0 80` is one
/// U+FFFD, where `from_utf8_lossy` gives three), and two bytes for a
/// three byte lead and a continuation byte followed by something else
/// (`ED A0 41` is U+FFFD and `A`). A sequence cut short by the end of
/// the input is one U+FFFD, however many bytes it has.
pub(crate) fn decode_utf8(bytes: &[u8]) -> String {
    let cont = |b: u8| b & 0xC0 == 0x80;
    let mut out = String::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let b1 = bytes[i];
        let rest = &bytes[i + 1..];
        // the bytes the sequence at `i` takes: a character, or the
        // length of a malformed sequence replaced with U+FFFD
        let (c, len) = match b1 {
            0x00..=0x7F => (char::from(b1), 1),
            0xC2..=0xDF => match rest.first() {
                None => (REPLACEMENT, 1),
                Some(&b2) if !cont(b2) => (REPLACEMENT, 1),
                Some(&b2) => (char::from_u32(u32::from(b1 & 0x1F) << 6 | u32::from(b2 & 0x3F)).unwrap(), 2),
            },
            0xE0..=0xEF => {
                let overlong = |b2: u8| b1 == 0xE0 && b2 & 0xE0 == 0x80;
                match *rest {
                    [b2, _, ..] if overlong(b2) || !cont(b2) => (REPLACEMENT, 1),
                    [_, b3, ..] if !cont(b3) => (REPLACEMENT, 2),
                    [b2, b3, ..] => {
                        let u = u32::from(b1 & 0x0F) << 12 | u32::from(b2 & 0x3F) << 6 | u32::from(b3 & 0x3F);
                        // an encoded surrogate is three bytes of malformed input
                        (char::from_u32(u).unwrap_or(REPLACEMENT), 3)
                    }
                    [b2] if overlong(b2) || !cont(b2) => (REPLACEMENT, 1),
                    // cut short by the end of the input
                    _ => (REPLACEMENT, 1 + rest.len()),
                }
            }
            0xF0..=0xF7 => {
                // the second byte the lead allows: the JDK checks it on its own
                // only where the lead is F0 or F4, the range of a code point
                let second_bad = |b2: u8| {
                    b1 > 0xF4
                        || b1 == 0xF0 && !(0x90..=0xBF).contains(&b2)
                        || b1 == 0xF4 && b2 & 0xF0 != 0x80
                        || !cont(b2)
                };
                match *rest {
                    [b2, b3, b4, ..] => {
                        let u = u32::from(b1 & 0x07) << 18
                            | u32::from(b2 & 0x3F) << 12
                            | u32::from(b3 & 0x3F) << 6
                            | u32::from(b4 & 0x3F);
                        if cont(b2) && cont(b3) && cont(b4) && (0x1_0000..=0x10_FFFF).contains(&u) {
                            (char::from_u32(u).unwrap(), 4)
                        } else if second_bad(b2) {
                            (REPLACEMENT, 1)
                        } else if !cont(b3) {
                            (REPLACEMENT, 2)
                        } else {
                            (REPLACEMENT, 3)
                        }
                    }
                    _ if b1 > 0xF4 => (REPLACEMENT, 1),
                    [b2, ..] if second_bad(b2) => (REPLACEMENT, 1),
                    [_, b3] if !cont(b3) => (REPLACEMENT, 2),
                    // cut short by the end of the input
                    _ => (REPLACEMENT, 1 + rest.len()),
                }
            }
            // a continuation byte, an overlong lead (C0, C1) or F8 to FF
            _ => (REPLACEMENT, 1),
        };
        out.push(c);
        i += len;
    }
    out
}

/// `Integer.parseInt(s)`: an optional `+` or `-`, then one or more digits,
/// in the `int` range. A digit is a UTF-16 unit `Character.digit(c, 10)`
/// gives a value for: an `Nd` character of the Basic Multilingual Plane,
/// so `٣` and `３` are digits and a digit outside the BMP (two surrogate
/// units in Java) is not. `None` where Java throws
/// `NumberFormatException`.
pub(crate) fn parse_int(s: &str) -> Option<i32> {
    let (negative, digits) = match s.as_bytes().first()? {
        b'-' => (true, &s[1..]),
        b'+' => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let limit: i64 = if negative { 1 << 31 } else { (1 << 31) - 1 };
    let mut value: i64 = 0;
    for c in digits.chars() {
        if u32::from(c) > 0xFFFF {
            return None;
        }
        value = value * 10 + i64::from(turbo_unicode::java::digit(c)?);
        if value > limit {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

/// `s.equalsIgnoreCase(target)` for an ASCII `target`. Java compares unit
/// by unit, taking two units as equal when they are, when their upper
/// cases are, or when the lower cases of their upper cases are. Besides
/// the ASCII letters themselves, that matches four characters to ASCII
/// letters: `ı` (U+0131) and `ſ` (U+017F), whose upper cases are `I` and
/// `S`, and `İ` (U+0130) and the Kelvin sign (U+212A), whose lower cases
/// are `i` and `k`.
pub(crate) fn equals_ignore_case_ascii(s: &str, target: &str) -> bool {
    debug_assert!(target.is_ascii());
    let mut chars = s.chars();
    for t in target.chars() {
        let Some(c) = chars.next() else {
            return false;
        };
        let matches = c == t
            || c.eq_ignore_ascii_case(&t)
            || match c {
                '\u{131}' | '\u{130}' => t.eq_ignore_ascii_case(&'i'),
                '\u{17F}' => t.eq_ignore_ascii_case(&'s'),
                '\u{212A}' => t.eq_ignore_ascii_case(&'k'),
                _ => false,
            };
        if !matches {
            return false;
        }
    }
    chars.next().is_none()
}

/// `MascIdentifiers.splitOnSpaces`: the runs of characters other than a
/// space, in order.
pub(crate) fn split_on_spaces(value: &str) -> Vec<&str> {
    value.split(' ').filter(|s| !s.is_empty()).collect()
}

/// `StringUtil.split(value, ' ')`, which splits as `String.split(" ")`
/// does: every space separates two fields, so leading and doubled spaces
/// give empty fields, and the empty fields at the end are removed (an
/// empty value is one empty field).
pub(crate) fn split_space(value: &str) -> Vec<&str> {
    if value.is_empty() {
        return vec![""];
    }
    let mut fields: Vec<&str> = value.split(' ').collect();
    while fields.last().is_some_and(|f| f.is_empty()) {
        fields.pop();
    }
    fields
}

/// `String.hashCode`: `s[0]*31^(n-1) + ... + s[n-1]` over the UTF-16
/// units, in wrapping `int` arithmetic.
pub(crate) fn string_hash(s: &str) -> i32 {
    s.encode_utf16().fold(0i32, |h, u| h.wrapping_mul(31).wrapping_add(i32::from(u)))
}

/// The order a `java.util.HashMap` made with the default capacity
/// iterates its keys in, for keys given in the order they were first put.
///
/// The map starts with 16 buckets and doubles them when it holds more
/// than three quarters as many keys, or when one bucket of a table of
/// fewer than 64 buckets gets a ninth key. A key goes in bucket
/// `(h ^ (h >>> 16)) & (buckets - 1)` of its hash `h`, at the end of the
/// bucket's list, and doubling splits each list in two keeping its order.
/// Iteration walks the buckets in order. A bucket of nine keys or more in
/// a table of 64 buckets or more becomes a tree, whose iteration order
/// this does not follow (it keeps list order): that takes nine keys whose
/// spread hashes agree on their low six bits or more, which the MASC ids
/// the readers put in their maps (dense ranges of small numbers, and a
/// header's few annotation names) do not have.
#[derive(Clone, Debug, Default)]
pub(crate) struct HashOrder {
    /// Each bucket's keys, as (spread hash, key).
    table: Vec<Vec<(u32, usize)>>,
    len: usize,
}

impl HashOrder {
    /// Adds the key `key` (any number the caller identifies it by) with
    /// hash `hash`. The caller adds each key once, when it is first put.
    pub(crate) fn insert(&mut self, hash: i32, key: usize) {
        if self.table.is_empty() {
            self.resize();
        }
        let h = spread(hash);
        let i = h as usize & (self.table.len() - 1);
        let before = self.table[i].len();
        self.table[i].push((h, key));
        if before >= 8 && self.table.len() < 64 {
            self.resize();
        }
        self.len += 1;
        if self.len > self.table.len() / 4 * 3 {
            self.resize();
        }
    }

    fn resize(&mut self) {
        let old = self.table.len();
        let new = if old == 0 { 16 } else { old * 2 };
        let mut table = vec![Vec::new(); new];
        for (j, bucket) in std::mem::take(&mut self.table).into_iter().enumerate() {
            for (h, key) in bucket {
                let to = if old != 0 && h as usize & old != 0 { j + old } else { j };
                table[to].push((h, key));
            }
        }
        self.table = table;
    }

    /// The keys in iteration order.
    pub(crate) fn order(&self) -> impl Iterator<Item = usize> + '_ {
        self.table.iter().flatten().map(|&(_, key)| key)
    }
}

fn spread(hash: i32) -> u32 {
    let h = hash as u32;
    h ^ (h >> 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decoded(bytes: &[u8]) -> String {
        decode_utf8(bytes).replace('\u{FFFD}', "?")
    }

    #[test]
    fn decode_utf8_replaces_as_the_jdk_does() {
        assert_eq!(decode_utf8("a\u{e9}\u{20ac}\u{1F600}\u{FEFF}".as_bytes()), "a\u{e9}\u{20ac}\u{1F600}\u{FEFF}");
        // an encoded surrogate is one replacement
        assert_eq!(decoded(b"\xED\xA0\x80x"), "?x");
        assert_eq!(decoded(b"\xED\xA0x"), "?x");
        // overlong forms: each byte on its own
        assert_eq!(decoded(b"\xC0\xAFx"), "??x");
        assert_eq!(decoded(b"\xE0\x80\xAFx"), "???x");
        assert_eq!(decoded(b"\xF0\x80\x80\xAFx"), "????x");
        // past U+10FFFF, and leads no code point has
        assert_eq!(decoded(b"\xF4\x90\x80\x80x"), "????x");
        assert_eq!(decoded(b"\xF5\x80\x80\x80x"), "????x");
        assert_eq!(decoded(b"\xF8\x88\x80\x80\x80x"), "?????x");
        // a sequence broken by another byte
        assert_eq!(decoded(b"\xE2\x28\xA1x"), "?(?x");
        assert_eq!(decoded(b"\xE2\x82x"), "?x");
        assert_eq!(decoded(b"\xF0\x9F\x98x"), "?x");
        assert_eq!(decoded(b"\xF0\x9Fx"), "?x");
        assert_eq!(decoded(b"\xF0\x9F\x98\xF0\x9F\x98\x80"), "?\u{1F600}");
        assert_eq!(decoded(b"\x80\xBF"), "??");
        // cut short by the end of the input: one replacement
        assert_eq!(decoded(b"x\xE2\x82"), "x?");
        assert_eq!(decoded(b"x\xF0\x9F\x98"), "x?");
        assert_eq!(decoded(b"x\xF0\x9F"), "x?");
        assert_eq!(decoded(b"x\xE0\x80"), "x??");
        assert_eq!(decoded(b"x\xF4\x90"), "x??");
        assert_eq!(decoded(b"x\xC3"), "x?");
    }

    #[test]
    fn parse_int_follows_java() {
        assert_eq!(parse_int("42"), Some(42));
        assert_eq!(parse_int("+7"), Some(7));
        assert_eq!(parse_int("-0"), Some(0));
        assert_eq!(parse_int("\u{0662}\u{0664}"), Some(24));
        assert_eq!(parse_int("\u{FF13}"), Some(3));
        assert_eq!(parse_int("2147483647"), Some(i32::MAX));
        assert_eq!(parse_int("-2147483648"), Some(i32::MIN));
        assert_eq!(parse_int("2147483648"), None);
        assert_eq!(parse_int(""), None);
        assert_eq!(parse_int("+"), None);
        assert_eq!(parse_int(" 1"), None);
        assert_eq!(parse_int("1a"), None);
        // MATHEMATICAL BOLD DIGIT ONE is two units in Java, neither a digit
        assert_eq!(parse_int("\u{1D7CF}"), None);
    }

    #[test]
    fn equals_ignore_case_matches_java() {
        assert!(equals_ignore_case_ascii("REGION", "region"));
        assert!(equals_ignore_case_ascii("reg\u{131}on", "region"));
        assert!(equals_ignore_case_ascii("reg\u{130}on", "region"));
        assert!(equals_ignore_case_ascii("\u{17F}", "s"));
        assert!(equals_ignore_case_ascii("\u{212A}", "k"));
        assert!(equals_ignore_case_ascii("primarydata", "primaryData"));
        assert!(!equals_ignore_case_ascii("regions", "region"));
        assert!(!equals_ignore_case_ascii("regio", "region"));
        assert!(!equals_ignore_case_ascii("g:region", "region"));
        assert!(!equals_ignore_case_ascii("r\u{e9}gion", "region"));
    }

    #[test]
    fn splits() {
        assert_eq!(split_on_spaces(" 1  2 "), ["1", "2"]);
        assert!(split_on_spaces("   ").is_empty());
        assert_eq!(split_space("1  2 "), ["1", "", "2"]);
        assert_eq!(split_space(" 1 2"), ["", "1", "2"]);
        assert_eq!(split_space(""), [""]);
        assert!(split_space("   ").is_empty());
    }

    #[test]
    fn string_hash_is_java_s() {
        // "Aa" and "BB" collide in Java
        assert_eq!(string_hash("Aa"), 2112);
        assert_eq!(string_hash("BB"), 2112);
        assert_eq!(string_hash(""), 0);
        assert_eq!(string_hash("f.text"), {
            let mut h = 0i32;
            for u in "f.text".encode_utf16() {
                h = h.wrapping_mul(31).wrapping_add(i32::from(u));
            }
            h
        });
        // a character outside the BMP counts as its two surrogates
        assert_eq!(string_hash("\u{1F600}"), 0xD83D * 31 + 0xDE00);
    }

    fn order_of(keys: &[i32]) -> Vec<i32> {
        let mut o = HashOrder::default();
        for (i, &k) in keys.iter().enumerate() {
            o.insert(k, i);
        }
        o.order().map(|i| keys[i]).collect()
    }

    #[test]
    fn hash_order_walks_buckets_then_insertion_order() {
        // 33 and 17 share bucket 1 of 16 with 1; 40 is in bucket 8
        assert_eq!(order_of(&[33, 17, 1, 2, 40]), [33, 17, 1, 2, 40]);
        assert_eq!(order_of(&[40, 2, 1, 17, 33]), [1, 17, 33, 2, 40]);
    }

    #[test]
    fn hash_order_doubles_past_three_quarters() {
        // the thirteenth key doubles the table to 32 buckets, which puts
        // 16 after 15 rather than next to 0
        let keys: Vec<i32> = (0..13).map(|i| if i == 1 { 16 } else { i }).collect();
        assert_eq!(order_of(&keys), [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16]);
        let twelve: Vec<i32> = (0..12).map(|i| if i == 1 { 16 } else { i }).collect();
        assert_eq!(order_of(&twelve), [0, 16, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
    }

    #[test]
    fn hash_order_doubles_on_a_ninth_key_in_one_bucket() {
        // nine keys in bucket 0 of 16: the ninth doubles the table, which
        // moves the odd multiples of 16 to bucket 16
        let keys: Vec<i32> = (0..9).map(|i| i * 16).collect();
        assert_eq!(order_of(&keys), [0, 32, 64, 96, 128, 16, 48, 80, 112]);
        let eight: Vec<i32> = (0..8).map(|i| i * 16).collect();
        assert_eq!(order_of(&eight), eight);
    }

    #[test]
    fn hash_order_spreads_the_high_bits() {
        // 65536 spreads to 65537, so it lands in bucket 1, after 1
        assert_eq!(order_of(&[65536, 1, 0]), [0, 65536, 1]);
    }
}
