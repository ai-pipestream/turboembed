//! Compiled character classes and predefined Java regex character sets.

/// The largest code point.
pub(super) const MAX: u32 = 0x10FFFF;

/// A compiled character class with an O(1) ASCII bitmap and sorted
/// disjoint non-ASCII code point ranges.
#[derive(Clone, Debug)]
pub(super) struct CharClass {
    ascii: u128,
    ranges: Box<[(u32, u32)]>,
}

impl CharClass {
    #[inline]
    pub(super) fn matches(&self, c: char) -> bool {
        let cp = c as u32;
        if cp < 0x80 {
            (self.ascii & (1u128 << cp)) != 0
        } else {
            self.ranges
                .binary_search_by(|&(lo, hi)| {
                    if cp < lo {
                        std::cmp::Ordering::Greater
                    } else if cp > hi {
                        std::cmp::Ordering::Less
                    } else {
                        std::cmp::Ordering::Equal
                    }
                })
                .is_ok()
        }
    }
}

/// A set of code points as sorted, disjoint, inclusive ranges.
#[derive(Clone, Debug, Default)]
pub(super) struct Set(Vec<(u32, u32)>);

impl Set {
    pub(super) fn add(&mut self, lo: u32, hi: u32) {
        self.0.push((lo, hi));
    }

    /// Adds `[lo, hi]`, and with `ci` each ASCII letter whose other case is
    /// in it: Java's `CIRange` and `BitClass.add` without `UNICODE_CASE`.
    pub(super) fn add_range(&mut self, lo: u32, hi: u32, ci: bool) {
        self.add(lo, hi);
        if ci {
            for c in (b'A'..=b'Z').chain(b'a'..=b'z') {
                let other = (c ^ 0x20) as u32;
                if (lo..=hi).contains(&other) {
                    self.add(c as u32, c as u32);
                }
            }
        }
    }

    pub(super) fn add_all(&mut self, ranges: &[(u32, u32)]) {
        self.0.extend_from_slice(ranges);
    }

    pub(super) fn add_complement(&mut self, ranges: &[(u32, u32)]) {
        let mut next = 0;
        for &(lo, hi) in ranges {
            if lo > next {
                self.add(next, lo - 1);
            }
            next = hi + 1;
        }
        if next <= MAX {
            self.add(next, MAX);
        }
    }

    pub(super) fn normalize(&mut self) {
        self.0.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(self.0.len());
        for &(lo, hi) in &self.0 {
            match out.last_mut() {
                Some(last) if lo <= last.1.saturating_add(1) => last.1 = last.1.max(hi),
                _ => out.push((lo, hi)),
            }
        }
        self.0 = out;
    }

    pub(super) fn negate(&mut self) {
        self.normalize();
        let ranges = std::mem::take(&mut self.0);
        self.add_complement(&ranges);
    }

    pub(super) fn intersects(&self, lo: u32, hi: u32) -> bool {
        self.0.iter().any(|&(a, b)| a <= hi && lo <= b)
    }

    pub(super) fn covers(&self, lo: u32, hi: u32) -> bool {
        self.0.iter().any(|&(a, b)| a <= lo && hi <= b)
    }

    /// Converts the normalized set into a [`CharClass`]. Surrogates are
    /// excluded because no `&str` holds one.
    pub(super) fn into_char_class(mut self) -> CharClass {
        self.normalize();
        let mut ascii = 0u128;
        let mut ranges = Vec::new();
        for (lo, hi) in self.0 {
            if lo < 0x80 {
                let ascii_hi = hi.min(0x7F);
                for cp in lo..=ascii_hi {
                    ascii |= 1u128 << cp;
                }
            }
            for (r_lo, r_hi) in [(lo.max(0x80), hi.min(0xD7FF)), (lo.max(0xE000), hi)] {
                if r_lo <= r_hi {
                    ranges.push((r_lo, r_hi));
                }
            }
        }
        CharClass { ascii, ranges: ranges.into_boxed_slice() }
    }
}

const DIGIT: &[(u32, u32)] = &[(0x30, 0x39)];
const WORD: &[(u32, u32)] = &[(0x30, 0x39), (0x41, 0x5A), (0x5F, 0x5F), (0x61, 0x7A)];
const SPACE: &[(u32, u32)] = &[(0x09, 0x0D), (0x20, 0x20)];
const HORIZONTAL: &[(u32, u32)] = &[
    (0x09, 0x09),
    (0x20, 0x20),
    (0xA0, 0xA0),
    (0x1680, 0x1680),
    (0x180E, 0x180E),
    (0x2000, 0x200A),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
];
const VERTICAL: &[(u32, u32)] = &[(0x0A, 0x0D), (0x85, 0x85), (0x2028, 0x2029)];

/// The class a predefined escape letter stands for, and whether negated.
pub(super) fn predefined(letter: u32) -> Option<(&'static [(u32, u32)], bool)> {
    let class = match char::from_u32(letter)? {
        'd' | 'D' => DIGIT,
        'w' | 'W' => WORD,
        's' | 'S' => SPACE,
        'h' | 'H' => HORIZONTAL,
        'v' | 'V' => VERTICAL,
        _ => return None,
    };
    Some((class, (letter as u8).is_ascii_uppercase()))
}
