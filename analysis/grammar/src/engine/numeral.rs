//! Reading a written numeral: digits with group and decimal separators.
//!
//! The reader looks at a whole chain of digit runs joined by single
//! separators and decides what the chain is: one number (`1,234,567.89`,
//! `1.111,11`, `12,34,567`, `1 234 567`), or a list of separate integers
//! (`192.168.1.2`, `22,24,26`). Which shapes count as one number is set by
//! a [`NumeralStyle`].

use crate::dec::Dec;

/// How a language writes numerals with digits.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct NumeralStyle {
    /// Whether `.` and `,` are both read as decimal or group separators,
    /// chosen by the shape of the chain: `1.111,11` and `1,111.11` are
    /// the same number. A lone dot always marks a fraction (`4.800` is
    /// 4.8); a lone comma groups when three digits follow (`12,567`) and
    /// marks a fraction otherwise (`1,3`). With `false`, `,` only groups
    /// and `.` only marks a fraction.
    pub mixed_separators: bool,
    /// Whether a single space, no-break space, narrow no-break space or
    /// thin space between groups of three digits groups them: `1 234 567`.
    pub space_groups: bool,
    /// Whether Indian grouping with commas is read: `12,34,567`.
    pub indian_groups: bool,
}

/// What a chain of digits starting at a position is.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Reading {
    /// One number, ending at byte `end`.
    One { end: usize, value: Dec },
    /// A chain that is no single number: every digit run in it up to byte
    /// `end` is an integer of its own.
    List { end: usize },
}

/// The most separators a chain may have before it is read as a list.
const MAX_RUNS: usize = 64;

/// A separator between two runs of a chain.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum Sep {
    Dot,
    Comma,
    Space,
}

/// The length in bytes of a grouping space at `i`, if there is one.
fn space_at(b: &[u8], i: usize) -> Option<usize> {
    match b.get(i..) {
        Some([b' ', ..]) => Some(1),
        // U+00A0 NO-BREAK SPACE
        Some([0xC2, 0xA0, ..]) => Some(2),
        // U+202F NARROW NO-BREAK SPACE, U+2009 THIN SPACE
        Some([0xE2, 0x80, 0xAF | 0x89, ..]) => Some(3),
        _ => None,
    }
}

/// The end of the run of ASCII digits starting at `i`.
pub fn digits_end(b: &[u8], mut i: usize) -> usize {
    while b.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    i
}

impl NumeralStyle {
    /// Reads the chain starting at byte `start`, which holds an ASCII
    /// digit. `buf` is scratch space for the digits.
    pub fn read(&self, text: &str, start: usize, buf: &mut Vec<u8>) -> Reading {
        let b = text.as_bytes();
        debug_assert!(b[start].is_ascii_digit());
        // Runs as (start, end) and the separator before each run after the
        // first.
        let mut runs: [(usize, usize); MAX_RUNS] = [(0, 0); MAX_RUNS];
        let mut seps: [Sep; MAX_RUNS] = [Sep::Dot; MAX_RUNS];
        let mut n = 1;
        runs[0] = (start, digits_end(b, start));
        loop {
            let at = runs[n - 1].1;
            let (sep, width) = match b.get(at) {
                Some(b'.') => (Sep::Dot, 1),
                Some(b',') => (Sep::Comma, 1),
                _ => match space_at(b, at) {
                    Some(w) if self.space_groups => (Sep::Space, w),
                    _ => break,
                },
            };
            let run_start = at + width;
            if !b.get(run_start).is_some_and(u8::is_ascii_digit) {
                break;
            }
            let run_end = digits_end(b, run_start);
            if sep == Sep::Space {
                // A space groups only three digits after one to three, in
                // a chain that has nothing but spaces before.
                let spaces_only = seps[1..n].iter().all(|&s| s == Sep::Space);
                if run_end - run_start != 3 || runs[0].1 - runs[0].0 > 3 || !spaces_only {
                    break;
                }
            }
            if n == MAX_RUNS {
                return Reading::List { end: run_end };
            }
            runs[n] = (run_start, run_end);
            seps[n] = sep;
            n += 1;
        }
        let end = runs[n - 1].1;
        let len = |i: usize| runs[i].1 - runs[i].0;
        let (int_runs, frac_run) = if n == 1 || self.groups(&seps[1..n], &len, n, false) {
            (n, None)
        } else if self.decimal_after(&seps[1..n], &len, n) {
            (n - 1, Some(n - 1))
        } else {
            return Reading::List { end };
        };
        buf.clear();
        for &(s, e) in &runs[..int_runs] {
            buf.extend_from_slice(&b[s..e]);
        }
        let int_len = buf.len();
        if let Some(f) = frac_run {
            buf.extend_from_slice(&b[runs[f].0..runs[f].1]);
        }
        Reading::One { end, value: Dec::from_digits(&buf[..int_len], &buf[int_len..]) }
    }

    /// Whether runs `0..n`, joined by `seps`, are one grouped integer.
    /// `prefix` says the chain goes on with a fraction, which lets a lone
    /// dot group (`1.111,11`).
    fn groups(&self, seps: &[Sep], len: &impl Fn(usize) -> usize, n: usize, prefix: bool) -> bool {
        let g = seps[0];
        if seps.iter().any(|&s| s != g) || n < 2 {
            return false;
        }
        let allowed = match g {
            Sep::Space => true,
            Sep::Comma => true,
            // A dot groups only where it cannot be the decimal point.
            Sep::Dot => self.mixed_separators && (n > 2 || prefix),
        };
        if !allowed {
            return false;
        }
        let thousands = (1..=3).contains(&len(0)) && (1..n).all(|i| len(i) == 3);
        let indian = self.indian_groups
            && g == Sep::Comma
            && n >= 3
            && (1..=2).contains(&len(0))
            && (1..n - 1).all(|i| len(i) == 2)
            && len(n - 1) == 3;
        thousands || indian
    }

    /// Whether the last run is a fraction after a grouped (or plain)
    /// integer part.
    fn decimal_after(&self, seps: &[Sep], len: &impl Fn(usize) -> usize, n: usize) -> bool {
        let last = seps[seps.len() - 1];
        let decimal = match last {
            Sep::Dot => true,
            Sep::Comma => self.mixed_separators,
            Sep::Space => false,
        };
        if !decimal {
            return false;
        }
        if n == 2 {
            return true;
        }
        let prefix = &seps[..seps.len() - 1];
        prefix.iter().all(|&s| s != last) && self.groups(prefix, len, n - 1, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EN: NumeralStyle = NumeralStyle { mixed_separators: true, space_groups: true, indian_groups: true };

    fn one(text: &str) -> (usize, String) {
        match EN.read(text, 0, &mut Vec::new()) {
            Reading::One { end, value } => (end, value.to_string()),
            r => panic!("{text:?}: {r:?}"),
        }
    }

    fn list(text: &str) -> usize {
        match EN.read(text, 0, &mut Vec::new()) {
            Reading::List { end } => end,
            r => panic!("{text:?}: {r:?}"),
        }
    }

    #[test]
    fn plain_and_grouped_integers() {
        assert_eq!(one("192."), (3, "192".into()));
        assert_eq!(one("1,234,567 x"), (9, "1234567".into()));
        assert_eq!(one("12,567"), (6, "12567".into()));
        assert_eq!(one("22,240,260,310."), (14, "22240260310".into()));
        assert_eq!(one("12,34,567"), (9, "1234567".into()));
        assert_eq!(one("1.234.567"), (9, "1234567".into()));
        assert_eq!(one("0000"), (4, "0".into()));
    }

    #[test]
    fn decimals_in_either_convention() {
        assert_eq!(one("4.800"), (5, "4.8".into()));
        assert_eq!(one("1,3."), (3, "1.3".into()));
        assert_eq!(one("5,00"), (4, "5".into()));
        assert_eq!(one("1.111,11, 1"), (8, "1111.11".into()));
        assert_eq!(one("1,111.11"), (8, "1111.11".into()));
        assert_eq!(one("12,34,567.89"), (12, "1234567.89".into()));
        assert_eq!(one("1,674,200.5 M"), (11, "1674200.5".into()));
    }

    #[test]
    fn spaces_group_threes() {
        assert_eq!(one("1 234 567"), (9, "1234567".into()));
        assert_eq!(one("40\u{202f}000 is"), (8, "40000".into()));
        assert_eq!(one("1\u{202f}234\u{202f}567.89"), (16, "1234567.89".into()));
        assert_eq!(one("423 0000"), (3, "423".into()));
        assert_eq!(one("1234 567"), (4, "1234".into()));
        assert_eq!(one("2  1/4"), (1, "2".into()));
        assert_eq!(one("12,345 678"), (6, "12345".into()));
    }

    #[test]
    fn chains_that_are_no_number_are_lists() {
        assert_eq!(list("192.168.1.2"), 11);
        assert_eq!(list("22,24,26,31."), 11);
        assert_eq!(list("1.2.3"), 5);
        assert_eq!(list("1,234.5.6"), 9);
    }

    #[test]
    fn strict_style_reads_one_convention() {
        let strict = NumeralStyle { mixed_separators: false, space_groups: false, indian_groups: false };
        let read = |t: &str| strict.read(t, 0, &mut Vec::new());
        assert_eq!(read("1,3"), Reading::List { end: 3 });
        assert!(matches!(read("1.234.567"), Reading::List { .. }));
        assert!(matches!(read("1,234.5"), Reading::One { end: 7, .. }));
        assert!(matches!(read("1 234"), Reading::One { end: 1, .. }));
        assert!(matches!(read("12,34,567"), Reading::List { .. }));
    }
}
