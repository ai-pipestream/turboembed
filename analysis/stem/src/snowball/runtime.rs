//! The state and primitives the generated Snowball stemmers run on: a port
//! of `SnowballProgram` and `Among` from OpenNLP (which are Snowball's Java
//! runtime), over UTF-16 code units as Java's `char[]` is.
//!
//! The generated code (`algorithms/`) reads and writes the fields directly,
//! as the Java code does with the protected fields of `SnowballProgram`.

/// One entry of a generated suffix or prefix table: the string, the index
/// of the longest entry that is a substring of it (or -1), the result the
/// lookup returns, and the routine that must also succeed for the entry to
/// match.
pub struct Among<T: 'static>(pub &'static [u16], pub i32, pub i32, pub Option<&'static Routine<T>>);

/// A routine of a generated stemmer, with its variables in `T`.
pub type Routine<T> = dyn Fn(&mut SnowballEnv, &mut T) -> bool + Sync;

/// The word being stemmed and the positions in it. Positions are indexes
/// of UTF-16 code units, as in Java.
#[derive(Clone)]
pub struct SnowballEnv {
    pub current: Vec<u16>,
    pub cursor: i32,
    pub limit: i32,
    pub limit_backward: i32,
    pub bra: i32,
    pub ket: i32,
}

impl SnowballEnv {
    /// `setCurrent`: takes the word and resets every position.
    pub fn new(current: Vec<u16>) -> SnowballEnv {
        let limit = current.len() as i32;
        SnowballEnv { current, cursor: 0, limit, limit_backward: 0, bra: 0, ket: limit }
    }

    /// `getCurrent`: the word as it is now.
    pub fn into_current(self) -> Vec<u16> {
        self.current
    }

    #[inline]
    fn at(&self, i: i32) -> u16 {
        self.current[i as usize]
    }

    #[inline]
    fn in_set(s: &[u8], ch: u16, min: u32, max: u32) -> bool {
        let ch = ch as u32;
        if ch > max || ch < min {
            return false;
        }
        let ch = ch - min;
        s[(ch >> 3) as usize] & (1 << (ch & 7)) != 0
    }

    pub fn in_grouping(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor >= self.limit || !Self::in_set(s, self.at(self.cursor), min, max) {
            return false;
        }
        self.cursor += 1;
        true
    }

    pub fn in_grouping_b(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor <= self.limit_backward || !Self::in_set(s, self.at(self.cursor - 1), min, max) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    pub fn out_grouping(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor >= self.limit || Self::in_set(s, self.at(self.cursor), min, max) {
            return false;
        }
        self.cursor += 1;
        true
    }

    pub fn out_grouping_b(&mut self, s: &[u8], min: u32, max: u32) -> bool {
        if self.cursor <= self.limit_backward || Self::in_set(s, self.at(self.cursor - 1), min, max) {
            return false;
        }
        self.cursor -= 1;
        true
    }

    pub fn eq_s(&mut self, s: &[u16]) -> bool {
        let n = s.len() as i32;
        if self.limit - self.cursor < n {
            return false;
        }
        let start = self.cursor as usize;
        if self.current[start..start + s.len()] != *s {
            return false;
        }
        self.cursor += n;
        true
    }

    pub fn eq_s_b(&mut self, s: &[u16]) -> bool {
        let n = s.len() as i32;
        if self.cursor - self.limit_backward < n {
            return false;
        }
        let start = (self.cursor - n) as usize;
        if self.current[start..start + s.len()] != *s {
            return false;
        }
        self.cursor -= n;
        true
    }

    /// Moves forward one code unit. The generated code checks the limit
    /// first.
    pub fn next_char(&mut self) {
        self.cursor += 1;
    }

    /// Moves back one code unit. The generated code checks the limit first.
    pub fn previous_char(&mut self) {
        self.cursor -= 1;
    }

    /// `hop n` forwards: fails when it would pass the limit.
    pub fn hop(&mut self, n: i32) -> bool {
        let c = self.cursor + n;
        if c < self.cursor || c > self.limit {
            return false;
        }
        self.cursor = c;
        true
    }

    /// `hop n` backwards: fails when it would pass the backward limit.
    pub fn hop_back(&mut self, n: i32) -> bool {
        let c = self.cursor - n;
        if c > self.cursor || c < self.limit_backward {
            return false;
        }
        self.cursor = c;
        true
    }

    /// Finds the longest entry of `v` that the text at the cursor starts
    /// with and whose routine (if any) succeeds, and returns its result,
    /// or 0. `v` is sorted by UTF-16 code units.
    pub fn find_among<T>(&mut self, v: &[Among<T>], context: &mut T) -> i32 {
        let mut i: i32 = 0;
        let mut j: i32 = v.len() as i32;
        let c = self.cursor;
        let l = self.limit;
        let mut common_i: i32 = 0;
        let mut common_j: i32 = 0;
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            let w = &v[k as usize];
            // Java's loop runs no times when `common` is past the end.
            for &unit in w.0.get(common as usize..).unwrap_or(&[]) {
                if c + common == l {
                    diff = -1;
                    break;
                }
                diff = self.at(c + common) as i32 - unit as i32;
                if diff != 0 {
                    break;
                }
                common += 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[i as usize];
            let n = w.0.len() as i32;
            if common_i >= n {
                self.cursor = c + n;
                let Some(method) = w.3 else { return w.2 };
                let res = method(self, context);
                self.cursor = c + n;
                if res {
                    return w.2;
                }
            }
            i = w.1;
            if i < 0 {
                return 0;
            }
        }
    }

    /// `find_among` backwards: the longest entry the text before the
    /// cursor ends with.
    pub fn find_among_b<T>(&mut self, v: &[Among<T>], context: &mut T) -> i32 {
        let mut i: i32 = 0;
        let mut j: i32 = v.len() as i32;
        let c = self.cursor;
        let lb = self.limit_backward;
        let mut common_i: i32 = 0;
        let mut common_j: i32 = 0;
        let mut first_key_inspected = false;
        loop {
            let k = i + ((j - i) >> 1);
            let mut diff: i32 = 0;
            let mut common = common_i.min(common_j);
            let w = &v[k as usize];
            for &unit in w.0[..w.0.len().saturating_sub(common as usize)].iter().rev() {
                if c - common == lb {
                    diff = -1;
                    break;
                }
                diff = self.at(c - 1 - common) as i32 - unit as i32;
                if diff != 0 {
                    break;
                }
                common += 1;
            }
            if diff < 0 {
                j = k;
                common_j = common;
            } else {
                i = k;
                common_i = common;
            }
            if j - i <= 1 {
                if i > 0 || j == i || first_key_inspected {
                    break;
                }
                first_key_inspected = true;
            }
        }
        loop {
            let w = &v[i as usize];
            let n = w.0.len() as i32;
            if common_i >= n {
                self.cursor = c - n;
                let Some(method) = w.3 else { return w.2 };
                let res = method(self, context);
                self.cursor = c - n;
                if res {
                    return w.2;
                }
            }
            i = w.1;
            if i < 0 {
                return 0;
            }
        }
    }

    /// Replaces the units from `c_bra` to `c_ket` with `s` and moves the
    /// limit and the cursor to match; returns the change in length.
    fn replace_s(&mut self, c_bra: i32, c_ket: i32, s: &[u16]) -> i32 {
        let adjustment = s.len() as i32 - (c_ket - c_bra);
        self.current.splice(c_bra as usize..c_ket as usize, s.iter().copied());
        self.limit += adjustment;
        if self.cursor >= c_ket {
            self.cursor += adjustment;
        } else if self.cursor > c_bra {
            self.cursor = c_bra;
        }
        adjustment
    }

    /// Replaces the slice between `bra` and `ket` with `s`. Always
    /// succeeds; the generated code tests the result.
    pub fn slice_from(&mut self, s: &[u16]) -> bool {
        let (bra, ket) = (self.bra, self.ket);
        self.replace_s(bra, ket, s);
        true
    }

    pub fn slice_del(&mut self) -> bool {
        self.slice_from(&[])
    }

    /// Replaces `c_bra..c_ket` with `s`, moving `bra` and `ket` when they
    /// are at or after `c_bra`.
    pub fn insert(&mut self, c_bra: i32, c_ket: i32, s: &[u16]) {
        let adjustment = self.replace_s(c_bra, c_ket, s);
        if c_bra <= self.bra {
            self.bra += adjustment;
        }
        if c_bra <= self.ket {
            self.ket += adjustment;
        }
    }

    /// A copy of the slice between `bra` and `ket`.
    pub fn slice_to(&self) -> Vec<u16> {
        self.current[self.bra as usize..self.ket as usize].to_vec()
    }
}

/// The UTF-16 length of a string, at compile time.
pub const fn utf16_len(s: &str) -> usize {
    let b = s.as_bytes();
    let mut i = 0;
    let mut n = 0;
    while i < b.len() {
        // A four-byte sequence is a supplementary character: two units.
        n += if b[i] >= 0xF0 { 2 } else { 1 };
        i += match b[i] {
            0x00..=0x7F => 1,
            0xC0..=0xDF => 2,
            0xE0..=0xEF => 3,
            _ => 4,
        };
    }
    n
}

/// A string as UTF-16 code units, at compile time. `N` is `utf16_len(s)`.
pub const fn utf16<const N: usize>(s: &str) -> [u16; N] {
    let b = s.as_bytes();
    let mut out = [0u16; N];
    let (mut i, mut o) = (0, 0);
    while i < b.len() {
        let (c, len) = match b[i] {
            0x00..=0x7F => (b[i] as u32, 1),
            0xC0..=0xDF => (((b[i] as u32 & 0x1F) << 6) | (b[i + 1] as u32 & 0x3F), 2),
            0xE0..=0xEF => {
                (((b[i] as u32 & 0x0F) << 12) | ((b[i + 1] as u32 & 0x3F) << 6) | (b[i + 2] as u32 & 0x3F), 3)
            }
            _ => (
                ((b[i] as u32 & 0x07) << 18)
                    | ((b[i + 1] as u32 & 0x3F) << 12)
                    | ((b[i + 2] as u32 & 0x3F) << 6)
                    | (b[i + 3] as u32 & 0x3F),
                4,
            ),
        };
        if c >= 0x10000 {
            let c = c - 0x10000;
            out[o] = 0xD800 | (c >> 10) as u16;
            out[o + 1] = 0xDC00 | (c & 0x3FF) as u16;
            o += 2;
        } else {
            out[o] = c as u16;
            o += 1;
        }
        i += len;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literals_are_utf16() {
        assert_eq!(u16s!(""), &[] as &[u16]);
        assert_eq!(u16s!("ab"), &[0x61, 0x62]);
        assert_eq!(u16s!("l\u{00F8}st"), &[0x6C, 0xF8, 0x73, 0x74]);
        assert_eq!(u16s!("\u{03B1}\u{20AC}\u{1D41A}"), &[0x3B1, 0x20AC, 0xD835, 0xDC1A]);
    }

    #[test]
    fn replace_moves_the_cursor_and_limit() {
        let mut env = SnowballEnv::new("abcdef".encode_utf16().collect());
        env.cursor = 4;
        env.bra = 1;
        env.ket = 3;
        env.slice_from(u16s!("XYZW"));
        assert_eq!(String::from_utf16(&env.current).unwrap(), "aXYZWdef");
        assert_eq!((env.cursor, env.limit), (6, 8));
        env.insert(0, 0, u16s!("<"));
        assert_eq!((env.bra, env.ket, env.limit), (2, 4, 9));
        assert_eq!(String::from_utf16(&env.current).unwrap(), "<aXYZWdef");
    }

    #[test]
    fn hops_stop_at_the_limits() {
        let mut env = SnowballEnv::new("abc".encode_utf16().collect());
        assert!(!env.hop(4));
        assert!(env.hop(3));
        assert!(!env.hop_back(4));
        assert!(env.hop_back(3));
        assert_eq!(env.cursor, 0);
    }
}
