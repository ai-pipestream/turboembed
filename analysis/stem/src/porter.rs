//! OpenNLP's `PorterStemmer`: Martin Porter's Java implementation of his
//! 1980 algorithm (release 3, as changed by Brian Goetz), not the Snowball
//! `porter` algorithm.
//!
//! It departs from the 1980 paper as Porter's own implementations do: step
//! 2 of the paper (here `step3`) maps -bli to -ble where the paper maps
//! -abli to -able, and also maps -logi to -log. Only the letters a, e, i,
//! o and u (and y after a consonant) are vowels; every other code unit, an
//! upper-case vowel, an accented letter or half of a surrogate pair, is a
//! consonant. Words of one or two code units are returned as they are.

use crate::stemmer::Stemmer;

/// OpenNLP's `opennlp.tools.stemmer.PorterStemmer`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct PorterStemmer;

impl Stemmer for PorterStemmer {
    fn stem_utf16(&self, word: &[u16], out: &mut Vec<u16>) {
        out.clear();
        out.extend_from_slice(word);
        let mut s = Porter { b: out, j: 0, k: word.len() as i32 - 1 };
        if s.k > 1 {
            s.step1();
            s.step2();
            s.step3();
            s.step4();
            s.step5();
            s.step6();
        }
        let len = (s.k + 1) as usize;
        out.truncate(len);
    }

    fn config_hash(&self) -> u64 {
        crate::Algorithm::Porter.config_hash()
    }
}

const fn units<const N: usize>(s: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = s[i] as u16;
        i += 1;
    }
    out
}

/// An ASCII literal as UTF-16 code units.
macro_rules! w {
    ($s:literal) => {{
        const U: [u16; $s.len()] = units($s);
        &U
    }};
}

/// The buffer and the two positions of the Java class: `k` is the last
/// unit of the word as it is now, `j` the last unit before the suffix the
/// last successful `ends` matched. The word starts at 0 (`k0`).
struct Porter<'a> {
    b: &'a mut Vec<u16>,
    j: i32,
    k: i32,
}

impl Porter<'_> {
    #[inline]
    fn at(&self, i: i32) -> u16 {
        self.b[i as usize]
    }

    /// Whether the unit at `i` is a consonant: anything but a, e, i, o
    /// and u, where y is a consonant at the start and after a vowel.
    fn cons(&self, mut i: i32) -> bool {
        // A y flips the answer for the unit before it.
        let mut flip = false;
        loop {
            match self.at(i) {
                0x61 | 0x65 | 0x69 | 0x6F | 0x75 => return flip,
                0x79 if i > 0 => {
                    flip = !flip;
                    i -= 1;
                }
                _ => return !flip,
            }
        }
    }

    /// The number of vowel-consonant sequences between 0 and `j`.
    fn m(&self) -> i32 {
        let mut n = 0;
        let mut i = 0;
        loop {
            if i > self.j {
                return n;
            }
            if !self.cons(i) {
                break;
            }
            i += 1;
        }
        i += 1;
        loop {
            loop {
                if i > self.j {
                    return n;
                }
                if self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            n += 1;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
        }
    }

    fn vowel_in_stem(&self) -> bool {
        (0..=self.j).any(|i| !self.cons(i))
    }

    fn doublec(&self, j: i32) -> bool {
        j >= 1 && self.at(j) == self.at(j - 1) && self.cons(j)
    }

    /// Consonant, vowel, consonant ending at `i`, the last not w, x or y.
    fn cvc(&self, i: i32) -> bool {
        if i < 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
            return false;
        }
        // Not w, x or y.
        !matches!(self.at(i), 0x77..=0x79)
    }

    fn ends(&mut self, s: &[u16]) -> bool {
        let l = s.len() as i32;
        let o = self.k - l + 1;
        if o < 0 || self.b[o as usize..(o + l) as usize] != *s {
            return false;
        }
        self.j = self.k - l;
        true
    }

    /// Replaces `j + 1..=k` with `s`.
    fn setto(&mut self, s: &[u16]) {
        let o = (self.j + 1) as usize;
        if o + s.len() > self.b.len() {
            // Never reached: a suffix is only lengthened after a longer one
            // was removed. Java would throw here.
            self.b.resize(o + s.len(), 0);
        }
        self.b[o..o + s.len()].copy_from_slice(s);
        self.k = self.j + s.len() as i32;
    }

    fn r(&mut self, s: &[u16]) {
        if self.m() > 0 {
            self.setto(s);
        }
    }

    /// Plurals and -ed or -ing.
    fn step1(&mut self) {
        if self.at(self.k) == 0x73 {
            if self.ends(w!(b"sses")) {
                self.k -= 2;
            } else if self.ends(w!(b"ies")) {
                self.setto(w!(b"i"));
            } else if self.at(self.k - 1) != 0x73 {
                self.k -= 1;
            }
        }
        if self.ends(w!(b"eed")) {
            if self.m() > 0 {
                self.k -= 1;
            }
        } else if (self.ends(w!(b"ed")) || self.ends(w!(b"ing"))) && self.vowel_in_stem() {
            self.k = self.j;
            if self.ends(w!(b"at")) {
                self.setto(w!(b"ate"));
            } else if self.ends(w!(b"bl")) {
                self.setto(w!(b"ble"));
            } else if self.ends(w!(b"iz")) {
                self.setto(w!(b"ize"));
            } else if self.doublec(self.k) {
                let ch = self.at(self.k);
                if !matches!(ch, 0x6C | 0x73 | 0x7A) {
                    self.k -= 1;
                }
            } else if self.m() == 1 && self.cvc(self.k) {
                self.setto(w!(b"e"));
            }
        }
    }

    /// A final y becomes i when the stem has a vowel.
    fn step2(&mut self) {
        if self.ends(w!(b"y")) && self.vowel_in_stem() {
            let k = self.k as usize;
            self.b[k] = 0x69;
        }
    }

    /// Double suffixes to single ones.
    fn step3(&mut self) {
        if self.k == 0 {
            return;
        }
        let table: &[(&[u16], &[u16])] = match self.at(self.k - 1) {
            0x61 => &[(w!(b"ational"), w!(b"ate")), (w!(b"tional"), w!(b"tion"))],
            0x63 => &[(w!(b"enci"), w!(b"ence")), (w!(b"anci"), w!(b"ance"))],
            0x65 => &[(w!(b"izer"), w!(b"ize"))],
            0x6C => &[
                (w!(b"bli"), w!(b"ble")),
                (w!(b"alli"), w!(b"al")),
                (w!(b"entli"), w!(b"ent")),
                (w!(b"eli"), w!(b"e")),
                (w!(b"ousli"), w!(b"ous")),
            ],
            0x6F => &[(w!(b"ization"), w!(b"ize")), (w!(b"ation"), w!(b"ate")), (w!(b"ator"), w!(b"ate"))],
            0x73 => &[
                (w!(b"alism"), w!(b"al")),
                (w!(b"iveness"), w!(b"ive")),
                (w!(b"fulness"), w!(b"ful")),
                (w!(b"ousness"), w!(b"ous")),
            ],
            0x74 => &[(w!(b"aliti"), w!(b"al")), (w!(b"iviti"), w!(b"ive")), (w!(b"biliti"), w!(b"ble"))],
            0x67 => &[(w!(b"logi"), w!(b"log"))],
            _ => &[],
        };
        self.first_rule(table);
    }

    /// -ic-, -full, -ness and the like.
    fn step4(&mut self) {
        let table: &[(&[u16], &[u16])] = match self.at(self.k) {
            0x65 => &[(w!(b"icate"), w!(b"ic")), (w!(b"ative"), w!(b"")), (w!(b"alize"), w!(b"al"))],
            0x69 => &[(w!(b"iciti"), w!(b"ic"))],
            0x6C => &[(w!(b"ical"), w!(b"ic")), (w!(b"ful"), w!(b""))],
            0x73 => &[(w!(b"ness"), w!(b""))],
            _ => &[],
        };
        self.first_rule(table);
    }

    /// Applies `r` for the first suffix of `table` the word ends with.
    fn first_rule(&mut self, table: &[(&[u16], &[u16])]) {
        for &(suffix, replacement) in table {
            if self.ends(suffix) {
                self.r(replacement);
                return;
            }
        }
    }

    /// -ant, -ence and the like, when the stem before them has m > 1.
    fn step5(&mut self) {
        if self.k == 0 {
            return;
        }
        let found = match self.at(self.k - 1) {
            0x61 => self.ends(w!(b"al")),
            0x63 => self.ends(w!(b"ance")) || self.ends(w!(b"ence")),
            0x65 => self.ends(w!(b"er")),
            0x69 => self.ends(w!(b"ic")),
            0x6C => self.ends(w!(b"able")) || self.ends(w!(b"ible")),
            0x6E => self.ends(w!(b"ant")) || self.ends(w!(b"ement")) || self.ends(w!(b"ment")) || self.ends(w!(b"ent")),
            0x6F => {
                (self.ends(w!(b"ion")) && self.j >= 0 && matches!(self.at(self.j), 0x73 | 0x74)) || self.ends(w!(b"ou"))
            }
            0x73 => self.ends(w!(b"ism")),
            0x74 => self.ends(w!(b"ate")) || self.ends(w!(b"iti")),
            0x75 => self.ends(w!(b"ous")),
            0x76 => self.ends(w!(b"ive")),
            0x7A => self.ends(w!(b"ize")),
            _ => false,
        };
        if found && self.m() > 1 {
            self.k = self.j;
        }
    }

    /// A final -e when m > 1, or m = 1 and not after consonant, vowel,
    /// consonant; and -ll to -l when m > 1.
    fn step6(&mut self) {
        self.j = self.k;
        if self.at(self.k) == 0x65 {
            let a = self.m();
            if a > 1 || a == 1 && !self.cvc(self.k - 1) {
                self.k -= 1;
            }
        }
        if self.at(self.k) == 0x6C && self.doublec(self.k) && self.m() > 1 {
            self.k -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_the_paper_examples() {
        for (word, stem) in [
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("cats", "cat"),
            ("feed", "feed"),
            ("agreed", "agre"),
            ("disabled", "disabl"),
            ("matting", "mat"),
            ("mating", "mate"),
            ("meeting", "meet"),
            ("milling", "mill"),
            ("messing", "mess"),
            ("meetings", "meet"),
            ("ion", "ion"),
            ("as", "as"),
        ] {
            assert_eq!(PorterStemmer.stem(word), stem, "{word}");
        }
    }
}
