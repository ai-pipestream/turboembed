//! Atomic fragment generators: foreign words, numbers, money, quantities, URLs, emails, and dates.

use super::Builder;
use super::tables::{
    ARABIC_WORDS, CJK_WORDS, CURRENCY_CODES, CURRENCY_SYMBOLS, CYRILLIC_WORDS, DE_WORDS, DEVANAGARI_WORDS, DOMAINS,
    EMAIL_LOCAL, ES_PT_WORDS, FR_WORDS, GREEK_WORDS, HEBREW_WORDS, PATHS, SCALE_WORDS, THAI_WORDS, UNICODE_DIGITS,
    UNITS,
};

impl Builder<'_> {
    pub(super) fn foreign_word(&mut self) {
        let list: &[&str] = match self.rng.below(10) {
            0 => DE_WORDS,
            1 => FR_WORDS,
            2 => ES_PT_WORDS,
            3 => GREEK_WORDS,
            4 => CYRILLIC_WORDS,
            5 => ARABIC_WORDS,
            6 => HEBREW_WORDS,
            7 => DEVANAGARI_WORDS,
            8 => THAI_WORDS,
            _ => CJK_WORDS,
        };
        let w = *self.rng.pick(list);
        self.out.push_str(w);
    }

    fn digits(&mut self, n: usize) {
        for _ in 0..n {
            let d = b'0' + self.rng.below(10) as u8;
            self.out.push(d as char);
        }
    }

    /// Integers, grouped either way, decimals, signs, scale words, Unicode digits.
    pub(super) fn number(&mut self) {
        match self.rng.below(100) {
            0..=24 => {
                let n = self.rng.range(1, 7);
                self.digits(n);
            }
            25..=39 => {
                // Grouped: US or EU, grouping and decimal marks.
                let eu = self.rng.chance(50);
                let groups = self.rng.range(2, 4);
                {
                    let n = self.rng.range(1, 3);
                    self.digits(n);
                }
                for _ in 1..groups {
                    self.out.push(if eu { '.' } else { ',' });
                    self.digits(3);
                }
                if self.rng.chance(50) {
                    self.out.push(if eu { ',' } else { '.' });
                    {
                        let n = self.rng.range(1, 3);
                        self.digits(n);
                    }
                }
            }
            40..=54 => {
                if self.rng.chance(40) {
                    let s = *self.rng.pick(&["-", "+", "−", "–"]);
                    self.out.push_str(s);
                }
                {
                    let n = self.rng.range(1, 4);
                    self.digits(n);
                }
                let mark = *self.rng.pick(&[".", ",", "·", "٫"]);
                self.out.push_str(mark);
                {
                    let n = self.rng.range(1, 6);
                    self.digits(n);
                }
            }
            55..=64 => {
                {
                    let n = self.rng.range(1, 3);
                    self.digits(n);
                }
                if self.rng.chance(50) {
                    self.out.push(' ');
                }
                let s = *self.rng.pick(SCALE_WORDS);
                self.out.push_str(s);
            }
            65..=74 => {
                let d = *self.rng.pick(UNICODE_DIGITS);
                self.out.push_str(d);
            }
            75..=84 => {
                {
                    let n = self.rng.range(1, 2);
                    self.digits(n);
                }
                let s = *self.rng.pick(&["st", "nd", "rd", "th", ".", "º", "ª", "e", "er"]);
                self.out.push_str(s);
            }
            85..=89 => {
                self.digits(1);
                let s = *self.rng.pick(&["/", "⁄", " / "]);
                self.out.push_str(s);
                self.digits(1);
            }
            90..=94 => {
                let s = *self.rng.pick(&["0x1F", "1e10", "2E-3", "0.", ".5", "1..2", "1.2.3", "3,", ",5", "--5", "5-"]);
                self.out.push_str(s);
            }
            _ => {
                // A long run of digits.
                let n = self.rng.range(20, 60);
                self.digits(n);
            }
        }
    }

    pub(super) fn money(&mut self) {
        if self.rng.chance(20) {
            self.out.push('-');
        }
        if self.rng.chance(65) {
            let s = *self.rng.pick(CURRENCY_SYMBOLS);
            self.out.push_str(s);
            if self.rng.chance(15) {
                self.out.push(' ');
            }
            self.number();
        } else {
            self.number();
            let s = *self.rng.pick(&[" ", "", "\u{00A0}"]);
            self.out.push_str(s);
            if self.rng.chance(50) {
                let c = *self.rng.pick(CURRENCY_CODES);
                self.out.push_str(c);
            } else {
                let c = *self.rng.pick(CURRENCY_SYMBOLS);
                self.out.push_str(c.trim());
            }
        }
    }

    pub(super) fn quantity(&mut self) {
        if self.rng.chance(15) {
            self.out.push('-');
        }
        self.number();
        let s = *self.rng.pick(&[" ", " ", "", "\u{00A0}", "\u{202F}"]);
        self.out.push_str(s);
        let u = *self.rng.pick(UNITS);
        self.out.push_str(u);
    }

    pub(super) fn url_or_email(&mut self) {
        if self.rng.chance(35) {
            let l = *self.rng.pick(EMAIL_LOCAL);
            self.out.push_str(l);
            self.out.push('@');
            let d = *self.rng.pick(DOMAINS);
            self.out.push_str(d);
            return;
        }
        let scheme = *self.rng.pick(&["https://", "http://", "www.", "", "ftp://", "HTTPS://", "mailto:"]);
        self.out.push_str(scheme);
        let d = *self.rng.pick(DOMAINS);
        self.out.push_str(d);
        let p = *self.rng.pick(PATHS);
        self.out.push_str(p);
    }

    pub(super) fn date_or_time(&mut self) {
        match self.rng.below(6) {
            0 => {
                self.out.push_str("20");
                self.digits(2);
                self.out.push('-');
                self.out.push('0');
                self.digits(1);
                self.out.push('-');
                self.out.push('1');
                self.digits(1);
            }
            1 => {
                self.digits(1);
                self.out.push('/');
                self.digits(2);
                self.out.push('/');
                self.digits(4);
            }
            2 => {
                self.digits(2);
                self.out.push('.');
                self.digits(2);
                self.out.push('.');
                self.digits(4);
            }
            3 => {
                self.digits(2);
                self.out.push(':');
                self.digits(2);
                if self.rng.chance(50) {
                    let s = *self.rng.pick(&[" p.m.", " a.m.", "pm", " PM", ":00"]);
                    self.out.push_str(s);
                }
            }
            4 => {
                let m = *self.rng.pick(&["Jan.", "Feb.", "March", "Dec.", "Sept.", "Oct"]);
                self.out.push_str(m);
                self.out.push(' ');
                {
                    let n = self.rng.range(1, 2);
                    self.digits(n);
                }
                if self.rng.chance(50) {
                    self.out.push_str(", 20");
                    self.digits(2);
                }
            }
            _ => {
                let s = *self.rng.pick(&["+1 (800) 555-0199", "555-1234", "+44 20 7946 0958", "1-800-FLOWERS"]);
                self.out.push_str(s);
            }
        }
    }
}
