//! The fragments the generator builds texts from, and the builder itself.
//!
//! A text is a few lines; a line is usually one or more sentences built
//! from a small grammar (a subject, a verb, an object and trimmings such
//! as an abbreviation, a number, a sum of money, a quantity, a URL, a quote
//! or a date), so that the sentence and token models see contexts like the
//! ones they were trained on, with the hard cases placed inside them. The
//! other kinds of line are the ones that break segmenters on their own: an
//! empty line, a whitespace only line, a very long token, a run of CJK or
//! Thai without spaces, right to left text with bidi controls, a heading, a
//! list item, a table row. The whole text is then mutated: zero width and
//! format characters inside words, combining marks on letters, Unicode
//! spaces for ASCII ones, CR LF or lone CR line ends, a byte order mark, no
//! final line end.
//!
//! Every character used is in the Unicode 15.0 repertoire, which is what
//! JDK 21 implements, so a difference between the two sides is never one of
//! Unicode version. The texts are valid UTF-8 by construction, so no lone
//! surrogate can be in an input: the Java side reads inputs as UTF-8 and
//! refuses a malformed file.

mod atoms;
mod lines;
mod tables;

use super::rng::Rng;
use tables::{
    ABBREVIATIONS, BIDI, CLOSE_QUOTES, COMBINING, CONTRACTIONS, CONTROLS, DASHES, EMOJI, EN_NAMES, EN_VERBS, EN_WORDS,
    OPEN_QUOTES, SEPARATORS, SOCIAL, SPACES, SYMBOLS, TERMINALS, TITLES, ZERO_WIDTH,
};

/// A fragment builder over one random source. Every method appends to the
/// text; the document method is the entry point.
pub struct Builder<'r> {
    pub(super) rng: &'r mut Rng,
    pub(super) out: String,
}

impl<'r> Builder<'r> {
    pub fn new(rng: &'r mut Rng) -> Builder<'r> {
        Builder { rng, out: String::new() }
    }

    /// A whole text, mutated.
    pub fn document(mut self) -> String {
        let lines = match self.rng.below(100) {
            0..=30 => 1,
            31..=60 => 2,
            61..=80 => 3,
            81..=92 => self.rng.range(4, 6),
            _ => self.rng.range(7, 14),
        };
        for i in 0..lines {
            self.line();
            if i + 1 < lines || self.rng.chance(70) {
                self.out.push('\n');
            }
        }
        let text = std::mem::take(&mut self.out);
        self.mutate(text)
    }

    fn line(&mut self) {
        match self.rng.below(100) {
            0..=44 => {
                let n = self.rng.range(1, 4);
                for i in 0..n {
                    self.sentence();
                    if i + 1 < n {
                        let sep = *self.rng.pick(SEPARATORS);
                        self.out.push_str(sep);
                    }
                }
            }
            45..=52 => self.fragment(),
            53..=57 => {}
            58..=62 => {
                let n = self.rng.range(1, 5);
                for _ in 0..n {
                    let s = *self.rng.pick(SPACES);
                    self.out.push_str(if s.len() > 3 { " " } else { s });
                }
            }
            63..=67 => self.long_token(),
            68..=72 => self.cjk_run(),
            73..=76 => self.rtl_line(),
            77..=79 => self.heading(),
            80..=84 => self.list_item(),
            85..=87 => self.table_row(),
            88..=90 => self.code_line(),
            91..=93 => self.hyphenated_lines(),
            94..=96 => self.mixed_script_line(),
            _ => self.dense_punctuation(),
        }
        if self.rng.chance(10) {
            let s = *self.rng.pick(SPACES);
            self.out.push_str(s);
        }
    }

    /// Subject, verb, object and trimmings, then a terminal.
    pub(super) fn sentence(&mut self) {
        if self.rng.chance(8) {
            let q = *self.rng.pick(OPEN_QUOTES);
            self.out.push_str(q);
        }
        self.subject();
        self.out.push(' ');
        let verb = *self.rng.pick(EN_VERBS);
        self.out.push_str(verb);
        self.out.push(' ');
        self.object();
        let trimmings = self.rng.range(0, 3);
        for _ in 0..trimmings {
            if self.rng.chance(30) {
                self.out.push(',');
            }
            self.out.push(' ');
            self.trimming();
        }
        if self.rng.chance(10) {
            self.out.push(' ');
            let q = *self.rng.pick(CLOSE_QUOTES);
            self.out.push_str(q);
        }
        let terminal = *self.rng.pick(TERMINALS);
        self.out.push_str(terminal);
    }

    fn subject(&mut self) {
        match self.rng.below(100) {
            0..=34 => {
                if self.rng.chance(60) {
                    let t = *self.rng.pick(TITLES);
                    self.out.push_str(t);
                    self.out.push(' ');
                }
                let n = *self.rng.pick(EN_NAMES);
                self.out.push_str(n);
            }
            35..=69 => {
                let w = *self.rng.pick(EN_WORDS);
                self.push_capitalized(w);
                self.out.push(' ');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            70..=79 => self.foreign_word(),
            80..=89 => self.number(),
            90..=94 => {
                let a = *self.rng.pick(ABBREVIATIONS);
                self.out.push_str(a);
            }
            _ => {
                let e = *self.rng.pick(EMOJI);
                self.out.push_str(e);
            }
        }
    }

    fn object(&mut self) {
        match self.rng.below(100) {
            0..=39 => {
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(50) {
                    self.out.push(' ');
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
            }
            40..=54 => self.money(),
            55..=69 => self.quantity(),
            70..=79 => self.number(),
            80..=89 => self.url_or_email(),
            _ => self.foreign_word(),
        }
    }

    /// One more thing a sentence can carry.
    fn trimming(&mut self) {
        match self.rng.below(100) {
            0..=9 => {
                let a = *self.rng.pick(ABBREVIATIONS);
                self.out.push_str(a);
                self.out.push(' ');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            10..=19 => self.number(),
            20..=27 => self.money(),
            28..=35 => self.quantity(),
            36..=43 => self.url_or_email(),
            44..=51 => self.date_or_time(),
            52..=59 => {
                let q = *self.rng.pick(OPEN_QUOTES);
                self.out.push_str(q);
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(40) {
                    let t = *self.rng.pick(TERMINALS);
                    self.out.push_str(t);
                }
                let q = *self.rng.pick(CLOSE_QUOTES);
                self.out.push_str(q);
            }
            60..=66 => {
                self.out.push('(');
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
                if self.rng.chance(40) {
                    self.out.push(' ');
                    self.number();
                }
                self.out.push(')');
            }
            67..=72 => {
                let c = *self.rng.pick(CONTRACTIONS);
                self.out.push_str(c);
            }
            73..=78 => {
                let s = *self.rng.pick(SOCIAL);
                self.out.push_str(s);
            }
            79..=84 => {
                let e = *self.rng.pick(EMOJI);
                self.out.push_str(e);
            }
            85..=89 => {
                let d = *self.rng.pick(DASHES);
                self.out.push_str(d);
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            90..=94 => {
                let s = *self.rng.pick(SYMBOLS);
                self.out.push_str(s);
            }
            _ => self.foreign_word(),
        }
    }

    fn fragment(&mut self) {
        let n = self.rng.range(1, 5);
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            match self.rng.below(5) {
                0 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                1 => self.number(),
                2 => self.foreign_word(),
                3 => {
                    let a = *self.rng.pick(ABBREVIATIONS);
                    self.out.push_str(a);
                }
                _ => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
            }
        }
    }

    fn push_capitalized(&mut self, w: &str) {
        let mut chars = w.chars();
        if let Some(c) = chars.next() {
            self.out.extend(c.to_uppercase());
            self.out.push_str(chars.as_str());
        }
    }

    /// The whole text changed in ways that reach into tokens: zero width
    /// and format characters inside words, combining marks after letters,
    /// Unicode spaces for ASCII spaces, control characters, other line ends,
    /// a byte order mark, repeated characters.
    fn mutate(&mut self, text: String) -> String {
        let mut text = text;
        if self.rng.chance(20) {
            text = self.insert_between_letters(&text, ZERO_WIDTH, 6);
        }
        if self.rng.chance(12) {
            text = self.insert_between_letters(&text, BIDI, 4);
        }
        if self.rng.chance(22) {
            text = self.insert_between_letters(&text, COMBINING, 8);
        }
        if self.rng.chance(18) {
            let space = *self.rng.pick(SPACES);
            let every = self.rng.range(1, 4);
            let mut n = 0;
            text = text
                .chars()
                .flat_map(|c| {
                    if c == ' ' {
                        n += 1;
                        if n % every == 0 {
                            return space.chars().collect::<Vec<_>>();
                        }
                    }
                    vec![c]
                })
                .collect();
        }
        if self.rng.chance(5) {
            text = self.insert_between_letters(&text, CONTROLS, 2);
        }
        match self.rng.below(100) {
            0..=19 => text = text.replace("\r\n", "\n").replace('\n', "\r\n"),
            20..=25 => text = text.replace("\r\n", "\n").replace('\n', "\r"),
            26..=29 => text = text.replace("\r\n", "\n").replace('\n', "\n\n"),
            30..=32 => text = text.replace("\r\n", "\n").replace('\n', " \n"),
            _ => {}
        }
        let bom = self.rng.chance(5);
        if self.rng.chance(8) {
            let lead = *self.rng.pick(SPACES);
            text.insert_str(0, lead);
        }
        if bom {
            text.insert(0, '\u{FEFF}');
        }
        text
    }

    /// `pieces` dropped in after letters, about one in `one_in` of them,
    /// so that the insertions land inside words.
    fn insert_between_letters(&mut self, text: &str, pieces: &[&str], per_hundred: usize) -> String {
        let mut out = String::with_capacity(text.len() + 16);
        for c in text.chars() {
            out.push(c);
            if c.is_alphabetic() && self.rng.chance(per_hundred) {
                let p = *self.rng.pick(pieces);
                out.push_str(p);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutate_keeps_bom_at_byte_zero_when_lead_space_also_fires() {
        let mut saw_both = false;
        for idx in 0..2000 {
            let mut rng = Rng::for_input(99, idx);
            let out = Builder::new(&mut rng).mutate("123\r\n456\n".into());
            assert!(!out.contains("\r\r"), "unexpected doubled CR in {out:?}");
            if out.contains('\u{FEFF}') {
                assert!(out.starts_with('\u{FEFF}'), "BOM displaced from byte 0 in {out:?}");
                let after_bom = &out['\u{FEFF}'.len_utf8()..];
                if !after_bom.starts_with('1') {
                    saw_both = true;
                }
            }
        }
        assert!(saw_both, "expected at least one seed where both BOM and leading space fired");
    }
}
