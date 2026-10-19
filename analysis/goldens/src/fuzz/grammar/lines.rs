//! Specialized line generators for fuzz inputs: long tokens, CJK runs, RTL lines, headings, lists, tables, code, hyphenated lines, mixed scripts, and dense punctuation.

use super::Builder;
use super::tables::{
    ARABIC_WORDS, BIDI, CJK_PUNCT, CJK_WORDS, CLOSE_QUOTES, COMBINING, DASHES, EMOJI, EN_WORDS, HEBREW_WORDS,
    MATH_LETTERS, OPEN_QUOTES, SYMBOLS, TERMINALS,
};

impl Builder<'_> {
    /// A token far longer than any model saw: 300 to 4000 bytes of one
    /// kind of character, or a long URL, or one emoji many times.
    pub(super) fn long_token(&mut self) {
        let kind = self.rng.below(7);
        let n = self.rng.range(300, 1500);
        match kind {
            0 => {
                for i in 0..n {
                    self.out.push((b'a' + (i % 26) as u8) as char);
                }
            }
            1 => {
                for i in 0..n {
                    self.out.push(if i % 7 == 6 { '-' } else { 'x' });
                }
            }
            2 => {
                self.out.push_str("https://example.com/");
                for i in 0..n {
                    self.out.push(if i % 11 == 10 { '/' } else { (b'a' + (i % 26) as u8) as char });
                }
            }
            3 => {
                let e = *self.rng.pick(EMOJI);
                for _ in 0..n / 8 {
                    self.out.push_str(e);
                }
            }
            4 => {
                for i in 0..n {
                    let w = CJK_WORDS[i % CJK_WORDS.len()];
                    self.out.push_str(w);
                }
            }
            5 => {
                self.out.push('a');
                let m = *self.rng.pick(COMBINING);
                for _ in 0..n / 4 {
                    self.out.push_str(m);
                }
            }
            _ => {
                // 254 letters and then a character outside the BMP, so a
                // cut after 255 UTF-16 units falls inside a pair.
                let k = *self.rng.pick(&[254usize, 255, 253, 509, 510]);
                for _ in 0..k {
                    self.out.push('b');
                }
                self.out.push('😀');
                self.out.push('c');
            }
        }
    }

    /// Chinese, Japanese or Korean without spaces, with its own punctuation.
    pub(super) fn cjk_run(&mut self) {
        let n = self.rng.range(3, 30);
        for _ in 0..n {
            if self.rng.chance(15) {
                let p = *self.rng.pick(CJK_PUNCT);
                self.out.push_str(p);
            }
            let w = *self.rng.pick(CJK_WORDS);
            self.out.push_str(w);
            if self.rng.chance(10) {
                let w = *self.rng.pick(EN_WORDS);
                self.out.push_str(w);
            }
            if self.rng.chance(10) {
                self.number();
            }
        }
        if self.rng.chance(60) {
            let p = *self.rng.pick(&["。", "！", "？", "…", "．", "。」"]);
            self.out.push_str(p);
        }
    }

    /// Arabic or Hebrew words with digits and Latin inside, with bidi
    /// controls around parts of it.
    pub(super) fn rtl_line(&mut self) {
        let n = self.rng.range(2, 10);
        let list = if self.rng.chance(50) { ARABIC_WORDS } else { HEBREW_WORDS };
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            if self.rng.chance(25) {
                let b = *self.rng.pick(BIDI);
                self.out.push_str(b);
            }
            match self.rng.below(10) {
                0 => self.number(),
                1 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                _ => {
                    let w = *self.rng.pick(list);
                    self.out.push_str(w);
                }
            }
            if self.rng.chance(15) {
                let b = *self.rng.pick(BIDI);
                self.out.push_str(b);
            }
        }
        let t = *self.rng.pick(&[".", "؟", "،", "!", "", "۔"]);
        self.out.push_str(t);
    }

    pub(super) fn heading(&mut self) {
        let n = self.rng.range(1, 5);
        for i in 0..n {
            if i > 0 {
                self.out.push(' ');
            }
            let w = *self.rng.pick(EN_WORDS);
            self.out.push_str(&w.to_uppercase());
        }
        if self.rng.chance(30) {
            self.out.push(':');
        }
    }

    pub(super) fn list_item(&mut self) {
        let m =
            *self.rng.pick(&["- ", "* ", "• ", "· ", "1. ", "2) ", "a. ", "IV. ", "– ", "\u{2014}", "\u{2023} ", "①"]);
        self.out.push_str(m);
        self.sentence();
    }

    pub(super) fn table_row(&mut self) {
        let n = self.rng.range(2, 5);
        self.out.push('|');
        for _ in 0..n {
            self.out.push(' ');
            match self.rng.below(3) {
                0 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                1 => self.number(),
                _ => self.money(),
            }
            self.out.push_str(" |");
        }
    }

    pub(super) fn code_line(&mut self) {
        let s = *self.rng.pick(&[
            "x = 5 * 6;",
            "if (a < b && c > d) { return a.b(c); }",
            "SELECT * FROM t WHERE x = 'y';",
            "let s = \"Hello, world!\";",
            "a[0] = b[1] + c[2];",
            "printf(\"%d\\n\", n);",
            "C:\\Users\\name\\file.txt",
            "/usr/local/bin/turbo-goldens",
            "{\"key\": [1, 2.5, -3e4, null]}",
            "<a href=\"https://example.com\">link</a>",
            "f(x) = x² + 2x − 1",
            "∀x∈ℝ: x² ≥ 0",
        ]);
        self.out.push_str(s);
    }

    /// A word broken at a line end with a hyphen or a soft hyphen.
    pub(super) fn hyphenated_lines(&mut self) {
        let w = *self.rng.pick(EN_WORDS);
        self.out.push_str(w);
        self.out.push(' ');
        let (a, b) = self.rng.pick(&[("exam", "ple"), ("sen", "tence"), ("cafe\u{0301}", "teria"), ("Fuß", "ball")]);
        self.out.push_str(a);
        let h = *self.rng.pick(&["-", "\u{00AD}", "‐", "–"]);
        self.out.push_str(h);
        if self.rng.chance(20) {
            self.out.push(' ');
        }
        let le = *self.rng.pick(&["\n", "\r\n", "\r", "\n\n", "\u{2028}"]);
        self.out.push_str(le);
        self.out.push_str(b);
        self.out.push(' ');
        let w = *self.rng.pick(EN_WORDS);
        self.out.push_str(w);
        self.out.push('.');
    }

    /// Words of several scripts and symbol runs side by side.
    pub(super) fn mixed_script_line(&mut self) {
        let n = self.rng.range(3, 12);
        for i in 0..n {
            if i > 0 && self.rng.chance(85) {
                self.out.push(' ');
            }
            match self.rng.below(6) {
                0 | 1 => self.foreign_word(),
                2 => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
                3 => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
                4 => {
                    let m = *self.rng.pick(MATH_LETTERS);
                    self.out.push_str(m);
                }
                _ => {
                    let e = *self.rng.pick(EMOJI);
                    self.out.push_str(e);
                }
            }
        }
        let t = *self.rng.pick(TERMINALS);
        self.out.push_str(t);
    }

    /// Punctuation on its own and piled up.
    pub(super) fn dense_punctuation(&mut self) {
        let n = self.rng.range(2, 12);
        for i in 0..n {
            if i > 0 && self.rng.chance(50) {
                self.out.push(' ');
            }
            match self.rng.below(5) {
                0 => {
                    let t = *self.rng.pick(TERMINALS);
                    self.out.push_str(t);
                }
                1 => {
                    let d = *self.rng.pick(DASHES);
                    self.out.push_str(d);
                }
                2 => {
                    let q = *self.rng.pick(OPEN_QUOTES);
                    self.out.push_str(q);
                    let q = *self.rng.pick(CLOSE_QUOTES);
                    self.out.push_str(q);
                }
                3 => {
                    let s = *self.rng.pick(SYMBOLS);
                    self.out.push_str(s);
                }
                _ => {
                    let w = *self.rng.pick(EN_WORDS);
                    self.out.push_str(w);
                }
            }
        }
    }
}
