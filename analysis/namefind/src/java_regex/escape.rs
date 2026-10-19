//! Escape sequence and character class parsing for [`super::parse::Translator`].

use crate::error::{Error, Feature};

use super::class::{MAX, Set, predefined};
use super::compile::{Assertion, Node};
use super::parse::{ASSERTION, CHAR, Piece, Translator};

impl Translator<'_> {
    /// An escape outside a class, at the backslash.
    pub(super) fn escape(&mut self) -> Result<(Node, Piece), Error> {
        self.at += 1;
        let Some(c) = self.read() else {
            return Err(self.syntax("unexpected internal error"));
        };
        let letter = char::from_u32(c).unwrap_or('\0');
        match letter {
            '1'..='9' | 'k' => Err(self.unsupported(Feature::Backreference)),
            'p' | 'P' => Err(self.unsupported(Feature::Property)),
            'A' => {
                self.anchored_empty = true;
                Ok((Node::Assert(Assertion::StartText), ASSERTION))
            }
            'z' => Ok((Node::Assert(Assertion::EndText), ASSERTION)),
            'Z' => {
                self.limits.dollar = true;
                Ok((Node::Assert(Assertion::EndText), ASSERTION))
            }
            'b' => {
                if self.peek_is('{') {
                    if self.peek_at(1) == Some('g' as u32) && self.peek_at(2) != Some('}' as u32) {
                        return Err(self.syntax("illegal/unsupported escape sequence"));
                    }
                    return Err(self.unsupported(Feature::Escape('b')));
                }
                self.limits.word_boundary = true;
                self.anchored_empty = true;
                Ok((Node::Assert(Assertion::AsciiWordBoundary), ASSERTION))
            }
            'B' => {
                self.limits.word_boundary = true;
                self.limits.mid_pair = true;
                Ok((Node::Assert(Assertion::AsciiNonWordBoundary), ASSERTION))
            }
            'N' => self.character_name(),
            'G' | 'R' | 'X' => Err(self.unsupported(Feature::Escape(letter))),
            _ => {
                if let Some((class, negated)) = predefined(c) {
                    let mut set = Set::default();
                    if negated {
                        set.add_complement(class);
                    } else {
                        set.add_all(class);
                    }
                    let node = self.add_class(set);
                    return Ok((node, CHAR));
                }
                self.at -= 1;
                let c = self.escaped_char()?;
                Ok((self.literal(c), CHAR))
            }
        }
    }

    /// `Pattern.N()`: validates `\N{name}` syntax before refusing it as [`Feature::Escape`]('N').
    fn character_name<T>(&mut self) -> Result<T, Error> {
        if self.read() != Some('{' as u32) {
            return Err(self.syntax("illegal character name escape sequence"));
        }
        let start = self.at;
        while let Some(c) = self.read() {
            if c == '}' as u32 {
                if self.at == start + 1 {
                    return Err(self.syntax("empty character name"));
                }
                return Err(self.unsupported(Feature::Escape('N')));
            }
        }
        Err(self.syntax("unclosed character name escape sequence"))
    }

    /// The character an escape that names one stands for, at the letter
    /// after the backslash: `escape` for the cases that return a value.
    pub(super) fn escaped_char(&mut self) -> Result<u32, Error> {
        let Some(c) = self.read() else {
            return Err(self.syntax("unexpected internal error"));
        };
        Ok(match char::from_u32(c).unwrap_or('\0') {
            '0' => self.octal()?,
            'a' => 0x07,
            'e' => 0x1B,
            'f' => 0x0C,
            'n' => 0x0A,
            'r' => 0x0D,
            't' => 0x09,
            'c' => match self.read() {
                Some(x) => x ^ 64,
                None => return Err(self.syntax("illegal control escape sequence")),
            },
            'x' => self.hex()?,
            'u' => self.unicode()?,
            l if l.is_ascii_alphanumeric() => return Err(self.syntax("illegal/unsupported escape sequence")),
            _ => c,
        })
    }

    /// `o()`: one to three octal digits after `\0`, the first of three at
    /// most 3.
    fn octal(&mut self) -> Result<u32, Error> {
        let is_octal = |c: Option<u32>| c.filter(|c| (0x30..=0x37).contains(c)).map(|c| c - 0x30);
        let Some(n) = is_octal(self.peek()) else {
            return Err(self.syntax("illegal octal escape sequence"));
        };
        self.at += 1;
        let Some(m) = is_octal(self.peek()) else { return Ok(n) };
        self.at += 1;
        match is_octal(self.peek()) {
            Some(o) if n <= 3 => {
                self.at += 1;
                Ok(n * 64 + m * 8 + o)
            }
            _ => Ok(n * 8 + m),
        }
    }

    /// `x()`: `\xhh` or `\x{h..}`.
    fn hex(&mut self) -> Result<u32, Error> {
        let hex = |c: Option<u32>| c.and_then(char::from_u32).and_then(|c| c.to_digit(16));
        if let Some(a) = hex(self.peek()) {
            self.at += 1;
            if let Some(b) = hex(self.peek()) {
                self.at += 1;
                return Ok(a * 16 + b);
            }
            return Err(self.syntax("illegal hexadecimal escape sequence"));
        }
        if self.peek_is('{') && hex(self.peek_at(1)).is_some() {
            self.at += 1;
            let mut cp = 0;
            while let Some(d) = hex(self.peek()) {
                self.at += 1;
                cp = (cp << 4) + d;
                if cp > MAX {
                    return Err(self.syntax("hexadecimal codepoint is too big"));
                }
            }
            if !self.peek_is('}') {
                return Err(self.syntax("unclosed hexadecimal escape sequence"));
            }
            self.at += 1;
            if (0xD800..=0xDFFF).contains(&cp) {
                return Err(self.unsupported(Feature::Surrogate));
            }
            return Ok(cp);
        }
        Err(self.syntax("illegal hexadecimal escape sequence"))
    }

    /// `\uhhhh`, joined with a following `\uhhhh` when the two make a
    /// surrogate pair, as Java joins them.
    fn unicode(&mut self) -> Result<u32, Error> {
        let unit = self.four_hex()?;
        if (0xD800..0xDC00).contains(&unit) && self.peek_is('\\') && self.peek_at(1) == Some('u' as u32) {
            let save = self.at;
            self.at += 2;
            let low = self.four_hex()?;
            if (0xDC00..0xE000).contains(&low) {
                return Ok(0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00));
            }
            self.at = save;
        }
        if (0xD800..=0xDFFF).contains(&unit) {
            return Err(self.unsupported(Feature::Surrogate));
        }
        Ok(unit)
    }

    fn four_hex(&mut self) -> Result<u32, Error> {
        let mut n = 0;
        for _ in 0..4 {
            match self.read().and_then(char::from_u32).and_then(|c| c.to_digit(16)) {
                Some(d) => n = n * 16 + d,
                None => return Err(self.syntax("illegal Unicode escape sequence")),
            }
        }
        Ok(n)
    }

    /// `clazz`, after the `[`.
    pub(super) fn class(&mut self) -> Result<(Node, Piece), Error> {
        let mut set = Set::default();
        let negated = self.peek_is('^');
        if negated {
            self.at += 1;
        }
        let mut empty = true;
        loop {
            let Some(c) = self.peek() else {
                return Err(self.syntax("unclosed character class"));
            };
            match char::from_u32(c).unwrap_or('\0') {
                '[' => return Err(self.unsupported(Feature::NestedClass)),
                '&' if self.peek_at(1) == Some('&' as u32) => {
                    return Err(self.unsupported(Feature::ClassIntersection));
                }
                ']' if !empty => {
                    self.at += 1;
                    break;
                }
                _ => self.class_item(&mut set)?,
            }
            empty = false;
        }
        if negated {
            set.negate();
        } else {
            set.normalize();
        }
        // Java's matcher can try the second half of a surrogate pair; a
        // class that matches it there but not the whole character at the
        // first half matches where UTF-8 character scanning cannot.
        if set.intersects(0xDC00, 0xDFFF) && !set.covers(0x10000, MAX) {
            self.limits.mid_pair = true;
        }
        let node = self.add_class(set);
        Ok((node, CHAR))
    }

    /// `range`: one character, range or predefined class of a class.
    fn class_item(&mut self, set: &mut Set) -> Result<(), Error> {
        let start = match self.class_char()? {
            ClassChar::Char(c) => c,
            ClassChar::Class(class, negated) => {
                if negated {
                    set.add_complement(class);
                } else {
                    set.add_all(class);
                }
                return Ok(());
            }
        };
        if self.peek_is('-') {
            match self.peek_at(1).and_then(char::from_u32) {
                Some('[' | ']') | None => {}
                Some(_) => {
                    self.at += 1;
                    let end = match self.class_char_in_range()? {
                        Some(end) if end >= start => end,
                        _ => return Err(self.syntax("illegal character range")),
                    };
                    set.add_range(start, end, self.ci);
                    return Ok(());
                }
            }
        }
        set.add_range(start, start, self.ci);
        Ok(())
    }

    /// One character of a class, or a predefined class.
    fn class_char(&mut self) -> Result<ClassChar, Error> {
        let c = self.read().unwrap_or(0);
        if c != '\\' as u32 {
            return Ok(ClassChar::Char(c));
        }
        let Some(letter) = self.peek() else {
            return Err(self.syntax("unexpected internal error"));
        };
        if letter == 'p' as u32 || letter == 'P' as u32 {
            return Err(self.unsupported(Feature::Property));
        }
        // Java reads \v as U+000B when a '-' follows it, so [\v-] holds
        // U+000B and '-'.
        if letter == 'v' as u32 && self.peek_at(1) == Some('-' as u32) {
            self.at += 1;
            return Ok(ClassChar::Char(0x0B));
        }
        if let Some((class, negated)) = predefined(letter) {
            self.at += 1;
            return Ok(ClassChar::Class(class, negated));
        }
        self.class_escape()
    }

    /// The end of a range: a character, or `None` for a predefined class,
    /// which Java refuses there.
    fn class_char_in_range(&mut self) -> Result<Option<u32>, Error> {
        let c = self.read().unwrap_or(0);
        if c != '\\' as u32 {
            return Ok(Some(c));
        }
        let Some(letter) = self.peek() else {
            return Err(self.syntax("unexpected internal error"));
        };
        if letter == 'v' as u32 {
            self.at += 1;
            return Ok(Some(0x0B));
        }
        if predefined(letter).is_some() {
            self.at += 1;
            return Ok(None);
        }
        match self.class_escape()? {
            ClassChar::Char(c) => Ok(Some(c)),
            ClassChar::Class(..) => Ok(None),
        }
    }

    /// An escape inside a class, at the letter after the backslash.
    fn class_escape(&mut self) -> Result<ClassChar, Error> {
        let letter = self.peek().and_then(char::from_u32).unwrap_or('\0');
        match letter {
            // Assertions and references are not escapes inside a class.
            '1'..='9' | 'A' | 'B' | 'G' | 'R' | 'X' | 'Z' | 'b' | 'k' | 'z' => {
                Err(self.syntax("illegal/unsupported escape sequence"))
            }
            'N' => {
                self.at += 1;
                self.character_name()
            }
            _ => self.escaped_char().map(ClassChar::Char),
        }
    }
}

enum ClassChar {
    Char(u32),
    Class(&'static [(u32, u32)], bool),
}
