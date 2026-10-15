//! gb18030 (and GBK, which the standard decodes the same way): 0x80 as
//! the euro sign, two-byte sequences through index gb18030, and four-byte
//! sequences through index gb18030 ranges.

use super::{Step, Stepper};
use crate::tables;

#[derive(Default)]
pub(crate) struct Gb18030 {
    first: u8,
    second: u8,
    third: u8,
}

/// The character of a two-byte sequence, if any.
pub(crate) fn pair(lead: u8, byte: u8) -> Option<char> {
    if !matches!(byte, 0x40..=0x7E | 0x80..=0xFE) {
        return None;
    }
    let offset = if byte < 0x7F { 0x40 } else { 0x41 };
    tables::gb18030((usize::from(lead) - 0x81) * 190 + usize::from(byte) - offset)
}

/// The character of a four-byte sequence whose bytes are each in range,
/// if any.
pub(crate) fn four(b1: u8, b2: u8, b3: u8, b4: u8) -> Option<char> {
    let pointer = (u32::from(b1) - 0x81) * (10 * 126 * 10)
        + (u32::from(b2) - 0x30) * (10 * 126)
        + (u32::from(b3) - 0x81) * 10
        + u32::from(b4)
        - 0x30;
    tables::gb18030_range(pointer)
}

impl Stepper for Gb18030 {
    const ASCII_PASSES: bool = true;

    fn step(&mut self, byte: u8) -> Step {
        if self.third != 0 {
            let (first, second, third) = (self.first, self.second, self.third);
            (self.first, self.second, self.third) = (0, 0, 0);
            if !(0x30..=0x39).contains(&byte) {
                // The second, third and this byte are decoded again.
                return Step::Error { restore: 3 };
            }
            return match four(first, second, third, byte) {
                Some(c) => Step::Char(c),
                None => Step::Error { restore: 0 },
            };
        }
        if self.second != 0 {
            if (0x81..=0xFE).contains(&byte) {
                self.third = byte;
                return Step::Pending;
            }
            (self.first, self.second) = (0, 0);
            return Step::Error { restore: 2 };
        }
        if self.first != 0 {
            if (0x30..=0x39).contains(&byte) {
                self.second = byte;
                return Step::Pending;
            }
            let lead = std::mem::take(&mut self.first);
            return match pair(lead, byte) {
                Some(c) => Step::Char(c),
                None => Step::Error { restore: u8::from(byte.is_ascii()) },
            };
        }
        match byte {
            0x00..=0x7F => Step::Char(char::from(byte)),
            0x80 => Step::Char('\u{20AC}'),
            0x81..=0xFE => {
                self.first = byte;
                Step::Pending
            }
            0xFF => Step::Error { restore: 0 },
        }
    }

    fn idle(&self) -> bool {
        self.first == 0
    }

    fn eof(&mut self) -> Option<Step> {
        if self.first == 0 {
            return None;
        }
        (self.first, self.second, self.third) = (0, 0, 0);
        Some(Step::Error { restore: 0 })
    }
}
