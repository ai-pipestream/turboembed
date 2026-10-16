//! Shift_JIS, the standard's decoder: ASCII and 0x80 as themselves,
//! half-width katakana at 0xA1 to 0xDF, and two-byte sequences through
//! index jis0208, with the Windows EUDC range as private use.

use super::{Step, Stepper};
use crate::tables;

#[derive(Default)]
pub(crate) struct ShiftJis {
    lead: u8,
}

/// What a byte with no lead pending is.
pub(crate) enum Single {
    Char(char),
    Lead,
    Error,
}

pub(crate) fn single(byte: u8) -> Single {
    match byte {
        0x00..=0x80 => Single::Char(char::from(byte)),
        0xA1..=0xDF => Single::Char(char::from_u32(0xFF61 - 0xA1 + u32::from(byte)).expect("katakana")),
        0x81..=0x9F | 0xE0..=0xFC => Single::Lead,
        _ => Single::Error,
    }
}

/// The character a lead and trail byte make, if any.
pub(crate) fn pair(lead: u8, byte: u8) -> Option<char> {
    let offset = if byte < 0x7F { 0x40 } else { 0x41 };
    let lead_offset = if lead < 0xA0 { 0x81 } else { 0xC1 };
    if !matches!(byte, 0x40..=0x7E | 0x80..=0xFC) {
        return None;
    }
    let pointer = (usize::from(lead) - lead_offset) * 188 + usize::from(byte) - offset;
    if (8836..=10715).contains(&pointer) {
        return char::from_u32((0xE000 - 8836 + pointer) as u32);
    }
    tables::jis0208(pointer)
}

impl Stepper for ShiftJis {
    const ASCII_PASSES: bool = true;

    fn step(&mut self, byte: u8) -> Step {
        if self.lead != 0 {
            let lead = std::mem::take(&mut self.lead);
            return match pair(lead, byte) {
                Some(c) => Step::Char(c),
                None => Step::Error { restore: u8::from(byte.is_ascii()) },
            };
        }
        match single(byte) {
            Single::Char(c) => Step::Char(c),
            Single::Lead => {
                self.lead = byte;
                Step::Pending
            }
            Single::Error => Step::Error { restore: 0 },
        }
    }

    fn idle(&self) -> bool {
        self.lead == 0
    }

    fn eof(&mut self) -> Option<Step> {
        if self.lead == 0 {
            return None;
        }
        self.lead = 0;
        Some(Step::Error { restore: 0 })
    }
}
