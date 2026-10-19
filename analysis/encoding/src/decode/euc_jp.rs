//! EUC-JP, the standard's decoder: JIS X 0208 pairs, half-width katakana
//! after 0x8E, and JIS X 0212 triples after 0x8F.

use super::{Step, Stepper};
use crate::tables;

#[derive(Default)]
pub(crate) struct EucJp {
    lead: u8,
    jis0212: bool,
}

/// Whether a byte with no lead pending starts a sequence.
pub(crate) fn is_lead(byte: u8) -> bool {
    matches!(byte, 0x8E | 0x8F | 0xA1..=0xFE)
}

/// The character of a JIS X 0208 (or, with `jis0212`, JIS X 0212) pair.
pub(crate) fn pair(lead: u8, byte: u8, jis0212: bool) -> Option<char> {
    if !(0xA1..=0xFE).contains(&lead) || !(0xA1..=0xFE).contains(&byte) {
        return None;
    }
    let pointer = (usize::from(lead) - 0xA1) * 94 + usize::from(byte) - 0xA1;
    if jis0212 { tables::jis0212(pointer) } else { tables::jis0208(pointer) }
}

impl Stepper for EucJp {
    const ASCII_PASSES: bool = true;

    fn step(&mut self, byte: u8) -> Step {
        if self.lead == 0x8E && (0xA1..=0xDF).contains(&byte) {
            self.lead = 0;
            return Step::Char(char::from_u32(0xFF61 - 0xA1 + u32::from(byte)).expect("katakana"));
        }
        if self.lead == 0x8F && (0xA1..=0xFE).contains(&byte) {
            self.jis0212 = true;
            self.lead = byte;
            return Step::Pending;
        }
        if self.lead != 0 {
            let lead = std::mem::take(&mut self.lead);
            let jis0212 = std::mem::take(&mut self.jis0212);
            return match pair(lead, byte, jis0212) {
                Some(c) => Step::Char(c),
                None => Step::Error { restore: u8::from(byte.is_ascii()) },
            };
        }
        if byte.is_ascii() {
            Step::Char(char::from(byte))
        } else if is_lead(byte) {
            self.lead = byte;
            Step::Pending
        } else {
            Step::Error { restore: 0 }
        }
    }

    fn idle(&self) -> bool {
        self.lead == 0
    }

    fn eof(&mut self) -> Option<Step> {
        if self.lead == 0 {
            return None;
        }
        (self.lead, self.jis0212) = (0, false);
        Some(Step::Error { restore: 0 })
    }
}
