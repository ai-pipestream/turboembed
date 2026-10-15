//! EUC-KR, the standard's decoder: code page 949 through index EUC-KR.

use super::{Step, Stepper};
use crate::tables;

#[derive(Default)]
pub(crate) struct EucKr {
    lead: u8,
}

/// Whether a byte with no lead pending starts a two-byte sequence.
pub(crate) fn is_lead(byte: u8) -> bool {
    (0x81..=0xFE).contains(&byte)
}

/// The character a lead and trail byte make, if any.
pub(crate) fn pair(lead: u8, byte: u8) -> Option<char> {
    if !(0x41..=0xFE).contains(&byte) {
        return None;
    }
    tables::euc_kr((usize::from(lead) - 0x81) * 190 + usize::from(byte) - 0x41)
}

impl Stepper for EucKr {
    const ASCII_PASSES: bool = true;

    fn step(&mut self, byte: u8) -> Step {
        if self.lead != 0 {
            let lead = std::mem::take(&mut self.lead);
            return match pair(lead, byte) {
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
        self.lead = 0;
        Some(Step::Error { restore: 0 })
    }
}
