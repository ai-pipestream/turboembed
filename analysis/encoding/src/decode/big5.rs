//! Big5, the standard's decoder: index Big5 with the HKSCS extensions,
//! and four pointers that decode to two characters each.

use super::{Step, Stepper};
use crate::tables;

#[derive(Default)]
pub(crate) struct Big5 {
    lead: u8,
}

/// Whether a byte with no lead pending starts a two-byte sequence.
pub(crate) fn is_lead(byte: u8) -> bool {
    (0x81..=0xFE).contains(&byte)
}

/// What a lead and trail byte make: one character, two, or nothing.
pub(crate) enum PairResult {
    One(char),
    Two(char, char),
    None,
}

pub(crate) fn pair(lead: u8, byte: u8) -> PairResult {
    let offset = if byte < 0x7F { 0x40 } else { 0x62 };
    if !matches!(byte, 0x40..=0x7E | 0xA1..=0xFE) {
        return PairResult::None;
    }
    let pointer = (usize::from(lead) - 0x81) * 157 + usize::from(byte) - offset;
    match pointer {
        1133 => PairResult::Two('\u{CA}', '\u{304}'),
        1135 => PairResult::Two('\u{CA}', '\u{30C}'),
        1164 => PairResult::Two('\u{EA}', '\u{304}'),
        1166 => PairResult::Two('\u{EA}', '\u{30C}'),
        _ => tables::big5(pointer).map_or(PairResult::None, PairResult::One),
    }
}

impl Stepper for Big5 {
    const ASCII_PASSES: bool = true;

    fn step(&mut self, byte: u8) -> Step {
        if self.lead != 0 {
            let lead = std::mem::take(&mut self.lead);
            return match pair(lead, byte) {
                PairResult::One(c) => Step::Char(c),
                PairResult::Two(a, b) => Step::Pair(a, b),
                PairResult::None => Step::Error { restore: u8::from(byte.is_ascii()) },
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
