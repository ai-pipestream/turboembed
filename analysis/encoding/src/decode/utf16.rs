//! UTF-16BE and UTF-16LE, the standard's shared UTF-16 decoder. Every
//! character is an edit (two or four bytes to its UTF-8). An unpaired
//! leading surrogate is one U+FFFD over its two bytes, and the code unit
//! after it is decoded again; an unpaired trailing surrogate is one U+FFFD
//! over its two bytes; an odd byte at the end is one U+FFFD over the bytes
//! still open.

use super::{Step, Stepper};

pub(crate) struct Utf16 {
    big_endian: bool,
    lead_byte: Option<u8>,
    lead_surrogate: Option<u16>,
}

impl Utf16 {
    pub(crate) fn new(big_endian: bool) -> Utf16 {
        Utf16 { big_endian, lead_byte: None, lead_surrogate: None }
    }
}

impl Stepper for Utf16 {
    const ASCII_PASSES: bool = false;

    fn step(&mut self, byte: u8) -> Step {
        let Some(lead) = self.lead_byte.take() else {
            self.lead_byte = Some(byte);
            return Step::Pending;
        };
        let unit = if self.big_endian { u16::from_be_bytes([lead, byte]) } else { u16::from_le_bytes([lead, byte]) };
        if let Some(high) = self.lead_surrogate.take() {
            if (0xDC00..=0xDFFF).contains(&unit) {
                let c = 0x10000 + ((u32::from(high) - 0xD800) << 10) + (u32::from(unit) - 0xDC00);
                return Step::Char(char::from_u32(c).expect("a surrogate pair is a scalar value"));
            }
            return Step::Error { restore: 2 };
        }
        match unit {
            0xD800..=0xDBFF => {
                self.lead_surrogate = Some(unit);
                Step::Pending
            }
            0xDC00..=0xDFFF => Step::Error { restore: 0 },
            _ => Step::Char(char::from_u32(u32::from(unit)).expect("not a surrogate")),
        }
    }

    fn idle(&self) -> bool {
        self.lead_byte.is_none() && self.lead_surrogate.is_none()
    }

    fn eof(&mut self) -> Option<Step> {
        if self.idle() {
            return None;
        }
        (self.lead_byte, self.lead_surrogate) = (None, None);
        Some(Step::Error { restore: 0 })
    }
}
