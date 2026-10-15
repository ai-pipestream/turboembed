//! ISO-2022-JP, the standard's decoder: escape sequences switch between
//! ASCII, JIS X 0201 Roman, half-width katakana and JIS X 0208. An escape
//! sequence writes nothing (one deletion edit over its three bytes), unless
//! it directly follows another escape sequence, which the standard treats
//! as an error.

use super::{Step, Stepper};
use crate::tables;

#[derive(Copy, Clone, PartialEq, Eq, Default)]
enum State {
    #[default]
    Ascii,
    Roman,
    Katakana,
    LeadByte,
    TrailByte,
    EscapeStart,
    Escape,
}

#[derive(Default)]
pub(crate) struct Iso2022Jp {
    state: State,
    output_state: State,
    lead: u8,
    output: bool,
}

impl Iso2022Jp {
    /// An error that ends an escape sequence: back to the state before the
    /// escape.
    fn escape_error(&mut self, restore: u8) -> Step {
        self.output = false;
        self.state = self.output_state;
        Step::Error { restore }
    }
}

impl Stepper for Iso2022Jp {
    // ASCII bytes are half-width katakana or JIS X 0208 in other states.
    const ASCII_PASSES: bool = false;

    fn step(&mut self, byte: u8) -> Step {
        match self.state {
            State::EscapeStart => {
                if byte == 0x24 || byte == 0x28 {
                    self.lead = byte;
                    self.state = State::Escape;
                    return Step::Pending;
                }
                self.escape_error(1)
            }
            State::Escape => {
                let lead = std::mem::take(&mut self.lead);
                let next = match (lead, byte) {
                    (0x28, 0x42) => Some(State::Ascii),
                    (0x28, 0x4A) => Some(State::Roman),
                    (0x28, 0x49) => Some(State::Katakana),
                    (0x24, 0x40 | 0x42) => Some(State::LeadByte),
                    _ => None,
                };
                let Some(next) = next else {
                    // The lead and this byte are decoded again.
                    return self.escape_error(2);
                };
                self.state = next;
                self.output_state = next;
                let wrote_nothing_since = std::mem::replace(&mut self.output, true);
                if wrote_nothing_since { Step::Error { restore: 0 } } else { Step::Drop }
            }
            _ if byte == 0x1B => {
                if self.state == State::TrailByte {
                    // The standard moves to escape start and reports an
                    // error for the lead byte. Decoding the escape byte
                    // again from the lead byte state does the same and
                    // keeps it out of the replacement.
                    self.state = State::LeadByte;
                    return Step::Error { restore: 1 };
                }
                self.state = State::EscapeStart;
                Step::Pending
            }
            State::TrailByte => {
                self.state = State::LeadByte;
                if !(0x21..=0x7E).contains(&byte) {
                    return Step::Error { restore: 0 };
                }
                let pointer = (usize::from(self.lead) - 0x21) * 94 + usize::from(byte) - 0x21;
                tables::jis0208(pointer).map_or(Step::Error { restore: 0 }, Step::Char)
            }
            state => {
                self.output = false;
                match (state, byte) {
                    (_, 0x0E | 0x0F) | (_, 0x80..=0xFF) => Step::Error { restore: 0 },
                    (State::Ascii, _) => Step::Char(char::from(byte)),
                    (State::Roman, 0x5C) => Step::Char('\u{A5}'),
                    (State::Roman, 0x7E) => Step::Char('\u{203E}'),
                    (State::Roman, _) => Step::Char(char::from(byte)),
                    (State::Katakana, 0x21..=0x5F) => {
                        Step::Char(char::from_u32(0xFF61 - 0x21 + u32::from(byte)).expect("katakana"))
                    }
                    (State::LeadByte, 0x21..=0x7E) => {
                        self.lead = byte;
                        self.state = State::TrailByte;
                        Step::Pending
                    }
                    _ => Step::Error { restore: 0 },
                }
            }
        }
    }

    fn idle(&self) -> bool {
        !matches!(self.state, State::TrailByte | State::EscapeStart | State::Escape)
    }

    fn eof(&mut self) -> Option<Step> {
        match self.state {
            State::EscapeStart => Some(self.escape_error(0)),
            // The lead (0x24 or 0x28) is decoded again.
            State::Escape => {
                self.lead = 0;
                Some(self.escape_error(1))
            }
            State::TrailByte => {
                self.state = State::LeadByte;
                Some(Step::Error { restore: 0 })
            }
            _ => None,
        }
    }
}
