//! Decoders for the detector's CJK candidates, fed one byte at a time.
//!
//! chardetng feeds encoding_rs decoders one byte per call and scores what
//! each call writes (UTF-16 code units) and whether it reports a
//! malformed sequence, and how long. These decoders give the same three
//! results for the same byte. Two details of that kind of feeding matter:
//!
//! - when a byte ends a sequence as an error and the standard decodes the
//!   byte again (an ASCII byte after a lead byte), the byte is not decoded
//!   again: the decoder reports it as not read and chardetng moves on, so
//!   the byte is dropped;
//! - the reported length counts the bytes of the sequence without the
//!   dropped byte.
//!
//! The lookups are the WHATWG index tables the text decoders use.

use crate::decode::{big5, euc_jp, euc_kr, gb18030, shift_jis};

/// What one byte made: up to two UTF-16 code units, and the length of a
/// malformed sequence it ended, if it ended one.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct Fed {
    pub units: [u16; 2],
    pub written: usize,
    pub malformed: Option<u8>,
}

impl Fed {
    fn char(c: char) -> Fed {
        let mut fed = Fed::default();
        fed.push(c);
        fed
    }

    fn malformed(len: u8) -> Fed {
        Fed { malformed: Some(len), ..Fed::default() }
    }

    fn push(&mut self, c: char) {
        let mut buf = [0u16; 2];
        for &u in c.encode_utf16(&mut buf).iter() {
            self.units[self.written] = u;
            self.written += 1;
        }
    }

    /// An error after a lead byte: an ASCII byte is dropped and not
    /// counted, any other byte is part of the sequence.
    fn bad_trail(len_with_byte: u8, byte: u8) -> Fed {
        Fed::malformed(if byte < 0x80 { len_with_byte - 1 } else { len_with_byte })
    }
}

#[derive(Default)]
pub(super) struct ShiftJisStream {
    lead: Option<u8>,
}

impl ShiftJisStream {
    pub(super) fn feed(&mut self, b: u8) -> Fed {
        if let Some(lead) = self.lead.take() {
            return shift_jis::pair(lead, b).map_or_else(|| Fed::bad_trail(2, b), Fed::char);
        }
        match shift_jis::single(b) {
            shift_jis::Single::Char(c) => Fed::char(c),
            shift_jis::Single::Lead => {
                self.lead = Some(b);
                Fed::default()
            }
            shift_jis::Single::Error => Fed::malformed(1),
        }
    }

    /// The end of the input: whether no sequence was left open.
    pub(super) fn finish(&mut self) -> bool {
        self.lead.take().is_none()
    }
}

#[derive(Default)]
pub(super) struct EucKrStream {
    lead: Option<u8>,
}

impl EucKrStream {
    pub(super) fn feed(&mut self, b: u8) -> Fed {
        if let Some(lead) = self.lead.take() {
            return euc_kr::pair(lead, b).map_or_else(|| Fed::bad_trail(2, b), Fed::char);
        }
        if b < 0x80 {
            Fed::char(char::from(b))
        } else if euc_kr::is_lead(b) {
            self.lead = Some(b);
            Fed::default()
        } else {
            Fed::malformed(1)
        }
    }

    pub(super) fn finish(&mut self) -> bool {
        self.lead.take().is_none()
    }
}

#[derive(Default)]
pub(super) struct Big5Stream {
    lead: Option<u8>,
}

impl Big5Stream {
    pub(super) fn feed(&mut self, b: u8) -> Fed {
        if let Some(lead) = self.lead.take() {
            return match big5::pair(lead, b) {
                big5::PairResult::One(c) => Fed::char(c),
                big5::PairResult::Two(c, d) => {
                    let mut fed = Fed::char(c);
                    fed.push(d);
                    fed
                }
                big5::PairResult::None => Fed::bad_trail(2, b),
            };
        }
        if b < 0x80 {
            Fed::char(char::from(b))
        } else if big5::is_lead(b) {
            self.lead = Some(b);
            Fed::default()
        } else {
            Fed::malformed(1)
        }
    }

    pub(super) fn finish(&mut self) -> bool {
        self.lead.take().is_none()
    }
}

#[derive(Default)]
pub(super) struct EucJpStream {
    /// The lead byte, 0 for none.
    lead: u8,
    /// Whether 0x8F came before `lead` (a JIS X 0212 sequence).
    jis0212: bool,
}

impl EucJpStream {
    pub(super) fn feed(&mut self, b: u8) -> Fed {
        let (lead, jis0212) = (std::mem::take(&mut self.lead), std::mem::take(&mut self.jis0212));
        match lead {
            0 => {}
            0x8E => {
                return if (0xA1..=0xDF).contains(&b) {
                    Fed::char(char::from_u32(0xFF61 - 0xA1 + u32::from(b)).expect("katakana"))
                } else {
                    Fed::bad_trail(2, b)
                };
            }
            0x8F => {
                if (0xA1..=0xFE).contains(&b) {
                    (self.lead, self.jis0212) = (b, true);
                    return Fed::default();
                }
                return Fed::bad_trail(2, b);
            }
            _ => {
                let len = if jis0212 { 3 } else { 2 };
                return euc_jp::pair(lead, b, jis0212).map_or_else(|| Fed::bad_trail(len, b), Fed::char);
            }
        }
        if b < 0x80 {
            Fed::char(char::from(b))
        } else if euc_jp::is_lead(b) {
            self.lead = b;
            Fed::default()
        } else {
            Fed::malformed(1)
        }
    }

    pub(super) fn finish(&mut self) -> bool {
        let open = self.lead != 0;
        (self.lead, self.jis0212) = (0, false);
        !open
    }
}

/// The bytes of an open gb18030 sequence.
#[derive(Copy, Clone, Default)]
enum Gb18030Pending {
    #[default]
    None,
    One(u8),
    Two(u8, u8),
    Three(u8, u8, u8),
}

/// GBK as encoding_rs decodes it (the gb18030 decoder). When a four-byte
/// sequence fails at its third or fourth byte, the standard decodes its
/// second byte (an ASCII digit) and third byte again; encoding_rs keeps
/// them and writes the digit at the start of the next call.
#[derive(Default)]
pub(super) struct GbkStream {
    pending: Gb18030Pending,
    pending_ascii: Option<u8>,
}

impl GbkStream {
    pub(super) fn feed(&mut self, b: u8) -> Fed {
        let mut fed = Fed::default();
        if let Some(digit) = self.pending_ascii.take() {
            fed.push(char::from(digit));
        }
        match std::mem::take(&mut self.pending) {
            Gb18030Pending::None => match b {
                0x00..=0x7F => fed.push(char::from(b)),
                0x80 => fed.push('\u{20AC}'),
                0x81..=0xFE => self.pending = Gb18030Pending::One(b),
                0xFF => fed.malformed = Some(1),
            },
            Gb18030Pending::One(first) => {
                if (0x30..=0x39).contains(&b) {
                    self.pending = Gb18030Pending::Two(first, b);
                } else {
                    match gb18030::pair(first, b) {
                        Some(c) => fed.push(c),
                        None => fed.malformed = Fed::bad_trail(2, b).malformed,
                    }
                }
            }
            Gb18030Pending::Two(first, second) => {
                if (0x81..=0xFE).contains(&b) {
                    self.pending = Gb18030Pending::Three(first, second, b);
                } else {
                    self.pending_ascii = Some(second);
                    fed.malformed = Some(1);
                }
            }
            Gb18030Pending::Three(first, second, third) => {
                if (0x30..=0x39).contains(&b) {
                    match gb18030::four(first, second, third, b) {
                        Some(c) => fed.push(c),
                        None => fed.malformed = Some(4),
                    }
                } else {
                    self.pending_ascii = Some(second);
                    self.pending = Gb18030Pending::One(third);
                    fed.malformed = Some(1);
                }
            }
        }
        fed
    }

    pub(super) fn finish(&mut self) -> bool {
        self.pending_ascii = None;
        matches!(std::mem::take(&mut self.pending), Gb18030Pending::None)
    }
}

/// UTF-8 validity, fed in any chunks.
#[derive(Default)]
pub(super) struct Utf8Stream {
    needed: u8,
    lower: u8,
    upper: u8,
}

impl Utf8Stream {
    /// Whether `bytes` continue valid UTF-8 (and, with `last`, end it).
    pub(super) fn feed(&mut self, bytes: &[u8], last: bool) -> bool {
        for &b in bytes {
            if self.needed == 0 {
                (self.needed, self.lower, self.upper) = match b {
                    0x00..=0x7F => continue,
                    0xC2..=0xDF => (1, 0x80, 0xBF),
                    0xE0 => (2, 0xA0, 0xBF),
                    0xED => (2, 0x80, 0x9F),
                    0xE1..=0xEF => (2, 0x80, 0xBF),
                    0xF0 => (3, 0x90, 0xBF),
                    0xF4 => (3, 0x80, 0x8F),
                    0xF1..=0xF3 => (3, 0x80, 0xBF),
                    _ => return false,
                };
                continue;
            }
            if !(self.lower..=self.upper).contains(&b) {
                return false;
            }
            (self.lower, self.upper) = (0x80, 0xBF);
            self.needed -= 1;
        }
        !(last && self.needed != 0)
    }
}
