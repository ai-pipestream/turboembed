//! The rule that tells a base64 run without a recognized signature from
//! text that merely uses base64 characters, an addition to OpenNLP (which
//! reports only runs it can identify).
//!
//! Base64 text of binary data draws its characters close to uniformly
//! from 64 symbols: upper case letters, lower case letters, digits and two
//! symbols. Two consecutive characters then belong to different classes
//! with probability 1 - (26² + 26² + 10² + 2²) / 64², about 0.64. Words,
//! identifiers and paths change class rarely: `AlphanumericRunsLikeThisOne`
//! changes at word starts only, a lower-case hash or an upper-case hex
//! string has no letters of the other case. A run is reported when:
//!
//! 1. it has at least `MIN_BARE_PAYLOAD` (32) characters without padding;
//! 2. it holds an upper case letter, a lower case letter and a digit;
//! 3. at least 45% of its adjacent character pairs change class.
//!
//! For 32 random base64 characters the change rate has a standard
//! deviation of about 0.086, so the third rule misses roughly one random
//! run in a hundred at that length and far fewer for longer runs; a missed
//! run is what OpenNLP does with every unidentified run.

use super::detect::MIN_BARE_PAYLOAD;
use super::payload::{Payload, digit};

/// The smallest share of adjacent pairs that change class, in percent.
pub(crate) const MIN_CLASS_CHANGE_PERCENT: usize = 45;

#[derive(Copy, Clone, PartialEq, Eq)]
enum Class {
    Upper,
    Lower,
    Digit,
    Symbol,
}

fn class(b: u8) -> Class {
    match b {
        b'A'..=b'Z' => Class::Upper,
        b'a'..=b'z' => Class::Lower,
        b'0'..=b'9' => Class::Digit,
        _ => Class::Symbol,
    }
}

/// Whether a valid base64 run looks like encoded data by the rules above.
/// Line breaks and padding are skipped.
pub(crate) fn looks_encoded(text: &[u8], payload: &Payload) -> bool {
    if !payload.valid || payload.encoded_len - payload.padding < MIN_BARE_PAYLOAD {
        return false;
    }
    let (mut upper, mut lower, mut digits) = (false, false, false);
    let (mut pairs, mut changes) = (0usize, 0usize);
    let mut previous = None;
    for &b in text[payload.start..payload.end].iter().filter(|&&b| digit(b).is_some()) {
        let c = class(b);
        upper |= c == Class::Upper;
        lower |= c == Class::Lower;
        digits |= c == Class::Digit;
        if let Some(p) = previous {
            pairs += 1;
            changes += usize::from(p != c);
        }
        previous = Some(c);
    }
    upper && lower && digits && changes * 100 >= MIN_CLASS_CHANGE_PERCENT * pairs
}
