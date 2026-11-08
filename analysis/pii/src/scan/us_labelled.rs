//! United States identifiers that are bare digit or near-digit shapes,
//! `UsIdentityPiiExtractor`. Their shapes alone say little, so:
//!
//! - Passports (`us-passport`): nine digits (older books), or one capital
//!   letter and eight digits (next-generation books and cards), with no
//!   separator. The kind
//!   needs a `passport` label in its word window by default.
//! - Driver's licences (`us-drivers-licence`): the shapes the states issue,
//!   taken together: seven to thirteen digits, one or two capital letters
//!   and four to fourteen digits, or two digits, three capitals and five
//!   digits; a hyphen may split two digits. The kind needs a licence
//!   label in its word window by default.
//! - ZIP codes (`us-zip`): five digits, or five, a hyphen and four, not
//!   all zeros, read only right after a state or territory code
//!   (`Springfield, IL 62704`) or a ZIP label (`ZIP: 62704`).

use turbo_analysis_layers::{PiiEvidence, PiiKind, PiiKinds};

use super::national::on_number_end;
use crate::hits::Hits;
use crate::text::{on_end, on_number_start, on_word_start};
use crate::window::find_any_label;

/// The USPS codes of the states, the District of Columbia, the
/// territories and the armed forces.
const STATES: [&[u8; 2]; 63] = [
    b"AL", b"AK", b"AZ", b"AR", b"CA", b"CO", b"CT", b"DE", b"FL", b"GA", b"HI", b"ID", b"IL", b"IN", b"IA", b"KS",
    b"KY", b"LA", b"ME", b"MD", b"MA", b"MI", b"MN", b"MS", b"MO", b"MT", b"NE", b"NV", b"NH", b"NJ", b"NM", b"NY",
    b"NC", b"ND", b"OH", b"OK", b"OR", b"PA", b"RI", b"SC", b"SD", b"TN", b"TX", b"UT", b"VT", b"VA", b"WA", b"WV",
    b"WI", b"WY", b"DC", b"AS", b"GU", b"MP", b"PR", b"VI", b"UM", b"FM", b"MH", b"PW", b"AA", b"AE", b"AP",
];

const ZIP_LABELS: [&[u8]; 4] = [b"zip code", b"postal code", b"zipcode", b"zip"];

pub(crate) fn scan(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    let want_passport = kinds.contains(PiiKind::UsPassport);
    let want_licence = kinds.contains(PiiKind::UsDriversLicence);
    let want_zip = kinds.contains(PiiKind::UsZip);
    if !(want_passport || want_licence || want_zip) {
        return;
    }
    let b = text.as_bytes();
    let mut value = String::new();
    let mut i = 0;
    while i < b.len() {
        let c = b[i];
        let joined = i > 0 && matches!(b[i - 1], b'-' | b'_');
        if !(c.is_ascii_digit() || c.is_ascii_uppercase())
            || !on_word_start(text, i)
            || !on_number_start(text, i)
            || joined
        {
            i += 1;
            continue;
        }
        let Some((end, letters, digits)) = read_token(text, i, &mut value) else {
            // Never read a value from inside a longer or joined token
            // (`1234-5678-9012-3456-7`, `X-538194027`).
            i = token_end(b, i);
            continue;
        };
        if want_passport && end - i == value.len() && passport_shape(&value, letters) {
            hits.add(i, end, PiiKind::UsPassport, PiiEvidence::Shape, |s| s.push_str(&value));
        }
        if want_licence && licence_shape(&value, letters, digits) {
            hits.add(i, end, PiiKind::UsDriversLicence, PiiEvidence::Shape, |s| s.push_str(&value));
        }
        if want_zip && let Some(zip_end) = zip_at(text, i) {
            hits.add_text(text, i, zip_end, PiiKind::UsZip, PiiEvidence::Shape);
        }
        i = end.max(i + 1);
    }
}

/// A run of capitals and digits from `start`, a hyphen allowed between
/// two digits, at a number end: its end, and how many letters and digits
/// it holds (`value` gets them without the hyphens).
fn read_token(text: &str, start: usize, value: &mut String) -> Option<(usize, usize, usize)> {
    let b = text.as_bytes();
    value.clear();
    let (mut letters, mut digits) = (0, 0);
    let mut p = start;
    while p < b.len() {
        let c = b[p];
        if c.is_ascii_uppercase() {
            letters += 1;
        } else if c.is_ascii_digit() {
            digits += 1;
        } else if c == b'-' && p > start && b[p - 1].is_ascii_digit() && b.get(p + 1).is_some_and(u8::is_ascii_digit) {
            p += 1;
            continue;
        } else {
            break;
        }
        value.push(c as char);
        p += 1;
        if value.len() > 16 {
            return None;
        }
    }
    (digits > 0 && on_end(text, p) && on_number_end(text, p, b".,-") && !joined_at(b, p))
        .then_some((p, letters, digits))
}

/// A `-` or `_` at `p` joins what follows to the token (`538194027-A`).
fn joined_at(b: &[u8], p: usize) -> bool {
    matches!(b.get(p), Some(b'-' | b'_'))
}

/// The end of the run of letters, digits, `-` and `_` from `i`.
fn token_end(b: &[u8], i: usize) -> usize {
    let n = b[i..].iter().take_while(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'_')).count();
    i + n.max(1)
}

fn passport_shape(v: &str, letters: usize) -> bool {
    let b = v.as_bytes();
    match letters {
        0 => b.len() == 9,
        1 => b.len() == 9 && b[0].is_ascii_uppercase(),
        _ => false,
    }
}

fn licence_shape(v: &str, letters: usize, digits: usize) -> bool {
    let b = v.as_bytes();
    let lead = b.iter().take_while(|c| c.is_ascii_uppercase()).count();
    match letters {
        0 => (7..=13).contains(&digits),
        1 | 2 => lead == letters && (4..=14).contains(&digits),
        3 => b.len() == 10 && b[..2].iter().all(u8::is_ascii_digit) && b[2..5].iter().all(u8::is_ascii_uppercase),
        _ => false,
    }
}

/// The end of a ZIP or ZIP+4 code at `i` read after a state code or a
/// ZIP label.
fn zip_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let five = b.get(i..i + 5)?;
    if !five.iter().all(u8::is_ascii_digit) || five == b"00000" {
        return None;
    }
    let mut end = i + 5;
    if b.get(end) == Some(&b'-') && b.get(end + 1..end + 5).is_some_and(|d| d.iter().all(u8::is_ascii_digit)) {
        end += 5;
    }
    if !on_end(text, end) || !on_number_end(text, end, b".,-") || joined_at(b, end) {
        return None;
    }
    (after_state(b, i) || find_any_label(text, i, &ZIP_LABELS, false).is_some()).then_some(end)
}

/// A state code, in capitals, ends one space before `i`, and no letter
/// or digit is before it.
fn after_state(b: &[u8], i: usize) -> bool {
    i >= 3
        && b[i - 1] == b' '
        && STATES.iter().any(|s| &b[i - 3..i - 1] == s.as_slice())
        && (i == 3 || !b[i - 4].is_ascii_alphanumeric())
}
