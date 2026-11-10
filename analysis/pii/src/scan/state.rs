//! United States states, `UsStatePiiExtractor`: the name of a state, the
//! District of Columbia or an inhabited territory (`Illinois`, `New York`,
//! `Puerto Rico`), as written in the table or all in capitals, at word
//! boundaries, with single spaces between its words. `Washington, D.C.`
//! and `Washington DC` are the District, `New York City` is a city, not
//! the state, and a name before a county word that makes a county of the
//! list (`Washington County`) is the county's. A two-letter USPS code (`IL`) is a common word or
//! abbreviation too (`IN`, `OR`, `ME`, `OK`), so it is a mention only in
//! capitals and only after a state label (`State: IL`) or before a ZIP
//! code that ends an address: after a comma (`Springfield, IL 62704`,
//! `Springfield,IL 62704`), in brackets with its ZIP code (`(IL 62704)`),
//! after a colon label (`Address: IL 62704`), after a capitalized word and
//! horizontal blanks when the ZIP code's three-digit prefix belongs to that
//! state (`Springfield IL 62704`, `Springfield\tIL 62704`,
//! `Saint Louis MO 63101`, `Reading PA 19601`), or alone on a line with its
//! ZIP code (`IL 62704`, `\u{a0}IL 62704`), not after another word or with
//! another state's prefix (`Order IN 12345`, `PART OR 90210`). The
//! normalized form is the USPS code.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use super::national::on_number_end;
use super::us_labelled::STATES;
use crate::hits::Hits;
use crate::tables::us_counties::{SUFFIXES, county};
use crate::tables::us_states::US_STATES;
use crate::text::{before, name_char, on_end, on_name_end, on_name_start};
use crate::window::find_any_label;

/// The labels a code is read after, longest first.
const LABELS: [&[u8]; 2] = [b"state of residence", b"state"];
/// What after `Washington` makes it the District.
const DISTRICT: [&[u8]; 4] = [b", D.C.", b" D.C.", b", DC", b" DC"];

pub(crate) fn scan(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i < b.len() {
        if !b[i].is_ascii_uppercase() || !on_name_start(text, i) {
            i += 1;
            continue;
        }
        let found = name_at(text, i).or_else(|| code_at(text, i).map(|end| (end, &text[i..end])));
        match found {
            Some((end, code)) => {
                hits.add(i, end, PiiKind::State, PiiEvidence::Shape, |s| s.push_str(code));
                i = end;
            }
            None => i += 1,
        }
    }
}

/// The end and code of the state name at `i`.
pub(super) fn name_at(text: &str, i: usize) -> Option<(usize, &'static str)> {
    let b = text.as_bytes();
    let (name, code) = US_STATES.iter().find(|(name, _)| {
        let n = name.as_bytes();
        b.get(i..i + n.len())
            .is_some_and(|w| w == n || (w.eq_ignore_ascii_case(n) && !w.iter().any(u8::is_ascii_lowercase)))
            && on_name_end(text, i + n.len())
    })?;
    let end = i + name.len();
    if *code == "WA"
        && let Some(d) = DISTRICT.iter().find(|d| b.get(end..end + d.len()).is_some_and(|w| w == **d))
        && on_name_end(text, end + d.len())
    {
        return Some((end + d.len(), "DC"));
    }
    if *code == "NY" && b.get(end..end + 5).is_some_and(|w| w.eq_ignore_ascii_case(b" city")) && on_end(text, end + 5) {
        return None;
    }
    // `Washington County` is a county, and the county scanner reads it;
    // a suffix word that makes no county (`the Texas area`, `Illinois
    // County`) leaves the state. Here, as for `New York City`, only a
    // letter or digit after the word makes it another word, so
    // `Washington County_x` is no state either.
    if b.get(end) == Some(&b' ')
        && SUFFIXES.iter().any(|s| {
            let at = end + 1;
            b.get(at..at + s.len()).is_some_and(|w| w.eq_ignore_ascii_case(s.as_bytes()))
                && on_end(text, at + s.len())
                && county(&text[i..at + s.len()]).is_some()
        })
    {
        return None;
    }
    Some((end, code))
}

/// The end of a two-capital USPS code at `i` that a state label is
/// before or a ZIP code after.
fn code_at(text: &str, i: usize) -> Option<usize> {
    let b = text.as_bytes();
    let code = b.get(i..i + 2)?;
    if !code.iter().all(u8::is_ascii_uppercase)
        || !on_name_end(text, i + 2)
        || !STATES.iter().any(|s| s.as_slice() == code)
    {
        return None;
    }
    let zip = b.get(i + 2) == Some(&b' ') && zip_end(text, i + 3).is_some_and(|end| address_like(text, i, end));
    (zip || find_any_label(text, i, &LABELS, false).is_some()).then_some(i + 2)
}

/// Whether a code at `code` before the ZIP code ending at `end` reads as
/// the end of an address, by what comes before it: a comma
/// (`Springfield, IL 62704`, `Springfield,IL 62704`), brackets around the
/// code and ZIP code (`(IL 62704)`), a colon label (`Address: IL 62704`),
/// a capitalized word and horizontal blanks when the ZIP code's
/// three-digit prefix belongs to that state (`Springfield IL 62704 is`,
/// `Springfield  IL 62704`, `Springfield\tIL 62704`,
/// `Saint Louis MO 63101`, `Reading PA 19601`), or only blanks back to the
/// start of a line when only blanks, or blanks and a clause's end, follow
/// the ZIP code (`IL 62704`, `\u{a0}IL 62704`, `IL 62704.`). A small word
/// or a prefix from another state (`(see IN 12345)`, `Order IN 12345`,
/// `PART OR 90210`), or a word after a line's code and number
/// (`IN 12345 units`), makes the number a count.
fn address_like(text: &str, code: usize, end: usize) -> bool {
    let b = text.as_bytes();
    // Only the blanks directly before the code and after the ZIP code are
    // walked, so each byte is looked at once however many codes a line
    // holds.
    let p = skip_blanks_back(b, code);
    if p > 0 && b[p - 1] == b',' && (p < code || before(text, p - 1).is_some_and(name_char)) {
        return true;
    }
    if p == 0 || b[p - 1] == b'\n' {
        return line_ends_after(text, end);
    }
    let after = skip_blanks(b, end);
    if (b[p - 1] == b'(' && b.get(after) == Some(&b')')) || (b[p - 1] == b'[' && b.get(after) == Some(&b']')) {
        return true;
    }
    if p < code && b[p - 1] == b':' && before(text, p - 1).is_some_and(name_char) {
        return line_ends_after(text, end);
    }
    p < code && zip_prefix_in_state(&b[code..code + 2], &b[code + 3..code + 6]) && capitalized_word_before(text, p)
}

/// Steps back over horizontal blanks (`' '`, `'\t'` and U+00A0) ending at `i`.
fn skip_blanks_back(b: &[u8], mut i: usize) -> usize {
    loop {
        if i > 0 && matches!(b[i - 1], b' ' | b'\t') {
            i -= 1;
        } else if i >= 2 && b[i - 2..i] == [0xC2, 0xA0] {
            i -= 2;
        } else {
            return i;
        }
    }
}

/// Steps forward over horizontal blanks (`' '`, `'\t'` and U+00A0) starting at `i`.
fn skip_blanks(b: &[u8], mut i: usize) -> usize {
    loop {
        if i < b.len() && matches!(b[i], b' ' | b'\t') {
            i += 1;
        } else if b.get(i..i + 2) == Some(&[0xC2, 0xA0]) {
            i += 2;
        } else {
            return i;
        }
    }
}

/// Whether the word ending at `end` is shaped like the last word of a
/// place name: letters and internal apostrophes or hyphens (`Reading`,
/// `O'Fallon`, `Winston-Salem`), no digit or `_`, starting with a
/// capital and ending with a word character.
fn capitalized_word_before(text: &str, end: usize) -> bool {
    let start = text[..end]
        .char_indices()
        .rev()
        .find(|&(_, c)| !(name_char(c) || matches!(c, '\'' | '\u{2019}' | '-')))
        .map_or(0, |(p, c)| p + c.len_utf8());
    let word = &text[start..end];
    !word.bytes().any(|b| b.is_ascii_digit() || b == b'_')
        && word.chars().next().is_some_and(char::is_uppercase)
        && word.chars().next_back().is_some_and(name_char)
}

/// Whether the three-digit ZIP prefix `zip` (`005`..`999`) is assigned by
/// the USPS to the state, territory or armed-forces designation `code`.
fn zip_prefix_in_state(code: &[u8], zip: &[u8]) -> bool {
    let p = u16::from(zip[0] - b'0') * 100 + u16::from(zip[1] - b'0') * 10 + u16::from(zip[2] - b'0');
    match code {
        b"AL" => (350..=369).contains(&p),
        b"AK" => (995..=999).contains(&p),
        b"AZ" => (850..=865).contains(&p),
        b"AR" => (716..=729).contains(&p),
        b"CA" => (900..=961).contains(&p),
        b"CO" => (800..=816).contains(&p),
        b"CT" => (60..=69).contains(&p),
        b"DE" => (197..=199).contains(&p),
        b"FL" => (320..=339).contains(&p) || (341..=349).contains(&p),
        b"GA" => (300..=319).contains(&p) || (398..=399).contains(&p),
        b"HI" => (967..=968).contains(&p),
        b"ID" => (832..=838).contains(&p),
        b"IL" => (600..=629).contains(&p),
        b"IN" => (460..=479).contains(&p),
        b"IA" => (500..=528).contains(&p),
        b"KS" => (660..=679).contains(&p),
        b"KY" => (400..=427).contains(&p),
        b"LA" => (700..=714).contains(&p),
        b"ME" => (39..=49).contains(&p),
        b"MD" => (206..=219).contains(&p),
        b"MA" => (10..=27).contains(&p) || p == 55,
        b"MI" => (480..=499).contains(&p),
        b"MN" => (550..=567).contains(&p),
        b"MS" => (386..=397).contains(&p),
        b"MO" => (630..=658).contains(&p),
        b"MT" => (590..=599).contains(&p),
        b"NE" => (680..=693).contains(&p),
        b"NV" => (889..=898).contains(&p),
        b"NH" => (30..=38).contains(&p),
        b"NJ" => (70..=89).contains(&p),
        b"NM" => (870..=884).contains(&p),
        b"NY" => p == 5 || p == 63 || (100..=149).contains(&p),
        b"NC" => (270..=289).contains(&p),
        b"ND" => (580..=588).contains(&p),
        b"OH" => (430..=459).contains(&p),
        b"OK" => (730..=732).contains(&p) || (734..=749).contains(&p),
        b"OR" => (970..=979).contains(&p),
        b"PA" => (150..=196).contains(&p),
        b"RI" => (28..=29).contains(&p),
        b"SC" => (290..=299).contains(&p),
        b"SD" => (570..=577).contains(&p),
        b"TN" => (370..=385).contains(&p),
        b"TX" => p == 733 || (750..=799).contains(&p) || p == 885,
        b"UT" => (840..=847).contains(&p),
        b"VT" => (50..=54).contains(&p) || (56..=59).contains(&p),
        b"VA" => p == 201 || (220..=246).contains(&p),
        b"WA" => (980..=994).contains(&p),
        b"WV" => (247..=268).contains(&p),
        b"WI" => (530..=549).contains(&p),
        b"WY" => (820..=831).contains(&p),
        b"DC" => p == 200 || (202..=205).contains(&p) || p == 569,
        b"AS" => p == 967,
        b"GU" | b"MP" | b"FM" | b"MH" | b"PW" => p == 969,
        b"PR" => (6..=7).contains(&p) || p == 9,
        b"VI" => p == 8,
        b"UM" => p == 968,
        b"AA" => p == 340,
        b"AE" => (90..=98).contains(&p),
        b"AP" => (962..=966).contains(&p),
        _ => false,
    }
}

/// Whether only blanks follow `end` to the end of its line or the text,
/// or blanks and then a mark that ends a clause (`IL 62704.`,
/// `IL 62704, USA`), not a word (`IN 12345 units`).
fn line_ends_after(text: &str, end: usize) -> bool {
    text[end..].chars().find(|&c| !matches!(c, ' ' | '\t' | '\r' | '\u{a0}')).is_none_or(|c| {
        matches!(c, '\n' | '\u{b}' | '\u{c}' | '\u{85}' | '\u{2028}' | '\u{2029}' | ',' | '.' | ';' | ')')
    })
}

/// The end of a whole ZIP or ZIP+4 token at `at` (`62704`, `62704-1234`),
/// with no letter, digit, `-` or `_` joined to it and not the start of a
/// decimal, grouped or split number (`12345abc`, `12345.67`, `12345,678`
/// and `12345/6` are none).
fn zip_end(text: &str, at: usize) -> Option<usize> {
    let b = text.as_bytes();
    let digits = |from: usize, n: usize| b.get(from..from + n).is_some_and(|d| d.iter().all(u8::is_ascii_digit));
    if !digits(at, 5) {
        return None;
    }
    let end = if b.get(at + 5) == Some(&b'-') && digits(at + 6, 4) { at + 10 } else { at + 5 };
    (on_number_end(text, end, b".,-/") && !matches!(b.get(end), Some(b'-' | b'_'))).then_some(end)
}
