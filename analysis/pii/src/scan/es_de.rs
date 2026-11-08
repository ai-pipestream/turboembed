//! Spanish (DNI, NIE, CIF) and German Personalausweis national identifier
//! scanners.

use turbo_analysis_layers::{ChecksumAlgorithm, PiiEvidence, PiiKind};

use super::national::{JOINERS, on_number_end};
use crate::checksum::{es, icao9303};
use crate::hits::Hits;
use crate::text::{before, on_number_start, on_word_start};

#[inline]
fn clean_token_start(text: &str, i: usize) -> bool {
    on_word_start(text, i) && !before(text, i).is_some_and(|c| matches!(c, '-' | '/' | '.' | '@'))
}

#[inline]
fn clean_token_end(text: &str, end: usize) -> bool {
    let b = text.as_bytes();
    on_number_end(text, end, JOINERS)
        && !(end + 1 < b.len() && matches!(b[end], b'-' | b'/') && b[end + 1].is_ascii_alphanumeric())
}

/// Scans Spanish DNIs: 8 ASCII digits + optional `-` + uppercase mod-23
/// control letter (`12345678Z`, `12345678-Z`).
pub(crate) fn scan_es_dni(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 9 <= b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) || before(text, i).is_some_and(|c| c == '/') {
            i += 1;
            continue;
        }
        if let Some((end, norm)) = read_es_dni(text, i) {
            hits.add(
                i,
                end,
                PiiKind::EsDni,
                PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Mod23, passed: true },
                |s| {
                    s.extend(norm.iter().map(|&c| c as char));
                },
            );
            i = end;
        } else {
            i += 1;
        }
    }
}

fn read_es_dni(text: &str, start: usize) -> Option<(usize, [u8; 9])> {
    let b = text.as_bytes();
    let digits = b.get(start..start + 8).filter(|d| d.iter().all(u8::is_ascii_digit))?;
    let mut p = start + 8;
    if b.get(p) == Some(&b'-') {
        p += 1;
    }
    let letter = *b.get(p).filter(|c| c.is_ascii_uppercase())?;
    let end = p + 1;
    if !clean_token_end(text, end) {
        return None;
    }
    let mut norm = [0u8; 9];
    norm[..8].copy_from_slice(digits);
    norm[8] = letter;
    es::dni(&norm).then_some((end, norm))
}

/// Scans Spanish NIEs: `X`, `Y`, or `Z` + optional `-` + 7 ASCII digits +
/// optional `-` + uppercase mod-23 control letter (`X1234567L`, `X-1234567-L`).
pub(crate) fn scan_es_nie(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 9 <= b.len() {
        if !matches!(b[i], b'X' | b'Y' | b'Z') || !clean_token_start(text, i) {
            i += 1;
            continue;
        }
        if let Some((end, norm)) = read_es_nie(text, i) {
            hits.add(
                i,
                end,
                PiiKind::EsNie,
                PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Mod23, passed: true },
                |s| {
                    s.extend(norm.iter().map(|&c| c as char));
                },
            );
            i = end;
        } else {
            i += 1;
        }
    }
}

fn read_es_nie(text: &str, start: usize) -> Option<(usize, [u8; 9])> {
    let b = text.as_bytes();
    let prefix = b[start];
    let mut p = start + 1;
    if b.get(p) == Some(&b'-') {
        p += 1;
    }
    let digits = b.get(p..p + 7).filter(|d| d.iter().all(u8::is_ascii_digit))?;
    p += 7;
    if b.get(p) == Some(&b'-') {
        p += 1;
    }
    let letter = *b.get(p).filter(|c| c.is_ascii_uppercase())?;
    let end = p + 1;
    if !clean_token_end(text, end) {
        return None;
    }
    let mut norm = [0u8; 9];
    norm[0] = prefix;
    norm[1..8].copy_from_slice(digits);
    norm[8] = letter;
    es::nie(&norm).then_some((end, norm))
}

/// Scans Spanish CIFs: organization type letter + optional `-` + 7 ASCII
/// digits + optional `-` + control digit or letter (`A58818501`, `Q-2826000-H`).
pub(crate) fn scan_es_cif(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 9 <= b.len() {
        if !b[i].is_ascii_uppercase() || !clean_token_start(text, i) {
            i += 1;
            continue;
        }
        if let Some((end, norm)) = read_es_cif(text, i) {
            hits.add(
                i,
                end,
                PiiKind::EsCif,
                PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::EsCif, passed: true },
                |s| {
                    s.extend(norm.iter().map(|&c| c as char));
                },
            );
            i = end;
        } else {
            i += 1;
        }
    }
}

fn read_es_cif(text: &str, start: usize) -> Option<(usize, [u8; 9])> {
    let b = text.as_bytes();
    let prefix = b[start];
    let mut p = start + 1;
    let first_hyphen = b.get(p) == Some(&b'-');
    if first_hyphen {
        p += 1;
    }
    let digits = b.get(p..p + 7).filter(|d| d.iter().all(u8::is_ascii_digit))?;
    p += 7;
    let second_hyphen = b.get(p) == Some(&b'-');
    if second_hyphen {
        p += 1;
    }
    let control = *b.get(p).filter(|c| c.is_ascii_digit() || c.is_ascii_uppercase())?;
    if second_hyphen && control.is_ascii_digit() && !first_hyphen {
        return None;
    }
    let end = p + 1;
    if !clean_token_end(text, end) {
        return None;
    }
    let mut norm = [0u8; 9];
    norm[0] = prefix;
    norm[1..8].copy_from_slice(digits);
    norm[8] = control;
    es::cif(&norm).then_some((end, norm))
}

/// Scans 10-character German Personalausweis document numbers (`T220001293`).
pub(crate) fn scan_de_personalausweis(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + 10 <= b.len() {
        if !b[i].is_ascii_uppercase() || !clean_token_start(text, i) {
            i += 1;
            continue;
        }
        let end = i + 10;
        if let Ok(candidate) = <&[u8; 10]>::try_from(&b[i..end])
            && clean_token_end(text, end)
            && icao9303::de_personalausweis(candidate)
        {
            hits.add(
                i,
                end,
                PiiKind::DePersonalausweis,
                PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Icao9303, passed: true },
                |s| {
                    s.push_str(&text[i..end]);
                },
            );
            i = end;
        } else {
            i += 1;
        }
    }
}
