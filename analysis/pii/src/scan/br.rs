//! Brazilian tax identifier scanners (`BrIdentityPiiExtractor`): CPF (`br-cpf`)
//! and CNPJ (`br-cnpj`).

use turbo_analysis_layers::{PiiConfidence, PiiKind, PiiKinds};

use super::fr::read_digit_groups;
use super::national::on_number_end;
use crate::checksum::mod11;
use crate::hits::Hits;
use crate::text::{before, on_number_start, on_word_start};

const BR_JOINERS: &[u8] = b".,-/ ";

/// Scans Brazilian CPFs and CNPJs enabled in `kinds`.
pub(crate) fn scan_br(text: &str, kinds: PiiKinds, hits: &mut Hits) {
    if kinds.contains(PiiKind::BrCnpj) {
        scan_cnpj(text, hits);
    }
    if kinds.contains(PiiKind::BrCpf) {
        scan_cpf(text, hits);
    }
}

fn scan_cpf(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut norm = [0u8; 11];
    let mut i = 0;
    while i + 11 <= b.len() {
        if !b[i].is_ascii_digit() || !on_number_start(text, i) || before(text, i).is_some_and(|c| c == '/') {
            i += 1;
            continue;
        }
        let end = read_punctuated_cpf(text, i, &mut norm)
            .or_else(|| read_digit_groups(text, i, &[3, 3, 3, 2], &mut norm))
            .or_else(|| read_digit_groups(text, i, &[11], &mut norm));
        if let Some(end) = end
            && on_number_end(text, end, BR_JOINERS)
            && mod11::br_cpf(&norm)
        {
            hits.add(i, end, PiiKind::BrCpf, PiiConfidence::Checksum, |s| {
                s.extend(norm.iter().map(|&c| c as char));
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

/// Reads `ddd.ddd.ddd-dd` (14 bytes) into `out`.
fn read_punctuated_cpf(text: &str, start: usize, out: &mut [u8; 11]) -> Option<usize> {
    let s = text.as_bytes().get(start..start + 14)?;
    if s[3] != b'.' || s[7] != b'.' || s[11] != b'-' {
        return None;
    }
    let parts = [&s[0..3], &s[4..7], &s[8..11], &s[12..14]];
    if !parts.iter().all(|p| p.iter().all(u8::is_ascii_digit)) {
        return None;
    }
    out[0..3].copy_from_slice(parts[0]);
    out[3..6].copy_from_slice(parts[1]);
    out[6..9].copy_from_slice(parts[2]);
    out[9..11].copy_from_slice(parts[3]);
    Some(start + 14)
}

fn scan_cnpj(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut norm = [0u8; 14];
    let mut i = 0;
    while i + 14 <= b.len() {
        if !(b[i].is_ascii_digit() || b[i].is_ascii_uppercase())
            || !on_word_start(text, i)
            || before(text, i).is_some_and(|c| matches!(c, '-' | '/' | '.' | '@'))
        {
            i += 1;
            continue;
        }
        let end = read_punctuated_cnpj(text, i, &mut norm).or_else(|| {
            if !b[i].is_ascii_digit() || !on_number_start(text, i) {
                return None;
            }
            read_digit_groups(text, i, &[2, 3, 3, 4, 2], &mut norm)
                .or_else(|| read_digit_groups(text, i, &[14], &mut norm))
        });
        if let Some(end) = end
            && on_number_end(text, end, BR_JOINERS)
            && mod11::br_cnpj(&norm)
        {
            hits.add(i, end, PiiKind::BrCnpj, PiiConfidence::Checksum, |s| {
                s.extend(norm.iter().map(|&c| c as char));
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

/// Reads `xx.xxx.xxx/xxxx-dd` (18 bytes, numeric or 2026 alphanumeric root/branch)
/// into `out`.
fn read_punctuated_cnpj(text: &str, start: usize, out: &mut [u8; 14]) -> Option<usize> {
    let s = text.as_bytes().get(start..start + 18)?;
    if s[2] != b'.' || s[6] != b'.' || s[10] != b'/' || s[15] != b'-' {
        return None;
    }
    let body = [&s[0..2], &s[3..6], &s[7..10], &s[11..15]];
    if !body.iter().all(|p| p.iter().all(|b| b.is_ascii_digit() || b.is_ascii_uppercase()))
        || !s[16..18].iter().all(u8::is_ascii_digit)
    {
        return None;
    }
    out[0..2].copy_from_slice(body[0]);
    out[2..5].copy_from_slice(body[1]);
    out[5..8].copy_from_slice(body[2]);
    out[8..12].copy_from_slice(body[3]);
    out[12..14].copy_from_slice(&s[16..18]);
    Some(start + 18)
}
