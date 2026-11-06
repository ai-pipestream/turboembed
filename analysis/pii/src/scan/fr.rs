//! French INSEE national and business identifier scanners: NIR (`fr-nir`),
//! SIREN (`fr-siren`), and SIRET (`fr-siret`).

use turbo_analysis_layers::{PiiConfidence, PiiKind};

use super::labelled::has_any_label;
use super::national::{JOINERS, on_number_end};
use crate::checksum::fr;
use crate::hits::Hits;
use crate::text::{before, on_number_start};

const NIR_GROUPS: &[&[usize]] = &[&[1, 2, 2, 2, 3, 3, 2], &[1, 2, 2, 3, 2, 3, 2], &[1, 2, 2, 5, 3, 2], &[13, 2], &[15]];

const SIRET_GROUPS: &[&[usize]] = &[&[3, 3, 3, 5], &[9, 5], &[14]];
const SIREN_LABELS: [&[u8]; 4] = [b"numero siren", b"no siren", b"n siren", b"siren"];

/// Scans 15-character French NIRs (bare or grouped by uniform spaces or
/// hyphens, with Corsica `2A`/`2B` department codes).
pub(crate) fn scan_fr_nir(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut norm = [0u8; 15];
    let mut i = 0;
    while i + 15 <= b.len() {
        if !matches!(b[i], b'1' | b'2' | b'3' | b'4' | b'7' | b'8')
            || !on_number_start(text, i)
            || before(text, i).is_some_and(|c| c == '/')
        {
            i += 1;
            continue;
        }
        let end = NIR_GROUPS.iter().find_map(|groups| read_nir_groups(text, i, groups, &mut norm));
        if let Some(end) = end
            && fr::nir(&norm)
        {
            hits.add(i, end, PiiKind::FrNir, PiiConfidence::Checksum, |s| {
                s.extend(norm.iter().map(|&c| c as char));
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

fn read_nir_groups(text: &str, start: usize, groups: &[usize], out: &mut [u8; 15]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    let mut sep = 0u8;
    for (g, &size) in groups.iter().enumerate() {
        if g > 0 {
            let c = *b.get(p)?;
            if (c != b' ' && c != b'-') || (g > 1 && c != sep) {
                return None;
            }
            sep = c;
            p += 1;
        }
        for _ in 0..size {
            let c = *b.get(p)?;
            let ok = if n == 6 { c.is_ascii_digit() || matches!(c, b'A' | b'B') } else { c.is_ascii_digit() };
            if !ok {
                return None;
            }
            out[n] = c;
            n += 1;
            p += 1;
        }
    }
    (n == 15 && on_number_end(text, p, JOINERS)).then_some(p)
}

/// Scans 9-digit French SIREN numbers: grouped (`ddd ddd ddd` or `ddd-ddd-ddd`)
/// or bare 9 digits after a `SIREN` label.
pub(crate) fn scan_fr_siren(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut norm = [0u8; 9];
    let mut i = 0;
    while i + 9 <= b.len() {
        if !matches!(b[i], b'1'..=b'9') || !on_number_start(text, i) || before(text, i).is_some_and(|c| c == '/') {
            i += 1;
            continue;
        }
        let end = read_digit_groups(text, i, &[3, 3, 3], &mut norm).or_else(|| {
            has_any_label(text, i, &SIREN_LABELS, true).then(|| read_digit_groups(text, i, &[9], &mut norm)).flatten()
        });
        if let Some(end) = end
            && fr::siren(&norm)
        {
            hits.add(i, end, PiiKind::FrSiren, PiiConfidence::Checksum, |s| {
                s.extend(norm.iter().map(|&c| c as char));
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

/// Scans 14-digit French SIRET numbers (`ddd ddd ddd ddddd`, `ddddddddd ddddd`,
/// or bare 14 digits).
pub(crate) fn scan_fr_siret(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut norm = [0u8; 14];
    let mut i = 0;
    while i + 14 <= b.len() {
        if !matches!(b[i], b'1'..=b'9') || !on_number_start(text, i) || before(text, i).is_some_and(|c| c == '/') {
            i += 1;
            continue;
        }
        let end = SIRET_GROUPS.iter().find_map(|groups| read_digit_groups(text, i, groups, &mut norm));
        if let Some(end) = end
            && b.get(end) != Some(&b'/')
            && fr::siret(&norm)
        {
            hits.add(i, end, PiiKind::FrSiret, PiiConfidence::Checksum, |s| {
                s.extend(norm.iter().map(|&c| c as char));
            });
            i = end;
        } else {
            i += 1;
        }
    }
}

pub(crate) fn read_digit_groups(text: &str, start: usize, groups: &[usize], out: &mut [u8]) -> Option<usize> {
    let b = text.as_bytes();
    let mut p = start;
    let mut n = 0;
    let mut sep = 0u8;
    for (g, &size) in groups.iter().enumerate() {
        if g > 0 {
            let c = *b.get(p)?;
            if (c != b' ' && c != b'-') || (g > 1 && c != sep) {
                return None;
            }
            sep = c;
            p += 1;
        }
        for _ in 0..size {
            let c = *b.get(p).filter(|c| c.is_ascii_digit())?;
            out[n] = c;
            n += 1;
            p += 1;
        }
    }
    (n == out.len() && on_number_end(text, p, JOINERS)).then_some(p)
}
