//! United States Employer Identification Numbers, `UsIdentityPiiExtractor`:
//! `dd-ddddddd` whose two-digit prefix is one the IRS assigns to a campus
//! or to online applications, and whose serial is not all zeros.

use turbo_analysis_layers::{PiiEvidence, PiiKind};

use super::national::on_number_end;
use crate::hits::Hits;
use crate::text::on_number_start;

const LENGTH: usize = 10;

/// Whether the IRS assigns EINs with prefix `p` (00 to 99). The
/// unassigned prefixes are 00, 07 to 09, 17 to 19, 28, 29, 49, 69, 70,
/// 78, 79, 89, 96 and 97.
pub(crate) const fn ein_prefix(p: u8) -> bool {
    matches!(p, 1..=6 | 10..=16 | 20..=27 | 30..=48 | 50..=68 | 71..=77 | 80..=88 | 90..=95 | 98 | 99)
}

/// EINs written `dd-ddddddd`.
pub(crate) fn scan_ein(text: &str, hits: &mut Hits) {
    let b = text.as_bytes();
    let mut i = 0;
    while i + LENGTH <= b.len() {
        if ein_at(text, i) {
            hits.add_text(text, i, i + LENGTH, PiiKind::UsEin, PiiEvidence::Shape);
            i += LENGTH;
        } else {
            i += 1;
        }
    }
}

fn ein_at(text: &str, i: usize) -> bool {
    let b = text.as_bytes();
    let w = &b[i..i + LENGTH];
    if !w[0].is_ascii_digit() || !on_number_start(text, i) || w[2] != b'-' {
        return false;
    }
    let digits = || w[..2].iter().chain(&w[3..]);
    if !digits().all(u8::is_ascii_digit) || !on_number_end(text, i + LENGTH, b".,-") {
        return false;
    }
    let prefix = (w[0] - b'0') * 10 + (w[1] - b'0');
    ein_prefix(prefix) && w[3..].iter().any(|&d| d != b'0')
}

#[cfg(test)]
mod tests {
    use super::ein_prefix;

    #[test]
    fn unassigned_prefixes() {
        let unassigned: Vec<u8> = (0..100).filter(|&p| !ein_prefix(p)).collect();
        assert_eq!(unassigned, [0, 7, 8, 9, 17, 18, 19, 28, 29, 49, 69, 70, 78, 79, 89, 96, 97]);
    }
}
