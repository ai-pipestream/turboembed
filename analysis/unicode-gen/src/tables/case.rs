//! `case.rs`: simple case mappings, the unconditional special lowercase
//! mappings and case folding.

use std::collections::BTreeMap;
use std::path::Path;

use super::{Inputs, emit_map};
use crate::emit::{RustFile, char_lit, str_lit};
use crate::ucd::{CODE_POINTS, UcdFile, to_char, to_string};

/// `CaseFolding.txt` by status, and the unconditional lowercase mappings
/// of `SpecialCasing.txt`.
pub struct CaseData {
    /// Status C: the folding shared by simple and full folding.
    pub common: BTreeMap<u32, Vec<u32>>,
    /// Status F: full folding where it differs from simple folding.
    pub full: BTreeMap<u32, Vec<u32>>,
    /// Status S: simple folding where it differs from full folding.
    pub simple: BTreeMap<u32, u32>,
    /// Status T: the Turkic foldings of I and U+0130.
    pub turkic: BTreeMap<u32, u32>,
    /// Lowercase mappings of `SpecialCasing.txt` that apply in every
    /// context and every language.
    pub special_lower: BTreeMap<u32, Vec<u32>>,
}

impl CaseData {
    pub fn read(dir: &Path) -> CaseData {
        Self::parse(&UcdFile::read(dir, "CaseFolding.txt"), &UcdFile::read(dir, "SpecialCasing.txt"))
    }

    fn parse(folding: &UcdFile, special: &UcdFile) -> CaseData {
        let mut data = CaseData {
            common: BTreeMap::new(),
            full: BTreeMap::new(),
            simple: BTreeMap::new(),
            turkic: BTreeMap::new(),
            special_lower: BTreeMap::new(),
        };
        for record in folding.records() {
            if record.field_count() != 4 || !record.field(3).is_empty() {
                record.fail("expected `code; status; mapping;`");
            }
            let cp = record.code_point(0);
            let mapping = record.code_points(2);
            let single = || match mapping[..] {
                [one] => one,
                _ => record.fail("a simple folding maps to one code point"),
            };
            let previous = match record.field(1) {
                "C" => {
                    single();
                    data.common.insert(cp, mapping.clone()).map(|_| ())
                }
                "F" => {
                    if mapping.is_empty() {
                        record.fail("empty case folding");
                    }
                    data.full.insert(cp, mapping.clone()).map(|_| ())
                }
                "S" => data.simple.insert(cp, single()).map(|_| ()),
                "T" => data.turkic.insert(cp, single()).map(|_| ()),
                status => record.fail(&format!("unknown status {status}")),
            };
            if previous.is_some() {
                record.fail("code point listed twice with one status");
            }
        }
        for record in special.records() {
            let unconditional = record.field_count() == 5 && record.field(4).is_empty();
            let conditional = record.field_count() == 6 && record.field(5).is_empty();
            if !unconditional && !conditional {
                record.fail("expected `code; lower; title; upper; (conditions;)`");
            }
            if unconditional {
                let lower = record.code_points(1);
                if lower.is_empty() {
                    record.fail("empty special lowercase mapping");
                }
                if data.special_lower.insert(record.code_point(0), lower).is_some() {
                    record.fail("code point listed twice");
                }
            }
        }
        data
    }

    /// The full folding of `cp` (statuses C and F), `None` when it folds to itself.
    pub fn full_folding(&self, cp: u32) -> Option<&Vec<u32>> {
        self.full.get(&cp).or_else(|| self.common.get(&cp))
    }

    /// The simple folding of `cp` (statuses C and S), `None` when it folds to itself.
    pub fn simple_folding(&self, cp: u32) -> Option<u32> {
        self.simple.get(&cp).copied().or_else(|| match self.common.get(&cp)?[..] {
            [one] => Some(one),
            _ => panic!("U+{cp:04X}: a status C folding maps to one code point"),
        })
    }
}

/// The special lowercase mappings that differ from the simple lowercase
/// mapping: the ones `String.toLowerCase` applies beyond `Character.toLowerCase`.
pub fn special_lower(inputs: &Inputs) -> BTreeMap<u32, Vec<u32>> {
    let simple = |cp: u32| inputs.data.get(cp).and_then(|data| data.lower).unwrap_or(cp);
    inputs
        .case
        .special_lower
        .iter()
        .filter(|(cp, lower)| **lower != [simple(**cp)])
        .map(|(&cp, l)| (cp, l.clone()))
        .collect()
}

pub fn render(inputs: &Inputs) -> String {
    let mut file = RustFile::new(
        "Simple case mappings, the unconditional special lowercase mappings, and case folding. Each mapping is a \
         sorted key array with a value array beside it; the property table has a bit per mapping that says whether \
         a code point has an entry, so most lookups skip the search.",
        &["UnicodeData.txt", "SpecialCasing.txt", "CaseFolding.txt"],
        inputs.version,
    );
    let simple = |pick: fn(&crate::unicode_data::CharData) -> Option<u32>| -> Vec<(u32, String)> {
        (0..CODE_POINTS as u32)
            .filter_map(|cp| inputs.data.get(cp).and_then(pick).map(|mapped| (cp, char_lit(to_char(mapped)))))
            .collect()
    };
    emit_map(
        &mut file,
        "LOWER",
        ["The code points with a simple lowercase mapping other than themselves.", "Their simple lowercase mappings."],
        "char",
        &simple(|data| data.lower),
    );
    emit_map(
        &mut file,
        "UPPER",
        ["The code points with a simple uppercase mapping other than themselves.", "Their simple uppercase mappings."],
        "char",
        &simple(|data| data.upper),
    );
    let special: Vec<(u32, String)> =
        special_lower(inputs).into_iter().map(|(cp, lower)| (cp, str_lit(&to_string(&lower)))).collect();
    emit_map(
        &mut file,
        "SPECIAL_LOWER",
        [
            "The code points whose unconditional lowercase mapping in SpecialCasing.txt differs from the simple one.",
            "Their lowercase mappings from SpecialCasing.txt.",
        ],
        "&str",
        &special,
    );
    let all_folded: Vec<u32> = folded_code_points(&inputs.case);
    let simple_folding: Vec<(u32, String)> = all_folded
        .iter()
        .filter_map(|&cp| inputs.case.simple_folding(cp).map(|folded| (cp, char_lit(to_char(folded)))))
        .collect();
    emit_map(
        &mut file,
        "FOLD_SIMPLE",
        ["The code points simple case folding changes (statuses C and S).", "Their simple case foldings."],
        "char",
        &simple_folding,
    );
    let full_folding: Vec<(u32, String)> = all_folded
        .iter()
        .filter_map(|&cp| inputs.case.full_folding(cp).map(|folded| (cp, str_lit(&to_string(folded)))))
        .collect();
    emit_map(
        &mut file,
        "FOLD_FULL",
        ["The code points full case folding changes (statuses C and F).", "Their full case foldings."],
        "&str",
        &full_folding,
    );
    let turkic: Vec<(u32, String)> =
        inputs.case.turkic.iter().map(|(&cp, &folded)| (cp, str_lit(&to_string(&[folded])))).collect();
    emit_map(
        &mut file,
        "FOLD_TURKIC",
        [
            "The code points with a Turkic case folding (status T), which replaces their other foldings.",
            "Their Turkic case foldings, one code point each.",
        ],
        "&str",
        &turkic,
    );
    file.finish()
}

/// Every code point with a C, F or S folding, in ascending order.
pub fn folded_code_points(case: &CaseData) -> Vec<u32> {
    let mut cps: Vec<u32> = case.common.keys().chain(case.full.keys()).chain(case.simple.keys()).copied().collect();
    cps.sort_unstable();
    cps.dedup();
    cps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(folding: &str, special: &str) -> CaseData {
        CaseData::parse(
            &UcdFile::from_text("CaseFolding.txt", folding),
            &UcdFile::from_text("SpecialCasing.txt", special),
        )
    }

    #[test]
    #[should_panic(expected = "CaseFolding.txt:1: a simple folding maps to one code point")]
    fn status_c_rejects_multi_code_point_mapping() {
        parse("00DF; C; 0073 0073;\n", "");
    }

    #[test]
    #[should_panic(expected = "CaseFolding.txt:1: empty case folding")]
    fn status_f_rejects_empty_mapping() {
        parse("00DF; F; ;\n", "");
    }

    #[test]
    #[should_panic(expected = "SpecialCasing.txt:1: empty special lowercase mapping")]
    fn special_casing_rejects_empty_unconditional_lowercase() {
        parse("", "0130; ; 0130; 0130;\n");
    }

    #[test]
    #[should_panic(expected = "SpecialCasing.txt:2: code point listed twice")]
    fn special_casing_rejects_duplicate_unconditional_code_point() {
        parse("", "0130; 0069 0307; 0130; 0130;\n0130; 0069; 0130; 0130;\n");
    }
}
