//! Reading property files: `PropList.txt`, `DerivedCoreProperties.txt`,
//! `emoji-data.txt`, `Scripts.txt` and `DerivedNormalizationProps.txt`.
//! Their records are `range ; Property` for binary properties and
//! `range ; Property ; Value` or `range ; Value` for the others.

use std::ops::RangeInclusive;

use crate::ucd::{CODE_POINTS, UcdFile};

/// Reads the binary properties `names` from `file`, one `Vec<bool>` per
/// name indexed by code point. Records of other properties are skipped. A
/// name that no record carries stops the generator, so a renamed property
/// cannot silently become empty.
pub fn binary(file: &UcdFile, names: &[&str]) -> Vec<Vec<bool>> {
    let mut sets = vec![vec![false; CODE_POINTS]; names.len()];
    let mut seen = vec![false; names.len()];
    for record in file.records() {
        if record.field_count() != 2 {
            continue;
        }
        let Some(which) = names.iter().position(|&name| name == record.field(1)) else { continue };
        seen[which] = true;
        for cp in record.range(0) {
            sets[which][cp as usize] = true;
        }
    }
    if let Some(missing) = seen.iter().position(|&s| !s) {
        panic!("{}: no record of property {}", file.name(), names[missing]);
    }
    sets
}

/// Reads the records `range ; value` of a file that assigns one value of
/// one enumerated property per line, such as `Scripts.txt`.
pub fn enumerated(file: &UcdFile) -> Vec<(RangeInclusive<u32>, String)> {
    file.records()
        .map(|record| {
            if record.field_count() != 2 {
                record.fail("expected `range ; value`");
            }
            (record.range(0), record.field(1).to_owned())
        })
        .collect()
}

/// Reads the records `range ; property ; value` of `property` from `file`.
pub fn valued(file: &UcdFile, property: &str) -> Vec<(RangeInclusive<u32>, String)> {
    let found: Vec<_> = file
        .records()
        .filter(|record| record.field_count() == 3 && record.field(1) == property)
        .map(|record| (record.range(0), record.field(2).to_owned()))
        .collect();
    if found.is_empty() {
        panic!("{}: no record of property {property}", file.name());
    }
    found
}
