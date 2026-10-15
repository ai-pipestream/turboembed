//! `grapheme_break.rs`: the UAX #29 Grapheme_Cluster_Break property and
//! the Indic_Conjunct_Break property of every code point, packed into one
//! byte.

use super::{Inputs, emit_trie};
use crate::emit::{RustFile, str_lit};
use crate::properties::{enumerated, valued};
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;

/// The Grapheme_Cluster_Break values as `GraphemeBreakProperty.txt` names
/// them, in the order of their numbers in the generated table, which is
/// the order of the library's `GraphemeBreak` enum. `Other` is the value
/// of every code point the file does not list.
const GRAPHEME_BREAKS: [&str; 14] = [
    "Other",
    "CR",
    "LF",
    "Control",
    "Extend",
    "ZWJ",
    "Regional_Indicator",
    "Prepend",
    "SpacingMark",
    "L",
    "V",
    "T",
    "LV",
    "LVT",
];

/// The Indic_Conjunct_Break values as `DerivedCoreProperties.txt` names
/// them, in the order of the library's `IndicConjunctBreak` enum. `None`
/// is the value of every code point the file does not list.
const INDIC_CONJUNCT_BREAKS: [&str; 4] = ["None", "Linker", "Consonant", "Extend"];

/// The Indic_Conjunct_Break value is stored above the four bits of the
/// Grapheme_Cluster_Break value.
const INCB_SHIFT: u8 = 4;

pub fn render(inputs: &Inputs) -> String {
    let property_file = inputs.file("GraphemeBreakProperty.txt");
    let mut values = vec![0u8; CODE_POINTS];
    for (range, name) in enumerated(&property_file) {
        let number = GRAPHEME_BREAKS
            .iter()
            .position(|&value| value == name)
            .unwrap_or_else(|| panic!("GraphemeBreakProperty.txt: unknown Grapheme_Cluster_Break value {name}"));
        assert!(number != 0, "GraphemeBreakProperty.txt lists the default value Other");
        for cp in range {
            if values[cp as usize] != 0 {
                panic!("GraphemeBreakProperty.txt: U+{cp:04X} is listed twice");
            }
            values[cp as usize] = number as u8;
        }
    }
    let core = inputs.file("DerivedCoreProperties.txt");
    for (range, name) in valued(&core, "InCB") {
        let number = INDIC_CONJUNCT_BREAKS
            .iter()
            .position(|&value| value == name)
            .unwrap_or_else(|| panic!("DerivedCoreProperties.txt: unknown InCB value {name}"));
        for cp in range {
            if values[cp as usize] >> INCB_SHIFT != 0 {
                panic!("DerivedCoreProperties.txt: U+{cp:04X} has two InCB values");
            }
            values[cp as usize] |= (number as u8) << INCB_SHIFT;
        }
    }
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "The UAX #29 Grapheme_Cluster_Break property of every code point in the low four bits, encoded as the \
         position of its value in NAMES (and in crate::grapheme::GraphemeBreak::ALL), and its \
         Indic_Conjunct_Break value above them, encoded as the position of the value in INCB_NAMES (and in \
         crate::grapheme::IndicConjunctBreak::ALL).",
        &["GraphemeBreakProperty.txt", "DerivedCoreProperties.txt"],
        inputs.version,
    );
    let names: Vec<String> = GRAPHEME_BREAKS.iter().map(|name| str_lit(name)).collect();
    file.blank();
    file.array(
        "The GraphemeBreakProperty.txt name of every value, in the order of their numbers.",
        "NAMES",
        "&str",
        &names,
    );
    let incb: Vec<String> = INDIC_CONJUNCT_BREAKS.iter().map(|name| str_lit(name)).collect();
    file.blank();
    file.array(
        "The DerivedCoreProperties.txt name of every Indic_Conjunct_Break value, in the order of their numbers.",
        "INCB_NAMES",
        "&str",
        &incb,
    );
    file.blank();
    file.constant(
        "The shift of the Indic_Conjunct_Break value in a stored byte.",
        "INCB_SHIFT",
        "u8",
        &INCB_SHIFT.to_string(),
    );
    emit_trie(&mut file, "the Grapheme_Cluster_Break and Indic_Conjunct_Break values", "u8", &trie);
    file.finish()
}
