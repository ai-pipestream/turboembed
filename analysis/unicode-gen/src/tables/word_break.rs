//! `word_break.rs`: the UAX #29 Word_Break property of every code point.

use super::{Inputs, emit_trie};
use crate::emit::{RustFile, str_lit};
use crate::properties::enumerated;
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;

/// The Word_Break values as `WordBreakProperty.txt` names them, in the
/// order of their numbers in the generated table. The library's
/// `WordBreak` enum uses the same order, which is the order of OpenNLP's
/// `WordBreak` enum. `Other` is the value of every code point the file
/// does not list.
const WORD_BREAKS: [&str; 19] = [
    "Other",
    "CR",
    "LF",
    "Newline",
    "Extend",
    "ZWJ",
    "Regional_Indicator",
    "Format",
    "Katakana",
    "Hebrew_Letter",
    "ALetter",
    "Single_Quote",
    "Double_Quote",
    "MidNumLet",
    "MidLetter",
    "MidNum",
    "Numeric",
    "ExtendNumLet",
    "WSegSpace",
];

pub fn render(inputs: &Inputs) -> String {
    let file_in = inputs.file("WordBreakProperty.txt");
    let mut values = vec![0u8; CODE_POINTS];
    for (range, name) in enumerated(&file_in) {
        let number = WORD_BREAKS
            .iter()
            .position(|&value| value == name)
            .unwrap_or_else(|| panic!("WordBreakProperty.txt: unknown Word_Break value {name}"));
        assert!(number != 0, "WordBreakProperty.txt lists the default value Other");
        for cp in range {
            if values[cp as usize] != 0 {
                panic!("WordBreakProperty.txt: U+{cp:04X} is listed twice");
            }
            values[cp as usize] = number as u8;
        }
    }
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "The UAX #29 Word_Break property of every code point, encoded as the position of its value in NAMES, \
         which is also its position in crate::word_break::WordBreak::ALL.",
        &["WordBreakProperty.txt"],
        inputs.version,
    );
    let names: Vec<String> = WORD_BREAKS.iter().map(|name| str_lit(name)).collect();
    file.blank();
    file.array(
        "The WordBreakProperty.txt name of every value, in the order of their numbers.",
        "NAMES",
        "&str",
        &names,
    );
    emit_trie(&mut file, "the Word_Break value", "u8", &trie);
    file.finish()
}
