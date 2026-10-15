//! `sentence_break.rs`: the UAX #29 Sentence_Break property of every code
//! point.

use super::{Inputs, emit_trie};
use crate::emit::{RustFile, str_lit};
use crate::properties::enumerated;
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;

/// The Sentence_Break values as `SentenceBreakProperty.txt` names them, in
/// the order of their numbers in the generated table. The library's
/// `SentenceBreak` enum uses the same order. `Other` is the value of every
/// code point the file does not list.
const SENTENCE_BREAKS: [&str; 15] = [
    "Other",
    "CR",
    "LF",
    "Extend",
    "Sep",
    "Format",
    "Sp",
    "Lower",
    "Upper",
    "OLetter",
    "Numeric",
    "ATerm",
    "SContinue",
    "STerm",
    "Close",
];

pub fn render(inputs: &Inputs) -> String {
    let file_in = inputs.file("SentenceBreakProperty.txt");
    let mut values = vec![0u8; CODE_POINTS];
    for (range, name) in enumerated(&file_in) {
        let number = SENTENCE_BREAKS
            .iter()
            .position(|&value| value == name)
            .unwrap_or_else(|| panic!("SentenceBreakProperty.txt: unknown Sentence_Break value {name}"));
        assert!(number != 0, "SentenceBreakProperty.txt lists the default value Other");
        for cp in range {
            if values[cp as usize] != 0 {
                panic!("SentenceBreakProperty.txt: U+{cp:04X} is listed twice");
            }
            values[cp as usize] = number as u8;
        }
    }
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "The UAX #29 Sentence_Break property of every code point, encoded as the position of its value in NAMES, \
         which is also its position in crate::sentence_break::SentenceBreak::ALL.",
        &["SentenceBreakProperty.txt"],
        inputs.version,
    );
    let names: Vec<String> = SENTENCE_BREAKS.iter().map(|name| str_lit(name)).collect();
    file.blank();
    file.array(
        "The SentenceBreakProperty.txt name of every value, in the order of their numbers.",
        "NAMES",
        "&str",
        &names,
    );
    emit_trie(&mut file, "the Sentence_Break value", "u8", &trie);
    file.finish()
}
