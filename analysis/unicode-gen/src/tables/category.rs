//! `category.rs`: the general category of every code point and the
//! decimal digit values.

use super::{Inputs, emit_trie};
use crate::emit::{RustFile, hex};
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;
use crate::unicode_data::DECIMAL_NUMBER;

pub fn render(inputs: &Inputs) -> String {
    let values: Vec<u8> = (0..CODE_POINTS as u32).map(|cp| inputs.data.category(cp)).collect();
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "The General_Category of every code point, encoded as the position of its value in \
         crate::category::GeneralCategory (Lu = 0 to Cn = 29), and the first code point of every run of decimal \
         digits.",
        &["UnicodeData.txt"],
        inputs.version,
    );
    emit_trie(&mut file, "the general category", "u8", &trie);
    let zeros: Vec<String> = decimal_zeros(inputs).into_iter().map(hex).collect();
    file.blank();
    file.array(
        "The digit zero of every run of ten decimal digits (General_Category Nd), in ascending order. Every Nd \
         character is in such a run, so its digit value is its distance from the nearest zero below it.",
        "DECIMAL_ZEROS",
        "u32",
        &zeros,
    );
    file.finish()
}

/// The zero of every run of Nd characters, after checking that every Nd
/// character has the digit value of its distance from that zero: the
/// library computes digit values that way.
fn decimal_zeros(inputs: &Inputs) -> Vec<u32> {
    let mut zeros = Vec::new();
    for cp in 0..CODE_POINTS as u32 {
        let Some(data) = inputs.data.get(cp).filter(|data| data.category == DECIMAL_NUMBER) else { continue };
        let value = data.decimal.unwrap_or_else(|| panic!("U+{cp:04X} is Nd without a decimal digit value"));
        if value == 0 {
            zeros.push(cp);
        }
        match zeros.last() {
            Some(&zero) if cp - zero == u32::from(value) => {}
            _ => panic!("U+{cp:04X} has digit value {value} but is not at that distance from a zero"),
        }
    }
    zeros
}
