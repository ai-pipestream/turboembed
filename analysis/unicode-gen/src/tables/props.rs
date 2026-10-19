//! `props.rs`: binary character properties and the bits that say which
//! mapping tables have an entry for a code point, packed into one `u32`
//! per code point.

use super::{Inputs, case, emit_trie};
use crate::emit::RustFile;
use crate::properties::binary;
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;

/// The UCD binary properties stored, by source file, in bit order.
const PROPERTIES: [(&str, &[(&str, &str)]); 3] = [
    (
        "PropList.txt",
        &[
            ("WHITE_SPACE", "White_Space"),
            ("DASH", "Dash"),
            ("QUOTATION_MARK", "Quotation_Mark"),
            ("REGIONAL_INDICATOR", "Regional_Indicator"),
            ("VARIATION_SELECTOR", "Variation_Selector"),
        ],
    ),
    (
        "DerivedCoreProperties.txt",
        &[
            ("DEFAULT_IGNORABLE", "Default_Ignorable_Code_Point"),
            ("ALPHABETIC", "Alphabetic"),
            ("LOWERCASE", "Lowercase"),
            ("UPPERCASE", "Uppercase"),
            ("CASED", "Cased"),
            ("CASE_IGNORABLE", "Case_Ignorable"),
        ],
    ),
    (
        "emoji-data.txt",
        &[
            ("EMOJI", "Emoji"),
            ("EMOJI_PRESENTATION", "Emoji_Presentation"),
            ("EMOJI_MODIFIER", "Emoji_Modifier"),
            ("EMOJI_MODIFIER_BASE", "Emoji_Modifier_Base"),
            ("EMOJI_COMPONENT", "Emoji_Component"),
            ("EXTENDED_PICTOGRAPHIC", "Extended_Pictographic"),
        ],
    ),
];

pub fn render(inputs: &Inputs) -> String {
    let mut values = vec![0u32; CODE_POINTS];
    let mut constants: Vec<(String, String)> = Vec::new();
    for (file_name, properties) in PROPERTIES {
        let file = inputs.file(file_name);
        let names: Vec<&str> = properties.iter().map(|&(_, name)| name).collect();
        for (set, &(constant, name)) in binary(&file, &names).iter().zip(properties) {
            let bit = add_bit(&mut constants, constant, &format!("The {name} property ({file_name})."));
            set_bits(&mut values, bit, |cp| set[cp as usize]);
        }
    }
    for (constant, doc, has) in mapping_bits(inputs) {
        let bit = add_bit(&mut constants, constant, doc);
        set_bits(&mut values, bit, has);
    }
    // Line_Break=SA (Complex_Context), the one Line_Break value the
    // segmenters need: the characters of the scripts written without
    // spaces whose words a dictionary has to find. It comes after the
    // mapping bits, which were assigned first.
    let line_break = inputs.file("LineBreak.txt");
    let complex = &binary(&line_break, &["SA"])[0];
    let bit = add_bit(&mut constants, "COMPLEX_CONTEXT", "Line_Break=SA, Complex_Context (LineBreak.txt).");
    set_bits(&mut values, bit, |cp| complex[cp as usize]);
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "Binary character properties, one bit each, and one bit per mapping table of case.rs and confusables.rs \
         that says whether the code point has an entry there.",
        &[
            "PropList.txt",
            "DerivedCoreProperties.txt",
            "emoji-data.txt",
            "UnicodeData.txt",
            "SpecialCasing.txt",
            "CaseFolding.txt",
            "confusables.txt",
            "LineBreak.txt",
        ],
        inputs.version,
    );
    for (i, (constant, doc)) in constants.iter().enumerate() {
        file.blank();
        file.constant(doc, constant, "u32", &format!("1 << {i}"));
    }
    emit_trie(&mut file, "the property bits", "u32", &trie);
    file.finish()
}

/// A bit of the property table: its constant name, its documentation, and
/// the test that says whether a code point has it.
type Bit<'a> = (&'static str, &'static str, Box<dyn Fn(u32) -> bool + 'a>);

/// The bits that say a code point has an entry in a mapping table.
fn mapping_bits(inputs: &Inputs) -> Vec<Bit<'_>> {
    let special_lower = case::special_lower(inputs);
    let folded = case::folded_code_points(&inputs.case);
    vec![
        (
            "HAS_LOWER",
            "The code point has a simple lowercase mapping other than itself (case.rs LOWER).",
            Box::new(|cp| inputs.data.get(cp).is_some_and(|data| data.lower.is_some())),
        ),
        (
            "HAS_UPPER",
            "The code point has a simple uppercase mapping other than itself (case.rs UPPER).",
            Box::new(|cp| inputs.data.get(cp).is_some_and(|data| data.upper.is_some())),
        ),
        (
            "HAS_SPECIAL_LOWER",
            "The code point has an entry in case.rs SPECIAL_LOWER.",
            Box::new(move |cp| special_lower.contains_key(&cp)),
        ),
        (
            "HAS_FOLD",
            "Case folding changes the code point (an entry in case.rs FOLD_SIMPLE or FOLD_FULL).",
            Box::new(move |cp| folded.binary_search(&cp).is_ok()),
        ),
        (
            "HAS_TURKIC_FOLD",
            "The code point has a Turkic case folding (case.rs FOLD_TURKIC).",
            Box::new(|cp| inputs.case.turkic.contains_key(&cp)),
        ),
        (
            "HAS_PROTOTYPE",
            "The code point has a confusable prototype (confusables.rs PROTOTYPE).",
            Box::new(|cp| inputs.confusables.prototypes.contains_key(&cp)),
        ),
    ]
}

fn add_bit(constants: &mut Vec<(String, String)>, name: &str, doc: &str) -> u32 {
    constants.push((name.to_owned(), doc.to_owned()));
    let bit = constants.len() - 1;
    assert!(bit < 32, "more than 32 property bits");
    1 << bit
}

fn set_bits(values: &mut [u32], bit: u32, has: impl Fn(u32) -> bool) {
    for (cp, value) in values.iter_mut().enumerate() {
        if has(cp as u32) {
            *value |= bit;
        }
    }
}
