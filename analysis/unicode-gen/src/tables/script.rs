//! `script.rs`: the `Script` enum and the script of every code point.

use std::collections::BTreeSet;

use super::{Inputs, emit_trie};
use crate::emit::{RustFile, str_lit};
use crate::properties::enumerated;
use crate::trie::Trie;
use crate::ucd::CODE_POINTS;

/// The value of code points `Scripts.txt` does not list.
const UNKNOWN: &str = "Unknown";

pub fn render(inputs: &Inputs) -> String {
    let file_in = inputs.file("Scripts.txt");
    let records = enumerated(&file_in);
    let mut names: BTreeSet<&str> = records.iter().map(|(_, name)| name.as_str()).collect();
    names.insert(UNKNOWN);
    let names: Vec<&str> = names.into_iter().collect();
    assert!(names.len() <= 256, "more than 256 scripts");
    let unknown = names.iter().position(|&name| name == UNKNOWN).expect("Unknown is in the list") as u8;
    let mut values = vec![unknown; CODE_POINTS];
    for (range, name) in &records {
        let number = names.binary_search(&name.as_str()).expect("every name is in the list") as u8;
        for cp in range.clone() {
            if values[cp as usize] != unknown {
                panic!("Scripts.txt: U+{cp:04X} is listed twice");
            }
            values[cp as usize] = number;
        }
    }
    let trie = Trie::build(&values);
    let mut file = RustFile::new(
        "The Script property: the Script enum, its values' names, and the script of every code point.",
        &["Scripts.txt"],
        inputs.version,
    );
    emit_enum(&mut file, &names);
    let all: Vec<String> = names.iter().map(|name| format!("Script::{}", variant(name))).collect();
    file.blank();
    file.array("Every script, in the order of their numbers in BLOCKS.", "ALL", "Script", &all);
    let quoted: Vec<String> = names.iter().map(|name| str_lit(name)).collect();
    file.blank();
    file.array("The Scripts.txt name of every script, in the order of ALL.", "NAMES", "&str", &quoted);
    emit_trie(&mut file, "the script", "u8", &trie);
    file.finish()
}

fn emit_enum(file: &mut RustFile, names: &[&str]) {
    file.blank();
    file.doc(
        "A value of the Unicode Script property. The variants are the long names of Scripts.txt without \
         underscores (Old_Italic is OldItalic); uppercased with the underscores kept they are the names of \
         java.lang.Character.UnicodeScript (OLD_ITALIC). Unknown is the script of unassigned code points.",
    );
    file.line("#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]");
    file.line("pub enum Script {");
    for name in names {
        file.line(&format!("    {},", variant(name)));
    }
    file.line("}");
}

fn variant(name: &str) -> String {
    name.replace('_', "")
}
