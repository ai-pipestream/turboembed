//! The generated table files, one module per file.

mod case;
mod category;
mod confusables;
mod emoji;
mod grapheme_break;
mod normalize;
mod props;
mod script;
mod word_break;

use std::fmt::Display;
use std::path::{Path, PathBuf};

use crate::GeneratedFile;
use crate::emit::{RustFile, dec};
use crate::trie::{INDEX_LEN, Trie};
use crate::ucd::UcdFile;
use crate::unicode_data::UnicodeData;

/// What several table files are generated from, read once.
pub struct Inputs {
    /// The UCD directory.
    pub dir: PathBuf,
    /// The Unicode version of the files.
    pub version: (u8, u8, u8),
    /// `UnicodeData.txt`.
    pub data: UnicodeData,
    /// `CaseFolding.txt` and `SpecialCasing.txt`.
    pub case: case::CaseData,
    /// `confusables.txt`.
    pub confusables: confusables::Confusables,
}

impl Inputs {
    /// Reads `name` from the UCD directory.
    pub fn file(&self, name: &str) -> UcdFile {
        UcdFile::read(&self.dir, name)
    }
}

pub fn render_all(dir: &Path) -> Vec<GeneratedFile> {
    let version = UcdFile::read(dir, "DerivedCoreProperties.txt").header_version();
    for name in [
        "CaseFolding.txt",
        "SpecialCasing.txt",
        "Scripts.txt",
        "PropList.txt",
        "DerivedNormalizationProps.txt",
        "WordBreakProperty.txt",
        "GraphemeBreakProperty.txt",
        "LineBreak.txt",
    ] {
        let file = UcdFile::read(dir, name);
        if file.header_version() != version {
            panic!("{name} is not from Unicode {version:?}");
        }
    }
    let inputs = Inputs {
        dir: dir.to_owned(),
        version,
        data: UnicodeData::read(dir),
        case: case::CaseData::read(dir),
        confusables: confusables::Confusables::read(dir),
    };
    vec![
        GeneratedFile { name: "mod.rs", contents: render_mod(version) },
        GeneratedFile { name: "category.rs", contents: category::render(&inputs) },
        GeneratedFile { name: "props.rs", contents: props::render(&inputs) },
        GeneratedFile { name: "script.rs", contents: script::render(&inputs) },
        GeneratedFile { name: "case.rs", contents: case::render(&inputs) },
        GeneratedFile { name: "normalize.rs", contents: normalize::render(&inputs) },
        GeneratedFile { name: "confusables.rs", contents: confusables::render(&inputs) },
        GeneratedFile { name: "emoji.rs", contents: emoji::render(&inputs) },
        GeneratedFile { name: "word_break.rs", contents: word_break::render(&inputs) },
        GeneratedFile { name: "grapheme_break.rs", contents: grapheme_break::render(&inputs) },
    ]
}

fn render_mod(version: (u8, u8, u8)) -> String {
    let mut file = RustFile::new(
        "The tables of turbo-unicode. Per-code-point values are two-stage tables (an INDEX of block numbers and \
         the BLOCKS they number, read by crate::trie); mappings are sorted key arrays searched by binary search \
         beside arrays of values.",
        &["the UCD files"],
        version,
    );
    file.blank();
    for module in
        ["case", "category", "confusables", "emoji", "grapheme_break", "normalize", "props", "script", "word_break"]
    {
        file.line(&format!("pub(crate) mod {module};"));
    }
    file.blank();
    let (major, minor, update) = version;
    file.constant(
        "The Unicode version of the files the tables were generated from.",
        "UNICODE_VERSION",
        "(u8, u8, u8)",
        &format!("({major}, {minor}, {update})"),
    );
    file.finish()
}

/// Appends the `INDEX` and `BLOCKS` arrays of a two-stage table.
fn emit_trie<T: Copy + Ord + Display>(file: &mut RustFile, what: &str, ty: &str, trie: &Trie<T>) {
    assert_eq!(trie.index.len(), INDEX_LEN);
    let index: Vec<String> = trie.index.iter().map(|&block| dec(block)).collect();
    file.blank();
    file.array(
        &format!("The stored block of each 256-code-point block of the code space, for {what}."),
        "INDEX",
        "u16",
        &index,
    );
    let blocks: Vec<String> = trie.blocks.iter().map(|&value| dec(value)).collect();
    file.blank();
    file.array(
        &format!("The {} distinct 256-entry blocks of {what}, one after another.", trie.block_count()),
        "BLOCKS",
        ty,
        &blocks,
    );
}

/// Appends a sorted key array and its value array: `{prefix}_KEYS` (code
/// points in ascending order, searched by binary search) and
/// `{prefix}_VALUES` (the value of each key at the same position).
fn emit_map(file: &mut RustFile, prefix: &str, docs: [&str; 2], value_ty: &str, entries: &[(u32, String)]) {
    assert!(entries.windows(2).all(|pair| pair[0].0 < pair[1].0), "{prefix} keys are not sorted and unique");
    let keys: Vec<String> = entries.iter().map(|(cp, _)| crate::emit::hex(*cp)).collect();
    let values: Vec<String> = entries.iter().map(|(_, value)| value.clone()).collect();
    file.blank();
    file.array(docs[0], &format!("{prefix}_KEYS"), "u32", &keys);
    file.blank();
    file.array(docs[1], &format!("{prefix}_VALUES"), value_ty, &values);
}
