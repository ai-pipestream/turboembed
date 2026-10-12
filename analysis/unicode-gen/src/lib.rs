//! Generates the tables of `turbo-unicode` from the Unicode Character
//! Database files.
//!
//! [`render`] reads the UCD directory and returns the source text of every
//! table file; [`generate`] writes them to a directory. The output depends
//! only on the input files: every map is iterated in sorted order and every
//! file is already in rustfmt layout, so running the generator twice gives
//! byte-identical files and a test can compare them with the committed
//! ones.

mod emit;
mod properties;
mod tables;
mod trie;
mod ucd;
mod unicode_data;

use std::fs;
use std::io;
use std::path::Path;

/// One generated file: its name in the output directory and its contents.
pub struct GeneratedFile {
    /// The file name, for example `category.rs`.
    pub name: &'static str,
    /// The Rust source text.
    pub contents: String,
}

/// Reads the UCD files in `ucd_dir` and returns the generated table files.
/// Panics with the file name and line number on malformed input.
pub fn render(ucd_dir: &Path) -> Vec<GeneratedFile> {
    tables::render_all(ucd_dir)
}

/// Writes the table files generated from `ucd_dir` into `out_dir`, which is
/// created when missing. Returns the files written.
pub fn generate(ucd_dir: &Path, out_dir: &Path) -> io::Result<Vec<GeneratedFile>> {
    let files = render(ucd_dir);
    fs::create_dir_all(out_dir)?;
    for file in &files {
        fs::write(out_dir.join(file.name), &file.contents)?;
    }
    Ok(files)
}
