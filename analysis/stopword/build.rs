//! Turns the bundled lists in `data/` into sorted slices of words, read the
//! way OpenNLP's `DictionaryStopwordFilter` reads them: each line trimmed
//! of Unicode white space, blank lines and lines starting with `#`
//! skipped, the rest split on Unicode white space into one entry. Rust's
//! `char::is_whitespace` is the Unicode `White_Space` property, which is
//! what OpenNLP's `StringUtil.trimUnicodeWhitespace` and
//! `splitOnUnicodeWhitespace` test.

use std::fmt::Write as _;
use std::path::Path;
use std::{env, fs};

/// The lists in `StopwordLanguage` order.
const CODES: [&str; 11] = ["bg", "da", "de", "en", "es", "fi", "fr", "it", "nl", "pt", "ru"];

fn main() {
    let dir = Path::new(&env::var("CARGO_MANIFEST_DIR").unwrap()).join("data");
    println!("cargo::rerun-if-changed=data");
    let mut out = String::new();
    let mut max_bytes = 0;
    for code in CODES {
        let path = dir.join(format!("{code}.txt"));
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut words: Vec<&str> = Vec::new();
        // BufferedReader.readLine ends a line at \n, \r or \r\n.
        for line in text.split(['\n', '\r']) {
            let line = line.trim_matches(char::is_whitespace);
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // Several words on a line make one multi-word entry in OpenNLP.
            // No bundled list has one; the lookup here handles single words
            // only, so a list that gains one must be handled first.
            assert!(!line.contains(char::is_whitespace), "{code}.txt: multi-word entry {line:?}");
            words.push(line);
        }
        words.sort_unstable();
        words.dedup();
        max_bytes = words.iter().map(|w| w.len()).max().unwrap_or(0).max(max_bytes);
        writeln!(out, "pub(crate) static {}: [&str; {}] = {words:?};", code.to_uppercase(), words.len()).unwrap();
    }
    writeln!(out, "/// The longest word of any list, in UTF-8 bytes.").unwrap();
    writeln!(out, "pub(crate) const MAX_BYTES: usize = {max_bytes};").unwrap();
    fs::write(Path::new(&env::var("OUT_DIR").unwrap()).join("lists.rs"), out).unwrap();
}
