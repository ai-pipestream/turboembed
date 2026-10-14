//! The bundles of the corpus inputs: a corpus directory written as one
//! text, as `MascGolden.writeBundle` reads it. A line `@@ file <path>`
//! starts a file whose content runs to the next line `@@ end`, `@@ hex
//! <path>` does the same with the bytes in hexadecimal, `@@ dir <path>`
//! makes an empty directory, and the other lines outside a file are empty
//! or comments starting with `#`.

use std::path::{Path, PathBuf};

/// Writes the files of a bundle under `root`, as `MascGolden.writeBundle`
/// does, refusing what it refuses.
pub fn write_bundle(text: &str, root: &Path) {
    let mut pos = 0;
    while pos < text.len() {
        let eol = text[pos..].find('\n').map(|i| pos + i);
        let line = &text[pos..eol.unwrap_or(text.len())];
        let next = eol.map_or(text.len(), |e| e + 1);
        if let Some((hex, path)) = line
            .strip_prefix("@@ file ")
            .map(|p| (false, p))
            .or_else(|| line.strip_prefix("@@ hex ").map(|p| (true, p)))
        {
            let file = resolve(root, path);
            let (end, after) = if text[next..].starts_with("@@ end") {
                (next, next + 6)
            } else {
                let end = next + text[next..].find("\n@@ end").unwrap_or_else(|| panic!("{line} has no @@ end"));
                (end, end + 7)
            };
            assert!(after >= text.len() || text.as_bytes()[after] == b'\n', "text after @@ end");
            let content = &text[next..end];
            std::fs::create_dir_all(file.parent().unwrap()).unwrap();
            let bytes = if hex { hex_bytes(content) } else { content.as_bytes().to_vec() };
            std::fs::write(&file, bytes).unwrap();
            pos = (after + 1).min(text.len());
        } else if let Some(path) = line.strip_prefix("@@ dir ") {
            std::fs::create_dir_all(resolve(root, path)).unwrap();
            pos = next;
        } else {
            assert!(line.is_empty() || line.starts_with('#'), "a line outside a file: {line}");
            pos = next;
        }
    }
}

fn resolve(root: &Path, relative: &str) -> PathBuf {
    assert!(
        !relative.is_empty() && !relative.starts_with('/') && !relative.split('/').any(|p| p == ".." || p == "."),
        "a path outside the corpus: {relative}"
    );
    root.join(relative)
}

fn hex_bytes(content: &str) -> Vec<u8> {
    let digits: Vec<u8> = content.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    assert!(digits.len().is_multiple_of(2), "an odd number of hex digits");
    digits.chunks(2).map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap()).collect()
}
