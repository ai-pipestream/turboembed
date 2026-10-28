//! Writes analysis/unicode/src/normalize/hf/tables.rs: the scalars where
//! the Unicode 9.0 tables of unicode_categories 0.1.1 and
//! unicode-normalization-alignments 0.1.12, which Hugging Face tokenizers
//! 0.23.2 normalizes with, differ from turbo-unicode. Run with
//! `cargo run --release` in this directory, then `cargo fmt --all` at the
//! root leaves the file unchanged.

use std::fmt::Write as _;

use turbo_unicode::GeneralCategory as G;
use turbo_unicode::category::general_category;
use turbo_unicode::normalize::{canonical_combining_class, canonical_decomposition, compose};
use unicode_categories::UnicodeCategories as _;
use unicode_normalization_alignments::char as una;

fn scalars() -> impl Iterator<Item = char> {
    (0..=0x10ffffu32).filter_map(char::from_u32)
}

/// Upstream's category bits, in the layout of `hf::categories`.
fn upstream_categories(c: char) -> u8 {
    (c.is_other() as u8)
        | ((c.is_mark_nonspacing() as u8) << 1)
        | ((c.is_punctuation() as u8) << 2)
        | ((c.is_mark() as u8) << 3)
        | ((c.is_number_decimal_digit() as u8) << 4)
        | ((c.is_punctuation_connector() as u8) << 5)
}

/// turbo-unicode's own category bits, before the frozen deltas.
fn current_categories(c: char) -> u8 {
    let g = general_category(c);
    (matches!(g, G::Cc | G::Cf | G::Co) as u8)
        | ((matches!(g, G::Mn) as u8) << 1)
        | ((g.is_punctuation() as u8) << 2)
        | ((g.is_mark() as u8) << 3)
        | ((matches!(g, G::Nd) as u8) << 4)
        | ((matches!(g, G::Pc) as u8) << 5)
}

/// Whether upstream decomposes `c` canonically.
fn upstream_decomposes(c: char) -> bool {
    let mut parts = Vec::new();
    una::decompose_canonical(c, |d| parts.push(d));
    parts != [c]
}

/// Every pair turbo-unicode composes, with its primary composite.
fn current_pairs() -> Vec<(char, char, char)> {
    scalars()
        .filter_map(|c| match canonical_decomposition(c)? {
            &[a, b] if compose(a, b) == Some(c) => Some((a, b, c)),
            _ => None,
        })
        .collect()
}

/// Consecutive scalars with the same wanted value where `differs`, as
/// `(first, last, wanted)` runs.
fn runs(differs: impl Fn(char) -> Option<u8>) -> Vec<(u32, u32, u8)> {
    let mut out: Vec<(u32, u32, u8)> = Vec::new();
    for c in scalars() {
        if let Some(want) = differs(c) {
            match out.last_mut() {
                Some(r) if r.1 + 1 == c as u32 && r.2 == want => r.1 = c as u32,
                _ => out.push((c as u32, c as u32, want)),
            }
        }
    }
    out
}

fn main() {
    let categories = runs(|c| (upstream_categories(c) != current_categories(c)).then(|| upstream_categories(c)));
    let ccc = runs(|c| {
        let want = una::canonical_combining_class(c);
        (want != canonical_combining_class(c)).then_some(want)
    });
    let new_nfd: Vec<u32> = scalars()
        .filter(|&c| canonical_decomposition(c).is_some() && !upstream_decomposes(c))
        .map(|c| c as u32)
        .collect();
    let mut blacklist: Vec<(u32, u32, u32)> = Vec::new();
    for (a, b, c) in current_pairs() {
        match una::compose(a, b) {
            Some(up) => assert_eq!(up, c, "upstream composes {a:?} {b:?} to another character"),
            None => blacklist.push((a as u32, b as u32, c as u32)),
        }
    }
    blacklist.sort_unstable();
    let mut out = String::from(
        "//! Frozen Hugging Face compatibility deltas over turbo-unicode, generated\n\
         //! from the Unicode 9.0 tables of unicode_categories 0.1.1 and\n\
         //! unicode-normalization-alignments 0.1.12 by scripts/hf-unicode-tables.\n\
         //! Do not edit by hand.\n",
    );
    for (name, table) in [("CATEGORIES", &categories), ("CCC", &ccc)] {
        writeln!(out, "\npub(super) const {name}: [(u32, u32, u8); {}] = [", table.len()).unwrap();
        for (a, b, v) in table {
            writeln!(out, "    (0x{a:x}, 0x{b:x}, {v}),").unwrap();
        }
        out.push_str("];\n");
    }
    writeln!(out, "\npub(super) const NEW_NFD: [u32; {}] = [", new_nfd.len()).unwrap();
    for row in new_nfd.chunks(12) {
        let row: Vec<String> = row.iter().map(|u| format!("0x{u:x},")).collect();
        writeln!(out, "    {}", row.join(" ")).unwrap();
    }
    out.push_str("];\n");
    out.push_str("\n/// Pairs turbo-unicode composes and upstream does not, with the composite.\n");
    writeln!(out, "pub(super) const COMPOSE_BLACKLIST: [(u32, u32); {}] = [", blacklist.len()).unwrap();
    let pairs: Vec<(String, u32)> = blacklist.iter().map(|&(a, b, c)| (format!("(0x{a:x}, 0x{b:x}),"), c)).collect();
    let width = pairs.iter().map(|p| p.0.len()).max().unwrap_or(0);
    for (pair, c) in pairs {
        writeln!(out, "    {pair:width$} // U+{c:04X}").unwrap();
    }
    out.push_str("];\n");
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../analysis/unicode/src/normalize/hf/tables.rs");
    std::fs::write(path, out).unwrap();
}
