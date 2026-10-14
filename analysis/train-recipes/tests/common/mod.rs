//! Corpora in MASC's layout (GrAF headers, text, quarks, sentence regions
//! and Penn tokens), written for the tests. They hold no MASC text.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// A token: its text, whether a space comes before it, and its tag.
pub struct Token {
    pub text: String,
    pub space_before: bool,
    pub tag: &'static str,
}

/// The sentences the corpora are made of. Tokens are written with a space
/// before them unless they start with `~` (which is dropped), so `dog ~'s`
/// is `dog's` in two tokens.
pub const SENTENCES: &[&str] = &[
    "Dr. Smith said that the dog ~'s food costs $ ~5.00 ~, not more ~.",
    "Mr. Jones left early ~.",
    "It was cold in the U.S. that week ~, and it rained ~.",
    "Who called ~? Nobody did ~!",
    "The St. Louis team won 3 ~- ~1 ~.",
    "Prices rose 4 ~% in May ~; they fell in June ~.",
    "She did n't go ~, did she ~?",
    "Ms. Brown met Prof. White at 10 a.m. on Monday ~.",
];

pub fn tokens(sentence: &str) -> Vec<Token> {
    sentence
        .split(' ')
        .map(|t| {
            let (text, space_before) = match t.strip_prefix('~') {
                Some(rest) => (rest, false),
                None => (t, true),
            };
            let tag = if text.chars().all(|c| c.is_ascii_punctuation()) { "." } else { "NN" };
            Token { text: text.to_owned(), space_before, tag }
        })
        .collect()
}

/// Writes one document of `sentences` (indices into `pool`) under `dir`
/// as `<name>.hdr`, `<name>.txt`, `<name>-seg.xml`, `<name>-s.xml` and
/// `<name>-penn.xml`.
pub fn write_document(dir: &Path, name: &str, pool: &[&str], sentences: &[usize]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut text = String::new();
    let mut seg = String::from("<graph xmlns=\"http://www.xces.org/ns/GrAF/1.0/\">\n");
    let mut s = seg.clone();
    let mut penn = seg.clone();
    let mut q = 0;
    for (n, &i) in sentences.iter().enumerate() {
        if n > 0 {
            text.push_str("  ");
        }
        let start = text.len();
        for (k, t) in tokens(pool[i]).iter().enumerate() {
            if k > 0 && t.space_before {
                text.push(' ');
            }
            let a = text.len();
            text.push_str(&t.text);
            writeln!(seg, "  <region xml:id=\"seg-r{q}\" anchors=\"{a} {}\"/>", text.len()).unwrap();
            writeln!(
                penn,
                "  <node xml:id=\"penn-n{q}\"><link targets=\"seg-r{q}\"/></node>\n  \
                 <a xml:id=\"penn-N{q}\" label=\"tok\" ref=\"penn-n{q}\" as=\"anc\"><fs><f name=\"msd\" value=\"{}\"/></fs></a>",
                t.tag
            )
            .unwrap();
            q += 1;
        }
        writeln!(
            s,
            "  <region xml:id=\"s-r{n}\" anchors=\"{start} {}\"/>\n  <node xml:id=\"s-n{n}\"><link targets=\"s-r{n}\"/></node>",
            text.len()
        )
        .unwrap();
    }
    text.push('\n');
    for g in [&mut seg, &mut s, &mut penn] {
        g.push_str("</graph>\n");
    }
    let header = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <documentHeader xmlns=\"http://www.xces.org/ns/GrAF/1.0/\" docId=\"TEST\" version=\"1.0.4\">\n  <profileDesc>\n    \
         <primaryData f.id=\"f.text\" loc=\"{name}.txt\"/>\n    <annotations>\n      \
         <annotation loc=\"{name}-s.xml\" f.id=\"f.s\">annotation</annotation>\n      \
         <annotation loc=\"{name}-seg.xml\" f.id=\"f.seg\">annotation</annotation>\n      \
         <annotation loc=\"{name}-penn.xml\" f.id=\"f.penn\">annotation</annotation>\n    </annotations>\n  \
         </profileDesc>\n</documentHeader>\n"
    );
    std::fs::write(dir.join(format!("{name}.hdr")), header).unwrap();
    std::fs::write(dir.join(format!("{name}.txt")), text).unwrap();
    std::fs::write(dir.join(format!("{name}-seg.xml")), seg).unwrap();
    std::fs::write(dir.join(format!("{name}-s.xml")), s).unwrap();
    std::fs::write(dir.join(format!("{name}-penn.xml")), penn).unwrap();
}

/// A fresh directory under the system's temporary directory.
pub fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-train-recipes-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A corpus of `documents` documents in two genre directories, each of
/// twelve sentences of `pool`, in an order that shifts by document.
pub fn write_corpus(root: &Path, documents: usize, pool: &[&str]) {
    for d in 0..documents {
        let genre = if d % 2 == 0 { "written/news" } else { "spoken/talk" };
        let sentences: Vec<usize> = (0..12).map(|k| (d * 3 + k * 5) % pool.len()).collect();
        write_document(&root.join(genre), &format!("doc{d:02}"), pool, &sentences);
    }
}
