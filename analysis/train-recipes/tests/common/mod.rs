//! Corpora in MASC's layout (GrAF headers, text, quarks, sentence regions
//! and Penn tokens), written for the tests. They hold no MASC text.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// A token: its text, whether a space comes before it, whether it is a
/// Penn token, and its tag, if it has one.
pub struct Token {
    pub text: String,
    pub space_before: bool,
    pub penn: bool,
    pub tag: Option<String>,
}

/// The sentences the corpora are made of. Tokens are written with a space
/// before them unless they start with `~` (which is dropped), so `dog ~'s`
/// is `dog's` in two tokens. A token that starts with `^` (after any `~`,
/// and dropped too) is in the text but not a Penn token, so a sentence of
/// such tokens has none. A token's Penn tag follows its last `/`, and a
/// token that ends in `/` has no tag (no `msd` feature); a token without
/// a `/` is tagged `.` when it is punctuation and `NN` otherwise.
pub const SENTENCES: &[&str] = &[
    "Dr./NNP Smith/NNP said/VBD that/IN the/DT dog/NN ~'s/POS food/NN costs/VBZ $/$ ~5.00/CD ~,/, not/RB more/JJR ~./.",
    "Mr./NNP Jones/NNP left/VBD early/RB ~./.",
    "It/PRP was/VBD cold/JJ in/IN the/DT U.S./NNP that/DT week/NN ~,/, and/CC it/PRP rained/VBD ~./.",
    "Who/WP called/VBD ~?/. Nobody/NN did/VBD ~!/.",
    "The/DT St./NNP Louis/NNP team/NN won/VBD 3/CD ~-/: ~1/CD ~./.",
    "Prices/NNS rose/VBD 4/CD ~%/NN in/IN May/NNP ~;/: they/PRP fell/VBD in/IN June/NNP ~./.",
    "She/PRP did/VBD n't/RB go/VB ~,/, did/VBD she/PRP ~?/.",
    "Ms./NNP Brown/NNP met/VBD Prof./NNP White/NNP at/IN 10/CD a.m./RB on/IN Monday/NNP ~./.",
];

pub fn tokens(sentence: &str) -> Vec<Token> {
    sentence
        .split(' ')
        .map(|t| {
            let (t, space_before) = match t.strip_prefix('~') {
                Some(rest) => (rest, false),
                None => (t, true),
            };
            let (t, penn) = match t.strip_prefix('^') {
                Some(rest) => (rest, false),
                None => (t, true),
            };
            let (text, tag) = match t.rsplit_once('/') {
                Some((text, "")) if !text.is_empty() => (text, None),
                Some((text, tag)) if !text.is_empty() => (text, Some(tag.to_owned())),
                _ if t.chars().all(|c| c.is_ascii_punctuation()) => (t, Some(".".to_owned())),
                _ => (t, Some("NN".to_owned())),
            };
            Token { text: text.to_owned(), space_before, penn, tag }
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
            if t.penn {
                let msd = t.tag.as_ref().map(|tag| format!("<f name=\"msd\" value=\"{tag}\"/>")).unwrap_or_default();
                writeln!(
                    penn,
                    "  <node xml:id=\"penn-n{q}\"><link targets=\"seg-r{q}\"/></node>\n  \
                     <a xml:id=\"penn-N{q}\" label=\"tok\" ref=\"penn-n{q}\" as=\"anc\"><fs>{msd}</fs></a>"
                )
                .unwrap();
            }
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

/// Documents enough for the gate: with [`SENTENCES_PER_DOCUMENT`] each,
/// eight are held out, 320 sentences, 32 sentence detector samples, more
/// than `turbo_eval::model_gate::MIN_UNITS`.
pub const GATED_DOCUMENTS: usize = 80;

/// The sentences of each document [`write_corpus`] writes.
pub const SENTENCES_PER_DOCUMENT: usize = 40;

/// A corpus of `documents` documents in two genre directories, each of
/// [`SENTENCES_PER_DOCUMENT`] sentences of `pool`, in an order that shifts
/// by document.
pub fn write_corpus(root: &Path, documents: usize, pool: &[&str]) {
    for d in 0..documents {
        let genre = if d % 2 == 0 { "written/news" } else { "spoken/talk" };
        let sentences: Vec<usize> = (0..SENTENCES_PER_DOCUMENT).map(|k| (d * 3 + k * 5) % pool.len()).collect();
        write_document(&root.join(genre), &format!("doc{d:03}"), pool, &sentences);
    }
}

/// Writes the document whose header is `header` (relative to `root`, as
/// a split lists it) again, of `sentences` of `pool`.
pub fn rewrite_document(root: &Path, header: &str, pool: &[&str], sentences: &[usize]) {
    let path = root.join(header);
    let name = path.file_stem().unwrap().to_str().unwrap().to_owned();
    write_document(path.parent().unwrap(), &name, pool, sentences);
}
