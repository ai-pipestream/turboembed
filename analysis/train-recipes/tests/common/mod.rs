//! Corpora in MASC's layout (GrAF headers, text, quarks, sentence regions,
//! Penn tokens, noun and verb chunks, and named entities), written for
//! the tests. They hold no MASC text.

#![allow(dead_code)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// A token: its text, whether a space comes before it, whether it is a
/// Penn token, its tag, if it has one, and its chunk tag, if it is in a
/// chunk.
pub struct Token {
    pub text: String,
    pub space_before: bool,
    pub penn: bool,
    pub tag: Option<String>,
    pub chunk: Option<String>,
}

/// The sentences the corpora are made of. Tokens are written with a space
/// before them unless they start with `~` (which is dropped), so `dog ~'s`
/// is `dog's` in two tokens. A token that starts with `^` (after any `~`,
/// and dropped too) is in the text but not a Penn token, so a sentence of
/// such tokens has none. A token's chunk tag follows a `|` at its end
/// (`B-NP`, `I-VP`), and a token without one is outside every chunk; a
/// document with a chunk tag has noun and verb chunk files. A token's
/// Penn tag follows its last `/`, and a token that ends in `/` has no tag
/// (no `msd` feature); a token without a `/` is tagged `.` when it is
/// punctuation and `NN` otherwise.
///
/// After the tokens, each ` | ` starts a named entity, as MASC's named
/// entity file links one to Penn tokens: its label, then the places of
/// its tokens among the sentence's tokens, in the order the file's edges
/// list them (these sentences list them first to last; MASC lists most
/// entities of several tokens last to first). The labels are MASC's
/// (`person`, `location`, `org`, `date`), and like MASC the sentences
/// label pronouns that refer to a person `person`. An entity may overlap
/// another, as `St. Louis` (a location) is inside `St. Louis team` (an
/// organization).
pub const SENTENCES: &[&str] = &[
    "Dr./NNP|B-NP Smith/NNP|I-NP said/VBD|B-VP that/IN the/DT|B-NP dog/NN|I-NP ~'s/POS|B-NP food/NN|I-NP \
     costs/VBZ|B-VP $/$|B-NP ~5.00/CD|I-NP ~,/, not/RB more/JJR ~./. | person 0 1",
    "Mr./NNP|B-NP Jones/NNP|I-NP left/VBD|B-VP early/RB ~./. | person 1",
    "It/PRP|B-NP was/VBD|B-VP cold/JJ in/IN the/DT|B-NP U.S./NNP|I-NP that/DT|B-NP week/NN|I-NP ~,/, and/CC \
     it/PRP|B-NP rained/VBD|B-VP ~./. | location 5 | date 6 7",
    "Who/WP|B-NP called/VBD|B-VP ~?/. Nobody/NN|B-NP did/VBD|B-VP ~!/.",
    "The/DT|B-NP St./NNP|I-NP Louis/NNP|I-NP team/NN|I-NP won/VBD|B-VP 3/CD|B-NP ~-/: ~1/CD|B-NP ~./. | org 1 2 3 \
     | location 1 2",
    "Prices/NNS|B-NP rose/VBD|B-VP 4/CD|B-NP ~%/NN|I-NP in/IN May/NNP|B-NP ~;/: they/PRP|B-NP fell/VBD|B-VP in/IN \
     June/NNP|B-NP ~./. | date 5 | date 10",
    "She/PRP|B-NP did/VBD|B-VP n't/RB|I-VP go/VB|I-VP ~,/, did/VBD|B-VP she/PRP|B-NP ~?/. | person 0 | person 6",
    "Ms./NNP|B-NP Brown/NNP|I-NP met/VBD|B-VP Prof./NNP|B-NP White/NNP|I-NP at/IN 10/CD|B-NP a.m./RB|I-NP on/IN \
     Monday/NNP|B-NP ~./. | person 1 | person 4 | date 6 7 | date 9",
];

/// A named entity of a sentence: its label and the places of its tokens
/// among the sentence's tokens.
pub struct Entity {
    pub label: String,
    pub tokens: Vec<usize>,
}

/// The tokens of `sentence`, and what follows them: its entities, each
/// after ` | `.
fn parts(sentence: &str) -> (&str, Vec<&str>) {
    let mut parts = sentence.split(" | ");
    let tokens = parts.next().unwrap_or_default();
    (tokens, parts.collect())
}

/// The named entities of `sentence`.
pub fn entities(sentence: &str) -> Vec<Entity> {
    parts(sentence)
        .1
        .into_iter()
        .map(|e| {
            let mut words = e.split(' ');
            let label = words.next().unwrap().to_owned();
            Entity { label, tokens: words.map(|w| w.parse().unwrap()).collect() }
        })
        .collect()
}

/// Writes `entities` after `tokens` as [`SENTENCES`] does.
fn with_entities(tokens: String, entities: &[Entity]) -> String {
    let mut out = tokens;
    for e in entities {
        write!(out, " | {}", e.label).unwrap();
        for t in &e.tokens {
            write!(out, " {t}").unwrap();
        }
    }
    out
}

pub fn tokens(sentence: &str) -> Vec<Token> {
    parts(sentence)
        .0
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
            let (t, chunk) = match t.rsplit_once('|') {
                Some((t, chunk)) => (t, Some(chunk.to_owned())),
                None => (t, None),
            };
            let (text, tag) = match t.rsplit_once('/') {
                Some((text, "")) if !text.is_empty() => (text, None),
                Some((text, tag)) if !text.is_empty() => (text, Some(tag.to_owned())),
                _ if t.chars().all(|c| c.is_ascii_punctuation()) => (t, Some(".".to_owned())),
                _ => (t, Some("NN".to_owned())),
            };
            Token { text: text.to_owned(), space_before, penn, tag, chunk }
        })
        .collect()
}

/// A token written back in the pool's notation, with `tag` and `chunk`.
fn written(t: &Token, tag: Option<&str>, chunk: Option<&str>) -> String {
    let space = if t.space_before { "" } else { "~" };
    let penn = if t.penn { "" } else { "^" };
    let chunk = chunk.map(|c| format!("|{c}")).unwrap_or_default();
    format!("{space}{penn}{}/{}{chunk}", t.text, tag.unwrap_or_default())
}

/// `pool` with each sentence's tags moved one token along, the last
/// token's to the first: a tagger trained on it can give every tag of
/// `pool` but has learned the wrong one for nearly every token. The chunk
/// tags stay with their tokens.
pub fn with_tags_rotated(pool: &[&str]) -> Vec<String> {
    pool.iter()
        .map(|sentence| {
            let tokens = tokens(sentence);
            let n = tokens.len();
            let written: Vec<String> = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| written(t, tokens[(i + n - 1) % n].tag.as_deref(), t.chunk.as_deref()))
                .collect();
            with_entities(written.join(" "), &entities(sentence))
        })
        .collect()
}

/// `pool` with each named entity's tokens moved one token along, the last
/// token's to the first: a name finder trained on it has learned to find
/// names one token after where they are.
pub fn with_names_shifted(pool: &[&str]) -> Vec<String> {
    pool.iter()
        .map(|sentence| {
            let n = tokens(sentence).len();
            let shifted: Vec<Entity> = entities(sentence)
                .into_iter()
                .map(|e| Entity { label: e.label, tokens: e.tokens.iter().map(|t| (t + 1) % n).collect() })
                .collect();
            with_entities(parts(sentence).0.to_owned(), &shifted)
        })
        .collect()
}

/// `pool` with each sentence's chunk tags moved one token along, the last
/// token's to the first: a chunker trained on it has learned chunks that
/// start and end one token late. The tags stay with their tokens.
pub fn with_chunks_rotated(pool: &[&str]) -> Vec<String> {
    pool.iter()
        .map(|sentence| {
            let tokens = tokens(sentence);
            let n = tokens.len();
            let written: Vec<String> = tokens
                .iter()
                .enumerate()
                .map(|(i, t)| written(t, t.tag.as_deref(), tokens[(i + n - 1) % n].chunk.as_deref()))
                .collect();
            with_entities(written.join(" "), &entities(sentence))
        })
        .collect()
}

/// The chunks of a sentence's tokens, each its kind's file prefix (`nc`
/// or `vc`) and its tokens' Penn numbers, read from the chunk tags as
/// `ChunkSample` reads phrases: `B-` starts a chunk, `I-` of the open
/// chunk's type goes on with it, any other `I-` starts one, and a token
/// without a chunk tag ends it.
fn chunks_of(tokens: &[(usize, &Token)]) -> Vec<(&'static str, Vec<usize>)> {
    let mut chunks: Vec<(&'static str, Vec<usize>)> = Vec::new();
    let mut open: Option<&str> = None;
    for &(q, t) in tokens {
        let Some(tag) = t.chunk.as_deref() else {
            open = None;
            continue;
        };
        assert!(t.penn, "a chunk tag on a token that is not a Penn token: {}", t.text);
        let (prefix, kind) = tag.split_at(2);
        let file = match kind {
            "NP" => "nc",
            "VP" => "vc",
            other => panic!("MASC has noun and verb chunks, not {other}"),
        };
        if prefix == "I-" && open == Some(kind) {
            chunks.last_mut().expect("an open chunk").1.push(q);
        } else {
            chunks.push((file, vec![q]));
            open = Some(kind);
        }
    }
    chunks
}

/// A chunk file of the chunks of `file` (`nc` or `vc`), each a node with
/// an `a` element and an edge to each of its Penn tokens, the edges last
/// to first as MASC writes them.
fn chunk_file(file: &str, chunks: &[(&'static str, Vec<usize>)]) -> String {
    let label = if file == "nc" { "nchunk" } else { "vchunk" };
    let mut xml = String::from("<graph xmlns=\"http://www.xces.org/ns/GrAF/1.0/\">\n");
    let mine: Vec<&Vec<usize>> = chunks.iter().filter(|(f, _)| *f == file).map(|(_, t)| t).collect();
    for n in 0..mine.len() {
        writeln!(xml, "  <node xml:id=\"{file}-n{n}\"/>\n  <a label=\"{label}\" ref=\"{file}-n{n}\" as=\"anc\"/>")
            .unwrap();
    }
    for (n, tokens) in mine.iter().enumerate().rev() {
        for q in tokens.iter().rev() {
            writeln!(xml, "  <edge from=\"{file}-n{n}\" to=\"penn-n{q}\"/>").unwrap();
        }
    }
    xml.push_str("</graph>\n");
    xml
}

/// Writes one document of `sentences` (indices into `pool`) under `dir`
/// as `<name>.hdr`, `<name>.txt`, `<name>-seg.xml`, `<name>-s.xml`,
/// `<name>-penn.xml` and `<name>-ne.xml`, the last with the sentences'
/// named entities, if any, and `<name>-nc.xml` and `<name>-vc.xml` when a
/// token has a chunk tag.
pub fn write_document(dir: &Path, name: &str, pool: &[&str], sentences: &[usize]) {
    std::fs::create_dir_all(dir).unwrap();
    let mut text = String::new();
    let mut seg = String::from("<graph xmlns=\"http://www.xces.org/ns/GrAF/1.0/\">\n");
    let mut s = seg.clone();
    let mut penn = seg.clone();
    let mut ne = seg.clone();
    let mut edges = String::new();
    let mut q = 0;
    let mut e = 0;
    let mut chunks = Vec::new();
    for (n, &i) in sentences.iter().enumerate() {
        if n > 0 {
            text.push_str("  ");
        }
        let start = text.len();
        let first_quark = q;
        for entity in entities(pool[i]) {
            writeln!(
                ne,
                "  <node xml:id=\"ne-n{e}\"/>\n  <a xml:id=\"ne-N{e}\" label=\"{}\" ref=\"ne-n{e}\" as=\"anc\"><fs/></a>",
                entity.label
            )
            .unwrap();
            for t in entity.tokens {
                writeln!(edges, "  <edge xml:id=\"ne-e{e}-{t}\" from=\"ne-n{e}\" to=\"penn-n{}\"/>", first_quark + t)
                    .unwrap();
            }
            e += 1;
        }
        let sentence = tokens(pool[i]);
        let numbered: Vec<(usize, &Token)> = sentence.iter().enumerate().map(|(k, t)| (q + k, t)).collect();
        chunks.extend(chunks_of(&numbered));
        for (k, t) in sentence.iter().enumerate() {
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
    ne.push_str(&edges);
    for g in [&mut seg, &mut s, &mut penn, &mut ne] {
        g.push_str("</graph>\n");
    }
    let chunked = sentences.iter().any(|&i| tokens(pool[i]).iter().any(|t| t.chunk.is_some()));
    let chunk_annotations = if chunked {
        format!(
            "      <annotation loc=\"{name}-nc.xml\" f.id=\"f.nc\">Noun chunks</annotation>\n      \
             <annotation loc=\"{name}-vc.xml\" f.id=\"f.vc\">Verb chunks</annotation>\n"
        )
    } else {
        String::new()
    };
    let header = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <documentHeader xmlns=\"http://www.xces.org/ns/GrAF/1.0/\" docId=\"TEST\" version=\"1.0.4\">\n  <profileDesc>\n    \
         <primaryData f.id=\"f.text\" loc=\"{name}.txt\"/>\n    <annotations>\n      \
         <annotation loc=\"{name}-s.xml\" f.id=\"f.s\">annotation</annotation>\n      \
         <annotation loc=\"{name}-seg.xml\" f.id=\"f.seg\">annotation</annotation>\n      \
         <annotation loc=\"{name}-penn.xml\" f.id=\"f.penn\">annotation</annotation>\n      \
         <annotation loc=\"{name}-ne.xml\" f.id=\"f.ne\">annotation</annotation>\n{chunk_annotations}    \
         </annotations>\n  </profileDesc>\n</documentHeader>\n"
    );
    std::fs::write(dir.join(format!("{name}.hdr")), header).unwrap();
    std::fs::write(dir.join(format!("{name}.txt")), text).unwrap();
    std::fs::write(dir.join(format!("{name}-seg.xml")), seg).unwrap();
    std::fs::write(dir.join(format!("{name}-s.xml")), s).unwrap();
    std::fs::write(dir.join(format!("{name}-penn.xml")), penn).unwrap();
    std::fs::write(dir.join(format!("{name}-ne.xml")), ne).unwrap();
    for file in ["nc", "vc"] {
        let path = dir.join(format!("{name}-{file}.xml"));
        if chunked {
            std::fs::write(path, chunk_file(file, &chunks)).unwrap();
        } else if path.exists() {
            std::fs::remove_file(path).unwrap();
        }
    }
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

/// Makes the Penn token of quark `quark` of the document whose header is
/// `header` (relative to `root`) cover that quark and the two after it,
/// whose own tokens are left as they are, so they lie inside it: a
/// sentence whose tokens are out of text order, as MASC 3.0.0's Penn layer
/// has once (`non-deterministic` over three quarks and `-` over the
/// second).
pub fn nest_token(root: &Path, header: &str, quark: usize) {
    let header = root.join(header);
    let name = header.file_stem().unwrap().to_str().unwrap();
    let penn = header.with_file_name(format!("{name}-penn.xml"));
    let xml = std::fs::read_to_string(&penn).unwrap();
    let one = format!("<link targets=\"seg-r{quark}\"/>");
    let three = format!("<link targets=\"seg-r{quark} seg-r{} seg-r{}\"/>", quark + 1, quark + 2);
    assert_eq!(xml.matches(&one).count(), 1, "{}", penn.display());
    std::fs::write(&penn, xml.replace(&one, &three)).unwrap();
}

/// Gives the header `header` (relative to `root`) a `textClass` naming the
/// categories `cat_ref` of MASC's taxonomy, as MASC's headers do (`WR TW`
/// for a document of the Twitter genre).
pub fn set_text_class(root: &Path, header: &str, cat_ref: &str) {
    let path = root.join(header);
    let xml = std::fs::read_to_string(&path).unwrap();
    let text_class = format!("<profileDesc>\n    <textClass catRef=\"{cat_ref}\"><domain>Test</domain></textClass>");
    assert_eq!(xml.matches("<profileDesc>").count(), 1, "{}", path.display());
    std::fs::write(&path, xml.replace("<profileDesc>", &text_class)).unwrap();
}

/// Adds a chunk over the Penn tokens `penn` (their quark numbers) to the
/// `file` chunk file (`nc` or `vc`) of the document whose header is
/// `header` (relative to `root`), numbered 100,000 plus `id` so it does
/// not clash with the chunks written.
pub fn add_chunk(root: &Path, header: &str, file: &str, id: usize, penn: &[usize]) {
    let header = root.join(header);
    let name = header.file_stem().unwrap().to_str().unwrap();
    let path = header.with_file_name(format!("{name}-{file}.xml"));
    let xml = std::fs::read_to_string(&path).unwrap();
    let label = if file == "nc" { "nchunk" } else { "vchunk" };
    let n = 100_000 + id;
    let mut chunk =
        format!("  <node xml:id=\"{file}-n{n}\"/>\n  <a label=\"{label}\" ref=\"{file}-n{n}\" as=\"anc\"/>\n");
    for q in penn {
        writeln!(chunk, "  <edge from=\"{file}-n{n}\" to=\"penn-n{q}\"/>").unwrap();
    }
    assert_eq!(xml.matches("</graph>").count(), 1, "{}", path.display());
    std::fs::write(&path, xml.replace("</graph>", &format!("{chunk}</graph>"))).unwrap();
}
