//! The MASC 3.0.0 held-out split as tokenizer gold (#1645): the same
//! documents the trainer recipes hold out (every tenth document in
//! corpus order, the Twitter genre left out), read with
//! `turbo-formats`' MASC reader. The recipes own the split and the
//! release digest in `turbo-train-recipes::masc`, which this crate
//! cannot depend on (the recipes depend on the eval crate), so the
//! split rule and the digest are restated here and the digest is
//! refused on mismatch: a corpus whose bytes differ from the recorded
//! release scores nothing.

use std::fmt::Write as _;
use std::path::Path;

use turbo_formats::masc::MascCorpus;

use super::tokenize::TokenDocument;
use crate::Result;
use crate::sha::{sha256, to_hex};

/// The recorded MASC 3.0.0 corpus digest (`turbo-train-recipes`'
/// `masc::RELEASE`, over `corpus_digest`'s manifest).
pub const MASC_CORPUS_SHA256: &str = "445449f535cac62e74b2712ad975d7b727e72b8276720dd27492266565ae0afd";

/// The recipes' split: documents at 9 modulo 10 in corpus order are
/// held out.
const EVERY: usize = 10;
const HELD_OUT_AT: usize = 9;

/// A held-out MASC gold side: its documents and how it was made.
pub struct MascGold {
    pub documents: Vec<TokenDocument>,
    /// The corpus digest the directory matched.
    pub sha256: String,
    /// Gold sentences with Penn tokens, over the held-out documents.
    pub sentences: usize,
}

/// The corpus digest of `dir`, as the recipes compute it: the SHA-256
/// of a `sha256sum`-style manifest of every regular file under it,
/// paths relative to `dir`, `/`-separated, sorted by path bytes. A
/// symbolic link, a non-regular entry, or a non-UTF-8 name is refused.
pub fn corpus_digest(dir: &Path) -> Result<String> {
    fn collect(dir: &Path, prefix: &str, out: &mut Vec<String>) -> Result<()> {
        for entry in std::fs::read_dir(dir).map_err(|e| format!("MASC: {}: {e}", dir.display()))? {
            let entry = entry.map_err(|e| format!("MASC: {}: {e}", dir.display()))?;
            let path = entry.path();
            let kind = entry.file_type().map_err(|e| format!("MASC: {}: {e}", path.display()))?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|n| format!("MASC: {}: the name {n:?} is not UTF-8", path.display()))?;
            let rel = format!("{prefix}{name}");
            if kind.is_symlink() {
                return Err(format!("MASC: {}: a symbolic link in the corpus directory", path.display()));
            } else if kind.is_dir() {
                collect(&path, &format!("{rel}/"), out)?;
            } else if kind.is_file() {
                out.push(rel);
            } else {
                return Err(format!("MASC: {}: not a regular file or directory", path.display()));
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    collect(dir, "", &mut files)?;
    files.sort();
    let mut manifest = String::new();
    for rel in &files {
        let path = dir.join(rel);
        let bytes = std::fs::read(&path).map_err(|e| format!("MASC: {}: {e}", path.display()))?;
        writeln!(manifest, "{}  {rel}", to_hex(&sha256(&bytes))).expect("writing to a String");
    }
    Ok(to_hex(&sha256(manifest.as_bytes())))
}

/// Whether a document header's `textClass` categories name the
/// Twitter genre (`TW`), the recipes' one exclusion.
fn header_excluded(header: &str) -> bool {
    let mut rest = header;
    while let Some(at) = rest.find("catRef") {
        rest = &rest[at + "catRef".len()..];
        let Some(eq) = rest.find('=') else { break };
        let after = rest[eq + 1..].trim_start();
        let Some(quote) = after.chars().next().filter(|c| *c == '"' || *c == '\'') else { continue };
        let Some(end) = after[1..].find(quote) else { break };
        if after[1..1 + end].split_whitespace().any(|code| code == "TW") {
            return true;
        }
        rest = &after[1 + end..];
    }
    false
}

/// Reads the held-out documents of the MASC release at `dir` (the
/// extracted `MASC-3.0.0` directory) as tokenizer gold: each
/// document's text is its gold sentences' raw text joined with one
/// space, as the CoNLL-U side joins `# text` lines, and its gold
/// words are the Penn tokens' text, by sentence.
///
/// # Errors
///
/// A directory whose digest is not the recorded release's, a corpus
/// with a skipped document (which would move every later document's
/// split place), an unreadable header, or an annotation offset the
/// reader refuses.
pub fn masc_test_documents(dir: &Path) -> Result<MascGold> {
    let digest = corpus_digest(dir)?;
    if digest != MASC_CORPUS_SHA256 {
        return Err(format!(
            "MASC: {} has corpus digest {digest}; the recorded MASC 3.0.0 release records {MASC_CORPUS_SHA256}",
            dir.display()
        ));
    }
    let corpus = MascCorpus::read(dir).map_err(|e| format!("MASC: {e}"))?;
    if !corpus.skipped().is_empty() {
        return Err(format!(
            "MASC: {} documents were skipped on read, so split places are not the recipes'",
            corpus.skipped().len()
        ));
    }
    let mut excluded = Vec::with_capacity(corpus.documents().len());
    for document in corpus.documents() {
        let header = std::fs::read(document.path()).map_err(|e| format!("MASC: {}: {e}", document.path().display()))?;
        excluded.push(header_excluded(&String::from_utf8_lossy(&header)));
    }
    let (documents, sentences) = masc_documents(&corpus, &excluded)?;
    Ok(MascGold { documents, sha256: digest, sentences })
}

/// The held-out documents of `corpus` as tokenizer gold, with their
/// gold sentence count: places at 9 modulo 10 in corpus order that
/// `excluded` does not mark, places counted over every document,
/// excluded or not, as the recipes count them.
pub fn masc_documents(corpus: &MascCorpus, excluded: &[bool]) -> Result<(Vec<TokenDocument>, usize)> {
    let mut documents = Vec::new();
    let mut sentences = 0usize;
    for (place, document) in corpus.documents().iter().enumerate() {
        if place % EVERY != HELD_OUT_AT || excluded.get(place).copied().unwrap_or(false) {
            continue;
        }
        let mut texts: Vec<String> = Vec::new();
        let mut gold: Vec<Vec<String>> = Vec::new();
        for sentence in document.sentences() {
            let Some(tokens) = sentence.tokens() else { continue };
            if tokens.is_empty() {
                continue;
            }
            let region =
                document.span(i64::from(sentence.start), i64::from(sentence.end)).map_err(|e| format!("MASC: {e}"))?;
            texts.push(document.text()[region.range()].to_owned());
            let mut forms = Vec::with_capacity(tokens.len());
            for span in document.token_spans(sentence).map_err(|e| format!("MASC: {e}"))? {
                forms.push(document.text()[span.range()].to_owned());
            }
            gold.push(forms);
            sentences += 1;
        }
        if !gold.is_empty() {
            documents.push(TokenDocument { text: texts.join(" "), sentences: gold });
        }
    }
    Ok((documents, sentences))
}

#[cfg(test)]
mod tests {
    use turbo_formats::masc::{MascDocument, MascFiles};

    use super::*;

    /// `aa bb.` as one MASC document of one sentence and three tokens.
    fn tiny_document() -> MascDocument {
        let seg = concat!(
            "<graph>",
            r#"<region xml:id="seg-r0" anchors="0 2"/>"#,
            r#"<region xml:id="seg-r1" anchors="3 5"/>"#,
            r#"<region xml:id="seg-r2" anchors="5 6"/>"#,
            "</graph>",
        );
        let penn = concat!(
            "<graph>",
            r#"<node xml:id="penn-n0"><link targets="seg-r0"/></node>"#,
            r#"<node xml:id="penn-n1"><link targets="seg-r1"/></node>"#,
            r#"<node xml:id="penn-n2"><link targets="seg-r2"/></node>"#,
            "</graph>",
        );
        MascDocument::parse(
            "t.hdr",
            MascFiles {
                text: b"aa bb.",
                segments: Some(seg.as_bytes()),
                sentences: Some(br#"<graph><region anchors="0 6"/></graph>"#),
                penn: Some(penn.as_bytes()),
                named_entities: None,
            },
        )
        .unwrap()
    }

    #[test]
    fn the_split_holds_out_every_tenth_document_and_skips_excluded_places() {
        let corpus = MascCorpus::from_documents((0..10).map(|_| tiny_document()).collect());
        let (documents, sentences) = masc_documents(&corpus, &[false; 10]).unwrap();
        assert_eq!(sentences, 1);
        assert_eq!(
            documents,
            [TokenDocument { text: "aa bb.".into(), sentences: vec![vec!["aa".into(), "bb".into(), ".".into()]] }]
        );
        let mut excluded = [false; 10];
        excluded[9] = true;
        let (documents, sentences) = masc_documents(&corpus, &excluded).unwrap();
        assert!(documents.is_empty() && sentences == 0);
    }

    #[test]
    fn twitter_headers_are_excluded_and_others_are_not() {
        assert!(header_excluded(r#"<textClass catRef="WR TW"/>"#));
        assert!(header_excluded("<textClass catRef='TW'/>"));
        assert!(!header_excluded(r#"<textClass catRef="WR NW"/>"#));
        assert!(!header_excluded("<textClass/>"));
    }

    #[test]
    fn digest_is_a_sorted_sha256sum_manifest() {
        let dir = std::env::temp_dir().join(format!("turbo-eval-masc-digest-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("b.txt"), b"two").unwrap();
        std::fs::write(dir.join("sub/a.txt"), b"one").unwrap();
        let manifest = format!("{}  b.txt\n{}  sub/a.txt\n", to_hex(&sha256(b"two")), to_hex(&sha256(b"one")));
        assert_eq!(corpus_digest(&dir).unwrap(), to_hex(&sha256(manifest.as_bytes())));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
