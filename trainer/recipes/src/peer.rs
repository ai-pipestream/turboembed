//! A peer system's tags on a corpus's test split (spaCy, Stanza), scored
//! as a trained head is scored: the same gold, the same majority
//! baseline, the same paired bootstrap, so the receipts stand side by
//! side.
//!
//! The split goes out as CoNLL-U with the gold words and no labels
//! (`conllu`), the peer tags those words as given, and its CoNLL-U comes
//! back in (`read`): the same sentences, word for word, the tag in the
//! column named.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::{Deserialize, Serialize};
use turbo_formats::conllu::ConlluWordReader;

use crate::corpus::{FileRecord, Sentence};
use crate::{Error, Result};

/// The CoNLL-U column a peer's tag is read from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Column {
    /// Column 4, the universal tag (spaCy `pos_`).
    Upos,
    /// Column 5, the language-specific tag (spaCy `tag_`, Penn for
    /// English).
    Xpos,
    /// Two tab-separated fields: original token and BIO entity label.
    Bio,
}

impl std::str::FromStr for Column {
    type Err = Error;
    fn from_str(s: &str) -> Result<Column> {
        match s {
            "upos" => Ok(Column::Upos),
            "xpos" => Ok(Column::Xpos),
            "bio" => Ok(Column::Bio),
            _ => Err(Error(format!("column {s:?}: upos, xpos or bio"))),
        }
    }
}

/// What a receipt records of a peer's predictions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PeerRecord {
    /// The peer and its version, as given (`spacy-3.8.7-en_core_web_trf-3.8.0`).
    pub name: String,
    pub column: Column,
    /// The predictions file, by digest.
    pub predictions: FileRecord,
    /// Tag spellings rewritten before scoring, peer's to the corpus's
    /// (spaCy's `-LRB-` is MASC's `(`): the same decision written another
    /// way. Empty when the tags are scored as the peer wrote them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub map: BTreeMap<String, String>,
}

/// A `--map` value, `FROM=TO` pairs separated by commas.
///
/// # Errors
///
/// A pair without `=`, an empty side, or a tag mapped twice.
pub fn parse_map(text: &str) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for pair in text.split(',') {
        let (from, to) = pair
            .split_once('=')
            .filter(|(f, t)| !f.is_empty() && !t.is_empty())
            .ok_or_else(|| Error(format!("--map {pair:?}: FROM=TO")))?;
        if out.insert(from.to_owned(), to.to_owned()).is_some() {
            return Err(Error(format!("--map maps {from:?} twice")));
        }
    }
    Ok(out)
}

/// `tags` with each tag `map` names rewritten.
pub fn apply(map: &BTreeMap<String, String>, tags: &mut [Vec<String>]) {
    for t in tags.iter_mut().flatten() {
        if let Some(to) = map.get(t.as_str()) {
            t.clone_from(to);
        }
    }
}

/// A word as a CoNLL-U form: a tab or line break in it (MASC has a word
/// broken across a line) is written as a space, which a form may hold.
pub fn form(word: &str) -> String {
    word.replace(['\t', '\n', '\r'], " ")
}

/// `sentences` as CoNLL-U: one block a sentence (`# sent_id = <n>`,
/// counted from 1), one line a word with its [`form`] and nothing else,
/// so a peer sees only the words.
pub fn conllu(sentences: &[Sentence]) -> String {
    let mut out = String::new();
    for (n, s) in sentences.iter().enumerate() {
        let _ = writeln!(out, "# sent_id = {}", n + 1);
        for (i, w) in s.words.iter().enumerate() {
            let _ = writeln!(out, "{}\t{}\t_\t_\t_\t_\t_\t_\t_\t_", i + 1, form(w));
        }
        out.push('\n');
    }
    out
}

/// The tags in `column` of a peer's CoNLL-U, one list a sentence of
/// `gold`.
///
/// # Errors
///
/// Text that is not UTF-8 or that the CoNLL-U reader refuses, or
/// sentences or words other than `gold`'s, naming the first that differs.
pub fn read(file: &str, bytes: &[u8], column: Column, gold: &[Sentence]) -> Result<Vec<Vec<String>>> {
    if column == Column::Bio {
        return crate::ner_peer::read(file, bytes, gold);
    }
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: not UTF-8: {e}")))?;
    let mut out = Vec::with_capacity(gold.len());
    for (n, sentence) in ConlluWordReader::new(text).map_err(|e| Error(format!("{file}: {e}")))?.enumerate() {
        let sentence = sentence.map_err(|e| Error(format!("{file}: {e}")))?;
        let Some(g) = gold.get(n) else {
            return Err(Error(format!("{file}: more than the {} sentences of the split", gold.len())));
        };
        let forms: Vec<&str> = sentence.words.iter().map(|w| w.form.text.as_ref()).collect();
        if forms != g.words.iter().map(|w| form(w)).collect::<Vec<_>>() {
            return Err(Error(format!("{file}: sentence {} has words {forms:?}, the split {:?}", n + 1, g.words)));
        }
        out.push(
            sentence
                .words
                .iter()
                .map(|w| match column {
                    Column::Upos => w.upos.text.to_string(),
                    Column::Xpos => w.xpos.text.to_string(),
                    Column::Bio => unreachable!("BIO is handled before CoNLL-U parsing"),
                })
                .collect(),
        );
    }
    if out.len() != gold.len() {
        return Err(Error(format!("{file}: {} sentences, the split {}", out.len(), gold.len())));
    }
    Ok(out)
}
