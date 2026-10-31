//! Strict word-aligned BIO peer exchange. Gold labels never leave in export.
use crate::corpus::Sentence;
use crate::{Error, Result};
use std::fmt::Write as _;

pub fn export(sentences: &[Sentence]) -> Result<String> {
    let mut out = String::new();
    for sentence in sentences {
        for word in &sentence.words {
            if word.is_empty() || word.contains(['\t', '\n', '\r']) {
                return Err(Error(
                    "BIO export cannot represent an empty token or a token containing a tab/newline".into(),
                ));
            }
            writeln!(out, "{word}\t_").expect("String write");
        }
        out.push('\n');
    }
    Ok(out)
}

pub fn read(file: &str, bytes: &[u8], gold: &[Sentence]) -> Result<Vec<Vec<String>>> {
    let text = std::str::from_utf8(bytes).map_err(|e| Error(format!("{file}: UTF-8: {e}")))?;
    let mut out = Vec::new();
    let mut current = Vec::new();
    for (line_no, line) in text.lines().chain(std::iter::once("")).enumerate() {
        let fail = |message: &str| Error(format!("{file}:{}: {message}", line_no + 1));
        if line.is_empty() {
            if !current.is_empty() {
                if gold.get(out.len()).is_none_or(|s| s.words.len() != current.len()) {
                    return Err(fail("prediction sentence does not have the gold token count"));
                }
                out.push(std::mem::take(&mut current));
            }
            continue;
        }
        let mut fields = line.split('\t');
        let word = fields.next().unwrap_or_default();
        let label = fields.next().ok_or_else(|| fail("expected token and BIO label"))?;
        if word.is_empty() || label.is_empty() || fields.next().is_some() {
            return Err(fail("expected exactly two nonempty tab-separated fields"));
        }
        let expected = gold.get(out.len()).and_then(|s| s.words.get(current.len()));
        if expected.map(String::as_str) != Some(word) {
            return Err(fail("prediction token differs from the gold token at this position"));
        }
        current.push(label.to_owned());
    }
    if out.len() != gold.len() {
        return Err(Error(format!("{file}: {} prediction sentences, {} gold sentences", out.len(), gold.len())));
    }
    Ok(out)
}

/// Validate after an explicit peer-to-corpus mapping, never infer a mapping.
pub fn validate(corpus: &str, labels: &[Vec<String>]) -> Result<()> {
    let known = |kind: &str| {
        crate::fixed_entity_types(corpus).is_some_and(|types| types.contains(&kind))
            || (corpus == crate::multiconer::ID
                && crate::multiconer::COARSE.iter().any(|(_, types)| types.contains(&kind)))
    };
    for (sentence, tags) in labels.iter().enumerate() {
        let mut previous = None;
        for (word, label) in tags.iter().enumerate() {
            if label == "O" {
                previous = None;
                continue;
            }
            let Some((prefix, kind)) = label.split_once('-') else {
                return Err(Error(format!("sentence {sentence} token {word}: not a BIO label: {label:?}")));
            };
            if !matches!(prefix, "B" | "I") || !known(kind) || (prefix == "I" && previous != Some(kind)) {
                return Err(Error(format!(
                    "sentence {sentence} token {word}: invalid {corpus} BIO sequence at {label:?}"
                )));
            }
            previous = Some(kind);
        }
    }
    Ok(())
}
