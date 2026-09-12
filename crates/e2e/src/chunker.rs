//! Stable paragraph / sentence ids for repeatable Embed batches.
//!
//! Ids are a function of the *source name* and the *order* of units in that
//! file. A SHA-pinned corpus therefore always produces the same ids:
//!
//! ```text
//! tiny-shakespeare:p0000          # first blank-line paragraph
//! tiny-shakespeare:p0000:s0001    # second sentence of that paragraph
//! ```

use std::fmt;

/// Paragraph (blank-line block) or sentence inside a paragraph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkKind {
    Paragraph,
    Sentence,
}

impl fmt::Display for ChunkKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Paragraph => f.write_str("paragraph"),
            Self::Sentence => f.write_str("sentence"),
        }
    }
}

/// One chunk of a named source, with a stable id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub id: String,
    pub kind: ChunkKind,
    pub text: String,
    pub source: String,
    pub para_index: usize,
    pub sent_index: Option<usize>,
}

impl Chunk {
    pub fn paragraph_id(source: &str, para_index: usize) -> String {
        format!("{source}:p{para_index:04}")
    }

    pub fn sentence_id(source: &str, para_index: usize, sent_index: usize) -> String {
        format!("{source}:p{para_index:04}:s{sent_index:04}")
    }
}

/// Split `text` into paragraphs and sentences with stable ids.
///
/// Paragraphs are separated by one or more blank lines. Sentences split on
/// `.`, `!`, or `?` followed by whitespace, with a small abbreviation list
/// so `Mr.` / `e.g.` stay intact. Ids do not hash the text — they follow
/// position — so a pinned file is a pinned batch.
pub fn chunk_text(source: &str, text: &str) -> Vec<Chunk> {
    let mut out = Vec::new();
    for (para_index, para) in split_paragraphs(text).into_iter().enumerate() {
        out.push(Chunk {
            id: Chunk::paragraph_id(source, para_index),
            kind: ChunkKind::Paragraph,
            text: para.clone(),
            source: source.to_string(),
            para_index,
            sent_index: None,
        });
        for (sent_index, sent) in split_sentences(&para).into_iter().enumerate() {
            out.push(Chunk {
                id: Chunk::sentence_id(source, para_index, sent_index),
                kind: ChunkKind::Sentence,
                text: sent,
                source: source.to_string(),
                para_index,
                sent_index: Some(sent_index),
            });
        }
    }
    out
}

/// Sentence chunks (or the paragraph itself when it has no sentence break).
/// This is the unit list sent to Embed for a soak / parity batch.
pub fn embed_units(source: &str, text: &str) -> Vec<Chunk> {
    use std::collections::BTreeMap;
    let all = chunk_text(source, text);
    let mut by_para: BTreeMap<usize, Vec<&Chunk>> = BTreeMap::new();
    for c in &all {
        by_para.entry(c.para_index).or_default().push(c);
    }
    let mut out = Vec::new();
    for group in by_para.values() {
        let sents: Vec<Chunk> = group
            .iter()
            .filter(|c| c.kind == ChunkKind::Sentence)
            .map(|c| (*c).clone())
            .collect();
        if sents.is_empty() {
            if let Some(p) = group.iter().find(|c| c.kind == ChunkKind::Paragraph) {
                out.push((*p).clone());
            }
        } else {
            out.extend(sents);
        }
    }
    out
}

/// Blank-line paragraphs, CR/LF normalized, empties dropped.
pub fn split_paragraphs(text: &str) -> Vec<String> {
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = Vec::new();
    let mut buf = String::new();
    for line in normalized.lines() {
        if line.trim().is_empty() {
            let trimmed = buf.trim();
            if !trimmed.is_empty() {
                out.push(trimmed.to_string());
            }
            buf.clear();
        } else {
            if !buf.is_empty() {
                buf.push('\n');
            }
            buf.push_str(line);
        }
    }
    let trimmed = buf.trim();
    if !trimmed.is_empty() {
        out.push(trimmed.to_string());
    }
    out
}

const ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e", "cf",
];

/// Split a paragraph into sentences. Deterministic; not a linguistic parser.
pub fn split_sentences(para: &str) -> Vec<String> {
    let para = para.trim();
    if para.is_empty() {
        return Vec::new();
    }
    let bytes = para.as_bytes();
    let mut out = Vec::new();
    let mut start = 0usize;
    let mut i = 0usize;
    while i < bytes.len() {
        let ch = bytes[i];
        if matches!(ch, b'.' | b'!' | b'?') {
            let prev = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            let token = prev
                .rsplit(|c: char| c.is_whitespace() || matches!(c, ',' | ';' | ':' | '(' | '['))
                .next()
                .unwrap_or("")
                .trim_matches(|c: char| !c.is_ascii_alphabetic() && c != '.')
                .to_ascii_lowercase();
            let is_abbrev = ch == b'.' && ABBREV.iter().any(|a| token == *a);
            let mut j = i + 1;
            while j < bytes.len() && matches!(bytes[j], b'"' | b'\'' | b')' | b']') {
                j += 1;
            }
            let at_end = j >= bytes.len();
            let next_ws = !at_end && bytes[j].is_ascii_whitespace();
            if !is_abbrev && (at_end || next_ws) {
                let end = j;
                let sent = para[start..end].trim();
                if !sent.is_empty() {
                    out.push(sent.to_string());
                }
                while j < bytes.len() && bytes[j].is_ascii_whitespace() {
                    j += 1;
                }
                start = j;
                i = j;
                continue;
            }
        }
        i += 1;
    }
    let tail = para[start..].trim();
    if !tail.is_empty() {
        // Only emit a sentence for a leftover when we already split on a
        // terminator (start > 0). A block with no `.?!` stays a paragraph
        // so embed_units can keep a single stable paragraph id.
        if start > 0 {
            out.push(tail.to_string());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXCERPT: &str =
        include_str!("../../../testdata/corpus/fixtures/tiny-shakespeare.excerpt.txt");

    #[test]
    fn excerpt_paragraph_ids_are_stable() {
        let chunks = chunk_text("tiny-shakespeare", EXCERPT);
        let paras: Vec<_> = chunks
            .iter()
            .filter(|c| c.kind == ChunkKind::Paragraph)
            .collect();
        assert_eq!(paras.len(), 12);
        assert_eq!(paras[0].id, "tiny-shakespeare:p0000");
        assert!(paras[0].text.starts_with("First Citizen:"));
        assert_eq!(paras[1].id, "tiny-shakespeare:p0001");
        assert_eq!(paras[11].id, "tiny-shakespeare:p0011");
        // Same bytes → same ids, even if we re-chunk.
        let again = chunk_text("tiny-shakespeare", EXCERPT);
        let ids: Vec<_> = chunks.iter().map(|c| c.id.as_str()).collect();
        let ids2: Vec<_> = again.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(ids, ids2);
    }

    #[test]
    fn sentences_split_on_terminators() {
        let sents = split_sentences(
            "Let us kill him, and we'll have corn at our own price.\nIs't a verdict?",
        );
        assert_eq!(sents.len(), 2);
        assert!(sents[0].starts_with("Let us kill him"));
        assert_eq!(sents[1], "Is't a verdict?");
    }

    #[test]
    fn abbreviations_do_not_split() {
        let sents = split_sentences("Dr. Caius met Mr. Marcius vs. the city.");
        assert_eq!(sents.len(), 1);
    }

    #[test]
    fn embed_units_are_sentences_not_duplicate_paragraphs() {
        let units = embed_units(
            "ex",
            "Hello world. Second sentence.\n\nAlone without a stop",
        );
        let ids: Vec<_> = units.iter().map(|c| c.id.as_str()).collect();
        assert!(ids.contains(&"ex:p0000:s0000"));
        assert!(ids.contains(&"ex:p0000:s0001"));
        assert!(ids.contains(&"ex:p0001"));
        assert!(
            !ids.contains(&"ex:p0000"),
            "paragraph with sentences is not re-embedded"
        );
    }

    #[test]
    fn empty_and_crlf_are_normalized() {
        assert!(split_paragraphs("").is_empty());
        assert_eq!(split_paragraphs("a\r\n\r\nb").len(), 2);
    }
}
