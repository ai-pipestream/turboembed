//! Whole-document examples retain cross-sentence annotations and source offsets.
use super::{Error, ReadError, Result, TYPES};
use crate::corpus::Sentence;
use std::path::PathBuf;
use turbo_formats::masc::MascDocument;

/// An original annotation cannot be represented by a flat BIO head.
#[derive(Debug)]
pub struct UnsupportedAnnotation {
    pub path: PathBuf,
    pub entity: i32,
    /// Original UTF-16 token bounds, never reconstructed text offsets.
    pub start: i32,
    pub end: i32,
    pub reason: &'static str,
}
impl std::fmt::Display for UnsupportedAnnotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: ne-n{} at UTF-16 {}..{}: {}",
            self.path.display(),
            self.entity,
            self.start,
            self.end,
            self.reason
        )
    }
}
impl std::error::Error for UnsupportedAnnotation {}

/// One sequence over the document's original Penn tokens. Encoder windowing
/// preserves the sequence; a sentence boundary never resets an entity label.
///
/// # Errors
///
/// Unknown, unresolved, overlapping and discontinuous annotations are refused.
pub fn document(doc: &MascDocument) -> Result<Option<Sentence>> {
    let fail = |reason| ReadError::Invalid(Error(format!("{}: {reason}", doc.path().display())));
    if !doc.has_named_entities() {
        if doc.entity_count() != 0 {
            return Err(fail("NE document has no Penn tokens"));
        }
        return Ok(None);
    }
    let all = doc.document_tokens().ok_or_else(|| fail("NE document has no Penn tokens"))?;
    if all.entity_pieces().len() != doc.entity_count() {
        return Err(fail("an original entity has no resolved Penn tokens"));
    }
    let tokens = all.tokens().ok_or_else(|| fail("document has no Penn tokens"))?;
    let spans = doc.token_spans(all)?;
    if spans.is_empty() || spans.iter().any(|s| s.start > s.end) || spans.windows(2).any(|w| w[0].end > w[1].start) {
        return Err(fail("empty, unordered or overlapping original Penn tokens"));
    }
    let mut labels = vec!["O".to_owned(); spans.len()];
    for piece in all.entity_pieces() {
        let e = &piece.entity;
        let error = |reason| {
            ReadError::Unsupported(UnsupportedAnnotation {
                path: doc.path().to_owned(),
                entity: piece.id,
                start: tokens[e.start as usize].start,
                end: tokens[e.end as usize - 1].end,
                reason,
            })
        };
        if e.tokens.iter().any(|&i| spans[i as usize].start == spans[i as usize].end) {
            return Err(error("entity references a zero-width source token"));
        }
        if piece.continues {
            return Err(error("unresolved entity token membership"));
        }
        if !e.is_contiguous() {
            return Err(error("discontinuous original annotation"));
        }
        let kind = match e.kind.as_deref() {
            Some("org") => "organization",
            Some(k) => k,
            None => return Err(error("missing entity type")),
        };
        if !TYPES.contains(&kind) {
            return Err(error("unsupported entity type"));
        }
        for (i, label) in labels[e.start as usize..e.end as usize].iter_mut().enumerate() {
            if label != "O" {
                return Err(error("overlapping original annotations"));
            }
            *label = format!("{}-{kind}", if i == 0 { "B" } else { "I" });
        }
    }
    if spans.iter().any(|s| s.start == s.end) {
        return Err(fail("zero-width source token outside an entity"));
    }
    let words = spans.iter().map(|s| doc.text()[s.start as usize..s.end as usize].to_owned()).collect();
    Ok(Some(Sentence { words, labels }))
}
