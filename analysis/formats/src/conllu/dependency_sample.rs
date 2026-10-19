//! Dependency samples with the alignment metadata that the Java stream omits.
use super::{ConlluDependencyReader, ConlluWordReader, Tagset};
use crate::sample::dependency::{DependencySample, DependencyTree};
use crate::source::Field;
use turbo_analysis::{Span, ViewId};
use turbo_unicode::WhitespaceMode;

/// A CoNLL-U multi-word token, indexing syntactic words with a zero-based,
/// half-open range. `form.span` addresses the original CoNLL-U source.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DependencyMultiwordToken {
    pub words: std::ops::Range<u32>,
    pub form: Field<'static>,
}
/// Lossless dependency/offset carrier. Source-field offsets, document-view word
/// spans and multi-word token ranges are separate, never inferred from forms.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConlluDependencySample {
    /// Original source bytes, including comments, MWT columns and empty nodes.
    /// Retained as provenance even when the dependency tree is replaced.
    pub source_text: String,
    pub sample: DependencySample<'static>,
    pub view: ViewId,
    pub word_spans: Vec<Span>,
    pub multiword_tokens: Vec<DependencyMultiwordToken>,
}
impl ConlluDependencySample {
    pub fn check(&self) -> Result<(), String> {
        if self.word_spans.len() != self.sample.len() {
            return Err("one view span per word required".into());
        }
        if self.word_spans.iter().any(|s| s.start > s.end) {
            return Err("reversed word span".into());
        }
        let mut end = 0;
        for m in &self.multiword_tokens {
            if m.words.start < end
                || m.words.start.checked_add(1).is_none_or(|start| m.words.end <= start)
                || m.words.end as usize > self.sample.len()
                || m.form.span.start > m.form.span.end
            {
                return Err("invalid or overlapping multi-word token range".into());
            }
            end = m.words.end;
        }
        Ok(())
    }
    /// Reads exactly one sentence. Offsets initially address the CoNLL-U source
    /// view, including the FORM fields of range rows. Callers mapping into a
    /// reconstructed text view can replace `view` and the view spans explicitly.
    /// Empty-node rows remain outside the basic dependency tree.
    pub fn read_sentence(source: &str, tagset: Tagset, mode: WhitespaceMode, view: ViewId) -> Result<Self, String> {
        if source.len() > u32::MAX as usize {
            return Err("source too long".into());
        }
        let mut reader = ConlluDependencyReader::new(source, tagset, mode).map_err(|e| e.to_string())?;
        let sample = reader.next().ok_or("no dependency sentence")?.map_err(|e| e.to_string())?;
        if reader.next().is_some() || !reader.skipped().is_empty() || !reader.node_id_issues().is_empty() {
            return Err("expected one valid dependency sentence".into());
        }
        let own = |f: &Field<'_>| Field::made(f.as_str().to_owned(), f.span);
        let tree = DependencyTree::new(
            sample.tree().heads().to_vec(),
            sample.tree().relations().iter().map(own).collect(),
            mode,
        )
        .map_err(|e| e.to_string())?;
        let word_spans = sample.tokens().iter().map(|f| f.span).collect();
        let owned = DependencySample::new(
            sample.tokens().iter().map(own).collect(),
            sample.tags().iter().map(own).collect(),
            tree,
            sample.source(),
        )
        .map_err(|e| e.to_string())?;
        let mut words = ConlluWordReader::new(source).map_err(|e| e.to_string())?;
        let sentence = words.next().ok_or("no syntactic sentence")?.map_err(|e| e.to_string())?;
        if words.next().is_some() || sentence.words.len() != owned.len() {
            return Err("expected one aligned syntactic sentence".into());
        }
        let multiword_tokens = sentence
            .multiwords
            .iter()
            .map(|m| DependencyMultiwordToken {
                words: m.words.start.get() as u32..m.words.end.get() as u32,
                form: own(&m.line_data.form),
            })
            .collect();
        let out = Self { source_text: source.into(), sample: owned, view, word_spans, multiword_tokens };
        out.check()?;
        Ok(out)
    }
}
