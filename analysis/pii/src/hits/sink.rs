//! The candidate sink a [`crate::CustomExtractor`] writes into.

use turbo_analysis::Span;
use turbo_analysis_layers::{CustomKinds, PiiDetectorRef, PiiEvidence, PiiKind, PiiScore};

use super::{Hits, SinkError};

/// Candidate sink passed to a [`crate::CustomExtractor`] during scanning.
/// Every candidate names its facts: shape, checksum or rule. Labels and
/// deny list hits are the detector's to record, so the sink refuses them.
pub struct PiiSink<'a> {
    text: &'a str,
    kinds: &'a CustomKinds,
    hits: &'a mut Hits,
}

impl<'a> PiiSink<'a> {
    pub(crate) fn new(text: &'a str, kinds: &'a CustomKinds, hits: &'a mut Hits) -> PiiSink<'a> {
        PiiSink { text, kinds, hits }
    }

    /// The detector's custom-kind table, which every custom kind handed to
    /// this sink must index.
    pub const fn custom_kinds(&self) -> &'a CustomKinds {
        self.kinds
    }

    /// The source text being scanned.
    pub const fn text(&self) -> &'a str {
        self.text
    }

    /// The entry a [`PiiScore::model`] names for the model `id` at
    /// `version`, added to the run's detector table when new.
    pub fn model(&mut self, id: &str, version: &str) -> Result<PiiDetectorRef, SinkError> {
        self.hits.detectors_mut().add(id, version).map_err(|e| SinkError::Score(Span::new(0, 0), e.to_string()))
    }

    fn check(
        &self,
        span: Span,
        kind: PiiKind,
        evidence: &[PiiEvidence],
        score: Option<PiiScore>,
    ) -> Result<(usize, usize), SinkError> {
        self.kinds.check(kind).map_err(SinkError::Kind)?;
        let (start, end) = (span.start as usize, span.end as usize);
        if start >= end
            || end > self.text.len()
            || !self.text.is_char_boundary(start)
            || !self.text.is_char_boundary(end)
        {
            return Err(SinkError::Span(span));
        }
        if evidence.is_empty() {
            return Err(SinkError::NoEvidence(span));
        }
        if let Some(foreign) =
            evidence.iter().find(|e| matches!(e, PiiEvidence::Label { .. } | PiiEvidence::DenyHit { .. }))
        {
            return Err(SinkError::ForeignEvidence(span, foreign.kind_name()));
        }
        if let Some(s) = score {
            s.check().map_err(|e| SinkError::Score(span, e))?;
            self.hits.detectors.check(s.model).map_err(|e| SinkError::Score(span, e.to_string()))?;
        }
        Ok((start, end))
    }

    /// Appends a candidate with `evidence`, an optional `score`, and the
    /// normalized form `write` writes into the shared candidate buffer,
    /// after verifying that `span` is a non-empty character-boundary range
    /// of [`PiiSink::text`], `kind` is built-in or in
    /// [`PiiSink::custom_kinds`], `evidence` is non-empty and holds no
    /// label or deny list fact, the score is in range, and the normalized
    /// form is non-empty.
    pub fn try_add_candidate(
        &mut self,
        span: Span,
        kind: PiiKind,
        evidence: &[PiiEvidence],
        score: Option<PiiScore>,
        write: impl FnOnce(&mut String),
    ) -> Result<(), SinkError> {
        let (start, end) = self.check(span, kind, evidence, score)?;
        if self.hits.add_candidate(start, end, kind, evidence, score, write) {
            Ok(())
        } else {
            Err(SinkError::EmptyValue(span))
        }
    }

    /// Appends a candidate whose normalized form is `normalized`.
    pub fn try_add(
        &mut self,
        span: Span,
        kind: PiiKind,
        evidence: &[PiiEvidence],
        normalized: &str,
    ) -> Result<(), SinkError> {
        self.try_add_candidate(span, kind, evidence, None, |s| s.push_str(normalized))
    }

    /// Appends a candidate whose normalized form is `normalized`.
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add`] would return an error.
    pub fn add(&mut self, span: Span, kind: PiiKind, evidence: &[PiiEvidence], normalized: &str) {
        self.try_add(span, kind, evidence, normalized).expect("valid candidate mention");
    }

    /// Appends a candidate whose normalized form is the covered slice of
    /// [`PiiSink::text`].
    pub fn try_add_text(&mut self, span: Span, kind: PiiKind, evidence: &[PiiEvidence]) -> Result<(), SinkError> {
        let text = self.text;
        self.try_add_candidate(span, kind, evidence, None, |s| s.push_str(&text[span.range()]))
    }

    /// Appends a candidate whose normalized form is the covered slice of
    /// [`PiiSink::text`].
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add_text`] would return an error.
    pub fn add_text(&mut self, span: Span, kind: PiiKind, evidence: &[PiiEvidence]) {
        self.try_add_text(span, kind, evidence).expect("valid candidate mention");
    }

    /// Appends a candidate whose normalized form `write` writes into the
    /// shared candidate buffer without allocating a temporary `String`.
    pub fn try_add_with(
        &mut self,
        span: Span,
        kind: PiiKind,
        evidence: &[PiiEvidence],
        write: impl FnOnce(&mut String),
    ) -> Result<(), SinkError> {
        self.try_add_candidate(span, kind, evidence, None, write)
    }

    /// Appends a candidate whose normalized form `write` writes into the
    /// shared candidate buffer.
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add_with`] would return an error.
    pub fn add_with(&mut self, span: Span, kind: PiiKind, evidence: &[PiiEvidence], write: impl FnOnce(&mut String)) {
        self.try_add_with(span, kind, evidence, write).expect("valid candidate mention");
    }
}
