//! Candidates and their order, OpenNLP's `Hits`: by start, longer first,
//! then by kind rank, then in the order they were found; equal mentions
//! once, the first kept. By default, a candidate strictly nested inside a
//! longer candidate is dropped; OpenNLP parity mode keeps it.

use std::cmp::Reverse;
use std::fmt;
use std::ops::Range;

use turbo_analysis::Span;
use turbo_analysis_layers::{Pii, PiiConfidence, PiiKind, PiiMention};

/// Why a candidate handed to [`PiiSink`] was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkError {
    /// The span is empty, past the end of the text, or not on UTF-8
    /// character boundaries.
    Span(Span),
    /// The kind name is empty.
    EmptyKind,
    /// A [`PiiKind::Custom`] name is longer than [`PiiKind::MAX_CUSTOM_NAME_BYTES`].
    KindTooLong(&'static str),
    /// A [`PiiKind::Custom`] uses one of the reserved built-in kind names.
    ReservedKind(&'static str),
    /// A [`PiiKind::Custom`] exceeds the [`PiiKind::MAX_CUSTOM_KINDS`] registry capacity.
    RegistryFull(&'static str),
    /// The normalized form is empty.
    EmptyValue(Span),
}

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SinkError::Span(s) => write!(f, "the span {}..{} is not a non-empty range of the text", s.start, s.end),
            SinkError::EmptyKind => f.write_str("the pii kind name must not be empty"),
            SinkError::KindTooLong(name) => {
                write!(f, "custom pii kind {name:?} is {} bytes (max {})", name.len(), PiiKind::MAX_CUSTOM_NAME_BYTES)
            }
            SinkError::ReservedKind(name) => {
                write!(f, "custom pii kind {name:?} uses a reserved built-in kind name")
            }
            SinkError::RegistryFull(name) => {
                write!(f, "custom pii kind {name:?} exceeds registry capacity ({})", PiiKind::MAX_CUSTOM_KINDS)
            }
            SinkError::EmptyValue(s) => {
                write!(f, "candidate at {}..{} has an empty normalized form", s.start, s.end)
            }
        }
    }
}

impl std::error::Error for SinkError {}

/// Candidate sink passed to a [`crate::CustomExtractor`] during scanning.
pub struct PiiSink<'a> {
    text: &'a str,
    hits: &'a mut Hits,
}

impl<'a> PiiSink<'a> {
    pub(crate) fn new(text: &'a str, hits: &'a mut Hits) -> PiiSink<'a> {
        PiiSink { text, hits }
    }

    /// The source text being scanned.
    pub const fn text(&self) -> &'a str {
        self.text
    }

    fn check_span(&self, span: Span, kind: PiiKind) -> Result<(usize, usize), SinkError> {
        if kind.name().is_empty() {
            return Err(SinkError::EmptyKind);
        }
        if let PiiKind::Custom(name) = kind {
            if name.len() > PiiKind::MAX_CUSTOM_NAME_BYTES {
                return Err(SinkError::KindTooLong(name));
            }
            if PiiKind::parse(name).is_some() {
                return Err(SinkError::ReservedKind(name));
            }
            if !PiiKind::remember_custom(kind) {
                return Err(SinkError::RegistryFull(name));
            }
        }
        let (start, end) = (span.start as usize, span.end as usize);
        if start >= end
            || end > self.text.len()
            || !self.text.is_char_boundary(start)
            || !self.text.is_char_boundary(end)
        {
            return Err(SinkError::Span(span));
        }
        Ok((start, end))
    }

    /// Appends a candidate whose normalized form is `normalized`, after
    /// verifying that `span` is a non-empty character-boundary range of
    /// [`PiiSink::text`] and `normalized` and `kind.name()` are non-empty.
    pub fn try_add(
        &mut self,
        span: Span,
        kind: PiiKind,
        confidence: PiiConfidence,
        normalized: &str,
    ) -> Result<(), SinkError> {
        let (start, end) = self.check_span(span, kind)?;
        if normalized.is_empty() {
            return Err(SinkError::EmptyValue(span));
        }
        PiiKind::remember_custom(kind);
        self.hits.add(start, end, kind, confidence, |s| s.push_str(normalized));
        Ok(())
    }

    /// Appends a candidate whose normalized form is `normalized`.
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add`] would return an error.
    pub fn add(&mut self, span: Span, kind: PiiKind, confidence: PiiConfidence, normalized: &str) {
        self.try_add(span, kind, confidence, normalized).expect("valid candidate mention");
    }

    /// Appends a candidate whose normalized form is the covered slice of
    /// [`PiiSink::text`].
    pub fn try_add_text(&mut self, span: Span, kind: PiiKind, confidence: PiiConfidence) -> Result<(), SinkError> {
        let (start, end) = self.check_span(span, kind)?;
        PiiKind::remember_custom(kind);
        self.hits.add_text(self.text, start, end, kind, confidence);
        Ok(())
    }

    /// Appends a candidate whose normalized form is the covered slice of
    /// [`PiiSink::text`].
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add_text`] would return an error.
    pub fn add_text(&mut self, span: Span, kind: PiiKind, confidence: PiiConfidence) {
        self.try_add_text(span, kind, confidence).expect("valid candidate mention");
    }

    /// Appends a candidate whose normalized form `write` writes into the
    /// shared candidate buffer without allocating a temporary `String`.
    pub fn try_add_with(
        &mut self,
        span: Span,
        kind: PiiKind,
        confidence: PiiConfidence,
        write: impl FnOnce(&mut String),
    ) -> Result<(), SinkError> {
        let (start, end) = self.check_span(span, kind)?;
        let from = self.hits.buf.len() as u32;
        write(&mut self.hits.buf);
        let to = self.hits.buf.len() as u32;
        if from == to {
            return Err(SinkError::EmptyValue(span));
        }
        PiiKind::remember_custom(kind);
        self.hits.hits.push(Hit { start: start as u32, end: end as u32, kind, confidence, normalized: from..to });
        Ok(())
    }

    /// Appends a candidate whose normalized form `write` writes into the
    /// shared candidate buffer.
    ///
    /// # Panics
    ///
    /// When [`PiiSink::try_add_with`] would return an error.
    pub fn add_with(&mut self, span: Span, kind: PiiKind, confidence: PiiConfidence, write: impl FnOnce(&mut String)) {
        self.try_add_with(span, kind, confidence, write).expect("valid candidate mention");
    }
}

/// One candidate, with its normalized form as a range of the shared
/// buffer.
#[derive(Clone, Debug)]
struct Hit {
    start: u32,
    end: u32,
    kind: PiiKind,
    confidence: PiiConfidence,
    normalized: Range<u32>,
}

/// The candidates of one text: the scanners append, [`Hits::resolve`]
/// orders them and drops repeats.
#[derive(Default, Debug)]
pub(crate) struct Hits {
    hits: Vec<Hit>,
    buf: String,
}

impl Hits {
    /// Adds a candidate whose normalized form `write` appends to the
    /// buffer it is given, so no string is made per candidate.
    pub(crate) fn add(
        &mut self,
        start: usize,
        end: usize,
        kind: PiiKind,
        confidence: PiiConfidence,
        write: impl FnOnce(&mut String),
    ) {
        let from = self.buf.len() as u32;
        write(&mut self.buf);
        let to = self.buf.len() as u32;
        self.hits.push(Hit { start: start as u32, end: end as u32, kind, confidence, normalized: from..to });
    }

    /// Adds a candidate whose normalized form is the text it covers.
    pub(crate) fn add_text(&mut self, text: &str, start: usize, end: usize, kind: PiiKind, confidence: PiiConfidence) {
        self.add(start, end, kind, confidence, |s| s.push_str(&text[start..end]));
    }

    fn normalized(&self, h: &Hit) -> &str {
        &self.buf[h.normalized.start as usize..h.normalized.end as usize]
    }

    /// Orders the candidates (a stable sort keeps the finding order of
    /// ties) and writes each distinct mention to `out`. Equal mentions
    /// share a span, so they sit in one run of that span. When
    /// `keep_nested` is `false`, any candidate whose span lies strictly
    /// inside an earlier, longer candidate's span is dropped.
    pub(crate) fn resolve(mut self, keep_nested: bool, out: &mut Pii) {
        self.hits.sort_by_key(|h| (h.start, Reverse(h.end), h.kind));
        let mut run_start = 0;
        let mut outer: Option<(u32, u32)> = None;
        for i in 0..self.hits.len() {
            let h = &self.hits[i];
            if i > 0 {
                let p = &self.hits[i - 1];
                if (p.start, p.end) != (h.start, h.end) {
                    run_start = i;
                }
            }
            if !keep_nested
                && outer.is_some_and(|(os, oe)| os <= h.start && h.end <= oe && (os < h.start || h.end < oe))
            {
                continue;
            }
            if outer.is_none_or(|(_, oe)| h.end > oe) {
                outer = Some((h.start, h.end));
            }
            let value = self.normalized(h);
            let repeat = self.hits[run_start..i].iter().any(|e| e.kind == h.kind && self.normalized(e) == value);
            if !repeat {
                out.push(PiiMention {
                    span: Span::new(h.start, h.end),
                    kind: h.kind,
                    confidence: h.confidence,
                    normalized: value,
                });
            }
        }
    }

    /// The resolved mentions as a layer over `view`.
    #[cfg(test)]
    pub(crate) fn into_layer(self, keep_nested: bool, view: turbo_analysis::ViewId) -> Pii {
        let mut out = Pii::new(view);
        self.resolve(keep_nested, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis::{Layer, ViewId};

    #[test]
    fn order_is_start_then_longer_then_rank_and_repeats_go() {
        let mut h = Hits::default();
        let f = PiiConfidence::Format;
        h.add(4, 6, PiiKind::Phone, f, |s| s.push('a'));
        h.add(0, 3, PiiKind::Phone, f, |s| s.push('b'));
        h.add(0, 5, PiiKind::Phone, f, |s| s.push('c'));
        h.add(0, 5, PiiKind::Email, f, |s| s.push('d'));
        h.add(0, 5, PiiKind::Email, f, |s| s.push('d'));
        h.add(0, 5, PiiKind::Email, f, |s| s.push('e'));
        let p = h.into_layer(true, ViewId::SOURCE);
        assert_eq!(p.check(), Ok(()));
        let rows: Vec<_> = p.mentions().map(|m| (m.span.start, m.span.end, m.kind, m.normalized.to_owned())).collect();
        assert_eq!(
            rows,
            [
                (0, 5, PiiKind::Email, "d".to_owned()),
                (0, 5, PiiKind::Email, "e".to_owned()),
                (0, 5, PiiKind::Phone, "c".to_owned()),
                (0, 3, PiiKind::Phone, "b".to_owned()),
                (4, 6, PiiKind::Phone, "a".to_owned()),
            ]
        );
    }

    fn resolved_with(keep_nested: bool, adds: &[(u32, u32, PiiKind, &str)]) -> Vec<(u32, u32, PiiKind, String)> {
        let mut h = Hits::default();
        for &(start, end, kind, value) in adds {
            h.add(start as usize, end as usize, kind, PiiConfidence::Format, |s| s.push_str(value));
        }
        h.into_layer(keep_nested, ViewId::SOURCE)
            .mentions()
            .map(|m| (m.span.start, m.span.end, m.kind, m.normalized.to_owned()))
            .collect()
    }

    fn resolved(adds: &[(u32, u32, PiiKind, &str)]) -> Vec<(u32, u32, PiiKind, String)> {
        resolved_with(true, adds)
    }

    fn row(start: u32, end: u32, kind: PiiKind, value: &str) -> (u32, u32, PiiKind, String) {
        (start, end, kind, value.to_owned())
    }

    #[test]
    fn nested_candidates_drop_by_default_and_same_or_crossing_spans_stay() {
        use PiiKind::*;
        // Prefix, interior and suffix spans inside 2..10 drop; a second
        // kind on 2..10 and a crossing span 8..14 stay.
        let adds = [
            (2, 10, Iban, "iban"),
            (2, 10, Card, "same-span"),
            (2, 6, Phone, "prefix"),
            (4, 8, Phone, "interior"),
            (4, 10, UkNhs, "suffix"),
            (8, 14, Email, "crossing"),
            (9, 12, Phone, "inside-crossing"),
        ];
        assert_eq!(
            resolved_with(false, &adds),
            [row(2, 10, Iban, "iban"), row(2, 10, Card, "same-span"), row(8, 14, Email, "crossing")]
        );
    }

    #[test]
    fn composite_orders() {
        use PiiKind::*;
        // Crossing candidates, in either order found.
        for adds in [[(4, 12, Email, "e"), (0, 8, Card, "c")], [(0, 8, Card, "c"), (4, 12, Email, "e")]] {
            assert_eq!(resolved(&adds), [row(0, 8, Card, "c"), row(4, 12, Email, "e")]);
        }
        assert_eq!(resolved(&[(2, 6, Phone, "p"), (2, 10, Card, "c")]), [row(2, 10, Card, "c"), row(2, 6, Phone, "p")]);
        // The rank decides an exact span tie, in either order found.
        for adds in [[(2, 10, Phone, "p"), (2, 10, Card, "c")], [(2, 10, Card, "c"), (2, 10, Phone, "p")]] {
            assert_eq!(resolved(&adds), [row(2, 10, Card, "c"), row(2, 10, Phone, "p")]);
        }
        // Equal ranks keep the order found.
        assert_eq!(resolved(&[(0, 5, Email, "b"), (0, 5, Email, "a")]), [row(0, 5, Email, "b"), row(0, 5, Email, "a")]);
        assert_eq!(resolved(&[(0, 4, Card, "c"), (4, 8, Phone, "p")]), [row(0, 4, Card, "c"), row(4, 8, Phone, "p")]);
        assert_eq!(
            resolved(&[(0, 6, Card, "a"), (10, 16, Card, "b"), (4, 12, Phone, "p"), (5, 7, Ipv4, "i")]),
            [row(0, 6, Card, "a"), row(4, 12, Phone, "p"), row(5, 7, Ipv4, "i"), row(10, 16, Card, "b")]
        );
        assert_eq!(resolved(&[(9, 25, Email, "x"), (9, 25, Email, "x")]), [row(9, 25, Email, "x")]);
        // Every pair of kinds on one span: the stronger first.
        for (i, &a) in PiiKind::ALL.iter().enumerate() {
            for &b in &PiiKind::ALL[i + 1..] {
                let want = [row(0, 5, a, "higher"), row(0, 5, b, "lower")];
                assert_eq!(resolved(&[(0, 5, a, "higher"), (0, 5, b, "lower")]), want);
                assert_eq!(resolved(&[(0, 5, b, "lower"), (0, 5, a, "higher")]), want);
            }
        }
    }
}
