//! SparseTermsResult: the turbo-analysis-layers `SparseTerms` layer's
//! columns (issue #708), document level (`SparseTerms<View>`), with its
//! source spans and occurrence spans.

use turbo_analysis::{Layer, ModelRef, Span, View};
use turbo_analysis_layers::SparseTerms;

use super::{MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::SparseTermsResult;

/// Document-level [`SparseTerms`] paired with optional view text and wire
/// [`OffsetMode`].
pub struct SparseTermsWire {
    pub text: String,
    pub offsets: OffsetMode,
    pub terms: SparseTerms<View>,
}

impl SparseTermsWire {
    pub fn new(terms: SparseTerms<View>, text: impl Into<String>, offsets: OffsetMode) -> Self {
        Self { text: text.into(), offsets, terms }
    }
}

fn spans_to_wire(spans: &[Span], text: &str, offsets: OffsetMode) -> (Vec<u32>, Vec<u32>) {
    if text.is_empty() && offsets == OffsetMode::Utf8 {
        (spans.iter().map(|s| s.start).collect(), spans.iter().map(|s| s.end).collect())
    } else {
        Offsetter::new(text, offsets).spans_to_wire(text, spans)
    }
}

fn sparse_to_msg(t: &SparseTerms<View>, text: &str, offsets: OffsetMode) -> SparseTermsResult {
    let has_occ = t.has_occurrences();
    let any_span = t.span.iter().any(|s| s.is_some());
    let present_spans: Vec<Span> = t.span.iter().flatten().copied().collect();
    let (span_start, span_end) =
        if any_span { spans_to_wire(&present_spans, text, offsets) } else { (Vec::new(), Vec::new()) };
    let (occ_start, occ_end) = if has_occ { spans_to_wire(&t.occ, text, offsets) } else { (Vec::new(), Vec::new()) };
    SparseTermsResult {
        model: Some(model_to_proto(&t.model)),
        row: t.row.clone(),
        term: t.term.clone(),
        weight: t.weight.clone(),
        has_span: if any_span { t.span.iter().map(|s| s.is_some()).collect() } else { Vec::new() },
        span_start,
        span_end,
        occ_start,
        occ_end,
        occ_at: if has_occ { t.occ_at.clone() } else { Vec::new() },
        text: text.to_string(),
        offsets: offsets.to_proto(),
    }
}

fn spans_from_wire(
    what: &str,
    starts: &[u32],
    ends: &[u32],
    text: &str,
    offsets: OffsetMode,
) -> Result<Vec<Span>, Failure> {
    if text.is_empty() && offsets == OffsetMode::Utf8 {
        Ok(starts.iter().zip(ends).map(|(&s, &e)| Span { start: s, end: e }).collect())
    } else {
        Offsetter::new(text, offsets).spans_from_wire(text, what, starts, ends)
    }
}

impl Wire for SparseTermsWire {
    type Msg = SparseTermsResult;

    fn to_msg(&self) -> SparseTermsResult {
        sparse_to_msg(&self.terms, &self.text, self.offsets)
    }

    fn from_msg(msg: &SparseTermsResult) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        let n = msg.term.len();
        if n as u64 > MAX_SPARSE_ENTRIES {
            return Err(over("max_sparse_entries", "sparse term entries", n as u64));
        }
        if msg.occ_start.len() as u64 > MAX_SPARSE_OCCURRENCES {
            return Err(over("max_sparse_occurrences", "sparse occurrence spans", msg.occ_start.len() as u64));
        }
        if msg.weight.len() != n {
            return Err(mismatched("weight", msg.weight.len(), n));
        }
        // The span columns are all empty, or all three hold one value per
        // entry (span_start one per present span).
        let spans_given = !msg.has_span.is_empty() || !msg.span_start.is_empty() || !msg.span_end.is_empty();
        if spans_given && !(msg.has_span.len() == n && msg.span_start.len() == msg.span_end.len()) {
            return Err(mismatched("has_span", msg.has_span.len(), n));
        }
        let spans = if msg.has_span.is_empty() {
            Vec::new()
        } else {
            let present = spans_from_wire("sparse term span", &msg.span_start, &msg.span_end, &msg.text, offsets)?;
            let mut spans = Vec::with_capacity(n);
            let mut k = 0;
            for &is_present in &msg.has_span {
                if !is_present {
                    spans.push(None);
                } else {
                    let Some(&span) = present.get(k) else {
                        return Err(mismatched("span_start", msg.span_start.len(), msg.span_end.len()));
                    };
                    spans.push(Some(span));
                    k += 1;
                }
            }
            if k != present.len() {
                return Err(mismatched("span_start", msg.span_start.len(), k));
            }
            spans
        };
        // The occurrence columns are all empty; or occ_at holds one end
        // per entry and the starts match the ends, the last end one past
        // the last occurrence; or the zero-occurrence form, one offset
        // per entry, all 0.
        if msg.occ_start.len() != msg.occ_end.len() {
            return Err(mismatched("occ_start", msg.occ_start.len(), msg.occ_end.len()));
        }
        let (occ, occ_at) = if msg.occ_start.is_empty() {
            if msg.occ_at.is_empty() {
                (Vec::new(), Vec::new())
            } else {
                if msg.occ_at.len() != n || msg.occ_at.iter().any(|&e| e != 0) {
                    return Err(Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        0,
                        "occ_at: zero-occurrence offsets are 0, one per entry",
                    ));
                }
                (Vec::new(), msg.occ_at.clone())
            }
        } else {
            if msg.occ_at.len() != n {
                return Err(mismatched("occ_at", msg.occ_at.len(), n));
            }
            if !msg.occ_at.windows(2).all(|w| w[0] <= w[1]) || msg.occ_at.last() != Some(&(msg.occ_start.len() as u32))
            {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    "occ_at: occurrence ends in entry order, the last one past the last occurrence",
                ));
            }
            let occ = spans_from_wire("sparse occurrence span", &msg.occ_start, &msg.occ_end, &msg.text, offsets)?;
            (occ, msg.occ_at.clone())
        };
        // The row column: one start per unit and one past the last entry,
        // every entry an index into term. Proved again by check() below,
        // but bounds are checked first so check() cannot slice out of range.
        if msg.row.is_empty() || msg.row[0] != 0 || msg.row.last().is_some_and(|&r| r as usize != n) {
            return Err(Failure::new(
                TURBO_E_INVALID_ARGUMENT,
                0,
                format!("row: one start per unit and one past the last of the {n} entries, the first 0"),
            ));
        }
        if msg.row.iter().any(|&r| r as usize > n) || !msg.row.windows(2).all(|w| w[0] <= w[1]) {
            return Err(Failure::new(TURBO_E_INVALID_ARGUMENT, 0, "row: starts in unit order, each an entry index"));
        }
        let model: ModelRef = msg.model.as_ref().map(model_from_proto).unwrap_or(ModelRef {
            id: String::new(),
            revision: String::new(),
            sha256: String::new(),
        });
        // push and push_occurrences build the span and occurrence columns;
        // nothing is assigned past them.
        let has_occ = !occ_at.is_empty();
        let mut out = SparseTerms::new(model);
        let mut e = 0;
        for w in msg.row.windows(2) {
            for k in w[0] as usize..w[1] as usize {
                out.push(msg.term[k], msg.weight[k], spans.get(e).copied().flatten());
                if has_occ {
                    let lo = if e == 0 { 0 } else { occ_at[e - 1] as usize };
                    out.push_occurrences(&occ[lo..occ_at[e] as usize]);
                }
                e += 1;
            }
            out.unit();
        }
        out.check().map_err(|e| Failure::new(TURBO_E_INVALID_ARGUMENT, 0, format!("sparse terms: {e}")))?;
        Ok(SparseTermsWire { text: msg.text.clone(), offsets, terms: out })
    }
}

impl Wire for SparseTerms<View> {
    type Msg = SparseTermsResult;

    fn to_msg(&self) -> SparseTermsResult {
        sparse_to_msg(self, "", OffsetMode::Utf8)
    }

    fn from_msg(msg: &SparseTermsResult) -> Result<Self, Failure> {
        SparseTermsWire::from_msg(msg).map(|w| w.terms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::TURBO_E_CAPACITY;

    fn model() -> ModelRef {
        ModelRef { id: "org/m".into(), revision: "r1".into(), sha256: "deadbeef".into() }
    }

    /// A document-level instance with spans and occurrences, like the
    /// embed step writes.
    fn sample() -> SparseTerms<View> {
        let mut t = SparseTerms::new(model());
        t.push(3, 0.5, Some(Span::new(0, 4)));
        t.push_occurrences(&[Span::new(0, 4)]);
        t.push(9, 1.25, None);
        t.push_occurrences(&[]);
        t.unit();
        t.push(1, 0.75, Some(Span::new(5, 9)));
        t.push_occurrences(&[Span::new(5, 7), Span::new(7, 9)]);
        t.unit();
        t
    }

    #[test]
    fn the_wire_round_trip_is_unchanged() {
        let t = sample();
        let msg = t.to_msg();
        let back = SparseTerms::from_msg(&msg).unwrap();
        assert_eq!(back.model, t.model);
        assert_eq!((back.row, back.term, back.weight), (t.row.clone(), t.term.clone(), t.weight.clone()));
        assert_eq!(back.span, t.span);
        assert_eq!(back.occ, t.occ);
        assert_eq!(back.occ_at, t.occ_at);
        // The canonical form: an absent span column stays absent, an
        // absent occurrence column stays absent.
        assert_eq!(msg.has_span.len(), msg.term.len());
        assert_eq!(msg.occ_at.len(), msg.occ_at.capacity().min(msg.occ_at.len()));
    }

    #[test]
    fn an_instance_without_spans_or_occurrences_round_trips() {
        let mut t = SparseTerms::new(model());
        t.push(3, 0.5, None);
        t.unit();
        let msg = t.to_msg();
        assert!(msg.has_span.is_empty() && msg.span_start.is_empty() && msg.span_end.is_empty());
        assert!(msg.occ_start.is_empty() && msg.occ_end.is_empty() && msg.occ_at.is_empty());
        let back = SparseTerms::from_msg(&msg).unwrap();
        assert_eq!(back.term, t.term);
        assert_eq!(back.weight, t.weight);
        assert!(back.span.iter().all(Option::is_none) && !back.has_occurrences());
    }

    #[test]
    fn an_empty_instance_is_row_zero_alone() {
        let t = SparseTerms::new(model());
        let msg = t.to_msg();
        assert_eq!(msg.row, [0]);
        let back = SparseTerms::from_msg(&msg).unwrap();
        assert_eq!(back.len(), 0);
    }

    fn refuse(r: Result<SparseTerms<View>, Failure>) -> Failure {
        match r {
            Err(e) => e,
            Ok(_) => panic!("a garbage message was accepted"),
        }
    }

    fn refused(mut f: impl FnMut(&mut SparseTermsResult)) -> Failure {
        let mut msg = sample().to_msg();
        f(&mut msg);
        refuse(SparseTerms::from_msg(&msg))
    }

    #[test]
    fn garbage_is_refused_without_panicking() {
        // Every corruption names what it broke; none panics.
        let e = refused(|m| {
            m.row.clear();
        });
        assert!(e.message.contains("row"), "{e:?}");
        let e = refused(|m| m.row = vec![1, 1]);
        assert!(e.message.contains("row"), "{e:?}");
        let e = refused(|m| m.row = vec![0, 99]);
        assert!(e.message.contains("row"), "{e:?}");
        let e = refused(|m| {
            m.weight.pop();
        });
        assert!(e.message.contains("weight"), "{e:?}");
        let e = refused(|m| {
            m.has_span.pop();
        });
        assert!(e.message.contains("has_span"), "{e:?}");
        let e = refused(|m| {
            m.has_span[1] = true;
        });
        assert!(e.message.contains("span_start"), "{e:?}");
        let e = refused(|m| {
            m.span_start[0] = 9;
        });
        assert!(e.message.contains("sparse terms"), "{e:?}");
        let e = refused(|m| {
            m.occ_at.pop();
        });
        assert!(e.message.contains("occ_at"), "{e:?}");
        let e = refused(|m| {
            m.occ_at[0] = 99;
        });
        assert!(e.message.contains("occ_at"), "{e:?}");
        let e = refused(|m| {
            m.weight[0] = f32::NAN;
        });
        assert!(e.message.contains("non-finite"), "{e:?}");
        let e = refused(|m| {
            m.weight[1] = 0.0;
        });
        assert!(e.message.contains("over 0"), "{e:?}");
        let e = refused(|m| {
            m.term[1] = 3;
        });
        assert!(e.message.contains("ascending"), "{e:?}");
        let e = refused(|m| {
            m.occ_start[0] = 40;
        });
        assert!(e.message.contains("occurrence"), "{e:?}");
        // A wire message built from nothing at all.
        let e = refuse(SparseTerms::from_msg(&SparseTermsResult::default()));
        assert!(e.message.contains("row"), "{e:?}");
    }

    #[test]
    fn occurrence_columns_cannot_drop_occurrences() {
        // The last offset must be one past the last occurrence: with it
        // short, the second occurrence would silently disappear.
        let msg = SparseTermsResult {
            model: Some(model_to_proto(&model())),
            row: vec![0, 1],
            term: vec![1],
            weight: vec![1.0],
            has_span: vec![true],
            span_start: vec![0],
            span_end: vec![100],
            occ_start: vec![0, 99],
            occ_end: vec![1, 100],
            occ_at: vec![1],
            ..Default::default()
        };
        let e = refuse(SparseTerms::from_msg(&msg));
        assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT);
        assert!(e.message.contains("occ_at"), "{e:?}");
        // And exactly at the count it is accepted.
        let msg = SparseTermsResult { occ_at: vec![2], ..msg };
        let back = SparseTerms::from_msg(&msg).unwrap();
        assert_eq!(back.occ, vec![Span::new(0, 1), Span::new(99, 100)]);
    }

    #[test]
    fn the_zero_occurrence_form_round_trips() {
        // One expanded term recording no occurrences carries occ_at
        // [0], one offset per entry, all zero. The layer's own form.
        let mut t = SparseTerms::new(model());
        t.push(5, 1.5, None);
        t.push_occurrences(&[]);
        t.unit();
        t.check().unwrap();
        assert!(t.has_occurrences());
        let msg = t.to_msg();
        assert!(msg.occ_start.is_empty() && msg.occ_end.is_empty());
        assert_eq!(msg.occ_at, [0]);
        let back = SparseTerms::from_msg(&msg).unwrap();
        assert!(back.has_occurrences());
        assert_eq!(back.occ_at, [0]);
        assert!(back.occurrences(0).is_empty());
        assert_eq!(back.check(), Ok(()));
        // Not all zero, or not one per entry, refuses by name.
        let e = refused(|m| {
            m.occ_at[0] = 9;
        });
        assert!(e.message.contains("occ_at"), "{e:?}");
    }

    #[test]
    fn over_the_entry_limit_names_it() {
        let msg = SparseTermsResult {
            row: vec![0, MAX_SPARSE_ENTRIES as u32 + 1],
            term: vec![1; MAX_SPARSE_ENTRIES as usize + 1],
            weight: vec![1.0; MAX_SPARSE_ENTRIES as usize + 1],
            ..Default::default()
        };
        let e = refuse(SparseTerms::from_msg(&msg));
        assert_eq!(e.code, TURBO_E_CAPACITY);
        assert!(e.message.contains("max_sparse_entries"), "{e:?}");
    }
}
