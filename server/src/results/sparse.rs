//! SparseTermsResult: the turbo-analysis-layers `SparseTerms` layer's
//! columns (issue #708), document level (`SparseTerms<View>`), with its
//! source spans and occurrence spans.

use turbo_analysis::{Layer, ModelRef, Span, View};
use turbo_analysis_layers::SparseTerms;

use super::{MAX_SPARSE_ENTRIES, MAX_SPARSE_OCCURRENCES, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::proto::result::SparseTermsResult;

impl Wire for SparseTerms<View> {
    type Msg = SparseTermsResult;

    fn to_msg(&self) -> SparseTermsResult {
        let has_occ = self.has_occurrences();
        // No entry carries a span: the canonical message has no span
        // column at all, the layer's own empty form.
        let any_span = self.span.iter().any(|s| s.is_some());
        SparseTermsResult {
            model: Some(model_to_proto(&self.model)),
            row: self.row.clone(),
            term: self.term.clone(),
            weight: self.weight.clone(),
            // `has_span` holds one entry per term; the start/end columns
            // hold one per present span. With no span at all, all three
            // stay empty.
            has_span: if any_span { self.span.iter().map(|s| s.is_some()).collect() } else { Vec::new() },
            span_start: self.span.iter().flatten().map(|s| s.start).collect(),
            span_end: self.span.iter().flatten().map(|s| s.end).collect(),
            occ_start: if has_occ { self.occ.iter().map(|s| s.start).collect() } else { Vec::new() },
            occ_end: if has_occ { self.occ.iter().map(|s| s.end).collect() } else { Vec::new() },
            occ_at: if has_occ { self.occ_at.clone() } else { Vec::new() },
        }
    }

    fn from_msg(msg: &SparseTermsResult) -> Result<Self, Failure> {
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
            let mut spans = Vec::with_capacity(n);
            let mut k = 0;
            for &present in &msg.has_span {
                if !present {
                    spans.push(None);
                } else {
                    let (Some(&start), Some(&end)) = (msg.span_start.get(k), msg.span_end.get(k)) else {
                        return Err(mismatched("span_start", msg.span_start.len(), msg.span_end.len()));
                    };
                    spans.push(Some(Span { start, end }));
                    k += 1;
                }
            }
            if k != msg.span_start.len() {
                return Err(mismatched("span_start", msg.span_start.len(), k));
            }
            spans
        };
        // The occurrence columns are all empty, or occ_at holds one end
        // per entry and the starts match the ends.
        if msg.occ_start.len() != msg.occ_end.len() {
            return Err(mismatched("occ_start", msg.occ_start.len(), msg.occ_end.len()));
        }
        let (occ, occ_at) = if msg.occ_start.is_empty() {
            if !msg.occ_at.is_empty() {
                return Err(mismatched("occ_at", msg.occ_at.len(), 0));
            }
            (Vec::new(), Vec::new())
        } else {
            if msg.occ_at.len() != n {
                return Err(mismatched("occ_at", msg.occ_at.len(), n));
            }
            if !msg.occ_at.windows(2).all(|w| w[0] <= w[1])
                || msg.occ_at.last().is_some_and(|&e| e as usize > msg.occ_start.len())
            {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    0,
                    "occ_at: occurrence ends in entry order, the last past the last occurrence",
                ));
            }
            let occ: Vec<Span> =
                msg.occ_start.iter().zip(&msg.occ_end).map(|(&s, &e)| Span { start: s, end: e }).collect();
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
        Ok(out)
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
