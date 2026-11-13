use super::*;
use crate::api::TURBO_E_CAPACITY;
use turbo_analysis_embed::TermKeyMode;

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

/// A single-unit document-level instance for impact quantization tests.
fn single_unit_sample() -> SparseTerms<View> {
    let mut t = SparseTerms::new(model());
    t.push(1, 0.25, Some(Span::new(0, 4)));
    t.push_occurrences(&[Span::new(0, 4)]);
    t.push(3, 1.50, Some(Span::new(5, 9)));
    t.push_occurrences(&[Span::new(5, 7), Span::new(7, 9)]);
    t.push(9, 2.75, None);
    t.push_occurrences(&[]);
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
    // absent occurrence column stays absent, and absent impact fields stay empty/0.
    assert_eq!(msg.has_span.len(), msg.term.len());
    assert_eq!(msg.occ_at.len(), msg.occ_at.capacity().min(msg.occ_at.len()));
    assert!(msg.impact.is_empty() && msg.term_key.is_empty() && msg.impact_bits == 0);
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

#[test]
fn quantized_impacts_and_error_certificates_round_trip_and_refuse_corruptions() {
    let t = single_unit_sample();
    for bits in [ImpactBits::U8, ImpactBits::U16] {
        let q = ImpactQuantizer::new(bits, 3.0)
            .unwrap()
            .with_doc_pruning(2, 0.1)
            .unwrap()
            .with_term_key_mode(TermKeyMode::CompactTokenId);
        let wire = SparseTermsWire::quantize_doc(&t, "hello world", OffsetMode::Utf8, &q, None).unwrap();
        let msg = wire.to_msg();
        assert_eq!(msg.term, [3, 9]);
        assert_eq!(msg.impact.len(), 2);
        assert_eq!(msg.impact_bits, bits.bits());
        assert_eq!(msg.term_key, ["#tok_3", "#tok_9"]);
        assert!(msg.impact_scale > 0.0 && msg.impact_w_max == 3.0);
        assert!(msg.impact_dot_error_bound > 0.0);

        let back = SparseTermsWire::from_msg(&msg).unwrap();
        let imp = back.impacts.as_ref().expect("impacts decoded");
        assert_eq!(imp.impacts(), wire.impacts.as_ref().unwrap().impacts());
        assert_eq!(imp.term_keys(), ["#tok_3", "#tok_9"]);
        assert_eq!(imp.certificate().dot_error_bound, msg.impact_dot_error_bound);
        assert_eq!(back.terms.span, [Some(Span::new(5, 9)), None]);
        assert_eq!(back.terms.occ, [Span::new(5, 7), Span::new(7, 9)]);

        // Corruptions on impact fields are refused with TURBO_E_INVALID_ARGUMENT.
        let mut bad = msg.clone();
        bad.impact_bits = 12;
        assert_eq!(SparseTermsWire::from_msg(&bad).err().unwrap().code, TURBO_E_INVALID_ARGUMENT);

        let mut bad = msg.clone();
        bad.impact.pop();
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("impact"));

        let mut bad = msg.clone();
        bad.impact[0] = 0;
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("outside"));

        if bits == ImpactBits::U8 {
            let mut bad = msg.clone();
            bad.impact[0] = 256;
            assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("outside"));
        }

        let mut bad = msg.clone();
        bad.term_key.pop();
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("term_key"));

        let mut bad = msg.clone();
        bad.term_key[0].clear();
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("term_key"));

        let mut bad = msg.clone();
        bad.impact_scale = 0.0;
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("impact_scale"));

        let mut bad = msg;
        bad.impact_dot_error_bound = -1.0;
        assert!(SparseTermsWire::from_msg(&bad).err().unwrap().message.contains("impact_dot_error_bound"));
    }
}
