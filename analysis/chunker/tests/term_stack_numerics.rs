use turbo_analysis::{Ix, Layer, ModelRef};
use turbo_analysis_layers::{Chunks, SparseTerms};
use turbo_chunker::terms::TermVectorStack;

fn sparse(weights: &[f32]) -> SparseTerms<Chunks> {
    let mut s = SparseTerms::new(ModelRef { id: "numerics".into(), revision: "1".into(), sha256: "s".into() });
    for (i, &weight) in weights.iter().enumerate() {
        s.push(i as u32, weight, None);
    }
    s.unit();
    s.check().unwrap();
    s
}

#[test]
fn finite_inputs_cannot_produce_infinite_stack_totals() {
    let s = sparse(&[2.0e38, 2.0e38]);
    for cap in [0, 1, 2] {
        let err = TermVectorStack::from_sparse_terms(&s, Ix::new(0), |_| Some("term"), cap).unwrap_err();
        assert!(err.contains("aggregate term weight"), "{err}");
    }
    let valid = sparse(&[1.0e38, 1.0e38]);
    let stack = TermVectorStack::from_sparse_terms(&valid, Ix::new(0), |_| Some("term"), 1).unwrap();
    assert!(stack.total_weight.is_finite() && stack.other_weight.is_finite());
    assert_eq!(stack.total_weight, 2.0e38);
    assert_eq!(stack.other_weight, 1.0e38);
}

#[test]
fn capped_tail_is_summed_without_subtracting_the_large_head() {
    let s = sparse(&[f32::MAX, 1.0, 0.5]);
    let stack = TermVectorStack::from_sparse_terms(&s, Ix::new(0), |_| Some("term"), 1).unwrap();
    assert_eq!(stack.other_term_count, 2);
    assert_eq!(stack.other_weight, 1.5);
    let all = TermVectorStack::from_sparse_terms(&s, Ix::new(0), |_| Some("term"), 3).unwrap();
    assert_eq!(all.other_weight, 0.0);
    assert_eq!(all.total_weight, f32::MAX);
}
