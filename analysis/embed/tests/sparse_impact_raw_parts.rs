//! `QuantizedSparseTerms::from_raw_parts` takes impact lists whose vocab
//! ids rise strictly; the merge joins that compute dot products rely on it.

use turbo::sparse::SparseTerm;
use turbo_analysis_embed::{ImpactBits, ImpactQuantizer, QuantizedSparseTerms};

#[test]
fn from_raw_parts_rejects_unsorted_or_duplicate_vocab_ids() {
    let q = ImpactQuantizer::new(ImpactBits::U8, 2.0).unwrap();
    let valid = q
        .quantize_doc_terms(&[SparseTerm { term: 3, weight: 1.0 }, SparseTerm { term: 10, weight: 0.5 }], None)
        .unwrap();
    let c = valid.certificate();

    // Descending vocab_id must be rejected.
    let err_desc = QuantizedSparseTerms::from_raw_parts(
        vec![(10, 64), (3, 128)],
        vec![0.5, 1.0],
        Vec::new(),
        c.bits,
        c.scale,
        c.w_max,
        c.l1_rounding_bound,
        c.dot_error_bound,
    )
    .unwrap_err();
    assert!(err_desc.to_string().contains("strictly increasing"), "{err_desc}");

    // Duplicate vocab_id must be rejected.
    let err_dup = QuantizedSparseTerms::from_raw_parts(
        vec![(3, 128), (3, 64)],
        vec![1.0, 0.5],
        Vec::new(),
        c.bits,
        c.scale,
        c.w_max,
        c.l1_rounding_bound,
        c.dot_error_bound,
    )
    .unwrap_err();
    assert!(err_dup.to_string().contains("strictly increasing"), "{err_dup}");

    // Strictly increasing vocab_id constructs and dots accurately.
    let rebuilt = QuantizedSparseTerms::from_raw_parts(
        valid.impacts().to_vec(),
        valid.weights().to_vec(),
        Vec::new(),
        c.bits,
        c.scale,
        c.w_max,
        c.l1_rounding_bound,
        c.dot_error_bound,
    )
    .unwrap();
    assert_eq!(rebuilt.dot_int(&valid), valid.dot_int(&valid));
    assert!((rebuilt.dot_float(&valid) - valid.dot_float(&valid)).abs() < 1e-6);
}
