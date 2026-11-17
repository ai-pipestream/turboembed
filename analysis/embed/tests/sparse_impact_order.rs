//! `QuantizedSparseTerms::from_raw_parts` holds the order its dot products
//! rest on: `dot_int` and `dot_float` are merge joins over term ids, so the
//! ids must rise strictly. A list out of order or with a repeated id used to
//! be taken and then gave a dot product that was too small (`#2343`).

use turbo_analysis_embed::{ImpactBits, QuantizedSparseTerms};

/// Two terms of weight 1.0 at 8 bits and `w_max` 1.0, in the order given.
fn raw(ids: &[u32]) -> Result<QuantizedSparseTerms, turbo::status::Error> {
    let impacts = ids.iter().map(|&id| (id, 255)).collect();
    let weights = vec![1.0; ids.len()];
    QuantizedSparseTerms::from_raw_parts(impacts, weights, Vec::new(), ImpactBits::U8, 255.0, 1.0, 1.0, 1.0)
}

#[test]
fn rising_term_ids_build_and_give_the_whole_dot_product() {
    let sorted = raw(&[1, 2]).unwrap();
    assert_eq!(sorted.dot_int(&sorted), 2 * 255 * 255);
    assert_eq!(sorted.dot_float(&sorted), 2.0);
    assert_eq!(sorted.dot_rescaled(&sorted), 2.0);
    assert!(raw(&[]).unwrap().is_empty());
    assert_eq!(raw(&[7]).unwrap().len(), 1);
}

#[test]
fn falling_term_ids_are_refused() {
    let err = raw(&[2, 1]).unwrap_err();
    assert_eq!(err.code, turbo::status::INVALID_ARGUMENT);
    assert!(err.message.contains("impact[1]") && err.message.contains("term 1"), "{}", err.message);
    assert!(raw(&[1, 3, 2]).is_err());
}

#[test]
fn a_repeated_term_id_is_refused() {
    let err = raw(&[4, 4]).unwrap_err();
    assert_eq!(err.code, turbo::status::INVALID_ARGUMENT);
    assert!(err.message.contains("term 4"), "{}", err.message);
    assert!(raw(&[1, 4, 4, 9]).is_err());
}
