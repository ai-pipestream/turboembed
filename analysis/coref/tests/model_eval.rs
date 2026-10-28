//! `CorefModel::link_probability` refuses feature arrays that do not
//! match and values that are not finite, and normalizes large finite
//! logits without overflowing to NaN.

use turbo_coref::{APART, CorefModel, LINK};
use turbo_train::{JavaString, WritableModel, write_gis_writable, write_qn_writable};

/// A GIS model with one predicate `x` weighted `link` then `apart`.
struct OneFeature {
    outcomes: [JavaString; 2],
    predicates: [JavaString; 1],
    weights: [f64; 2],
}

impl WritableModel for OneFeature {
    fn outcome_labels(&self) -> &[JavaString] {
        &self.outcomes
    }

    fn pred_labels(&self) -> &[JavaString] {
        &self.predicates
    }

    fn pattern(&self, _: usize) -> &[u32] {
        &[0, 1]
    }

    fn parameters(&self, _: usize) -> &[f64] {
        &self.weights
    }
}

fn model(link: f64, apart: f64) -> CorefModel {
    let m = OneFeature {
        outcomes: [JavaString::new(LINK), JavaString::new(APART)],
        predicates: [JavaString::new("x")],
        weights: [link, apart],
    };
    CorefModel::package("eng", write_gis_writable(&m).unwrap(), false, 0, None, 0).unwrap()
}

/// The same one-predicate model as a QN model.
fn qn_model(link: f64, apart: f64) -> CorefModel {
    let m = OneFeature {
        outcomes: [JavaString::new(LINK), JavaString::new(APART)],
        predicates: [JavaString::new("x")],
        weights: [link, apart],
    };
    CorefModel::package("eng", write_qn_writable(&m).unwrap(), false, 0, None, 0).unwrap()
}

fn x() -> Vec<String> {
    vec!["x".to_owned()]
}

#[test]
fn moderate_weights_give_the_softmax() {
    let m = model(1.0, 0.0);
    let e = std::f64::consts::E;
    let p = m.link_probability(&x(), Some(&[1.0])).unwrap();
    assert!((p - e / (e + 1.0)).abs() < 1e-12, "{p}");
    let binary = m.link_probability(&x(), None).unwrap();
    assert!((binary - p).abs() < 1e-12, "{binary} {p}");
    // An unknown feature leaves the uniform prior.
    let p = m.link_probability(&["y".to_owned()], Some(&[1.0])).unwrap();
    assert!((p - 0.5).abs() < 1e-12, "{p}");
}

#[test]
fn values_must_match_the_names() {
    let m = model(1.0, 0.0);
    assert!(m.link_probability(&x(), Some(&[])).is_err());
    assert!(m.link_probability(&x(), Some(&[1.0, 2.0])).is_err());
}

#[test]
fn values_must_be_finite() {
    let m = model(1.0, 0.0);
    for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(m.link_probability(&x(), Some(&[v])).is_err(), "{v}");
    }
}

#[test]
fn large_logits_do_not_overflow() {
    let m = model(1000.0, 0.0);
    let p = m.link_probability(&x(), Some(&[1.0])).unwrap();
    assert_eq!(p, 1.0);
    let p = m.link_probability(&x(), None).unwrap();
    assert_eq!(p, 1.0);
    let p = m.link_probability(&x(), Some(&[-1.0])).unwrap();
    assert!(p.is_finite() && p < 1e-300, "{p}");
    let huge = model(f64::MAX, 0.0);
    match huge.link_probability(&x(), Some(&[2.0])) {
        Ok(p) => assert!(p.is_finite() && (0.0..=1.0).contains(&p), "{p}"),
        Err(e) => assert!(e.contains("not finite"), "{e}"),
    }
}

#[test]
fn moderate_qn_weights_give_the_softmax() {
    let m = qn_model(1.0, 0.0);
    let e = std::f64::consts::E;
    let p = m.link_probability(&x(), Some(&[1.0])).unwrap();
    assert!((p - e / (e + 1.0)).abs() < 1e-12, "{p}");
    let binary = m.link_probability(&x(), None).unwrap();
    assert!((binary - p).abs() < 1e-12, "{binary} {p}");
}

#[test]
fn large_equal_qn_logits_remain_uniform() {
    // max + ln(sum) rounds back to max at this magnitude, so a QN
    // normalization through the log-sum-exp would leave every outcome
    // at exp(0) = 1 instead of 1 / 2.
    let m = qn_model(1e300, 1e300);
    let p = m.link_probability(&x(), Some(&[1.0])).unwrap();
    assert!((p - 0.5).abs() < 1e-12, "{p}");
    let binary = m.link_probability(&x(), None).unwrap();
    assert!((binary - 0.5).abs() < 1e-12, "{binary}");
}
