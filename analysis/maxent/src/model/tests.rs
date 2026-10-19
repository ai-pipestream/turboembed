use super::*;
use turbo_jvm::strict_math::{exp, log};

/// A model with outcomes a, b, c and predicates p (a: 1, b: 2),
/// q (c: -0.5), and two whose modified UTF-8 differs from UTF-8:
/// "n\0" (a: 3) and "e\u{1F600}" (b: 3).
fn model(kind: ModelKind) -> MaxentModel {
    let mut predicates = HashMap::default();
    predicates.insert(Box::from(&b"p"[..]), 0);
    predicates.insert(Box::from(&b"q"[..]), 1);
    predicates.insert(Box::from(&b"n\xC0\x80"[..]), 2);
    predicates.insert(Box::from(&b"e\xED\xA0\xBD\xED\xB8\x80"[..]), 3);
    let outcomes = vec![JavaString::new("a"), JavaString::new("b"), JavaString::new("c")];
    MaxentModel {
        kind,
        outcome_texts: outcome_texts(&outcomes),
        outcomes,
        predicates,
        pred_start: vec![0, 2, 3, 4, 5],
        pred_outcome: vec![0, 1, 2, 0, 1],
        pred_param: vec![1.0, 2.0, -0.5, 3.0, 3.0],
        start: if kind == ModelKind::Gis { log(1.0 / 3.0) } else { 0.0 },
        outcome_totals: if kind == ModelKind::NaiveBayes { vec![4.0, 5.0, -0.5] } else { Vec::new() },
        vocabulary: if kind == ModelKind::NaiveBayes { 4.0 } else { 0.0 },
    }
}

fn eval(m: &MaxentModel, context: &[&str]) -> Vec<f64> {
    let mut out = vec![f64::NAN; m.num_outcomes()];
    m.eval(context, &mut out);
    let mut bytes = vec![f64::NAN; m.num_outcomes()];
    let encoded: Vec<Vec<u8>> = context
        .iter()
        .map(|f| {
            let mut v = Vec::new();
            push_str_modified_utf8(&mut v, f);
            v
        })
        .collect();
    let encoded: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
    m.eval_modified_utf8(&encoded, &mut bytes);
    assert_eq!(out, bytes, "eval and eval_modified_utf8 disagree on {context:?}");
    out
}

#[test]
fn predicates_are_listed_as_modified_utf8() {
    let m = model(ModelKind::Gis);
    let mut names: Vec<&[u8]> = m.predicates().collect();
    names.sort();
    assert_eq!(names, [&b"e\xED\xA0\xBD\xED\xB8\x80"[..], b"n\xC0\x80", b"p", b"q"]);
    assert!(names.iter().all(|n| m.has_predicate_modified_utf8(n)));
}

#[test]
fn gis_adds_the_uniform_prior_then_normalizes() {
    let got = eval(&model(ModelKind::Gis), &["p", "q", "unknown"]);
    let r = log(1.0 / 3.0);
    let e = [exp(r + 1.0), exp(r + 2.0), exp(r - 0.5)];
    let s = e[0] + e[1] + e[2];
    assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
}

#[test]
fn qn_subtracts_the_log_sum_of_exps() {
    let got = eval(&model(ModelKind::Qn), &["p", "p"]);
    let x = [2.0f64, 4.0, 0.0];
    let lse = 4.0 + log(exp(2.0 - 4.0) + exp(0.0) + exp(0.0 - 4.0));
    assert_eq!(got, x.iter().map(|v| exp(v - lse)).collect::<Vec<_>>());
}

#[test]
fn perceptron_scales_by_the_largest_magnitude_above_one() {
    let got = eval(&model(ModelKind::Perceptron), &["p", "q"]);
    let e = [exp(1.0 / 2.0), exp(2.0 / 2.0), exp(-0.5 / 2.0)];
    let s = e[0] + e[1] + e[2];
    assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
    // Sums all within [-1, 1] are not scaled.
    let got = eval(&model(ModelKind::Perceptron), &["q"]);
    let e = [exp(0.0), exp(0.0), exp(-0.5)];
    let s = e[0] + e[1] + e[2];
    assert_eq!(got, vec![e[0] / s, e[1] / s, e[2] / s]);
}

#[test]
fn naive_bayes_smooths_until_the_pattern_ends_then_adds_the_priors() {
    // p has parameters for a and b only, so `NaiveBayesModel.eval` stops
    // before c and c gets no smoothed term. c's total is negative, so its
    // prior's log is NaN and its probability is 0.
    let got = eval(&model(ModelKind::NaiveBayes), &["p", "unknown"]);
    let total = 4.0 + 5.0 + -0.5;
    let a = log((1.0 + 0.05) / (4.0 + 0.05 * 4.0)) + log(4.0 / total);
    let b = log((2.0 + 0.05) / (5.0 + 0.05 * 4.0)) + log(5.0 / total);
    let max = a.max(b);
    let (ea, eb) = (exp(a - max), exp(b - max));
    let s = ea + eb;
    assert_eq!(got, vec![ea / s, eb / s, 0.0]);
    // q has a parameter for c only: a and b get the smoothed zero count.
    let got = eval(&model(ModelKind::NaiveBayes), &["q"]);
    let a = log(0.05 / (4.0 + 0.05 * 4.0)) + log(4.0 / total);
    let b = log(0.05 / (5.0 + 0.05 * 4.0)) + log(5.0 / total);
    let max = a.max(b);
    let (ea, eb) = (exp(a - max), exp(b - max));
    let s = ea + eb;
    assert_eq!(got, vec![ea / s, eb / s, 0.0]);
}

#[test]
fn an_empty_context_gives_uniform_probabilities() {
    for kind in [ModelKind::Gis, ModelKind::Qn, ModelKind::Perceptron] {
        let got = eval(&model(kind), &[]);
        assert!(got.iter().all(|&p| (p - 1.0 / 3.0).abs() < 1e-15), "{kind:?}: {got:?}");
    }
}

#[test]
fn str_features_are_looked_up_as_modified_utf8() {
    let m = model(ModelKind::Qn);
    let x = [3.0f64, 3.0, 0.0];
    let lse = 3.0 + log(exp(0.0) + exp(0.0) + exp(0.0 - 3.0));
    let want: Vec<f64> = x.iter().map(|v| exp(v - lse)).collect();
    assert_eq!(eval(&m, &["n\0", "e\u{1F600}"]), want);

    // The same features given as plain UTF-8 bytes match nothing.
    let mut out = [0.0; 3];
    m.eval_modified_utf8(&[b"n\0", "e\u{1F600}".as_bytes()], &mut out);
    let mut empty = [0.0; 3];
    m.eval_modified_utf8(&[], &mut empty);
    assert_eq!(out, empty);

    assert!(m.has_predicate("n\0"));
    assert!(m.has_predicate("e\u{1F600}"));
    assert!(m.has_predicate_modified_utf8(b"n\xC0\x80"));
    assert!(!m.has_predicate_modified_utf8(b"n\0"));
}

#[test]
fn argmax_keeps_the_first_maximum() {
    assert_eq!(best_outcome(&[0.2, 0.4, 0.4]), 1);
    assert_eq!(best_outcome(&[0.5, 0.1, 0.4]), 0);
    assert_eq!(best_outcome(&[f64::NAN, 0.1, 0.4]), 0);
}

#[test]
fn outcome_lookups() {
    let mut m = model(ModelKind::Qn);
    assert_eq!(m.outcome_index("b"), Some(1));
    assert_eq!(m.outcome_index(&JavaString::new("b")), Some(1));
    assert_eq!(m.outcome_index("z"), None);
    assert_eq!(*m.outcome(2), "c");
    assert_eq!(m.outcomes().len(), 3);
    assert_eq!(m.outcome_texts().unwrap(), ["a", "b", "c"]);
    assert_eq!(m.outcome_str(1), Some("b"));
    // An outcome holding half of a surrogate pair is not text; one
    // holding a whole character outside the BMP is.
    let half = JavaString::from_utf16(&[0x44, 0x30, 0xD83D]);
    m.outcomes[1] = half.clone();
    m.outcomes[2] = JavaString::new("😀");
    m.outcome_texts = outcome_texts(&m.outcomes);
    assert_eq!(m.outcome_texts(), Err(&half));
    assert_eq!(m.outcome_str(1), None);
    assert_eq!(m.outcome_str(2), Some("😀"));
    assert_eq!(m.outcome_index("😀"), Some(2));
    assert_eq!(m.outcome_index(&half), Some(1));
    assert_eq!(m.outcome_index("D0\u{FFFD}"), None);
    assert_eq!(m.num_predicates(), 4);
    assert_eq!(m.num_parameters(), 5);
    assert!(m.has_predicate("q"));
    assert!(!m.has_predicate("r"));
}

#[test]
#[should_panic(expected = "one probability per outcome")]
fn a_wrong_output_length_panics() {
    model(ModelKind::Gis).eval(&[], &mut [0.0; 2]);
}

#[test]
#[should_panic(expected = "one probability per outcome")]
fn a_wrong_output_length_panics_for_bytes() {
    model(ModelKind::Gis).eval_modified_utf8(&[], &mut [0.0; 2]);
}
