use super::*;

fn names(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| (*s).to_owned()).collect()
}

#[test]
fn labels_name_their_types_in_order() {
    let (labels, types) = labels(&names(&["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-", "X"]));
    assert_eq!(types, ["MISC", "PER"]);
    assert_eq!(
        labels,
        [
            Label::Outside,
            Label::Begin(0),
            Label::Inside(0),
            Label::Begin(1),
            Label::Inside(1),
            Label::Outside,
            Label::Outside
        ]
    );
}

#[test]
fn chunks_overlap_as_abstract_dl_cuts_them() {
    assert_eq!(chunk_ranges(0, 250, 50).unwrap(), []);
    assert_eq!(chunk_ranges(10, 250, 50).unwrap(), [(0, 10)]);
    assert_eq!(chunk_ranges(600, 250, 50).unwrap(), [(0, 250), (200, 450), (400, 600)]);
    assert_eq!(chunk_ranges(450, 250, 50).unwrap(), [(0, 250), (200, 450)]);
    assert!(chunk_ranges(10, 5, 5).is_err());
    assert!(chunk_ranges(10, 0, 0).is_err());
    assert_eq!(words(" a\u{a0}bb\u{3000} c "), [(1, 2), (4, 6), (10, 11)]);
}

#[test]
fn probabilities_follow_label_probability() {
    let p = label_probability(&[1.0, 2.0, 3.0], 2);
    let e = |x: f64| x.exp();
    assert!((p - e(0.0) / (e(-2.0) + e(-1.0) + e(0.0))).abs() < 1e-15);
    // NaN is left out of both the argmax and the sum.
    assert_eq!(max_index(&[f32::NAN, 1.0, 1.0]), Some(1));
    assert!((label_probability(&[f32::NAN, 0.0, 0.0], 1) - 0.5).abs() < 1e-15);
    // Positive infinities share the probability.
    assert_eq!(label_probability(&[f32::INFINITY, 1.0, f32::INFINITY], 0), 0.5);
    assert_eq!(label_probability(&[f32::INFINITY, 1.0], 1), 0.0);
    // Nothing but negative infinity: every label alike.
    assert_eq!(label_probability(&[f32::NEG_INFINITY; 4], 2), 0.25);
    assert_eq!(max_index(&[f32::NAN, f32::NAN]), None);
}

/// Scores whose argmax is `label` with a margin `m` over the rest.
fn row(n: usize, label: usize, m: f32) -> Vec<f32> {
    (0..n).map(|i| if i == label { m } else { 0.0 }).collect()
}

#[test]
fn begin_and_inside_runs_become_entities() {
    let (labels, _) = labels(&names(&["O", "B-PER", "I-PER", "B-LOC", "I-LOC"]));
    // [CLS] John Smith went to New York [SEP], pieces with their bytes.
    let pieces =
        [None, Some((0, 4)), Some((5, 10)), Some((11, 15)), Some((16, 18)), Some((19, 22)), Some((23, 27)), None];
    let picks = [(0, 1.0), (1, 3.0), (2, 1.0), (0, 2.0), (0, 2.0), (3, 2.0), (4, 2.0), (0, 1.0)];
    let scores: Vec<f32> = picks.iter().flat_map(|&(l, m)| row(5, l, m)).collect();
    let got = decode(&pieces, &scores, &labels).unwrap();
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].start, got[0].end, got[0].kind), (0, 10, 0));
    // The lower of the two pieces' probabilities.
    assert!((got[0].probability - label_probability(&row(5, 2, 1.0), 2)).abs() < 1e-15);
    assert_eq!((got[1].start, got[1].end, got[1].kind), (19, 27, 1));
    // An inside run of another type ends the entity; a lone inside starts nothing.
    let picks = [(1, 2.0), (4, 2.0), (2, 2.0)];
    let scores: Vec<f32> = picks.iter().flat_map(|&(l, m)| row(5, l, m)).collect();
    let got = decode(&pieces[1..4], &scores, &labels).unwrap();
    assert_eq!(got.iter().map(|c| (c.start, c.end)).collect::<Vec<_>>(), [(0, 4)]);
    // A begin on a special piece covers nothing and is skipped.
    let scores: Vec<f32> = [(1, 2.0), (0, 2.0)].iter().flat_map(|&(l, m)| row(5, l, m)).collect();
    assert!(decode(&[None, Some((0, 4))], &scores, &labels).unwrap().is_empty());
    assert!(decode(&[None], &[f32::NAN; 5], &labels).is_err());
    assert!(decode(&[None, None], &[0.0; 5], &labels).is_err());
}

#[test]
fn overlaps_keep_the_longest_then_the_most_probable() {
    let text = "Jos\u{e9} Mar\u{ed}a of Bogot\u{e1} D.C.";
    let c = |start, end, probability| Candidate { start, end, kind: 0, probability };
    // The same entity from two chunks, and a shorter one inside it.
    let got = resolve(text, vec![c(0, 5, 0.9), c(0, 12, 0.6), c(6, 12, 0.99), c(16, 28, 0.7), c(16, 28, 0.8)]);
    assert_eq!(got, [c(0, 12, 0.6), c(16, 28, 0.8)]);
    // Adjacent spans both stay.
    assert_eq!(resolve(text, vec![c(6, 12, 0.5), c(0, 5, 0.5)]), [c(0, 5, 0.5), c(6, 12, 0.5)]);
    // Length is Java's: UTF-16 units, so `Bogotá` (6) is not longer than `D.C.`
    // padded to the same bytes.
    assert_eq!(resolve(text, vec![c(16, 23, 0.5), c(17, 24, 0.9)]), [c(17, 24, 0.9)]);
}
