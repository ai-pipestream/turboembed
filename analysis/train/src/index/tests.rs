use super::*;

fn event(outcome: &str, context: &str) -> Event {
    Event::from_strs(outcome, &context.split(' ').filter(|c| !c.is_empty()).collect::<Vec<_>>())
}

fn valued(outcome: &str, context: &str, values: &[f32]) -> Event {
    let context: Vec<JavaString> = context.split(' ').map(JavaString::new).collect();
    Event::with_values(outcome, context, Some(values.to_vec())).unwrap()
}

/// The predicate labels as text.
fn labels(ix: &DataIndex) -> Vec<String> {
    ix.pred_labels().iter().map(|l| l.to_string_lossy()).collect()
}

fn params(kind: IndexerKind, cutoff: i32) -> IndexParams {
    IndexParams::new(kind, cutoff, true)
}

#[test]
fn predicates_are_counted_cut_and_sorted_in_java_order() {
    let events = [event("s", "b a a"), event("n", "Z a 😀 z"), event("s", "b")];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, 2)).unwrap();
    // "a" occurs three times (twice in one event), "b" twice, the rest once.
    assert_eq!(labels(&ix), ["a", "b"]);
    assert_eq!(ix.pred_counts(), [3, 2]);
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, 0)).unwrap();
    // Java's order: upper case before lower case, and the emoji's
    // surrogates (U+D83D) before "z" (U+007A) is false: 'z' < U+D83D.
    assert_eq!(labels(&ix), ["Z", "a", "b", "z", "😀"]);
    assert_eq!(ix.pred_counts(), [1, 3, 2, 1, 1]);
    // A cutoff below zero keeps everything too.
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, -7)).unwrap();
    assert_eq!(ix.pred_labels().len(), 5);
}

#[test]
fn outcomes_are_numbered_in_order_of_first_sight_dropped_events_included() {
    let events = [event("n", "rare"), event("s", "a"), event("x", "a"), event("s", "a b")];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, 2)).unwrap();
    // "rare" is under the cutoff, so the first event is dropped, but
    // its outcome "n" keeps number 0: OpenNLP's quirk.
    assert_eq!(ix.outcome_labels(), ["n", "s", "x"]);
    assert_eq!(ix.num_events(), 3);
    assert_eq!(labels(&ix), ["a"]);
    // "s a" and "s a b" merge once "b" is cut: outcome 1 seen twice.
    assert_eq!(ix.num_unique_events(), 2);
    assert_eq!(ix.outcomes(), [1, 2]);
    assert_eq!(ix.times_seen(), [2, 1]);
    assert_eq!(ix.context(0), [0]);
}

#[test]
fn events_sort_by_outcome_then_predicates_then_length() {
    let events = [event("b", "q p"), event("a", "q"), event("a", "p q"), event("a", "p"), event("a", "p q")];
    let ix = DataIndex::index(&events, &params(IndexerKind::TwoPass, 0)).unwrap();
    assert_eq!(ix.outcome_labels(), ["b", "a"]);
    assert_eq!(labels(&ix), ["p", "q"]);
    // Outcome 0 ("b") first; within outcome 1: [0], [0 1] x2, [1].
    assert_eq!(ix.outcomes(), [0, 1, 1, 1]);
    assert_eq!(ix.context(0), [1, 0]);
    assert_eq!(ix.context(1), [0]);
    assert_eq!(ix.context(2), [0, 1]);
    assert_eq!(ix.context(3), [1]);
    assert_eq!(ix.times_seen(), [1, 1, 2, 1]);
    assert!(!ix.has_values());
    assert_eq!(ix.values(0), None);
}

#[test]
fn duplicate_predicates_in_one_event_are_kept() {
    let events = [event("s", "a a b"), event("s", "a b"), event("n", "b")];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, 0)).unwrap();
    assert_eq!(ix.pred_counts(), [3, 3]);
    // [0 0 1] sorts before [0 1]: the second ids differ, 0 before 1.
    assert_eq!(ix.context(0), [0, 0, 1]);
    assert_eq!(ix.context(1), [0, 1]);
    assert_eq!(ix.times_seen(), [1, 1, 1]);
}

#[test]
fn unsorted_events_keep_their_order_and_are_not_merged() {
    let events = [event("s", "a"), event("n", "a"), event("s", "a")];
    let ix = DataIndex::index(&events, &IndexParams::new(IndexerKind::OnePass, 0, false)).unwrap();
    assert_eq!(ix.num_unique_events(), 3);
    assert_eq!(ix.outcomes(), [0, 1, 0]);
    assert_eq!(ix.times_seen(), [1, 1, 1]);
}

#[test]
fn no_surviving_event_is_insufficient_data() {
    let events = [event("s", "a"), event("n", "b")];
    for sort in [true, false] {
        let p = IndexParams::new(IndexerKind::OnePass, 2, sort);
        assert_eq!(DataIndex::index(&events, &p), Err(Error::InsufficientTrainingData(NO_EVENTS)));
        assert_eq!(DataIndex::index(&[], &p), Err(Error::InsufficientTrainingData(NO_EVENTS)));
    }
    // An event with an empty context is dropped too.
    assert_eq!(
        DataIndex::index(&[event("s", "")], &params(IndexerKind::OnePass, 0)),
        Err(Error::InsufficientTrainingData(NO_EVENTS))
    );
}

#[test]
fn values_order_events_and_are_kept_by_the_real_value_indexer() {
    let events = [valued("s", "a b", &[2.0, 1.0]), valued("s", "a b", &[1.0, 1.0]), event("s", "a b")];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePassRealValue, 0)).unwrap();
    // The second and third events are equal: a missing value counts 1.0.
    assert_eq!(ix.num_unique_events(), 2);
    assert_eq!(ix.times_seen(), [2, 1]);
    assert!(ix.has_values());
    // The merged event is the first of its run in the stable sort: the
    // one with values, so they are kept.
    assert_eq!(ix.values(0), Some(&[1.0f32, 1.0][..]));
    assert_eq!(ix.values(1), Some(&[2.0f32, 1.0][..]));
    assert_eq!(ix.values(2), None);
    // The other indexers drop them, but sort by them all the same.
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePass, 0)).unwrap();
    assert!(!ix.has_values());
    assert_eq!(ix.times_seen(), [2, 1]);
    assert_eq!(ix.values(2), None);
}

#[test]
fn a_merged_event_without_values_leaves_none() {
    let events = [event("s", "a"), valued("s", "a", &[1.0])];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePassRealValue, 0)).unwrap();
    assert_eq!(ix.num_unique_events(), 1);
    assert_eq!(ix.values(0), None);
}

#[test]
fn values_are_read_at_filtered_positions_after_a_cutoff() {
    // "x" is cut; the surviving predicate "b" is compared by the value
    // at position 0, which is x's value. OpenNLP's quirk: 3.0 vs 1.0
    // keeps the two events apart although b's values are equal.
    let events = [valued("s", "x b", &[3.0, 1.0]), valued("s", "b", &[1.0]), event("s", "b")];
    let ix = DataIndex::index(&events, &params(IndexerKind::OnePassRealValue, 2)).unwrap();
    assert_eq!(labels(&ix), ["b"]);
    assert_eq!(ix.num_unique_events(), 2);
    assert_eq!(ix.times_seen(), [2, 1]);
    assert_eq!(ix.values(0), Some(&[1.0f32][..]));
    // The kept array is the full one, two values for one predicate.
    assert_eq!(ix.values(1), Some(&[3.0f32, 1.0][..]));
}

#[test]
fn float_compare_is_javas() {
    assert_eq!(float_compare(1.0, 2.0), Ordering::Less);
    assert_eq!(float_compare(-0.0, 0.0), Ordering::Less);
    assert_eq!(float_compare(0.0, 0.0), Ordering::Equal);
    assert_eq!(float_compare(f32::NAN, f32::INFINITY), Ordering::Greater);
    assert_eq!(float_compare(f32::NAN, f32::from_bits(0x7f80_0001)), Ordering::Equal);
    assert_eq!(float_compare(f32::NEG_INFINITY, f32::MIN), Ordering::Less);
}

#[test]
fn two_pass_refuses_strings_write_utf_cannot_write() {
    let long = "é".repeat(32768); // 65536 bytes of modified UTF-8
    let events = [Event::from_strs("s", &[&long, "a"]), event("n", "a")];
    assert_eq!(
        DataIndex::index(&events, &params(IndexerKind::TwoPass, 0)),
        Err(Error::StringTooLong { what: "context", bytes: 65536 })
    );
    assert_eq!(
        DataIndex::index(&[Event::from_strs(long.clone(), &["a"])], &params(IndexerKind::TwoPass, 0)),
        Err(Error::StringTooLong { what: "outcome", bytes: 65536 })
    );
    // Exactly 65535 bytes is fine, and OnePass has no limit.
    let fits = format!("{}a", "é".repeat(32767));
    assert_eq!(JavaString::new(&fits).len(), 65535);
    assert!(DataIndex::index(&[Event::from_strs("s", &[&fits])], &params(IndexerKind::TwoPass, 0)).is_ok());
    assert!(DataIndex::index(&events, &params(IndexerKind::OnePass, 0)).is_ok());
}

#[test]
fn predicates_with_lone_surrogates_are_indexed_and_sorted_by_units() {
    let half = JavaString::from_utf16(&[0x70, 0x3D, 0xD83D]);
    let events = [
        Event::new("T", vec![half.clone(), JavaString::new("p=z")]),
        Event::new("F", vec![half.clone(), JavaString::new("p=\u{FFFF}")]),
    ];
    let ix = DataIndex::index(&events, &params(IndexerKind::TwoPass, 0)).unwrap();
    // "p=z" (0x7A) before the lone high surrogate (0xD83D) before U+FFFF.
    assert_eq!(ix.pred_labels(), [JavaString::new("p=z"), half, JavaString::new("p=\u{FFFF}")]);
    assert_eq!(ix.context(0), [1, 0]);
    assert_eq!(ix.context(1), [1, 2]);
}

#[test]
fn outcomes_with_lone_surrogates_are_numbered_by_their_units() {
    // Two lemma classes that differ only in which half of the pair they
    // delete, and the same class seen again.
    let high = JavaString::from_utf16(&[0x44, 0x30, 0xD83D]);
    let low = JavaString::from_utf16(&[0x44, 0x30, 0xDE00]);
    let events = [
        Event::new(high.clone(), vec![JavaString::new("a")]),
        Event::new(low.clone(), vec![JavaString::new("a")]),
        Event::new(high.clone(), vec![JavaString::new("a")]),
    ];
    let ix = DataIndex::index(&events, &params(IndexerKind::TwoPass, 0)).unwrap();
    assert_eq!(ix.outcome_labels(), [high, low]);
    assert_eq!(ix.outcomes(), [0, 1]);
    assert_eq!(ix.times_seen(), [2, 1]);
}

#[test]
fn index_params_come_from_the_training_parameters() {
    let p = TrainingParameters::new();
    assert_eq!(IndexParams::from_training_parameters(&p).unwrap(), IndexParams::default());
    assert_eq!(IndexerKind::default(), IndexerKind::TwoPass);
    let mut p = TrainingParameters::new();
    p.put_str(Key::DATA_INDEXER, "OnePassRealValue");
    p.put_str(Key::CUTOFF, "3");
    p.put_bool(Key::SORT, false);
    assert_eq!(
        IndexParams::from_training_parameters(&p).unwrap(),
        IndexParams::new(IndexerKind::OnePassRealValue, 3, false)
    );
    p.put_str(Key::DATA_INDEXER, "onepass");
    assert_eq!(IndexParams::from_training_parameters(&p), Err(Error::UnknownIndexer("onepass".into())));
}
