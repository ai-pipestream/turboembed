use super::*;

#[test]
fn qn_sessions_carry_reset_and_do_not_change_native_search() {
    let golden = read_golden("ner-qn-fixture");
    let file = open(&fixtures(), &golden);
    let model = file.maxent(namefind::MODEL_ENTRY_NAME).unwrap();
    assert_eq!(model.kind(), turbo_maxent::ModelKind::Qn);
    let search = BeamSearch::<str>::new(golden.beam, &model).unwrap();
    let codec = NameCodec::for_model(&file);
    let input = tokens(&golden.sentences[0]);
    let context = || DefaultNameContext::for_model(&file, PatternMode::Utf16Units).unwrap();
    let bits = |s: Sequence<'_>| sequences(&[s]);
    let native = bits(search.best_sequence(&input, &mut context(), &codec).unwrap());
    let mut a = search.opennlp();
    let mut b = search.opennlp();
    let first = bits(a.best_sequence(&input, &mut context(), &codec).unwrap());
    assert_eq!(first, bits(b.best_sequence(&input, &mut context(), &codec).unwrap()));
    assert_eq!(first[0], golden.sentences[0].sequences[0]);
    assert_ne!(first, native, "QN carry already affects later tokens in the first sentence");
    let empty = a.best_sequence::<&str>(&[], &mut context(), &codec).unwrap();
    assert!(empty.outcomes().is_empty());
    let second = bits(a.best_sequence(&input, &mut context(), &codec).unwrap());
    assert_eq!(second, bits(b.best_sequence(&input, &mut context(), &codec).unwrap()));
    assert_ne!(first, second, "state must persist across calls");
    a.reset();
    assert_eq!(first, bits(a.best_sequence(&input, &mut context(), &codec).unwrap()));
    assert_eq!(native, bits(search.best_sequence(&input, &mut context(), &codec).unwrap()));
    let (scored, _) = search.best_sequence_scored(&input, &mut context(), &codec).unwrap();
    assert_eq!(native, bits(scored));

    a.reset();
    b.reset();
    assert!(a.best_sequences(0, &input, BeamSearch::<str>::DEFAULT_MIN_SCORE, &mut context(), &codec).is_empty());
    b.best_sequence(&input, &mut context(), &codec).unwrap();
    assert_eq!(
        bits(a.best_sequence(&input, &mut context(), &codec).unwrap()),
        bits(b.best_sequence(&input, &mut context(), &codec).unwrap()),
        "asking for zero results still runs the evaluations"
    );
}

#[test]
fn seeded_qn_evaluation_uses_the_callers_buffer_and_checks_its_size() {
    let golden = read_golden("ner-qn-fixture");
    let file = open(&fixtures(), &golden);
    let model = file.maxent(namefind::MODEL_ENTRY_NAME).unwrap();
    let mut context = DefaultNameContext::for_model(&file, PatternMode::Utf16Units).unwrap();
    let mut features = Features::new();
    context.context(0, &tokens(&golden.sentences[0]), &[], &mut features);
    let mut fresh = vec![0.0; model.num_outcomes()];
    let mut seeded = fresh.clone();
    model.eval_features(features.iter(), &mut fresh);
    model.eval_features_seeded(features.iter(), &mut seeded);
    assert_eq!(fresh, seeded);
    model.eval_features_seeded(features.iter(), &mut seeded);
    assert_ne!(fresh, seeded);
    model.eval_features(features.iter(), &mut seeded);
    assert_eq!(fresh, seeded, "native evaluation overwrites a previously seeded output");
    assert!(std::panic::catch_unwind(|| model.eval_features_seeded(features.iter(), &mut [])).is_err());
}
