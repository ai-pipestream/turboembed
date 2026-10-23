use std::collections::HashMap;

use super::*;
use crate::neural::bilstm_model::char_vocabulary_from_strs;

fn tiny_corpus() -> Vec<PosSample> {
    vec![
        PosSample::new(["The", "cat", "sat"], ["D", "N", "V"]).unwrap(),
        PosSample::new(["A", "dog", "ran"], ["D", "N", "V"]).unwrap(),
    ]
}

fn usage_corpus() -> Vec<PosSample> {
    let distinct = vec![
        PosSample::new(["the", "dog", "barks"], ["DT", "NN", "VBZ"]).unwrap(),
        PosSample::new(["the", "cat", "sleeps"], ["DT", "NN", "VBZ"]).unwrap(),
        PosSample::new(["a", "bird", "sings"], ["DT", "NN", "VBZ"]).unwrap(),
        PosSample::new(["dogs", "bark"], ["NNS", "VBP"]).unwrap(),
        PosSample::new(["cats", "sleep"], ["NNS", "VBP"]).unwrap(),
    ];
    let mut out = Vec::with_capacity(40 * distinct.len());
    for _ in 0..40 {
        out.extend_from_slice(&distinct);
    }
    out
}

#[test]
fn serialization_round_trip_preserves_scores_pretrained_and_adapter() {
    let settings =
        BilstmSettings::new(8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap();
    let model = BilstmPosTrainer::train(&tiny_corpus(), settings.clone()).unwrap();
    let mut buf = Vec::new();
    model.serialize(&mut buf).unwrap();
    let loaded = BilstmPosModel::load(&buf[..]).unwrap();
    assert_eq!(model.tags(), loaded.tags());
    let sent = ["The", "dog", "sat"];
    assert_eq!(model.score(&sent).unwrap(), loaded.score(&sent).unwrap());

    // Pretrained table + unseen lexicon word
    let vec_fn = |w: &str| Some(vec![w.len() as f32, 1.0]);
    let extra = vec!["unseen".to_owned()];
    let model_p =
        BilstmPosTrainer::train_with_vectors(&tiny_corpus(), settings, Some(vec_fn), Some(&extra), None).unwrap();
    buf.clear();
    model_p.serialize(&mut buf).unwrap();
    let loaded_p = BilstmPosModel::load(&buf[..]).unwrap();
    assert_eq!(model_p.word_representation("unseen").unwrap(), loaded_p.word_representation("unseen").unwrap());

    // Adapter model (ONLP-BLPT-5)
    let adapter_settings =
        BilstmSettings::new(8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, true)
            .unwrap();
    let model_a =
        BilstmPosTrainer::train_with_vectors(&tiny_corpus(), adapter_settings, Some(vec_fn), Some(&extra), None)
            .unwrap();
    buf.clear();
    model_a.serialize(&mut buf).unwrap();
    assert_eq!(&buf[2..2 + buf[1] as usize], b"ONLP-BLPT-5");
    let loaded_a = BilstmPosModel::load(&buf[..]).unwrap();
    let sent4 = ["The", "dog", "sat", "unseen"];
    assert_eq!(model_a.score(&sent4).unwrap(), loaded_a.score(&sent4).unwrap());
}

#[test]
fn serialization_round_trip_preserves_one_and_two_layer_crf() {
    for layers in [1, 2] {
        let settings = BilstmSettings::new(
            8, 4, 4, 8, 3, 2, 5e-3, 5.0, 0.1, 1, 12, 7, 2, 0.0, 0, true, layers, 0.0, 0.0, 1.0, 0.0, false,
        )
        .unwrap();
        let model = BilstmPosTrainer::train(&tiny_corpus(), settings).unwrap();
        let mut buf = Vec::new();
        model.serialize(&mut buf).unwrap();
        let expected_magic = if layers == 1 { b"ONLP-BLPT-2" } else { b"ONLP-BLPT-4" };
        assert_eq!(&buf[2..2 + buf[1] as usize], expected_magic);
        let loaded = BilstmPosModel::load(&buf[..]).unwrap();
        assert!(loaded.is_crf());
        let sent = ["The", "dog", "sat"];
        assert_eq!(BilstmPosTagger::new(model).tag(&sent).unwrap(), BilstmPosTagger::new(loaded).tag(&sent).unwrap());
    }
}

#[test]
fn usage_and_edge_cases_match_java_deterministic_outputs() {
    let settings = BilstmSettings::new(
        16, 8, 8, 16, 60, 8, 0.05, 5.0, 0.0, 1, 16, 17, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false,
    )
    .unwrap();
    let model = BilstmPosTrainer::train(&usage_corpus(), settings).unwrap();
    let tagger = BilstmPosTagger::jvm_parity(model.clone());

    assert_eq!(tagger.tag(&["the", "cat", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
    assert_eq!(tagger.tag(&["dogs", "bark"]).unwrap(), vec!["NNS", "VBP"]);
    assert_eq!(tagger.tag(&["dog"]).unwrap(), vec!["VBZ"]);
    let packed = BilstmPosTagger::new(model.clone());
    for s in [&["the", "cat", "barks"][..], &["dogs", "bark"], &["dog"]] {
        assert_eq!(packed.tag(s).unwrap(), tagger.tag(s).unwrap(), "{s:?}");
    }
    assert!(tagger.tag(&Vec::<&str>::new()).unwrap().is_empty());

    let overlong = "z".repeat(40);
    let prefix = &overlong[..model.max_word_length()];
    assert_eq!(tagger.tag(&[prefix]).unwrap(), tagger.tag(&[overlong.as_str()]).unwrap());

    // An empty token has no characters, so as in Java its character
    // states stay zero instead of running the character LSTMs.
    assert!(model.char_ids("").is_empty());
    let empty = model.word_representation("").unwrap();
    let word_dim = model.word_embeddings[0].len();
    let char_hidden = model.char_forward.hidden_size();
    assert!(empty[word_dim..word_dim + 2 * char_hidden].iter().all(|&v| v.to_bits() == 0));
    assert_eq!(tagger.tag(&[""]).unwrap().len(), 1);

    let unicode_tags = tagger.tag(&["İstanbul", "🐕"]).unwrap();
    assert_eq!(unicode_tags.len(), 2);
    assert_eq!(unicode_tags, tagger.tag(&["İstanbul", "🐕"]).unwrap());

    // 2-layer CRF usage test
    let accurate = BilstmSettings::new(
        16, 8, 8, 16, 60, 8, 0.05, 5.0, 0.0, 1, 16, 17, 1, 0.0, 0, true, 2, 0.0, 0.0, 1.0, 0.0, false,
    )
    .unwrap();
    let crf_model = BilstmPosTrainer::train(&usage_corpus(), accurate).unwrap();
    for crf_tagger in [BilstmPosTagger::jvm_parity(crf_model.clone()), BilstmPosTagger::new(crf_model)] {
        assert_eq!(crf_tagger.tag(&["the", "cat", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
        assert_eq!(crf_tagger.tag(&["dogs", "bark"]).unwrap(), vec!["NNS", "VBP"]);
    }
}

#[test]
fn full_model_gradients_match_finite_differences() {
    const EPSILON: f64 = 1e-6;
    const TOLERANCE: f64 = 1e-5;

    let corpus = vec![
        MultiTaskSample::new(
            ["The", "cat", "sat"],
            ["DET", "NOUN", "VERB"],
            Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
            Some(vec!["_".into(), "Number=Sing".into(), "Tense=Past".into()]),
        )
        .unwrap(),
        MultiTaskSample::new(
            ["A", "dog", "ran"],
            ["DET", "NOUN", "VERB"],
            Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
            Some(vec!["_".into(), "Number=Sing".into(), "Tense=Past".into()]),
        )
        .unwrap(),
        MultiTaskSample::new(
            ["Cats", "sleep"],
            ["NOUN", "VERB"],
            Some(vec!["NNS".into(), "VBP".into()]),
            Some(vec!["Number=Plur".into(), "Tense=Pres".into()]),
        )
        .unwrap(),
    ];
    let sample = &corpus[0];

    // Check CRF + 2-layer + Aux + Adapter gradients
    let crf_settings =
        BilstmSettings::new(4, 3, 3, 4, 1, 2, 1e-3, 5.0, 0.0, 1, 10, 7, 1, 0.0, 0, true, 2, 0.0, 0.0, 0.7, 0.0, true)
            .unwrap();
    let vec_fn = |_: &str| Some(vec![0.5f32, -0.25f32]);
    let mut ctx = TrainingContext::build(&corpus, crf_settings, Some(vec_fn), None).unwrap();
    let mut worker = ctx.new_worker();
    ctx.reset_worker(&mut worker);
    ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
    let analytic_out_w = worker.buffers()[ctx.idx_out_w].clone();
    let analytic_tr = worker.buffers()[ctx.idx_crf_tr.unwrap()].clone();
    let analytic_aw = worker.buffers()[ctx.idx_adapter_w.unwrap()].clone();

    for o in 0..analytic_out_w.len() {
        for j in 0..analytic_out_w[o].len() {
            let orig = ctx.testing_output_weights_mut()[o][j];
            ctx.testing_output_weights_mut()[o][j] = orig + EPSILON;
            let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_output_weights_mut()[o][j] = orig - EPSILON;
            let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_output_weights_mut()[o][j] = orig;
            let num = (plus - minus) / (2.0 * EPSILON);
            assert!((analytic_out_w[o][j] - num).abs() <= TOLERANCE);
        }
    }
    for j in 0..analytic_tr.len() {
        for k in 0..analytic_tr[j].len() {
            let orig = ctx.testing_transition_weights_mut().unwrap()[j][k];
            ctx.testing_transition_weights_mut().unwrap()[j][k] = orig + EPSILON;
            let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_transition_weights_mut().unwrap()[j][k] = orig - EPSILON;
            let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_transition_weights_mut().unwrap()[j][k] = orig;
            let num = (plus - minus) / (2.0 * EPSILON);
            assert!((analytic_tr[j][k] - num).abs() <= TOLERANCE);
        }
    }
    for r in 0..analytic_aw.len() {
        for c in 0..analytic_aw[r].len() {
            let orig = ctx.testing_adapter_weights_mut().unwrap()[r][c];
            ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig + EPSILON;
            let plus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig - EPSILON;
            let minus = ctx.sentence_gradients(sample, &mut JavaRng::new(0), &mut worker).unwrap();
            ctx.testing_adapter_weights_mut().unwrap()[r][c] = orig;
            let num = (plus - minus) / (2.0 * EPSILON);
            assert!((analytic_aw[r][c] - num).abs() <= TOLERANCE);
        }
    }

    let _ = char_vocabulary_from_strs([("*UNK*", 0)]);
}

#[test]
fn pretrained_tuning_clips_extreme_gradients_and_rejects_non_finite_updates() {
    let corpus = vec![MultiTaskSample::new(["The", "cat"], ["D", "N"], None, None).unwrap()];
    let settings =
        BilstmSettings::new(4, 2, 2, 4, 1, 1, 1e-3, 5.0, 0.0, 1, 8, 7, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.5, false)
            .unwrap();
    let vec_fn = |_: &str| Some(vec![1.0f32, 2.0f32]);
    let mut ctx = TrainingContext::build(&corpus, settings, Some(vec_fn), None).unwrap();

    // Bug 4: new_worker() buffer count matches optimizer parameter count without trailing dummy buffer.
    let worker = ctx.new_worker();
    assert_eq!(worker.buffers().len(), ctx.optimizer.new_gradient_buffers().len());

    // Bug 1a: large finite tuning gradient (1e200 and f64::MAX) must clip cleanly to max_grad_norm (5.0) instead of overflowing to +inf and zeroing gradients.
    let mut touched = HashMap::new();
    touched.insert(0, vec![3e200, 4e200]);
    ctx.step_batch(1.0, 1e-3, 1, &mut touched).unwrap();
    assert!((touched[&0][0] - 3.0).abs() < 1e-9);
    assert!((touched[&0][1] - 4.0).abs() < 1e-9);

    let mut max_touched = HashMap::new();
    max_touched.insert(0, vec![f64::MAX, f64::MAX]);
    ctx.step_batch(1.0, 1.0, 1, &mut max_touched).unwrap();
    let expected_max_clip = 5.0 / 2.0_f64.sqrt();
    assert!((max_touched[&0][0] - expected_max_clip).abs() < 1e-12);
    assert!((max_touched[&0][1] - expected_max_clip).abs() < 1e-12);

    // Combined dense + sparse f64::MAX gradients must clip together without dense global_norm() overflowing to +inf.
    ctx.optimizer.gradient_mut(ctx.idx_out_b)[0].fill(f64::MAX);
    let mut both_max_touched = HashMap::new();
    both_max_touched.insert(0, vec![f64::MAX, f64::MAX]);
    ctx.step_batch(1.0, 1e-3, 1, &mut both_max_touched).unwrap();
    assert!((ctx.optimizer.gradient(ctx.idx_out_b)[0][0] - 2.5).abs() < 1e-12);
    assert!((both_max_touched[&0][0] - 2.5).abs() < 1e-12);

    // Finite total_sq (2.5e301) with tiny max_grad_norm (1e-300) must not underflow clip = limit / norm to 0.0.
    let tiny_limit_settings = BilstmSettings::new(
        4, 2, 2, 4, 1, 1, 1e-3, 1e-300, 0.0, 1, 8, 7, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.5, false,
    )
    .unwrap();
    let mut tiny_ctx = TrainingContext::build(&corpus, tiny_limit_settings, Some(vec_fn), None).unwrap();
    tiny_ctx.optimizer.gradient_mut(tiny_ctx.idx_out_b)[0].copy_from_slice(&[3e150, 0.0]);
    let mut tiny_touched = HashMap::new();
    tiny_touched.insert(0, vec![0.0, -4e150]);
    tiny_ctx.step_batch(1.0, 1.0, 1, &mut tiny_touched).unwrap();
    assert!((tiny_ctx.optimizer.gradient(tiny_ctx.idx_out_b)[0][0] - 6e-301).abs() <= 6e-301 * 1e-12);
    assert!((tiny_touched[&0][1] - -8e-301).abs() <= 8e-301 * 1e-12);

    // Bug 1b: NaN tuning gradient must return NonFiniteScore.
    let mut nan_touched = HashMap::new();
    nan_touched.insert(0, vec![f64::NAN, 1.0]);
    assert!(matches!(ctx.step_batch(1.0, 1e-3, 2, &mut nan_touched), Err(NeuralError::NonFiniteScore(_))));

    // Bug 1c: tuned pretrained weight exceeding f32::MAX must return NonFiniteScore in to_model().
    ctx.testing_pretrained_trainable_mut().unwrap()[0][0] = 1e100;
    assert!(matches!(ctx.to_model(), Err(NeuralError::NonFiniteScore(_))));
}

#[test]
fn primary_and_auxiliary_heads_reject_non_finite_weights() {
    let corpus = vec![
        MultiTaskSample::new(
            ["The", "cat"],
            ["D", "N"],
            Some(vec!["DT".into(), "NN".into()]),
            Some(vec!["Def".into(), "Sing".into()]),
        )
        .unwrap(),
    ];
    let settings =
        BilstmSettings::new(4, 2, 2, 4, 1, 1, 1e-3, 5.0, 0.0, 1, 8, 7, 2, 0.0, 0, false, 2, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap();
    let mut ctx = TrainingContext::build(&corpus, settings, None::<fn(&str) -> Option<Vec<f32>>>, None).unwrap();
    assert!(ctx.testing_word_backward2_mut().is_some());
    assert!(ctx.testing_feats_bias_mut().is_some());

    let mut worker = ctx.new_worker();
    let orig_b0 = ctx.testing_output_bias_mut()[0];
    ctx.testing_output_bias_mut()[0] = f64::NAN;
    assert!(matches!(
        ctx.sentence_gradients(&corpus[0], &mut JavaRng::new(0), &mut worker),
        Err(NeuralError::NonFiniteScore(_))
    ));
    ctx.testing_output_bias_mut()[0] = orig_b0;

    ctx.testing_feats_bias_mut().unwrap()[0] = f64::INFINITY;
    assert!(matches!(
        ctx.sentence_gradients(&corpus[0], &mut JavaRng::new(0), &mut worker),
        Err(NeuralError::NonFiniteScore(_))
    ));
}

#[test]
fn validation_rejects_mismatched_lengths_empty_tags_and_unseen_tags() {
    let settings =
        BilstmSettings::new(4, 2, 2, 4, 1, 1, 1e-3, 5.0, 0.0, 1, 8, 7, 1, 0.0, 0, false, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap();

    // Empty corpus / empty tags in TrainingContext::build
    let empty_sample = MultiTaskSample { sentence: vec![], tags: vec![], xpos: None, feats: None };
    assert!(matches!(
        TrainingContext::build(&[empty_sample], settings.clone(), None::<fn(&str) -> Option<Vec<f32>>>, None),
        Err(NeuralError::InvalidArgument(_))
    ));

    // Mutated sample with mismatched tags length
    let bad_sample =
        MultiTaskSample { sentence: vec!["The".into(), "cat".into()], tags: vec!["D".into()], xpos: None, feats: None };
    assert!(matches!(
        BilstmPosTrainer::train_multi_task(
            &[bad_sample],
            settings.clone(),
            None::<fn(&str) -> Option<Vec<f32>>>,
            None,
            None,
        ),
        Err(NeuralError::InvalidArgument(_))
    ));

    // Unseen tag passed to sentence_gradients returns InvalidArgument instead of panicking
    let valid_corpus = vec![MultiTaskSample::new(["The", "cat"], ["D", "N"], None, None).unwrap()];
    let ctx = TrainingContext::build(&valid_corpus, settings, None::<fn(&str) -> Option<Vec<f32>>>, None).unwrap();
    let mut worker = ctx.new_worker();
    let unseen = MultiTaskSample::new(["The", "cat"], ["D", "UNSEEN"], None, None).unwrap();
    assert!(matches!(
        ctx.sentence_gradients(&unseen, &mut JavaRng::new(0), &mut worker),
        Err(NeuralError::InvalidArgument(_))
    ));
}
