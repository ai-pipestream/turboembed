use super::*;
use serde_json::Value;
fn ec() -> turbo::deberta::Config {
    turbo::deberta::Config {
        hidden: 32,
        heads: 2,
        intermediate: 48,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    }
}
fn config() -> Config {
    Config { hidden: 16, max_width: 3, entity_token: 12, separator_token: 13, cls_token: 1, eos_token: 2 }
}
#[test]
fn real_checkpoint_heads_match_upstream_python() {
    let reference: Value = serde_json::from_str(include_str!("../tests/fixtures/reference.json")).unwrap();
    let heads = heads::Heads::load(&config(), &ec(), include_bytes!("../tests/fixtures/heads.safetensors")).unwrap();
    let input: Vec<Vec<f32>> = serde_json::from_value(reference["input"].clone()).unwrap();
    let width = input[0].len();
    let states: Vec<f32> = input.concat();
    let words: Vec<usize> = (2..input.len()).collect();
    let scores: Vec<f32> = heads.score(&states, width, &[0, 1], &words, 3).unwrap().iter().map(|c| c.logit).collect();
    let expected: Vec<f32> = serde_json::from_value(reference["logits"].clone()).unwrap();
    assert_eq!(scores.len(), expected.len());
    for (a, b) in scores.iter().zip(expected) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }
}
#[test]
fn named_limits_and_layout_rejections() {
    assert!(ExtractionSchema::new(vec![], 0.5, Decode::Flat).unwrap_err().contains("max_labels"));
    assert!(ExtractionSchema::new(vec!["a".into(); 33], 0.5, Decode::Flat).unwrap_err().contains("max_labels"));
    for t in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(ExtractionSchema::new(vec!["a".into()], t, Decode::Flat).is_err());
    }
    assert!(ExtractionSchema::new(vec!["a".repeat(129)], 0.5, Decode::Flat).unwrap_err().contains("max_label_bytes"));
    let mut c = config();
    c.max_width = usize::MAX;
    assert!(c.check(&ec()).unwrap_err().contains("max_width"));
    let p = PackedInput { ids: vec![1, 12, 4, 13, 5, 2], prompts: vec![1], words: vec![4] };
    p.check(&config(), &ec(), 1).unwrap();
    for bad in [vec![usize::MAX], vec![0], vec![5], vec![4, 4]] {
        let mut p = p.clone();
        p.words = bad;
        assert!(p.check(&config(), &ec(), 1).is_err());
    }
    let mut p = p.clone();
    p.words = vec![4; 385];
    assert!(p.check(&config(), &ec(), 1).unwrap_err().contains("max_tokens"));
}
#[test]
fn heads_reject_bad_shapes_missing_and_nonfinite() {
    let bytes = include_bytes!("../tests/fixtures/heads.safetensors");
    assert!(heads::Heads::load(&config(), &ec(), &bytes[..10]).is_err());
    let mut c = config();
    c.hidden = 18;
    assert!(heads::Heads::load(&c, &ec(), bytes).is_err());
    let mut bytes = bytes.to_vec();
    let n = bytes.len();
    bytes[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(heads::Heads::load(&config(), &ec(), &bytes).is_err());
}
#[test]
fn nested_decode_is_stable_and_rejects_crossings() {
    let candidates = vec![
        Candidate { start: 0, end: 4, label: 0, logit: 2.0, score: 0.9 },
        Candidate { start: 1, end: 3, label: 1, logit: 1.0, score: 0.8 },
        Candidate { start: 2, end: 5, label: 0, logit: 0.5, score: 0.7 },
    ];
    let schema = |mode| ExtractionSchema::new(vec!["a".into(), "b".into()], 0.5, mode).unwrap();
    assert_eq!(decode(candidates.clone(), &schema(Decode::Nested)).len(), 2);
    assert_eq!(decode(candidates, &schema(Decode::Flat)).len(), 1);
}

#[test]
fn schema_and_special_tokens_refuse_ambiguous_prompts() {
    for labels in [vec![" ".into()], vec!["person".into(), "person".into()], vec!["<<ENT>> person".into()]] {
        assert!(ExtractionSchema::new(labels, 0.5, Decode::Flat).is_err());
    }
    let mut c = config();
    c.entity_token = c.cls_token;
    assert!(c.check(&ec()).is_err());
    c = config();
    c.entity_token = u32::MAX;
    assert!(c.check(&ec()).is_err());
    let good = PackedInput { ids: vec![1, 12, 4, 13, 5, 2], prompts: vec![1], words: vec![4] };
    for ids in [vec![1, 12, 4, 13, 13, 2], vec![1, 12, 4, 13, 12, 2], vec![1, 12, 4, 5, 5, 2], vec![1; 33]] {
        let mut p = good.clone();
        p.ids = ids;
        assert!(p.check(&config(), &ec(), 1).is_err());
    }
    for prompts in [vec![], vec![usize::MAX], vec![2]] {
        let mut p = good.clone();
        p.prompts = prompts;
        assert!(p.check(&config(), &ec(), 1).is_err());
    }
}

#[test]
fn decode_threshold_and_tie_order_match_reference() {
    let schema = ExtractionSchema::new(vec!["a".into(), "b".into()], 0.5, Decode::Nested).unwrap();
    let candidates = vec![
        Candidate { start: 0, end: 2, label: 0, logit: 1.0, score: 0.75 },
        Candidate { start: 0, end: 2, label: 1, logit: 1.0, score: 0.75 },
        Candidate { start: 2, end: 3, label: 1, logit: 0.0, score: 0.5 },
        Candidate { start: 3, end: 4, label: 0, logit: 1.0, score: 0.75 },
    ];
    let selected = decode(candidates, &schema);
    assert_eq!(selected.iter().map(|x| (x.start, x.label)).collect::<Vec<_>>(), vec![(0, 0), (3, 0)]);
    assert_eq!(heads::sigmoid(f32::MAX), 1.0);
    assert_eq!(heads::sigmoid(-f32::MAX), 0.0);
}

#[test]
fn decisions_reject_nonfinite_and_out_of_range_values() {
    use turbo_analysis::Layer;
    for threshold in [f32::NAN, f32::INFINITY, -0.1, 1.1] {
        assert!(Decisions { logits: vec![], threshold }.check().is_err());
    }
    for logit in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert!(Decisions { logits: vec![logit], threshold: 0.5 }.check().is_err());
    }
    Decisions { logits: vec![-1000.0, 1000.0], threshold: 0.0 }.check().unwrap();
}

#[path = "boundary_tests.rs"]
mod boundaries;

#[path = "window_tests.rs"]
mod windows;

#[test]
fn head_workspace_retains_buffers_and_resets_recurrent_state() {
    let reference: Value = serde_json::from_str(include_str!("../tests/fixtures/reference.json")).unwrap();
    let heads = heads::Heads::load(&config(), &ec(), include_bytes!("../tests/fixtures/heads.safetensors")).unwrap();
    let input: Vec<Vec<f32>> = serde_json::from_value(reference["input"].clone()).unwrap();
    let width = input[0].len();
    let states = input.concat();
    let words: Vec<_> = (2..input.len()).collect();
    let mut work = heads::Work::default();
    let expected = heads.score(&states, width, &[0, 1], &words, 3).unwrap();
    assert_eq!(heads.score_with_work(&states, width, &[0, 1], &words, 3, &mut work).unwrap(), expected);
    let allocations = work.allocations();
    assert!(allocations.iter().all(|(_, n)| *n > 0));
    let short = &words[..1];
    assert_eq!(
        heads.score_with_work(&states, width, &[0, 1], short, 3, &mut work).unwrap(),
        heads.score(&states, width, &[0, 1], short, 3).unwrap()
    );
    assert_eq!(heads.score_with_work(&states, width, &[0, 1], &words, 3, &mut work).unwrap(), expected);
    assert_eq!(work.allocations(), allocations, "all 9 head buffers retained after warm-up");
}
