use super::*;

#[test]
fn sigmoid_is_stable_at_the_tails() {
    assert_eq!(sigmoid(0.0), 0.5);
    assert_eq!(sigmoid(-1000.0), 0.0);
    assert_eq!(sigmoid(1000.0), 1.0);
    assert!(sigmoid(-100.0).is_finite() && sigmoid(-100.0) > 0.0);
    assert_eq!(sigmoid_f64(0.0), 0.5);
    assert_eq!(sigmoid_f64(-1000.0), 0.0);
    assert_eq!(sigmoid_f64(1000.0), 1.0);
}

#[test]
fn linear_and_head_reject_overflowing_misaligned_and_non_finite_inputs() {
    let err = Linear::new(usize::MAX, 2, vec![], vec![0.0, 0.0]).unwrap_err();
    assert_eq!(err.code, INVALID_SHAPE);

    let cls = Linear::new(2, 2, vec![1.0, 0.0, 0.0, 1.0], vec![0.0, 0.0]).unwrap();
    let head = Head {
        kind: ClassifierKind::Sequence,
        hidden: 2,
        labels: vec!["neg".into(), "pos".into()],
        pooler: Some(cls.clone()),
        classifier: cls,
    };
    assert!(head.logits(&[1.0, 2.0]).is_ok());
    assert_eq!(head.logits(&[f32::NAN, 1.0]).unwrap_err().code, INVALID_ARGUMENT);

    let zero_hidden = Head { hidden: 0, ..head.clone() };
    assert_eq!(zero_hidden.logits(&[]).unwrap_err().code, INVALID_ARGUMENT);

    let bad_cls = Head { hidden: 3, ..head.clone() };
    assert_eq!(bad_cls.logits(&[1.0, 2.0, 3.0]).unwrap_err().code, INVALID_SHAPE);

    // Short and long mutated classifier/pooler weight and bias storage are rejected
    // across logits, logits_into, probabilities, and probabilities_into.
    let mut short_cls = head.clone();
    short_cls.classifier.weight_mut().clear();
    let mut out_f32 = [0.0f32; 2];
    let mut out_f64 = [0.0f64; 2];
    assert_eq!(short_cls.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(short_cls.logits_into(&[1.0, 2.0], &mut out_f32).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(short_cls.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(short_cls.probabilities_into(&[1.0, 2.0], &mut out_f64).unwrap_err().code, INVALID_SHAPE);

    let mut long_cls = head.clone();
    long_cls.classifier.weight_mut().extend([1.0, 1.0]);
    long_cls.classifier.bias_mut().push(0.0);
    assert_eq!(long_cls.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(long_cls.logits_into(&[1.0, 2.0], &mut out_f32).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(long_cls.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(long_cls.probabilities_into(&[1.0, 2.0], &mut out_f64).unwrap_err().code, INVALID_SHAPE);

    let mut bad_pooler = head;
    bad_pooler.pooler.as_mut().unwrap().bias_mut().pop();
    assert_eq!(bad_pooler.logits(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
    assert_eq!(bad_pooler.probabilities(&[1.0, 2.0]).unwrap_err().code, INVALID_SHAPE);
}

/// The batched packed product names the same logits as one state at
/// a time, with and without a pooler, and a refused batch writes
/// nothing.
#[test]
fn batched_logits_match_one_state_at_a_time() {
    let (hidden, labels, rows) = (5, 3, 7);
    let mut state = 7u64;
    let mut next = move || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let layer = |inp: usize, out: usize, next: &mut dyn FnMut() -> f32| {
        Linear::new(inp, out, (0..inp * out).map(|_| next()).collect(), (0..out).map(|_| next()).collect()).unwrap()
    };
    let classifier = layer(hidden, labels, &mut next);
    let pooler = layer(hidden, hidden, &mut next);
    let states: Vec<f32> = (0..rows * hidden).map(|_| next()).collect();
    for with_pooler in [false, true] {
        let head = Head {
            kind: ClassifierKind::Sequence,
            hidden,
            labels: (0..labels).map(|i| i.to_string()).collect(),
            pooler: with_pooler.then(|| pooler.clone()),
            classifier: classifier.clone(),
        };
        let batched = head.logits(&states).unwrap();
        let mut one_at_a_time = Vec::new();
        for s in states.chunks_exact(hidden) {
            one_at_a_time.extend(head.logits(s).unwrap());
        }
        assert_eq!(batched, one_at_a_time, "pooler: {with_pooler}");
        let probs = head.probabilities(&states).unwrap();
        let mut want = vec![0.0f64; labels];
        softmax(&batched[..labels], &mut want).unwrap();
        assert_eq!(&probs[..labels], &want[..], "pooler: {with_pooler}");
    }
    let head = Head {
        kind: ClassifierKind::Sequence,
        hidden,
        labels: (0..labels).map(|i| i.to_string()).collect(),
        pooler: None,
        classifier,
    };
    let mut bad = states.clone();
    bad[hidden] = f32::INFINITY;
    let mut out = vec![9.0f32; rows * labels];
    assert_eq!(head.logits_into(&bad, &mut out).unwrap_err().code, INVALID_ARGUMENT);
    assert!(out.iter().all(|&v| v == 9.0), "a refused batch writes nothing");
}

/// Mutating the public weights is observed by the next product,
/// before and after the packed form exists, and a clone compares
/// equal only while it also products equal.
#[test]
fn mutated_weights_are_the_ones_applied() {
    let mut layer = Linear::new(2, 1, vec![1.0, 2.0], vec![0.0]).unwrap();
    let mut out = [0.0f32; 1];
    layer.weight_mut()[0] = 3.0;
    layer.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [5.0], "mutation before first apply");
    layer.weight_mut()[0] = 10.0;
    layer.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [12.0], "mutation after packing");
    layer.bias_mut()[0] = 1.0;
    layer.apply(&[1.0, 1.0], &mut out);
    assert_eq!(out, [13.0], "bias mutation");
    let clone = layer.clone();
    assert_eq!(layer, clone);
    let mut clone_out = [0.0f32; 1];
    clone.apply(&[1.0, 1.0], &mut clone_out);
    assert_eq!(out, clone_out, "equal layers, equal products");
}
