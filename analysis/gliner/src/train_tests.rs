use super::*;

const GOLD: [(usize, usize, usize); 1] = [(0, 1, 0)];
/// Hidden size two (the BiLSTM needs an even one), output factor two.
const H: usize = 2;
const F: usize = 2;

/// Which heads a test trainer holds, in the order its tensors follow.
#[derive(Clone, Copy, Debug, Default)]
struct Heads {
    linear: bool,
    boundaries: bool,
    prompt: bool,
    recurrent: bool,
    projection: bool,
    /// Zero stands for the output factor.
    prompt_factor: usize,
}

/// Small values of both signs from a fixed sequence.
struct Values(u32);

impl Values {
    fn tensor(&mut self, len: usize) -> Vec<f32> {
        (0..len)
            .map(|_| {
                self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((self.0 >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 0.8
            })
            .collect()
    }

    fn mlp(&mut self, input: usize, width: usize) -> [Vec<f32>; 4] {
        [self.tensor(width * input), self.tensor(width), self.tensor(H * width), self.tensor(H)]
    }
}

fn trainer(heads: Heads) -> OutputTrainer {
    let prompt_factor = if heads.prompt_factor == 0 { F } else { heads.prompt_factor };
    // A seed whose activations sit away from zero, where a finite
    // difference would straddle a ReLU kink.
    let mut values = Values(11);
    let mut params =
        if heads.linear { vec![values.tensor(H * 2 * H), values.tensor(H)] } else { values.mlp(2 * H, F * H).to_vec() };
    if heads.boundaries {
        params.extend(values.mlp(H, F * H));
        params.extend(values.mlp(H, F * H));
    }
    if heads.prompt {
        params.extend(values.mlp(H, prompt_factor * H));
    }
    if heads.recurrent {
        for _ in 0..2 {
            params.extend([2 * H * H, 2 * H * (H / 2), 2 * H, 2 * H].map(|len| values.tensor(len)));
        }
    }
    if heads.projection {
        params.extend([values.tensor(3 * H), values.tensor(H)]);
    }
    OutputTrainer {
        hidden: H,
        factor: F,
        prompt_factor,
        base: params.clone(),
        params,
        anchor: 0.0,
        linear: heads.linear,
        boundaries: heads.boundaries,
        prompt: heads.prompt,
        recurrent: heads.recurrent,
        projection: heads.projection,
    }
}

fn fixture() -> (TrainingInput, Vec<Candidate>) {
    let input = TrainingInput {
        words: vec![0.6, -0.2, 0.4, 0.3],
        projected_words: vec![0.5, -0.3, 0.2, 0.7],
        encoder_inputs: Vec::new(),
        spans: vec![0.5, -0.3, 0.2, 0.8, 0.1, 0.4, -0.2, 0.6],
        prompts: vec![0.7, 0.2],
        prompt_inputs: vec![0.4, -0.1],
        pairs: vec![(0, 1), (1, 2)],
    };
    let candidates = vec![
        Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
        Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
    ];
    (input, candidates)
}

fn check_gradients(heads: Heads) {
    let (mut input, candidates) = fixture();
    if heads.projection {
        input.encoder_inputs = vec![0.4, -0.1, 0.3, 0.6, -0.2, 0.4, 0.3, 0.2, 0.5];
    }
    let mut trainer = trainer(heads);
    trainer.set_anchor(0.25).unwrap();
    let mut grads = trainer.zero_grads();
    let loss = trainer.gradients(&input, &candidates, &GOLD, &mut grads).unwrap();
    assert!(loss.is_finite());
    for (tensor, analytic) in grads.iter().enumerate() {
        for (at, &analytic) in analytic.iter().enumerate() {
            let original = trainer.params[tensor][at];
            trainer.params[tensor][at] = original + 0.001;
            let plus = trainer.gradients(&input, &candidates, &GOLD, &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original - 0.001;
            let minus = trainer.gradients(&input, &candidates, &GOLD, &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original;
            let expected = (plus - minus) / 0.002;
            assert!(
                (analytic as f64 - expected).abs() < 0.002,
                "{heads:?} tensor {tensor} at {at}: {analytic} != {expected}"
            );
        }
    }
}

/// Every supported combination: output form, boundary heads, prompt head,
/// and the BiLSTM where the boundary heads are trained.
fn combinations() -> Vec<Heads> {
    let mut out = Vec::new();
    for linear in [false, true] {
        for boundaries in [false, true] {
            for prompt in [false, true] {
                for recurrent in [false, true] {
                    if !recurrent || boundaries {
                        out.push(Heads { linear, boundaries, prompt, recurrent, prompt_factor: 0, ..Heads::default() });
                    }
                }
            }
        }
    }
    out
}

#[test]
fn negative_only_sentences_train_and_empty_inputs_are_safe() {
    for heads in combinations() {
        let (input, candidates) = fixture();
        let trainer = trainer(heads);
        let mut grads = trainer.zero_grads();
        let loss = trainer.gradients(&input, &candidates, &[], &mut grads).unwrap();
        assert!(loss.is_finite() && loss != 0.0, "{heads:?}");
        assert!(grads.iter().flatten().any(|&gradient| gradient != 0.0), "{heads:?}");
    }

    let (mut input, _) = fixture();
    input.pairs.clear();
    input.spans.clear();
    let trainer = trainer(Heads::default());
    let mut grads = trainer.zero_grads();
    assert_eq!(trainer.gradients(&input, &[], &[], &mut grads).unwrap(), 0.0);
    assert!(grads.iter().flatten().all(|&gradient| gradient == 0.0));
}

#[test]
fn gradients_match_finite_differences_for_every_head_combination() {
    let counts: Vec<usize> = combinations().into_iter().map(|heads| trainer(heads).params.len()).collect();
    // Twelve modes; 6 and 22 are linear output with a prompt head.
    assert_eq!(counts, [4, 8, 12, 20, 16, 24, 2, 6, 10, 18, 14, 22]);
    for heads in combinations() {
        check_gradients(heads);
    }
}

#[test]
fn the_prompt_head_keeps_its_own_width() {
    // Output and boundary factor two, prompt factor four.
    for heads in combinations().into_iter().filter(|heads| heads.prompt) {
        let heads = Heads { prompt_factor: 4, ..heads };
        let built = trainer(heads);
        assert_eq!(built.params[built.prompt_offset().unwrap()].len(), 4 * H * H);
        check_gradients(heads);
    }
}

#[test]
fn the_layout_follows_the_heads_not_the_tensor_count() {
    for heads in combinations() {
        let trainer = trainer(heads);
        let output = if heads.linear { 2 } else { 4 };
        assert_eq!(trainer.boundary_offset(), heads.boundaries.then_some(output));
        let before_prompt = output + if heads.boundaries { 8 } else { 0 };
        assert_eq!(trainer.prompt_offset(), heads.prompt.then_some(before_prompt));
        let before_recurrent = before_prompt + if heads.prompt { 4 } else { 0 };
        assert_eq!(trainer.recurrent_offset(), heads.recurrent.then_some(before_recurrent));
        assert_eq!(trainer.projection_offset(), None);
        assert_eq!(trainer.params.len(), before_recurrent + if heads.recurrent { 8 } else { 0 });
    }
}

#[test]
fn a_gradient_buffer_of_another_mode_is_refused() {
    let (input, candidates) = fixture();
    let modes = combinations();
    for own in &modes {
        let own = trainer(*own);
        for other in &modes {
            let other = trainer(*other);
            let same = own.params.iter().map(Vec::len).eq(other.params.iter().map(Vec::len));
            let mut grads = other.zero_grads();
            let result = own.gradients(&input, &candidates, &GOLD, &mut grads);
            if same {
                assert!(result.is_ok());
            } else {
                // Shorter or longer, it is refused before anything is written.
                assert!(result.unwrap_err().contains("shape mismatch"));
                assert!(grads.iter().flatten().all(|&g| g == 0.0));
            }
        }
    }
    // A tensor of the wrong length at the end of the trainer's own buffer.
    let all = trainer(Heads { boundaries: true, prompt: true, recurrent: true, ..Heads::default() });
    let mut wrong = all.zero_grads();
    wrong[23].push(0.0);
    assert!(all.gradients(&input, &candidates, &GOLD, &mut wrong).is_err());
}

#[test]
fn weight_decay_reaches_every_matrix_and_no_bias() {
    let projected = [false, true].map(|linear| Heads {
        linear,
        boundaries: true,
        prompt: true,
        recurrent: true,
        projection: true,
        prompt_factor: 0,
    });
    for heads in combinations().into_iter().chain(projected) {
        let mut trainer = trainer(heads);
        let before = trainer.params.clone();
        let settings = Settings { lr: 0.01, weight_decay: 0.1, total: 1, ..Settings::default() };
        let mut optimizer = output_optimizer(settings, &trainer).unwrap();
        // No gradient: only the decay moves a tensor.
        trainer.step(&mut optimizer, &mut trainer.zero_grads()).unwrap();
        let recurrent = trainer.recurrent_offset().unwrap_or(before.len());
        let projection = trainer.projection_offset().unwrap_or(before.len());
        for (at, (old, new)) in before.iter().zip(&trainer.params).enumerate() {
            // Dense and projection layers are weight then bias; a BiLSTM
            // direction is weight_ih, weight_hh, bias_ih, bias_hh.
            let matrix = if at >= projection {
                (at - projection).is_multiple_of(2)
            } else if at < recurrent {
                at.is_multiple_of(2)
            } else {
                (at - recurrent) % 4 < 2
            };
            assert_eq!(old != new, matrix, "{heads:?} tensor {at}");
        }
    }
}

#[test]
fn projection_gradients_match_finite_differences() {
    for linear in [false, true] {
        for prompt_factor in [0, 4] {
            let heads =
                Heads { linear, boundaries: true, prompt: true, recurrent: true, projection: true, prompt_factor };
            let built = trainer(heads);
            let output = if linear { 2 } else { 4 };
            assert_eq!(built.recurrent_offset(), Some(output + 12));
            assert_eq!(built.projection_offset(), Some(output + 20));
            assert_eq!(built.params.len(), output + 22);
            check_gradients(heads);
            let (mut input, candidates) = fixture();
            input.encoder_inputs = vec![0.4, -0.1, 0.3, 0.6, -0.2, 0.4, 0.3, 0.2, 0.5];
            let (_, dencoder) =
                built.gradients_with_encoder(&input, &candidates, &GOLD, &mut built.zero_grads()).unwrap();
            for (at, &actual) in dencoder.iter().enumerate() {
                let original = input.encoder_inputs[at];
                input.encoder_inputs[at] = original + 0.001;
                let plus = built.gradients(&input, &candidates, &GOLD, &mut built.zero_grads()).unwrap();
                input.encoder_inputs[at] = original - 0.001;
                let minus = built.gradients(&input, &candidates, &GOLD, &mut built.zero_grads()).unwrap();
                input.encoder_inputs[at] = original;
                let expected = (plus - minus) / 0.002;
                assert!((actual as f64 - expected).abs() < 0.002, "encoder input {at}");
            }
        }
    }
}
