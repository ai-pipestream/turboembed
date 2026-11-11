use super::*;

const GOLD: [(usize, usize, usize); 1] = [(0, 1, 0)];

/// Which heads a test trainer holds, in the order its tensors follow.
#[derive(Clone, Copy, Default)]
struct Heads {
    linear: bool,
    boundaries: bool,
    prompt: bool,
    prompt_factor: usize,
}

fn mlp(width: usize) -> [Vec<f32>; 4] {
    let cycle = |n: usize, shift: usize| (0..n).map(|i| [0.2, -0.1, 0.3, -0.4, 0.1][(i + shift) % 5]).collect();
    [cycle(width, 0), cycle(width, 2), cycle(width, 1), vec![0.2]]
}

/// A trainer of hidden size one and factor two with the given heads.
fn trainer(heads: Heads) -> OutputTrainer {
    let prompt_factor = if heads.prompt_factor == 0 { 2 } else { heads.prompt_factor };
    let mut params = if heads.linear {
        vec![vec![0.3, -0.4], vec![0.2]]
    } else {
        vec![vec![0.2, -0.1, 0.3, 0.4], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]
    };
    if heads.boundaries {
        params.extend(mlp(2));
        params.extend(mlp(2));
    }
    if heads.prompt {
        params.extend(mlp(prompt_factor));
    }
    OutputTrainer {
        hidden: 1,
        factor: 2,
        prompt_factor,
        linear: heads.linear,
        boundaries: heads.boundaries,
        prompt: heads.prompt,
        base: params.clone(),
        params,
        anchor: 0.0,
    }
}

fn fixture() -> (TrainingInput, Vec<Candidate>) {
    let input = TrainingInput {
        words: vec![0.6, -0.2],
        spans: vec![0.5, -0.3, 0.2, 0.8],
        prompts: vec![0.7],
        prompt_inputs: vec![0.4],
        pairs: vec![(0, 1), (1, 2)],
    };
    let candidates = vec![
        Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
        Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
    ];
    (input, candidates)
}

fn check_gradients(heads: Heads) {
    let (input, candidates) = fixture();
    let mut trainer = trainer(heads);
    trainer.set_anchor(0.25).unwrap();
    for tensor in 0..trainer.params.len() {
        for at in 0..trainer.params[tensor].len() {
            let original = trainer.params[tensor][at];
            let mut grads = trainer.zero_grads();
            let loss = trainer.gradients(&input, &candidates, &GOLD, &mut grads).unwrap();
            assert!(loss.is_finite());
            trainer.params[tensor][at] = original + 0.001;
            let plus = trainer.gradients(&input, &candidates, &GOLD, &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original - 0.001;
            let minus = trainer.gradients(&input, &candidates, &GOLD, &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original;
            let expected = (plus - minus) / 0.002;
            assert!(
                (grads[tensor][at] as f64 - expected).abs() < 0.002,
                "tensor {tensor} at {at}: {} != {expected}",
                grads[tensor][at]
            );
        }
    }
}

/// Every combination of output form, boundary heads and prompt head.
fn combinations() -> Vec<Heads> {
    let mut out = Vec::new();
    for linear in [false, true] {
        for boundaries in [false, true] {
            for prompt in [false, true] {
                out.push(Heads { linear, boundaries, prompt, prompt_factor: 2 });
            }
        }
    }
    out
}

#[test]
fn gradients_match_finite_differences_for_every_head_combination() {
    let counts: Vec<usize> = combinations().into_iter().map(|heads| trainer(heads).params.len()).collect();
    // Linear output with a prompt head is six tensors, like no other mode.
    assert_eq!(counts, [4, 8, 12, 16, 2, 6, 10, 14]);
    for heads in combinations() {
        check_gradients(heads);
    }
}

#[test]
fn the_prompt_head_keeps_its_own_width() {
    // Output and boundary factor two, prompt factor four.
    for linear in [false, true] {
        for boundaries in [false, true] {
            let heads = Heads { linear, boundaries, prompt: true, prompt_factor: 4 };
            let prompt = &trainer(heads).params;
            assert_eq!(prompt[prompt.len() - 4].len(), 4);
            check_gradients(heads);
        }
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
        assert_eq!(trainer.params.len(), before_prompt + if heads.prompt { 4 } else { 0 });
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
    let all = trainer(Heads { boundaries: true, prompt: true, ..Heads::default() });
    let mut wrong = all.zero_grads();
    wrong[15].push(0.0);
    assert!(all.gradients(&input, &candidates, &GOLD, &mut wrong).is_err());
}
