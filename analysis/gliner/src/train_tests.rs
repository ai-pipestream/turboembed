use super::*;

fn trainer(params: Vec<Vec<f32>>, boundaries: bool, prompt: bool, recurrent: bool) -> OutputTrainer {
    OutputTrainer { hidden: 1, factor: 2, base: params.clone(), params, anchor: 0.0, boundaries, prompt, recurrent }
}

fn fixture() -> (OutputTrainer, TrainingInput, Vec<Candidate>) {
    let params = vec![vec![0.2, -0.1, 0.3, 0.4], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]];
    let trainer = trainer(params, false, false, false);
    let input = TrainingInput {
        words: vec![0.6, -0.2],
        projected_words: vec![0.6, -0.2],
        spans: vec![0.5, -0.3, 0.2, 0.8],
        prompts: vec![0.7],
        prompt_inputs: vec![0.4],
        pairs: vec![(0, 1), (1, 2)],
    };
    let candidates = vec![
        Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
        Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
    ];
    (trainer, input, candidates)
}

fn check_gradients(trainer: &mut OutputTrainer, input: &TrainingInput, candidates: &[Candidate]) {
    trainer.set_anchor(0.25).unwrap();
    for tensor in 0..trainer.params.len() {
        for at in 0..trainer.params[tensor].len() {
            let original = trainer.params[tensor][at];
            let mut grads = trainer.zero_grads();
            let loss = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut grads).unwrap();
            assert!(loss.is_finite());
            trainer.params[tensor][at] = original + 0.001;
            let plus = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original - 0.001;
            let minus = trainer.gradients(input, candidates, &[(0, 1, 0)], &mut trainer.zero_grads()).unwrap();
            trainer.params[tensor][at] = original;
            let expected = (plus - minus) / 0.002;
            assert!(
                (grads[tensor][at] as f64 - expected).abs() < 0.002,
                "tensor {tensor} at {at}: {} != {expected}, plus {plus}, minus {minus}",
                grads[tensor][at]
            );
        }
    }
}

#[test]
fn output_gradients_match_finite_differences() {
    let (mut trainer, input, candidates) = fixture();
    check_gradients(&mut trainer, &input, &candidates);
}

#[test]
fn linear_output_gradients_match_finite_differences() {
    let (_, input, candidates) = fixture();
    let params = vec![vec![0.3, -0.4], vec![0.2]];
    let mut trainer = trainer(params, false, false, false);
    check_gradients(&mut trainer, &input, &candidates);
}

#[test]
fn boundary_gradients_match_finite_differences() {
    let (output, input, candidates) = fixture();
    let mut params = output.params;
    for _ in 0..2 {
        params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
    }
    let mut trainer = trainer(params, true, false, false);
    check_gradients(&mut trainer, &input, &candidates);
}

#[test]
fn linear_boundary_gradients_match_finite_differences() {
    let (_, input, candidates) = fixture();
    let mut params = vec![vec![0.3, -0.4], vec![0.2]];
    for _ in 0..2 {
        params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
    }
    let mut trainer = trainer(params, true, false, false);
    check_gradients(&mut trainer, &input, &candidates);
}

#[test]
fn prompt_gradients_match_finite_differences() {
    let (output, input, candidates) = fixture();
    let mut params = output.params;
    params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
    let mut trainer = trainer(params, false, true, false);
    check_gradients(&mut trainer, &input, &candidates);
}

#[test]
fn all_head_gradients_match_finite_differences() {
    let (output, input, candidates) = fixture();
    let mut params = output.params;
    for _ in 0..3 {
        params.extend([vec![0.2, -0.1], vec![0.1, -0.2], vec![0.3, -0.4], vec![0.2]]);
    }
    let mut trainer = trainer(params, true, true, false);
    check_gradients(&mut trainer, &input, &candidates);
}

fn recurrent_fixture() -> (OutputTrainer, TrainingInput, Vec<Candidate>) {
    let shapes = [16, 4, 8, 2, 8, 4, 8, 2, 8, 4, 8, 2, 8, 4, 8, 2, 8, 4, 4, 4, 8, 4, 4, 4];
    let params: Vec<Vec<f32>> = shapes
        .iter()
        .enumerate()
        .map(|(tensor, &len)| (0..len).map(|at| (tensor * len + at) as f32 * 0.001 - 0.2).collect())
        .collect();
    let trainer = OutputTrainer {
        hidden: 2,
        factor: 2,
        base: params.clone(),
        params,
        anchor: 0.0,
        boundaries: true,
        prompt: true,
        recurrent: true,
    };
    let input = TrainingInput {
        words: vec![0.6, -0.2, 0.4, 0.3],
        projected_words: vec![0.6, -0.2, 0.4, 0.3],
        spans: vec![0.5, -0.3, 0.2, 0.8, 0.1, 0.4, -0.2, 0.6],
        prompts: vec![0.7, 0.2],
        prompt_inputs: vec![0.4, -0.1],
        pairs: vec![(0, 1), (1, 2)],
    };
    let candidates = vec![
        Candidate { start: 0, end: 1, label: 0, logit: 0.1, score: 0.6 },
        Candidate { start: 1, end: 2, label: 0, logit: 0.2, score: 0.7 },
    ];
    (trainer, input, candidates)
}

#[test]
fn recurrent_gradients_match_finite_differences() {
    let (mut trainer, input, candidates) = recurrent_fixture();
    check_gradients(&mut trainer, &input, &candidates);
}
