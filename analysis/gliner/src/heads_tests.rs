//! The heads' BiLSTM on turbo::cpu::lstm (#859) against PyTorch's LSTM
//! formula in f64, its two biases added separately, on the real-weight
//! head slice: one sequence of 1, 2 and 40 projected words.
use super::{Heads, Work};
use turbo::safetensors::{Dtype, File};

const HEADS: &[u8] = include_bytes!("../tests/fixtures/heads.safetensors");

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// One direction of `torch.nn.LSTM` over `x` [n, k], states in row order.
fn torch(x: &[Vec<f64>], suffix: &str) -> Vec<Vec<f64>> {
    let f = File::parse("heads", HEADS).unwrap();
    let t = |name: &str| -> Vec<f64> {
        let rank = if name.starts_with("bias") { 1 } else { 2 };
        let tensor = f.get(&format!("rnn.lstm.{name}_l0{suffix}"), Dtype::F32, rank).unwrap();
        tensor.f32s().into_iter().map(f64::from).collect()
    };
    let (wi, wh, bi, bh) = (t("weight_ih"), t("weight_hh"), t("bias_ih"), t("bias_hh"));
    let (g, k) = (bi.len(), x[0].len());
    let h = g / 4;
    let (mut state, mut cell) = (vec![0.0; h], vec![0.0; h]);
    let order: Vec<usize> = if suffix.is_empty() { (0..x.len()).collect() } else { (0..x.len()).rev().collect() };
    let mut out = vec![Vec::new(); x.len()];
    for i in order {
        let dot = |w: &[f64], v: &[f64], r: usize, n: usize| (0..n).map(|c| w[r * n + c] * v[c]).sum::<f64>();
        let gate: Vec<f64> = (0..g).map(|r| dot(&wi, &x[i], r, k) + bi[r] + dot(&wh, &state, r, h) + bh[r]).collect();
        for j in 0..h {
            cell[j] = sigmoid(gate[h + j]) * cell[j] + sigmoid(gate[j]) * gate[2 * h + j].tanh();
            state[j] = sigmoid(gate[3 * h + j]) * cell[j].tanh();
        }
        out[i] = state.clone();
    }
    out
}

#[test]
fn the_bilstm_matches_torch_at_1_2_and_40_words() {
    let config =
        crate::Config { hidden: 16, max_width: 3, entity_token: 12, separator_token: 13, cls_token: 1, eos_token: 2 };
    let ec = turbo::deberta::Config {
        hidden: 32,
        heads: 2,
        intermediate: 48,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let heads = Heads::load(&config, &ec, HEADS).unwrap();
    let mut seed = 7u32;
    let mut work = Work::default();
    for n in [1, 2, 40] {
        let x: Vec<f32> = (0..n * 16)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (seed >> 8) as f32 / (1 << 24) as f32 * 4.0 - 2.0
            })
            .collect();
        heads.words(&x, n, &mut work).unwrap();
        let rows: Vec<Vec<f64>> = x.chunks(16).map(|r| r.iter().map(|&v| f64::from(v)).collect()).collect();
        let (fwd, bwd) = (torch(&rows, ""), torch(&rows, "_reverse"));
        let want = fwd.iter().zip(&bwd).flat_map(|(f, b)| f.iter().chain(b).copied());
        let worst = work.words.iter().zip(want).map(|(&a, b)| (f64::from(a) - b).abs()).fold(0.0, f64::max);
        assert_eq!(work.words.len(), n * 16);
        assert!(worst < 1e-5, "{n} words: max abs diff {worst:e}");
    }
}
