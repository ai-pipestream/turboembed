//! BERT-family forward and backward, including packed biases and tables.

use turbo_trainer::encoder::{Dropout, Encoder, Geometry, Work};
use turbo_trainer::rng::Rng;

fn encoder() -> Encoder<f64> {
    let geo = Geometry {
        layers: 2,
        hidden: 4,
        heads: 1,
        intermediate: 8,
        eps: 1e-5,
        window: usize::MAX,
        global_every: 0,
        max_positions: 8,
        global_theta: 10_000.0,
        local_theta: 10_000.0,
    };
    Encoder::random_bert(geo, 8, 10, 2, 1, &mut Rng::new(31)).unwrap()
}

fn loss(e: &Encoder<f64>, ids: &[u32], dout: &[f64]) -> f64 {
    let mut w = Work::default();
    e.forward_at(ids, 11, &mut w).unwrap();
    w.out.iter().zip(dout).map(|(a, b)| a * b).sum()
}

#[test]
fn bert_gradients_match_finite_differences_under_dropout() {
    let e = encoder().with_dropout(Dropout { embedding: 0.2, attention: 0.3, mlp: 0.25, seed: 5 }).unwrap();
    let ids = [1, 2, 3];
    let dout: Vec<f64> = (0..12).map(|i| (i % 7) as f64 * 0.1 - 0.3).collect();
    let mut w = Work::default();
    e.forward_at(&ids, 11, &mut w).unwrap();
    assert!(w.out.iter().all(|v| v.is_finite()));
    let mut analytic = e.grads();
    e.backward(&w, &dout, &mut analytic).unwrap();
    for (s, slot) in analytic.iter().enumerate().filter(|(_, slot)| !slot.is_empty()) {
        for j in [0, slot.len() - 1] {
            let mut plus = e.clone();
            let mut minus = e.clone();
            plus.params_mut_for_test(s, j, 0.000_001);
            minus.params_mut_for_test(s, j, -0.000_001);
            let numeric = (loss(&plus, &ids, &dout) - loss(&minus, &ids, &dout)) / 0.000_002;
            assert!((numeric - slot[j]).abs() < 0.000_01, "slot {s} value {j}: {numeric} vs {}", slot[j]);
        }
    }
}

trait Perturb {
    fn params_mut_for_test(&mut self, slot: usize, index: usize, delta: f64);
}

impl Perturb for Encoder<f64> {
    fn params_mut_for_test(&mut self, slot: usize, index: usize, delta: f64) {
        let mut p = self.params().to_vec();
        p[slot][index] += delta;
        self.set_params(p).unwrap();
    }
}
