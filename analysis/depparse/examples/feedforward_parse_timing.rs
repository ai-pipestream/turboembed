//! Times feedforward parsing with a model read back from bytes, the way a
//! deployed model is loaded.
//!
//! `cargo run --release -p turbo-depparse --example feedforward_parse_timing`
//!
//! Trains a small Chen-Manning model (embedding 50, hidden 200) on seeded
//! synthetic sentences, round-trips it through `to_bytes`/`from_bytes`, then
//! reports greedy and beam-4 parse time per sentence (best of five rounds).
use std::time::Instant;
use turbo_analysis::Ix;
use turbo_analysis_layers::Tokens;
use turbo_depparse::*;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 33) % n as u64) as usize
    }
}

fn sentence(rng: &mut Lcg, len: usize) -> (Vec<String>, Vec<String>, Vec<Option<u32>>, Vec<String>) {
    let words = (0..len).map(|_| format!("w{}", rng.next(3000))).collect();
    let tags = (0..len).map(|_| format!("T{}", rng.next(17))).collect();
    // A projective chain: each token's head is the next one; the last is the root.
    let heads = (0..len).map(|i| (i + 1 < len).then_some(i as u32 + 1)).collect();
    let labels = (0..len).map(|i| if i + 1 < len { format!("l{}", rng.next(30)) } else { "root".into() }).collect();
    (words, tags, heads, labels)
}

fn main() {
    let mut rng = Lcg(7);
    let train: Vec<DependencySample> = (0..300)
        .map(|_| {
            let (w, t, h, l) = sentence(&mut rng, 12);
            let graph = DependencyGraph::<Tokens>::new(h.into_iter().map(|h| h.map(Ix::new)).collect(), l).unwrap();
            DependencySample::new(w, t, graph).unwrap()
        })
        .collect();
    let settings = FeedforwardSettings {
        embedding_size: 50,
        hidden_size: 200,
        epochs: 1,
        batch_size: 64,
        learning_rate: 0.01,
        l2: 0.0,
        dropout: 0.0,
        word_cutoff: 1,
        seed: 17,
        layout: FeatureLayout::ChenManning18,
        activation: Activation::Cube,
    };
    let trained = FeedforwardDependencyTrainer::train(&train, &settings).unwrap();
    let model = FeedforwardDependencyModel::from_bytes(&trained.to_bytes().unwrap()).unwrap();
    let test: Vec<_> = (0..400).map(|_| sentence(&mut rng, 20)).collect();
    for beam in [1, 4] {
        let parser = FeedforwardDependencyParser::new(model.clone(), beam).unwrap();
        let mut best = f64::MAX;
        let mut arcs = 0usize;
        for _ in 0..5 {
            let start = Instant::now();
            for (w, t, _, _) in &test {
                let w = w.iter().map(String::as_str).collect::<Vec<_>>();
                let t = t.iter().map(String::as_str).collect::<Vec<_>>();
                let g = parser.parse::<Tokens>(&w, &t).unwrap();
                for i in 0..g.len() {
                    let kids = g.children(Ix::new(i as u32)).unwrap();
                    arcs = arcs.wrapping_mul(31).wrapping_add(kids.iter().map(|c| c.get() + 1).sum::<usize>());
                }
            }
            best = best.min(start.elapsed().as_secs_f64());
        }
        println!(
            "beam {beam}: {:.1} us/sentence ({} sentences of 20, checksum {arcs:x})",
            best * 1e6 / test.len() as f64,
            test.len()
        );
    }
}
