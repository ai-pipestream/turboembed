//! The biaffine parser in training: its loss gradients against central
//! finite differences in f64, and its arc and relation scores against
//! the core's inference path.

use turbo::head::parser::Work as CoreWork;
use turbo::manifest::{Adapter, Biaffine, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::Parser;
use turbo_trainer::parser::Work;
use turbo_trainer::rng::Rng;

const EPS: f64 = 1e-6;

fn spec(adapter: Option<(u32, u32)>) -> HeadSpec {
    HeadSpec {
        name: "deps".into(),
        layer: "turboembed:dependencies".into(),
        kind: HeadKind::BiaffineParser,
        labels: ["root", "nsubj", "obj"].map(String::from).to_vec(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode: Decode::Mst,
        biaffine: Some(Biaffine { arc: 3, rel: 2 }),
        pos_input: None,
        artifact: "weights".into(),
        tensor_prefix: "deps".into(),
    }
}

fn random(rng: &mut Rng, n: usize, scale: f64) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(scale)).collect()
}

#[test]
fn parser_loss_gradients_match_finite_differences() {
    let hidden = 4;
    // Words 1..4: 2 on the root, 1 and 3 on 2, 4 on 3.
    let (heads, rels) = ([2u32, 0, 2, 3], [1u32, 0, 2, 1]);
    for (k, adapter) in [None, Some((2, 1))].into_iter().enumerate() {
        let s = spec(adapter);
        let mut rng = Rng::new(40 + k as u64);
        let mut model = Parser::<f64>::new(&s, hidden, &mut rng).unwrap();
        for i in 0..model.params().len() {
            model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.4));
        }
        let x = random(&mut rng, heads.len() * hidden, 1.0);
        let mut grads = model.zero_grads();
        model.loss(&x, &heads, &rels, &mut grads, &mut Work::default()).unwrap();
        for (i, g) in grads.iter().enumerate() {
            let name = model.names()[i].clone();
            for (j, &analytic) in g.iter().enumerate() {
                let f = |v: f64| {
                    let mut m = model.clone();
                    m.param_mut(i)[j] = v;
                    m.loss(&x, &heads, &rels, &mut m.zero_grads(), &mut Work::default()).unwrap()
                };
                let p = model.params()[i][j];
                let numeric = (f(p + EPS) - f(p - EPS)) / (2.0 * EPS);
                let tol = 1e-6 * (1.0 + numeric.abs().max(analytic.abs()));
                assert!(
                    (numeric - analytic).abs() <= tol,
                    "case {k} {name}[{j}]: numeric {numeric}, analytic {analytic}"
                );
            }
        }
    }
}

#[test]
fn trained_weights_score_as_the_core_scores_them() {
    let hidden = 6;
    let s = spec(Some((3, 1)));
    let mut rng = Rng::new(9);
    let mut model = Parser::<f32>::new(&s, hidden, &mut rng).unwrap();
    for i in 0..model.params().len() {
        model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.3) as f32);
    }
    let core = model.to_core().unwrap();
    let x: Vec<f32> = (0..5 * hidden).map(|_| rng.uniform(1.0) as f32).collect();
    let mut w = Work::default();
    let n = model.forward(&x, &mut w).unwrap();
    let words: Vec<u32> = (0..n as u32).collect();
    let (mut arcs, mut cw) = (Vec::new(), CoreWork::default());
    core.arcs(&x, &words, &mut arcs, &mut cw).unwrap();
    let worst = w.arcs().iter().zip(&arcs).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(worst < 1e-4, "arc scores differ by {worst}");
}

#[test]
fn a_tagger_spec_or_bad_gold_is_refused() {
    let mut tagger = spec(None);
    tagger.kind = HeadKind::TokenTagger;
    tagger.biaffine = None;
    tagger.decode = Decode::Argmax;
    assert!(Parser::<f64>::new(&tagger, 4, &mut Rng::new(1)).unwrap_err().message.contains("not a biaffine parser"));
    let model = Parser::<f64>::new(&spec(None), 4, &mut Rng::new(1)).unwrap();
    let x = vec![0.1; 8];
    let mut g = model.zero_grads();
    let mut w = Work::default();
    assert!(model.loss(&x, &[0, 2], &[0], &mut g, &mut w).is_err(), "relations for every word");
    assert!(model.loss(&x, &[1, 0], &[0, 0], &mut g, &mut w).is_err(), "word 1 on itself");
    assert!(model.loss(&x, &[0, 3], &[0, 0], &mut g, &mut w).is_err(), "head past the words");
    assert!(model.loss(&x, &[0, 1], &[0, 3], &mut g, &mut w).is_err(), "relation past the labels");
}
