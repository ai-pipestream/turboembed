//! A batch's gradients spread over threads: the same bytes for any
//! thread count, and within f32 rounding of the single-thread reference
//! (each example's loss added straight into one buffer, as the loop did
//! before), for a BiLSTM tagger with hashed embeddings under a CRF, the
//! heaviest head there is; a run on several threads trains the same
//! weights as a run on one, and logs and times its epochs.

use turbo::manifest::{Adapter, Bilstm, Decode, HashEmbed, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;
use turbo_trainer::{Config, Example, Tagger, batch_gradients, fit};

const HIDDEN: usize = 12;
const WORDS: [&str; 8] = ["Ann", "met", "Bob", "in", "Paris", "running", "the", "stone"];

fn spec() -> HeadSpec {
    HeadSpec {
        name: "t".into(),
        layer: "turboembed:test".into(),
        kind: HeadKind::TokenTagger,
        labels: ["O", "B-X", "I-X"].map(String::from).to_vec(),
        scheme: Scheme::Bio,
        word_pooling: WordPooling::First,
        adapter: Some(Adapter { bilstm: Bilstm { hidden: 8, layers: 2 } }),
        decode: Decode::Crf,
        biaffine: None,
        // 2048 rows of 8: past the size at which the sum splits by element.
        pos_input: None,
        hash_embed: Some(HashEmbed { dim: 8, rows: [1024, 256, 512, 256] }),
        layer_mix: None,
        span: None,
        relation: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "t".into(),
    }
}

/// Random sentences of 2 to 12 words, the label 0 or a B-X I-X* span.
fn data(seed: u64, n: usize) -> Vec<Example> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let len = 2 + rng.below(11);
            let (mut features, mut labels, mut forms) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..len {
                features.extend((0..HIDDEN).map(|_| rng.uniform(1.0) as f32));
                let prev = labels.last().copied().unwrap_or(0);
                labels.push(match rng.below(3) {
                    0 => 0,
                    1 => 1,
                    _ if prev == 0 => 1,
                    _ => 2,
                });
                forms.push(WORDS[rng.below(WORDS.len())].to_string());
            }
            Example { features, labels, forms }
        })
        .collect()
}

#[test]
fn bilstm_batch_gradients_match_the_single_thread_reference() {
    let s = spec();
    let mut rng = Rng::new(5);
    let mut model = Tagger::<f32>::new(&s, HIDDEN, &mut rng).unwrap();
    for i in 0..model.params().len() {
        model.param_mut(i).iter_mut().for_each(|v| *v += rng.uniform(0.2) as f32);
    }
    let examples = data(6, 13);

    // The reference: one buffer, each example's loss added in turn.
    let (mut want, mut want_loss, mut w) = (model.zero_grads(), 0.0, Work::default());
    for e in &examples {
        let forms = e.forms().unwrap();
        want_loss += model.loss_forms(&e.features, Some(&forms), &e.labels, &mut want, &mut w).unwrap();
    }

    let (one, one_loss) = batch_gradients(&model, &examples, 1).unwrap();
    for threads in [2, 4, 7, 32] {
        let (got, loss) = batch_gradients(&model, &examples, threads).unwrap();
        assert_eq!(loss.to_bits(), one_loss.to_bits(), "{threads} threads");
        for (i, (a, b)) in one.iter().zip(&got).enumerate() {
            let same = a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
            assert!(same, "{threads} threads: tensor {} differs", model.names()[i]);
        }
    }
    assert!((one_loss - want_loss).abs() <= 1e-9 * want_loss.abs(), "{one_loss} vs {want_loss}");
    let mut moved = 0;
    for (i, (a, b)) in one.iter().zip(&want).enumerate() {
        let scale = b.iter().fold(0f32, |m, v| m.max(v.abs()));
        for (j, (x, y)) in a.iter().zip(b).enumerate() {
            let tol = 1e-5 * (1.0 + scale);
            assert!((x - y).abs() <= tol, "{}[{j}]: batch {x}, reference {y}", model.names()[i]);
        }
        moved += b.iter().any(|v| *v != 0.0) as usize;
    }
    assert_eq!(moved, want.len(), "every tensor has a gradient");
}

#[test]
fn a_run_on_several_threads_trains_the_weights_of_a_run_on_one() {
    let s = spec();
    let (train, dev) = (data(7, 40), data(8, 10));
    let c = Config { epochs: 3, batch: 8, threads: 1, ..Config::default() };
    let (a, ra) = fit(&s, HIDDEN, &train, &dev, &c).unwrap();
    let (b, rb) = fit(&s, HIDDEN, &train, &dev, &Config { threads: 4, verbose: true, ..c }).unwrap();
    assert_eq!(a.tensors(), b.tensors());
    assert_eq!((&ra.train_loss, &ra.dev, ra.best_epoch), (&rb.train_loss, &rb.dev, rb.best_epoch));
    assert_eq!(rb.epoch_seconds.len(), rb.dev.len());
    assert!(rb.epoch_seconds.iter().all(|s| s.is_finite() && *s >= 0.0), "{:?}", rb.epoch_seconds);
}

#[test]
fn a_refused_example_is_reported_whatever_thread_runs_it() {
    let s = spec();
    let model = Tagger::<f32>::new(&s, HIDDEN, &mut Rng::new(1)).unwrap();
    let mut examples = data(9, 9);
    examples[6].forms.pop();
    for threads in [1, 3] {
        let e = batch_gradients(&model, &examples, threads).unwrap_err();
        assert!(e.message.contains("example 6"), "{e:?}");
    }
}
