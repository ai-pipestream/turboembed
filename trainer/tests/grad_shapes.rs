//! `Tagger::loss` refuses gradient buffers shaped other than its weights
//! (too few tensors, none, or tensors of the wrong length), for every
//! head layout, before it runs: an error, never a panic, with the
//! caller's buffers and the weights untouched.

use turbo::manifest::{Adapter, Bilstm, Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::Tagger;
use turbo_trainer::rng::Rng;
use turbo_trainer::tagger::Work;

const SENTINEL: f32 = 7.25;

fn spec(decode: Decode, adapter: Option<(u32, u32)>) -> HeadSpec {
    let labels: &[&str] = if decode == Decode::Crf { &["O", "B-X", "I-X"] } else { &["A", "B", "C"] };
    HeadSpec {
        name: "probe".into(),
        layer: "turboembed:probe".into(),
        kind: HeadKind::TokenTagger,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme: if decode == Decode::Crf { Scheme::Bio } else { Scheme::Tags },
        word_pooling: WordPooling::First,
        adapter: adapter.map(|(hidden, layers)| Adapter { bilstm: Bilstm { hidden, layers } }),
        decode,
        biaffine: None,
        pos_input: None,
        artifact: "weights".into(),
        tensor_prefix: "probe".into(),
    }
}

/// Every wrongly shaped buffer set for weights of these lengths.
fn wrong(lens: &[usize]) -> Vec<(String, Vec<Vec<f32>>)> {
    let full = |lens: &[usize]| lens.iter().map(|&n| vec![SENTINEL; n]).collect::<Vec<_>>();
    let mut out = vec![
        ("no tensors".to_string(), Vec::new()),
        ("empty tensors".to_string(), lens.iter().map(|_| Vec::new()).collect()),
        ("one tensor short".to_string(), full(&lens[..lens.len() - 1])),
        ("one tensor too many".to_string(), {
            let mut g = full(lens);
            g.push(vec![SENTINEL; 1]);
            g
        }),
    ];
    if lens.len() > 2 {
        // A CRF or adapter head given only two tensors' worth.
        out.push(("two tensors only".to_string(), full(&lens[..2])));
    }
    for i in 0..lens.len() {
        for d in [-1i64, 1] {
            let mut g = full(lens);
            let n = (lens[i] as i64 + d).max(0) as usize;
            if n != lens[i] {
                g[i] = vec![SENTINEL; n];
                out.push((format!("tensor {i} of {n} values"), g));
            }
        }
    }
    out
}

#[test]
fn wrongly_shaped_gradients_are_refused_with_every_buffer_untouched() {
    let hidden = 3;
    let features = [0.5f32, -0.25, 1.0, 0.75, 0.0, -1.0];
    let mut checked = 0;
    for decode in [Decode::Argmax, Decode::Crf] {
        for adapter in [None, Some((2, 1)), Some((2, 2))] {
            let s = spec(decode, adapter);
            let t = Tagger::<f32>::new(&s, hidden, &mut Rng::new(5)).unwrap();
            let before = t.params().to_vec();
            let lens: Vec<usize> = before.iter().map(Vec::len).collect();
            for (what, mut grads) in wrong(&lens) {
                let snapshot = grads.clone();
                let r = t.loss(&features, &[0, 0], &mut grads, &mut Work::default());
                assert!(r.is_err(), "{decode:?} {adapter:?}, {what}: {r:?}");
                assert_eq!(grads, snapshot, "{decode:?} {adapter:?}, {what}: the buffers changed");
                assert_eq!(t.params(), &before[..], "{decode:?} {adapter:?}, {what}: the weights changed");
                checked += 1;
            }
            // The right shape still trains.
            let mut grads = t.zero_grads();
            assert!(t.loss(&features, &[0, 0], &mut grads, &mut Work::default()).unwrap().is_finite());
            assert!(grads.iter().flatten().any(|g| *g != 0.0));
        }
    }
    assert!(checked > 6 * 5, "{checked}");
}
