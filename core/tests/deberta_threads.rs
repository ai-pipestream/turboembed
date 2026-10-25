#![cfg(feature = "cpu")]
//! The DeBERTa-v3 encoder on its pool of threads: the same bits for any
//! thread count, for a batch and for its rows alone, and for calls made
//! at once on one encoder.
#[path = "deberta_support/mod.rs"]
mod deberta_support;

use deberta_support::weights;
use turbo::deberta::{Config, Encoder};

/// Enough heads, layers and positions that every thread takes some of
/// each product and head.
fn config() -> Config {
    Config {
        hidden: 64,
        intermediate: 160,
        heads: 4,
        layers: 2,
        vocabulary: 50,
        max_sequence: 160,
        position_buckets: 32,
        max_relative_positions: 64,
        layer_norm_eps: 1e-7,
    }
}

fn encoder(threads: usize) -> Encoder {
    let c = config();
    Encoder::from_safetensors(c.clone(), &weights(&c)).unwrap().with_threads(threads).unwrap()
}

/// Sequences of 1 to 160 positions, with no mask, padding at the end,
/// and masked positions inside.
fn inputs() -> Vec<(Vec<u32>, Vec<bool>)> {
    let mut out = Vec::new();
    for (n, every) in [(1, 0), (7, 0), (64, 0), (65, 5), (130, 3), (160, 0)] {
        let ids = (0..n).map(|i| (i * 13 % 50) as u32).collect();
        let mask = (0..n).map(|i| every == 0 || i % every != 1).collect();
        out.push((ids, mask));
    }
    let mut padded: Vec<u32> = (0..100).map(|i| (i * 7 % 50) as u32).collect();
    padded.resize(150, 0);
    out.push((padded, (0..150).map(|i| i < 100).collect()));
    out
}

#[test]
fn every_thread_count_gives_the_bits_of_one_thread() {
    let one = encoder(1);
    assert_eq!(one.threads(), 1);
    let want: Vec<Vec<Vec<f32>>> = inputs().iter().map(|(ids, mask)| one.encode_layers(ids, mask).unwrap()).collect();
    for threads in [2, 3, 4, 7] {
        let e = encoder(threads);
        assert_eq!(e.threads(), threads);
        for (i, ((ids, mask), want)) in inputs().iter().zip(&want).enumerate() {
            assert_eq!(&e.encode_layers(ids, mask).unwrap(), want, "{threads} threads, input {i}");
            assert_eq!(e.encode(ids, mask).unwrap().data(), want.last().unwrap(), "{threads} threads, input {i}");
        }
    }
}

#[test]
fn a_batch_gives_each_row_the_bits_it_has_alone() {
    let one = encoder(1);
    let mut rows = inputs();
    // A row that fails fails alone.
    rows.insert(2, (vec![99], vec![true]));
    let views: Vec<(&[u32], &[bool])> = rows.iter().map(|(i, m)| (i.as_slice(), m.as_slice())).collect();
    for threads in [1, 3, 4] {
        let got = encoder(threads).encode_batch(&views);
        assert_eq!(got.len(), rows.len());
        for (i, ((ids, mask), r)) in rows.iter().zip(&got).enumerate() {
            match (one.encode(ids, mask), r) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.data(), b.data(), "{threads} threads, row {i}");
                    assert_eq!(a.mask(), b.mask());
                }
                (Err(a), Err(b)) => assert_eq!(a.message, b.message),
                (a, b) => panic!("row {i}: alone {:?}, in the batch {:?}", a.is_ok(), b.is_ok()),
            }
        }
    }
    assert!(encoder(2).encode_batch(&[]).is_empty());
}

#[test]
fn calls_made_at_once_share_the_encoder_and_its_bits() {
    let e = encoder(4);
    let want: Vec<Vec<f32>> = inputs().iter().map(|(ids, mask)| e.encode(ids, mask).unwrap().data().to_vec()).collect();
    std::thread::scope(|s| {
        for _ in 0..4 {
            s.spawn(|| {
                for _ in 0..5 {
                    for ((ids, mask), want) in inputs().iter().zip(&want) {
                        assert_eq!(e.encode(ids, mask).unwrap().data(), want);
                    }
                }
            });
        }
    });
}

#[test]
fn a_thread_count_outside_1_to_1024_is_refused() {
    let c = config();
    for n in [0, 1025] {
        let e = Encoder::from_safetensors(c.clone(), &weights(&c)).unwrap().with_threads(n);
        assert!(e.err().unwrap().message.contains("1 to 1024"), "{n}");
    }
}
