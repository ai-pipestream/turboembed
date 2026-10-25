//! Compare warm packed projections with the scalar loop replaced by #1085.
//! Run in release mode. The batch reference includes input/weight validation;
//! the measured public Head has no pooler. These are projection timings,
//! not model-inference throughput.

use std::{hint::black_box, time::Instant};

use turbo::{
    head::{Head, Linear},
    manifest::ClassifierKind,
};

#[inline(never)]
fn scalar_apply(l: &Linear, x: &[f32], out: &mut [f32]) {
    assert_eq!(x.len(), l.in_dim);
    assert_eq!(out.len(), l.out_dim);
    assert_eq!(l.bias().len(), l.out_dim);
    assert_eq!(l.weight().len(), l.out_dim.checked_mul(l.in_dim).unwrap());
    for (dst, (row, &b)) in out.iter_mut().zip(l.weight().chunks_exact(l.in_dim).zip(l.bias())) {
        let mut sum = b;
        for (&w, &xi) in row.iter().zip(x) {
            sum += w * xi;
        }
        *dst = sum;
    }
}

#[inline(never)]
fn scalar_batch(h: &Head, states: &[f32], out: &mut [f32]) {
    assert!(h.hidden > 0 && !h.labels.is_empty());
    assert_eq!((h.classifier.in_dim, h.classifier.out_dim), (h.hidden, h.labels.len()));
    h.classifier.validate("classifier").unwrap();
    assert!(states.len().is_multiple_of(h.hidden));
    assert_eq!(out.len(), states.len() / h.hidden * h.labels.len());
    assert!(states.iter().all(|v| v.is_finite()));
    for (state, dst) in states.chunks_exact(h.hidden).zip(out.chunks_exact_mut(h.labels.len())) {
        scalar_apply(&h.classifier, state, dst);
    }
}

fn layer(hidden: usize, labels: usize) -> Linear {
    Linear::new(
        hidden,
        labels,
        (0..hidden * labels).map(|i| ((i % 97) as f32 - 48.0) * 0.001).collect(),
        vec![0.01; labels],
    )
    .unwrap()
}

fn main() {
    let hidden = 768;
    for labels in [2, 32, 768] {
        let l = layer(hidden, labels);
        let x = vec![0.3; hidden];
        let mut out = vec![0.0; labels];
        for _ in 0..50 {
            l.apply(black_box(&x), black_box(&mut out));
            scalar_apply(&l, black_box(&x), black_box(&mut out));
        }
        let iterations = if labels < 768 { 5000 } else { 100 };
        for round in 0..6 {
            for packed in if round % 2 == 0 { [false, true] } else { [true, false] } {
                let t = Instant::now();
                for _ in 0..iterations {
                    if packed {
                        l.apply(black_box(&x), black_box(&mut out));
                    } else {
                        scalar_apply(black_box(&l), black_box(&x), black_box(&mut out));
                    }
                    black_box(&out);
                }
                println!(
                    "kind=single hidden={hidden} labels={labels} rows=1 round={round} packed={packed} ns_per_call={}",
                    t.elapsed().as_nanos() / iterations
                );
            }
        }
    }
    for (labels, rows) in [(32, 32), (768, 32)] {
        let h = Head {
            kind: ClassifierKind::Sequence,
            hidden,
            labels: (0..labels).map(|i| i.to_string()).collect(),
            pooler: None,
            classifier: layer(hidden, labels),
        };
        let x = vec![0.3; rows * hidden];
        let mut out = vec![0.0; rows * labels];
        for _ in 0..20 {
            h.logits_into(black_box(&x), black_box(&mut out)).unwrap();
            scalar_batch(&h, black_box(&x), black_box(&mut out));
        }
        let iterations = if labels < 768 { 200 } else { 20 };
        for round in 0..6 {
            for packed in if round % 2 == 0 { [false, true] } else { [true, false] } {
                let t = Instant::now();
                for _ in 0..iterations {
                    if packed {
                        h.logits_into(black_box(&x), black_box(&mut out)).unwrap();
                    } else {
                        scalar_batch(black_box(&h), black_box(&x), black_box(&mut out));
                    }
                    black_box(&out);
                }
                println!(
                    "kind=batch hidden={hidden} labels={labels} rows={rows} round={round} packed={packed} ns_per_call={}",
                    t.elapsed().as_nanos() / iterations
                );
            }
        }
    }
}
