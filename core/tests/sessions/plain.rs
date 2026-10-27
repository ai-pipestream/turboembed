//! The encoder against plain arithmetic: every pooling and option
//! against PlainBert in f64, and the bundle's pooling and normalization
//! as the default.

use crate::common::*;
use crate::support::*;
use turbo::*;

/// Every pooling, with and without normalization, cut or not, with token
/// types and with padding on either side, against PlainBert in f64.
#[test]
fn every_option_matches_the_arithmetic_written_plainly() {
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let l = tiny();
    let s = session(&l);
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    // Row 0 right-padded, row 1 left-padded, row 2 with a masked token in
    // the middle and types of 1 after its first token.
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types.clone());

    let mut worst = 0f64;
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            s.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = s.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let (ids, mask, ty) = (&t.ids[at.clone()], &t.mask[at.clone()], &types[at]);
                let want = plain.embed(ids, mask, ty, pooling, 32, normalize == TURBO_NORMALIZE_L2);
                for (g, w) in row.iter().zip(&want) {
                    let d = (*g as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "pooling {pooling} normalize {normalize} row {r}: {g} vs {w}");
                }
            }
        }
    }
    println!("largest difference from the f64 encoder: {worst:.3e}");
}

#[test]
fn the_bundles_pooling_and_normalization_are_the_default() {
    let l = tiny();
    let mi = l.info();
    assert_eq!((mi.pooling, mi.normalize), (TURBO_POOLING_MEAN, TURBO_NORMALIZE_L2));
    let s = session(&l);
    let explicit = opts(|o| {
        o.pooling = TURBO_POOLING_MEAN;
        o.normalize = TURBO_NORMALIZE_L2;
    });
    assert_eq!(s.embed(&TEXTS, None).unwrap(), s.embed(&TEXTS, Some(&explicit)).unwrap());
    let raw = s.embed(&TEXTS, Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    for v in s.embed(&TEXTS, None).unwrap() {
        let n: f64 = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        assert!((n - 1.0).abs() < 1e-6, "{n}");
    }
    assert!(raw.iter().any(|v| (v.iter().map(|x| x * x).sum::<f32>().sqrt() - 1.0).abs() > 1e-3));
}
