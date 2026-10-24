#![cfg(feature = "turboquant")]

use turbo_analysis::ModelRef;
use turbo_vector_index::{IndexBuilder, Metric, Normalization, QuantizedIndex, Space, TurboQuant};

fn rows() -> Vec<Vec<f32>> {
    include_bytes!("data/potion-32-159.f32")
        .chunks_exact(1024)
        .map(|row| {
            let (values, _) = row.as_chunks::<4>();
            values.iter().map(|&b| f32::from_le_bytes(b)).collect()
        })
        .collect()
}
fn space() -> Space {
    Space::new(
        ModelRef {
            id: "minishlab/potion-base-8M".into(),
            revision: "bf8b056651a2c21b8d2565580b8569da283cab23".into(),
            sha256: "f65d0f325faadc1e121c319e2faa41170d3fa07d8c89abd48ca5358d9a223de2".into(),
        },
        256,
        Normalization::None,
    )
}
fn index(bits: u8, calibrate: bool, n: usize) -> QuantizedIndex<usize> {
    let rows = rows();
    let mut b = IndexBuilder::new(space(), Metric::Cosine).unwrap();
    for i in 0..n {
        b.push(&rows[i % rows.len()], i).unwrap();
    }
    b.build_quantized(TurboQuant::new(bits, calibrate).unwrap()).unwrap()
}

#[test]
fn compact_real_model_codes_preserve_space_calibration_and_scores() {
    for calibrate in [false, true] {
        let mut sizes = Vec::new();
        for bits in 2..=4 {
            let original = index(bits, calibrate, 1024);
            let bytes = original.to_compact_bytes();
            let loaded = QuantizedIndex::from_compact_bytes(&bytes, (0..1024).collect()).unwrap();
            assert_eq!(loaded.space(), original.space());
            assert_eq!(loaded.metric(), original.metric());
            assert_eq!(loaded.config(), original.config());
            assert_eq!(*loaded.meta(127), 127);
            assert_eq!(loaded.to_compact_bytes(), bytes);
            for q in rows().iter().step_by(17) {
                assert_eq!(loaded.top_k(&space(), q, 10).unwrap(), original.top_k(&space(), q, 10).unwrap());
            }
            sizes.push(bytes.len());
        }
        assert_eq!(sizes[2] - sizes[1], 1024 * 256 / 8);
        assert_eq!(sizes[1] - sizes[0], 1024 * 256 / 8);
    }
}

#[test]
fn compact_images_refuse_truncated_lengths_flags_and_bad_values() {
    let bytes = index(3, true, 128).to_compact_bytes();
    let load = |b: &[u8]| QuantizedIndex::from_compact_bytes(b, (0..128).collect::<Vec<_>>());
    for end in (0..bytes.len()).step_by(97).chain([bytes.len() - 1]) {
        assert!(load(&bytes[..end]).is_err(), "accepted prefix {end}");
    }
    for (at, value) in [(0, b'X'), (8, b'X'), (48, 9), (49, 2), (50, 2), (51, 2)] {
        let mut bad = bytes.clone();
        bad[at] = value;
        assert!(load(&bad).is_err());
    }
    let mut bad = bytes.clone();
    bad[52..56].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(load(&bad).is_err());
    let mut bad = bytes.clone();
    bad[56..64].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(load(&bad).is_err());
    for scale in [0.0, -1.0, 1e-100, 1e100, f64::NAN, f64::INFINITY] {
        let mut bad = bytes.clone();
        bad[64..72].copy_from_slice(&scale.to_le_bytes());
        assert!(load(&bad).is_err());
    }
    let mut bad = bytes.clone();
    bad.push(0);
    assert!(load(&bad).is_err());
    assert!(QuantizedIndex::from_compact_bytes(&bytes, vec![(); 127]).is_err());
}

#[test]
fn large_real_model_index_matches_an_exhaustive_estimate_sort() {
    for bits in [2, 4] {
        let index = index(bits, false, 32_769);
        let raw = turbovec::TurboQuantIndex::from_bytes(&index.to_bytes()).unwrap();
        for q in rows().iter().step_by(43) {
            let norm = q.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>().sqrt();
            let q: Vec<f32> = q.iter().map(|&x| (f64::from(x) / norm) as f32).collect();
            let all = raw.try_search(&q, raw.len()).unwrap();
            let mut expected: Vec<_> = all.indices.iter().zip(&all.scores).collect();
            expected.sort_by(|(ai, ascore), (bi, bscore)| bscore.total_cmp(ascore).then(ai.cmp(bi)));
            let actual = index.top_k(&space(), &q, 10).unwrap();
            assert_eq!(
                actual.iter().map(|h| h.row as i64).collect::<Vec<_>>(),
                expected[..10].iter().map(|x| *x.0).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn compact_dot_scores_keep_the_original_vector_scale_and_empty_space() {
    let mut b = IndexBuilder::new(space(), Metric::Dot).unwrap();
    let source = rows();
    for (i, row) in source.iter().enumerate() {
        b.push(&row.iter().map(|x| x * 1000.0).collect::<Vec<_>>(), i).unwrap();
    }
    let original = b.build_quantized(TurboQuant::new(3, false).unwrap()).unwrap();
    let loaded = QuantizedIndex::from_compact_bytes(&original.to_compact_bytes(), (0..128).collect()).unwrap();
    assert_eq!(loaded.metric(), Metric::Dot);
    assert_eq!(loaded.top_k(&space(), &source[19], 10).unwrap(), original.top_k(&space(), &source[19], 10).unwrap());
    let empty = IndexBuilder::<()>::new(space(), Metric::Dot)
        .unwrap()
        .build_quantized(TurboQuant::new(3, true).unwrap())
        .unwrap();
    let loaded = QuantizedIndex::<()>::from_compact_bytes(&empty.to_compact_bytes(), vec![]).unwrap();
    assert!(loaded.top_k(&space(), &source[19], 10).unwrap().is_empty());
}
