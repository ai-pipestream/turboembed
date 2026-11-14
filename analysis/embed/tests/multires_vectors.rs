//! Conformance and single-pass benchmark tests for `#2344` (`MultiResVectors`).

#[path = "vectors/common.rs"]
mod common;

use std::time::Instant;

use common::*;
use turbo_analysis::{Document, Ix, Span, Spans, View, ViewId};
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Chunks, Sentences, Subwords};
use turbo_sha256::Sha256;

fn matryoshka_bundle() -> Edited {
    Edited::new(|m| {
        m.embed.as_mut().unwrap().output_dims = vec![8, 16, 32];
    })
}

#[test]
fn single_pass_multires_matches_separate_matryoshka_passes_and_digests_across_batches() {
    let edited = matryoshka_bundle();
    let sub_ann = SubwordAnnotator::open(&edited.0).unwrap();
    let full_ann = VectorAnnotator::open(&edited.0, &VectorOptions::default()).unwrap();
    let m8_ann =
        VectorAnnotator::open(&edited.0, &VectorOptions { output_dim: 8, ..VectorOptions::default() }).unwrap();
    let m16_ann =
        VectorAnnotator::open(&edited.0, &VectorOptions { output_dim: 16, ..VectorOptions::default() }).unwrap();

    let mut docs = vec![
        {
            let (t, s) = text_of(3);
            document(&t, &s, &[(0, 2, 0, None), (2, 3, 0, None)])
        },
        {
            let (t, s) = text_of(4);
            document(&t, &s, &[(0, 2, 0, None), (2, 4, 0, None)])
        },
        {
            let (t, s) = text_of(2);
            document(&t, &s, &[(0, 2, 0, None)])
        },
        {
            let (t, s) = text_of(3);
            document(&t, &s, &[(0, 1, 0, None), (1, 3, 0, None)])
        },
    ];
    for d in &mut docs {
        let view = d.get::<View>().unwrap();
        let sw = sub_ann.pieces(view).unwrap();
        d.add(&turbo_analysis::Key::DEFAULT, sw, turbo_analysis::Provenance::by("test")).unwrap();
    }

    let opts = MultiResOptions {
        matryoshka_dims: vec![16, 8, 8],
        emit_row_sha256: true,
        #[cfg(feature = "turboquant")]
        turboquant: Some((4, false)),
        #[cfg(not(feature = "turboquant"))]
        turboquant: None,
    };

    let mr_inputs: Vec<MultiResInput<'_>> = docs
        .iter()
        .map(|d| MultiResInput {
            subwords: d.get::<Subwords>().unwrap(),
            sentences: Some(&d.get::<Sentences>().unwrap().spans),
            chunks: Some(&d.get::<Chunks>().unwrap().spans),
            doc_span: None,
        })
        .collect();

    let batch4: Vec<MultiResVectors> =
        full_ann.embed_multires("test", &mr_inputs, &opts).into_iter().map(Result::unwrap).collect();
    // All 4 documents, their 7 chunks, and their 12 sentences fit in a single
    // encoder session run!
    assert_eq!(full_ann.last_shape().runs, 1, "expected 1 encoder forward pass for batch of 4");

    // Batch-size invariance: batch of 1 and batch of 2 must produce bit-identical
    // vectors, inv_norm, and SHA-256 digests.
    for (idx, single_in) in mr_inputs.iter().enumerate() {
        let one = full_ann
            .embed_multires("test", std::slice::from_ref(single_in), &opts)
            .into_iter()
            .next()
            .unwrap()
            .unwrap();
        assert_eq!(one, batch4[idx], "batch=1 vs batch=4 mismatch on doc {idx}");
    }
    for (pair_idx, pair) in mr_inputs.chunks(2).enumerate() {
        let two: Vec<MultiResVectors> =
            full_ann.embed_multires("test", pair, &opts).into_iter().map(Result::unwrap).collect();
        assert_eq!(two[0], batch4[pair_idx * 2]);
        assert_eq!(two[1], batch4[pair_idx * 2 + 1]);
    }

    // Compare every tier bit-for-bit against dedicated VectorAnnotator runs.
    let plain_inputs: Vec<Input<'_>> = docs
        .iter()
        .map(|d| Input {
            subwords: d.get::<Subwords>().unwrap(),
            sentences: Some(&d.get::<Sentences>().unwrap().spans),
            chunks: Some(&d.get::<Chunks>().unwrap().spans),
            levels: None,
        })
        .collect();
    let ref_full = full_ann.embed("ref_full", &plain_inputs);
    let ref_8 = m8_ann.embed("ref_8", &plain_inputs);
    let ref_16 = m16_ann.embed("ref_16", &plain_inputs);

    for d in 0..docs.len() {
        let got = &batch4[d];
        let want_full = ref_full[d].as_ref().unwrap();
        let want_8 = ref_8[d].as_ref().unwrap();
        let want_16 = ref_16[d].as_ref().unwrap();

        assert_eq!(got.full.dim, 32);
        assert_eq!(got.full.data, want_full.chunks.as_ref().unwrap().data);
        assert_eq!(got.prefixes.len(), 2);
        assert_eq!(got.prefixes[0].dim, 8);
        assert_eq!(got.prefixes[0].data, want_8.chunks.as_ref().unwrap().data);
        assert_eq!(got.prefixes[1].dim, 16);
        assert_eq!(got.prefixes[1].data, want_16.chunks.as_ref().unwrap().data);

        let sent = got.sentences.as_ref().unwrap();
        assert_eq!(sent.dim, 32);
        assert_eq!(sent.data, want_full.sentences.as_ref().unwrap().data);

        for tier in [&got.full, &got.prefixes[0], &got.prefixes[1], sent] {
            let mut stream = Sha256::new();
            for v in &tier.data {
                stream.update(&v.to_le_bytes());
            }
            assert_eq!(tier.content_sha256, stream.finish().0);
            assert_eq!(tier.row_sha256.len(), tier.rows as usize);
            for r in 0..tier.rows as usize {
                let row = &tier.data[r * tier.dim as usize..(r + 1) * tier.dim as usize];
                assert!(is_unit(row));
                assert!((tier.inv_norm[r] - 1.0).abs() < 1e-5);
                assert!(tier.raw_inv_norm[r] > 0.0);
                assert_eq!(tier.row_sha256[r], row_le_sha256(row));
            }
        }

        #[cfg(feature = "turboquant")]
        {
            use turbo_vector_index::{Normalization, QuantizedIndex, Space};
            assert_eq!(got.full.turboquant_bits, 4);
            assert_eq!(got.full.turboquant_rows.len(), got.full.rows as usize);
            let q = QuantizedIndex::from_compact_bytes(&got.full.turboquant_compact, got.full.turboquant_rows.clone())
                .unwrap();
            let space = Space::new(got.model.clone(), 32, Normalization::Unit);
            let hits = q.top_k(&space, &got.full.data[..32], 1).unwrap();
            assert_eq!(hits[0].row, 0);
        }
    }

    // Whole-document mode (`chunks: None`) pools document + sentences in 1 pass,
    // and handles empty whitespace spans cleanly (`inv_norm == 0.0`).
    let ws_doc = Document::new("   ");
    let ws_sw = sub_ann.pieces(ws_doc.get::<View>().unwrap()).unwrap();
    let mut ws_sents = Spans::<Sentences>::new(ViewId::SOURCE);
    ws_sents.push(Span::new(0, 3));
    let ws_res = full_ann
        .embed_multires(
            "test",
            &[MultiResInput {
                subwords: &ws_sw,
                sentences: Some(&ws_sents),
                chunks: None,
                doc_span: Some(Span::new(0, 3)),
            }],
            &opts,
        )
        .remove(0)
        .unwrap();
    assert_eq!(ws_res.full.empty, vec![0]);
    assert_eq!(ws_res.full.inv_norm, vec![0.0]);
    assert_eq!(ws_res.sentences.as_ref().unwrap().empty, vec![0]);
    assert!(ws_res.full.turboquant_compact.is_empty());
    let _ = Ix::<Chunks>::new(0);

    // Reject undeclared Matryoshka dim.
    let bad = full_ann.embed_multires(
        "test",
        &mr_inputs[..1],
        &MultiResOptions { matryoshka_dims: vec![12], ..Default::default() },
    );
    assert!(bad[0].is_err());

    // Measure single-pass vs separate 3-tier pass timing receipt.
    let iters = 20usize;
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = full_ann.embed_multires("bench", &mr_inputs, &opts);
    }
    let single_pass_us = t0.elapsed().as_micros() as f64 / iters as f64;

    let t1 = Instant::now();
    for _ in 0..iters {
        let _ = full_ann.embed("bench", &plain_inputs);
        let _ = m8_ann.embed("bench", &plain_inputs);
        let _ = m16_ann.embed("bench", &plain_inputs);
    }
    let separate_us = t1.elapsed().as_micros() as f64 / iters as f64;
    eprintln!(
        "multires_2344_receipt: batch=4 single_pass_us={single_pass_us:.1} separate_3pass_us={separate_us:.1} speedup={:.2}x",
        separate_us / single_pass_us.max(1.0)
    );
}
