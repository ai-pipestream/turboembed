//! Hybrid BM25 + dense retrieval tests by hand on the toy evaluation set.

use super::*;
use crate::analysis::{Analysis, Sentencer};
use crate::chunkers::SentenceWindows;
use crate::embedder::stub::Stub;
use crate::pipeline::tests::set;
use crate::pipeline::{Pipeline, Titles};
use crate::set::Query;

fn pipeline(hybrid: Hybrid) -> Pipeline {
    Pipeline {
        analysis: Analysis { sentences: Sentencer::Newline, structure: None },
        chunker: Box::new(SentenceWindows::new(1).unwrap()),
        embedder: Box::new(Stub::new(&["cat", "dog", "rust", "tea"])),
        titles: Titles::Ignore,
        lexical: None,
        hybrid: Some(hybrid),
    }
}

#[test]
fn weighted_hybrid_fuses_dense_and_bm25_by_hand() {
    // At bm25_weight = 0.0, min-max normalized dense scores [1, 1/sqrt(2), 0]
    // preserve the exact dense ranking and scores.
    let zero_w = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Weighted { bm25_weight: 0.0 } };
    let o0 = pipeline(zero_w).evaluate(&set(None)).unwrap();
    let half = f64::from(std::f32::consts::FRAC_1_SQRT_2);
    let q1_0: Vec<_> = o0.run.hits("q1").iter().map(|h| (h.doc.as_str(), h.score)).collect();
    assert_eq!(q1_0.iter().map(|h| h.0).collect::<Vec<_>>(), ["d1", "d2", "d3"]);
    assert!((q1_0[0].1 - 1.0).abs() < 1e-6);
    assert!((q1_0[1].1 - half).abs() < 1e-6);
    assert!((q1_0[2].1 - 0.0).abs() < 1e-6);
    assert_eq!(o0.cost.tokens_embedded, 32);

    // At bm25_weight = 0.5 on q1 ("cat"):
    // Chunks:
    //   c0 (d1 "cat purrs", 2 terms): dense = 1.0 (norm 1.0), bm25 = max (norm 1.0) -> fused = 1.0
    //   c3 (d2 "cat sleeps with dog", 4 terms): dense = 1/sqrt(2),
    //      bm25 = idf / (1 + 1.2 * (0.25 + 0.75 * 4 / 2.4)) vs c0 idf / (1 + 1.2 * (0.25 + 0.75 * 2 / 2.4))
    //      so norm_bm25(c3) = (1 + 1.2 * 0.875) / (1 + 1.2 * 1.5) = 2.05 / 2.80.
    //   All other chunks: dense = 0 (norm 0), bm25 = 0 (norm 0) -> fused = 0.0.
    let half_w = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Weighted { bm25_weight: 0.5 } };
    let o = pipeline(half_w.clone()).evaluate(&set(None)).unwrap();
    let q1: Vec<_> = o.run.hits("q1").iter().map(|h| (h.doc.as_str(), h.score)).collect();
    assert_eq!(q1.iter().map(|h| h.0).collect::<Vec<_>>(), ["d1", "d2", "d3"]);
    let want_d2 = 0.5 * half + 0.5 * (2.05 / 2.80);
    assert!((q1[0].1 - 1.0).abs() < 1e-6);
    assert!((q1[1].1 - want_d2).abs() < 1e-6, "got {}, want {want_d2}", q1[1].1);
    assert!((q1[2].1 - 0.0).abs() < 1e-6);

    // A query on "purrs", which is outside the stub embedder's 4-word
    // vocabulary: dense scores every chunk 0 (so dense alone ranks d3 > d2 > d1
    // by tie-break), whereas hybrid's BM25 component gives c0 (d1) norm_bm25 = 1.0
    // and ranks d1 first.
    let mut s = set(None);
    s.queries.push(Query { id: "q4".into(), text: "purrs".into() });
    let o_q4 = pipeline(half_w).evaluate(&s).unwrap();
    let q4: Vec<_> = o_q4.run.hits("q4").iter().map(|h| (h.doc.as_str(), h.score)).collect();
    assert_eq!(q4[0].0, "d1");
    assert!((q4[0].1 - 0.5).abs() < 1e-12);
}

#[test]
fn rrf_hybrid_fuses_reciprocal_ranks_by_hand() {
    let rrf = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Rrf { k: 60, bm25_weight: 0.5 } };
    let o = pipeline(rrf).evaluate(&set(None)).unwrap();
    // On q1 ("cat"), over the 5 chunks:
    // Dense order: c0 (d1 [0,9), score 1.0) rank 1; c3 (d2 [10,29), score 0.707) rank 2;
    //   c4 (d3 [0,9), 0) rank 3; c2 (d2 [0,9), 0) rank 4; c1 (d1 [10,23), 0) rank 5.
    // BM25 order (only chunks with > 0 matches): c0 rank 1, c3 rank 2; c4 has 0 BM25.
    // Fused chunk scores:
    //   c0 (d1): 0.5 / 61 + 0.5 / 61 = 1.0 / 61
    //   c3 (d2): 0.5 / 62 + 0.5 / 62 = 1.0 / 62
    //   c4 (d3): 0.5 / 63 + 0.0 = 0.5 / 63
    let q1: Vec<_> = o.run.hits("q1").iter().map(|h| (h.doc.as_str(), h.score)).collect();
    assert_eq!(q1.iter().map(|h| h.0).collect::<Vec<_>>(), ["d1", "d2", "d3"]);
    assert!((q1[0].1 - 1.0 / 61.0).abs() < 1e-12);
    assert!((q1[1].1 - 1.0 / 62.0).abs() < 1e-12);
    assert!((q1[2].1 - 0.5 / 63.0).abs() < 1e-12);
}

#[test]
fn the_query_s_own_document_is_left_out_in_both_fusions() {
    for fusion in [Fusion::Weighted { bm25_weight: 0.4 }, Fusion::Rrf { k: 60, bm25_weight: 0.4 }] {
        let mut s = set(None);
        s.queries.push(Query { id: "d2".into(), text: "dog cat".into() });
        let o = pipeline(Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion }).evaluate(&s).unwrap();
        assert!(o.run.hits("d2").iter().all(|h| h.doc != "d2"));
        assert!(o.chunks.hits("d2").iter().all(|h| h.doc != "d2"));
        assert_eq!(o.run.hits("d2")[0].doc, "d1");
    }
}

#[test]
fn the_record_names_hybrid_retrieval_and_settings() {
    let w = Hybrid {
        model: Bm25 { k1: 1.2, b: 0.75 },
        terms: TermRule::English,
        fusion: Fusion::Weighted { bm25_weight: 0.35 },
    };
    let d = pipeline(w).description();
    let r = d.retrieval.unwrap();
    assert_eq!(r.name, "hybrid");
    let kv: Vec<_> = r.config.into_iter().map(|s| (s.name, s.value)).collect();
    assert_eq!(
        kv,
        [
            ("fusion".into(), "weighted".into()),
            ("bm25_weight".into(), "0.35".into()),
            ("k1".into(), "1.2".into()),
            ("b".into(), "0.75".into()),
            ("terms".into(), "english".into()),
        ]
    );

    let rrf = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Rrf { k: 60, bm25_weight: 0.5 } };
    let r2 = pipeline(rrf).description().retrieval.unwrap();
    let kv2: Vec<_> = r2.config.into_iter().map(|s| (s.name, s.value)).collect();
    assert_eq!(
        kv2,
        [
            ("fusion".into(), "rrf".into()),
            ("rrf_k".into(), "60".into()),
            ("bm25_weight".into(), "0.5".into()),
            ("k1".into(), "1.2".into()),
            ("b".into(), "0.75".into()),
            ("terms".into(), "folded".into()),
        ]
    );
}

#[test]
fn invalid_fusion_and_mismatched_inputs_are_refused() {
    assert!(Fusion::Rrf { k: 0, bm25_weight: 0.5 }.check().is_err());
    assert!(Fusion::Rrf { k: 60, bm25_weight: -0.1 }.check().is_err());
    assert!(Fusion::Rrf { k: 60, bm25_weight: 1.1 }.check().is_err());
    assert!(Fusion::Weighted { bm25_weight: f64::NAN }.check().is_err());
    assert!(Fusion::Weighted { bm25_weight: 1.5 }.check().is_err());

    let h = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion: Fusion::Weighted { bm25_weight: 0.5 } };
    let texts = vec!["cat".to_string()];
    let bm25 = h.index(&texts, &[0]).unwrap();
    let docs = ["d".to_string()];
    let span = [Span::new(0, 3).unwrap()];
    let q = [("q", "cat")];
    let ok = HybridSearch { bm25: &bm25, dim: 2, vectors: &[1.0, 0.0], doc: &[0], span: &span, docs: &docs };
    assert!(h.search(&ok, &q, &[1.0, 0.0]).is_ok());
    assert!(h.search(&HybridSearch { dim: 0, ..ok }, &q, &[]).is_err());
    assert!(h.search(&HybridSearch { doc: &[], ..ok }, &q, &[1.0, 0.0]).is_err());
    assert!(h.search(&HybridSearch { span: &[], ..ok }, &q, &[1.0, 0.0]).is_err());
    assert!(h.search(&HybridSearch { vectors: &[1.0], ..ok }, &q, &[1.0, 0.0]).is_err());
    assert!(h.search(&ok, &q, &[1.0]).is_err());
    assert!(h.search(&HybridSearch { doc: &[1], ..ok }, &q, &[1.0, 0.0]).is_err());

    // Overflowing rows * dim or queries.len() * dim returns Err without panicking.
    let texts2 = vec!["cat".to_string(), "dog".to_string()];
    let bm25_2 = h.index(&texts2, &[0, 1]).unwrap();
    let docs2 = ["d1".to_string(), "d2".to_string()];
    let spans2 = [Span::new(0, 3).unwrap(), Span::new(0, 3).unwrap()];
    let overflow_chunks =
        HybridSearch { bm25: &bm25_2, dim: usize::MAX, vectors: &[], doc: &[0, 1], span: &spans2, docs: &docs2 };
    assert!(h.search(&overflow_chunks, &q, &[]).is_err());
    let q2 = [("q1", "cat"), ("q2", "dog")];
    let ok2_vecs = vec![0.0f32; 4];
    let ok2 = HybridSearch { bm25: &bm25_2, dim: 2, vectors: &ok2_vecs, doc: &[0, 1], span: &spans2, docs: &docs2 };
    assert!(h.search(&HybridSearch { dim: usize::MAX, ..ok2 }, &q2, &[]).is_err());

    // Non-finite vector entries or dot products are rejected before fusion.
    for fusion in [Fusion::Rrf { k: 60, bm25_weight: 0.5 }, Fusion::Weighted { bm25_weight: 0.5 }] {
        let h_f = Hybrid { model: Bm25::LUCENE, terms: TermRule::Folded, fusion };
        let nan_doc = HybridSearch {
            bm25: &bm25_2,
            dim: 1,
            vectors: &[f32::NAN, 1.0],
            doc: &[0, 1],
            span: &spans2,
            docs: &docs2,
        };
        assert!(h_f.search(&nan_doc, &[("q", "none")], &[1.0]).is_err());
        let ok_doc =
            HybridSearch { bm25: &bm25_2, dim: 1, vectors: &[1.0, 0.0], doc: &[0, 1], span: &spans2, docs: &docs2 };
        assert!(h_f.search(&ok_doc, &[("q", "none")], &[f32::NAN]).is_err());
        let huge_doc = HybridSearch {
            bm25: &bm25_2,
            dim: 2,
            vectors: &[f32::MAX, f32::MAX, 1.0, 0.0],
            doc: &[0, 1],
            span: &spans2,
            docs: &docs2,
        };
        assert!(h_f.search(&huge_doc, &[("q", "none")], &[f32::MAX, f32::MAX]).is_err());
    }
}
