use turbo::manifest::Manifest;

#[test]
fn rerank_manifest_requires_a_named_relevance_head() {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("../../testdata/tiny-bert-bundle/manifest.json")).unwrap();
    v["task"] = "TASK_RERANK".into();
    v.as_object_mut().unwrap().remove("embed");
    v["rerank"] = serde_json::json!({"max_seq": 16, "max_batch": 2, "query_tokens": 4, "pair_template": "PAIR_BERT"});
    let error = Manifest::parse(&serde_json::to_vec(&v).unwrap()).unwrap_err();
    assert!(error.message.contains("classifier"), "{}", error.message);
}

#[path = "rerank/common.rs"]
mod common;
use std::path::Path;
use turbo::{
    bundle::Bundle,
    rerank::{RerankOptions, Reranker},
    tokenizer::Tokenizer,
};
fn fixture() -> common::Fixture {
    common::Fixture::new(Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap())
}

#[test]
fn real_encoder_pair_scores_match_reference_and_window_maximum() {
    let f = fixture();
    let mut r = Reranker::open(&f.dir, 1).unwrap();
    let cases = f.cases();
    let pairs: Vec<_> = cases.iter().map(|(q, p)| (q.as_str(), p.as_str())).collect();
    let actual = r.score(&pairs, RerankOptions::default()).unwrap();
    for (got, want) in actual.scores.iter().zip(f.reference()) {
        assert!((got - want).abs() < 1e-4, "{got} vs {want}");
    }
    assert_eq!(actual.model.manifest_sha256, Bundle::open(&f.dir).unwrap().manifest_sha256);
    let mut many = pairs.clone();
    many.extend_from_slice(&pairs);
    assert_eq!(r.score(&many, RerankOptions::default()).unwrap().scores, actual.scores.repeat(2));
    assert!(r.score(&[], RerankOptions::default()).unwrap().scores.is_empty());
    assert!(r.score(&pairs, RerankOptions { max_tokens: 33, ..Default::default() }).is_err());
    assert!(r.score(&pairs, RerankOptions { stride: u32::MAX, ..Default::default() }).is_err());
    assert_eq!(r.score(&pairs, RerankOptions::default()).unwrap().scores, actual.scores);
}

#[test]
fn pair_budgets_segment_ids_and_overlap_cover_the_passage() {
    let f = fixture();
    let b = Bundle::open(&f.dir).unwrap();
    let t = Tokenizer::load(&b).unwrap();
    let q = "query ".repeat(20);
    let p = "passage ".repeat(45);
    let windows = t.encode_pair(&q, &p, 8, 32, 7).unwrap();
    assert!(windows.len() > 1);
    for w in &windows {
        assert!(w.ids.len() <= 32);
        assert_eq!(w.ids[0], 101);
        assert_eq!(w.ids[9], 102);
        assert_eq!(*w.ids.last().unwrap(), 102);
        assert!(w.types[..10].iter().all(|&x| x == 0));
        assert!(w.types[10..].iter().all(|&x| x == 1));
    }
    assert_eq!(windows[0].passage.start, 0);
    assert_eq!(windows.last().unwrap().passage.end, 45);
    assert!(windows.windows(2).all(|w| w[1].passage.start <= w[0].passage.end));
    for (budget, limit) in [(29, 32), (8, 4), (u32::MAX, 32)] {
        assert!(t.encode_pair("q", "p", budget, limit, 0).is_err());
    }
}

#[test]
fn malformed_task_head_and_pair_template_are_named_errors() {
    let mut f = fixture();
    f.manifest["classifier"]["labels"] = serde_json::json!(["a", "b"]);
    f.save();
    assert!(Bundle::open(&f.dir).err().unwrap().message.contains("classifier"));
    f.manifest["classifier"]["labels"] = serde_json::json!(["LABEL_0"]);
    f.manifest["rerank"]["pair_template"] = "PAIR_UNKNOWN".into();
    f.save();
    assert!(Bundle::open(&f.dir).err().unwrap().message.contains("rerank.pair_template"));
    f.manifest["task"] = "TASK_UNKNOWN".into();
    f.save();
    assert!(Bundle::open(&f.dir).err().unwrap().message.contains("TASK_UNKNOWN"));
}
