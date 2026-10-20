//! The analysis result layer over gRPC with a real client against the
//! server in-process on the CPU: every result kind crosses the wire and
//! back unchanged from its in-process layer, and the capability listing
//! follows LoadBundle and UnloadBundle. Nothing is mocked.

mod common;

use tonic::Code;
use turbo_analysis::{Ix, Key, Layer, Span, View, ViewId};
use turbo_analysis_embed::{Answers, Reranked};
use turbo_kserve::proto::result::*;
use turbo_kserve::results::Wire;
use turbo_kserve::results::{extraction::Extraction, rerank::RerankScores, tokens::TokenVectors};

use common::*;

#[tokio::test]
async fn sparse_terms_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let t = sparse_terms();
    let msg = <turbo_analysis_layers::SparseTerms<View> as Wire>::to_msg(&t);
    let back = c.round_trip_sparse_terms(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg, "the answer is the request");
    let decoded = turbo_analysis_layers::SparseTerms::from_msg(&back).unwrap();
    assert_eq!(decoded.model, t.model);
    assert_eq!(decoded.term, t.term);
    assert_eq!(decoded.weight, t.weight);
    assert_eq!(decoded.span, t.span);
    assert_eq!(decoded.occ, t.occ);
    assert_eq!(decoded.occ_at, t.occ_at);
}

#[tokio::test]
async fn token_vectors_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let (v, w) = token_vectors();
    let t = TokenVectors::new(&v, &w);
    let msg = t.to_msg();
    let back = c.round_trip_token_vectors(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = TokenVectors::from_msg(&back).unwrap();
    assert_eq!(decoded.dim, 3);
    assert_eq!(decoded.data, t.data);
    assert_eq!(decoded.empty, [1]);
    assert_eq!(decoded.view, 0);
    assert_eq!(decoded.windows, t.windows);
    assert_eq!(decoded.vectors().unwrap().data, v.data);
    assert_eq!(decoded.windows_layer(4).unwrap().spans.columns(), w.spans.columns());
}

#[tokio::test]
async fn rerank_scores_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let r = reranked();
    let m = RerankScores::new(&r);
    let msg = m.to_msg();
    let back = c.round_trip_rerank_scores(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = RerankScores::from_msg(&back).unwrap();
    assert_eq!(decoded.model, r.model);
    assert_eq!(decoded.source, "turbo:chunks");
    assert_eq!(decoded.instance, "candidates");
    assert_eq!(decoded.items, r.items.iter().map(|i| i.get() as u32).collect::<Vec<_>>());
    assert_eq!(decoded.scores, r.scores);
    // Named instances of one layer keep their identity on the wire.
    for name in ["shortlist-a", "shortlist-b"] {
        let r = Reranked {
            source: Key::<turbo_analysis_layers::Chunks>::named(name),
            items: vec![Ix::new(0)],
            scores: vec![1.0],
            model: model_ref(),
        };
        let decoded = RerankScores::from_msg(&RerankScores::new(&r).to_msg()).unwrap();
        assert_eq!(decoded.instance, name);
    }
}

#[tokio::test]
async fn extraction_answers_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let a = answers();
    let x = Extraction::new(&a, model_ref());
    let msg = x.to_msg();
    let back = c.round_trip_extraction(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = Extraction::from_msg(&back).unwrap();
    assert_eq!(decoded.model, x.model);
    assert_eq!(decoded.spans, x.spans);
    assert_eq!(decoded.scores, x.scores);
    assert_eq!(decoded.probabilities, x.probabilities);
    assert_eq!(decoded.answers().unwrap().spans.columns(), a.spans.columns());
}

#[tokio::test]
async fn discontinuous_answers_keep_their_fragments_over_the_wire() {
    let (_s, mut c) = serve().await;
    // An answer in two separated pieces: the envelope [0, 10), the
    // fragments [0, 2) and [8, 10). Without the fragment columns the
    // gap would decode as answer text.
    let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
    spans.push_fragments(&[Span::new(0, 2), Span::new(8, 10)]);
    let a = Answers { spans, scores: vec![6.0], probabilities: vec![0.8], null_score: -1.0, null_probability: 0.2 };
    a.check().unwrap();
    let x = Extraction::new(&a, model_ref());
    assert_eq!(x.spans, [Span::new(0, 10)]);
    assert_eq!(x.fragments, [Span::new(0, 2), Span::new(8, 10)]);
    let msg = x.to_msg();
    let back = c.round_trip_extraction(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = Extraction::from_msg(&back).unwrap();
    assert_eq!(decoded.fragments, [Span::new(0, 2), Span::new(8, 10)]);
    assert_eq!(decoded.frag_at, [2]);
    let layer = decoded.answers().unwrap();
    assert_eq!(layer.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), [Span::new(0, 2), Span::new(8, 10)]);
}

#[tokio::test]
async fn the_capability_listing_matches_the_loaded_bundle() {
    let (_s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let b = the_one_bundle(&list);
    let m = manifest();
    assert_eq!(b.name, NAME);
    assert_eq!(b.model.as_ref().unwrap().id, m["model"]["id"]);
    assert_eq!(b.model.as_ref().unwrap().revision, m["model"]["revision"]);
    assert!(!b.model.as_ref().unwrap().sha256.is_empty());
    assert_eq!(b.task, "TASK_EMBED");
    assert_eq!(b.roles, ["PROMPT_QUERY"], "the bundle carries a query prefix only");
    assert_eq!(b.widths, [m["embed"]["dim"].as_u64().unwrap() as u32]);
    assert_eq!(b.context_limit, m["embed"]["max_seq"].as_u64().unwrap() as u32);
    assert_eq!(b.batch_limit, m["embed"]["max_batch"].as_u64().unwrap() as u32);
    assert!(b.schema.is_none(), "no merged task reads an extraction schema yet");
    let limits = list.limits.as_ref().unwrap();
    assert!(limits.max_sparse_entries > 0 && limits.max_sparse_occurrences >= limits.max_sparse_entries);
    assert!(limits.max_token_vector_bytes > 0 && limits.max_rerank_items > 0 && limits.max_answers > 0);
}

#[tokio::test]
async fn the_listing_follows_load_and_unload() {
    let (s, mut c) = serve().await;
    // The same bundle under a second name, as one bundle at two tiers would be.
    let bundle = tiny().to_str().unwrap().to_string();
    let req = load_req("second", &bundle, 1, 0);
    let loaded = c.load_bundle(req).await.unwrap().into_inner();
    let b = loaded.capabilities.as_ref().unwrap();
    assert_eq!(b.name, "second");
    assert_eq!(b.widths, [32]);
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), [NAME, "second"]);

    // The new name serves inference too.
    let mut k = kserve(&s).await;
    assert_eq!(embed_shape(&mut k, "second", &[b"hello"]).await, [1, 32]);

    // Unload: the listing shrinks and the name is refused as never served.
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
    let gone = k.model_infer(texts_req("second", &[b"hello"])).await.expect_err("gone");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));
    let gone = c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.expect_err("not served");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));

    // The freed name serves again.
    c.load_bundle(load_req("second", &bundle, 1, 0)).await.unwrap();
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();

    // A name that is taken, and requests that cannot mean anything, refuse
    // by name; a bundle that cannot load leaves nothing served.
    let e = c.load_bundle(load_req(NAME, &bundle, 1, 0)).await.expect_err("the name is taken");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    for bad in [
        load_req("x", "", 1, 0),
        load_req("x", &bundle, 0, 0),
        load_req("x", &bundle, 1, 9),
        load_req("x", "/no/such/bundle", 1, 0),
    ] {
        c.load_bundle(bad).await.expect_err("refused");
    }
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
}
