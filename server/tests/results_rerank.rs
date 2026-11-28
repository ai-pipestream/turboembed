//! Rerank scores and their query token counts over gRPC.

mod common;

use turbo_analysis::{Ix, Key};
use turbo_analysis_embed::Reranked;
use turbo_kserve::results::{Wire, rerank::RerankScores};

use common::*;

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
    assert_eq!(decoded.query.unwrap().seen, Some(20));
    // Named instances of one layer keep their identity on the wire.
    for name in ["shortlist-a", "shortlist-b"] {
        let r = Reranked {
            source: Key::<turbo_analysis_layers::Chunks>::named(name),
            items: vec![Ix::new(0)],
            scores: vec![1.0],
            model: model_ref(),
            query_tokens: None,
        };
        let decoded = RerankScores::from_msg(&RerankScores::new(&r).to_msg()).unwrap();
        assert_eq!(decoded.instance, name);
    }
}
