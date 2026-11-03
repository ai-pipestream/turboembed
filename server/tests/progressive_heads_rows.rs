//! Encoder forwards across a whole `AnalyzeDocumentProgressive` request
//! (#1832), counted at the encoder (`turbo_analysis_embed::forwarded_rows`)
//! so every path a request takes is in the count, not just the head
//! group's. Adding compatible head steps adds no encoder read: a request
//! of one head reads the rows a request of every head reads, a sentence
//! at a time and over the whole view.
//!
//! One test in its own binary: the count is the process's.

#[path = "progressive_heads/bundle.rs"]
mod bundle;
mod common;
#[path = "progressive_heads/harness.rs"]
mod harness;

use bundle::*;
use harness::*;
use turbo_analysis_embed::forwarded_rows;

const DOC: &str = "Zoë met Ann in São Paulo, and the quiet old dog slept by the river del Río. Ann walked \
                   the dog while Zoë watched the river and the old dog. Then Zoë and Ann left São Paulo.";

async fn rows(client: &mut common::Analysis, steps: &[&str]) -> u64 {
    let before = forwarded_rows();
    complete(&run(client, request(DOC, "", steps)).await);
    forwarded_rows() - before
}

#[tokio::test]
async fn compatible_head_steps_add_no_encoder_rows_to_a_request() {
    let dir = bundle("rows-all", |m| {
        let heads = m["heads"].as_array_mut().unwrap();
        heads.extend([morph_head(), span_head(), relation_head(&["PER", "LOC"]), split_head()]);
    });
    let (server, mut client) = serve_models(vec![model(&dir, "heads", MAX_SEQ)]).await;
    let base = ["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"];
    let with = |extra: &[&'static str]| -> Vec<&'static str> { base.iter().chain(extra).copied().collect() };

    assert_eq!(rows(&mut client, &base).await, 0, "rule steps read no encoder");
    let one = rows(&mut client, &with(&["turbo-tagger:pos"])).await;
    assert!(one > 3, "each sentence in several windows: {one} rows");
    for extra in [
        &["turbo-tagger:spans"][..],
        &["turbo-tagger:entities", "turbo-tagger:relations"],
        &[
            "turbo-tagger:pos",
            "turbo-tagger:morphology",
            "turbo-tagger:entities",
            "turbo-tagger:dependencies",
            "turbo-tagger:lemmas",
            "turbo-tagger:spans",
            "turbo-tagger:relations",
            "turbo-tagger:words",
        ],
        &["turbo-tagger:heads"],
    ] {
        assert_eq!(rows(&mut client, &with(extra)).await, one, "{extra:?}");
    }
    server.stop().await;

    // Over the whole view: the sentence head, coreference and every
    // sentence-scoped head from one read.
    let dir = bundle("rows-whole", |m| {
        m["heads"] = serde_json::json!([heads()[0].clone(), span_head(), sentence_head(), split_head(), coref_head()])
    });
    let (server, mut client) = serve_models(vec![model(&dir, "whole", MAX_SEQ)]).await;
    let tokens = ["turbo-tokenize:uax29-untyped"];
    let whole = |extra: &[&'static str]| -> Vec<&'static str> { tokens.iter().chain(extra).copied().collect() };
    let one = rows(&mut client, &whole(&["turbo-tagger:sentences"])).await;
    assert!(one > 3, "the view in several windows: {one} rows");
    let all = whole(&[
        "turbo-tagger:sentences",
        "turbo-tagger:pos",
        "turbo-tagger:spans",
        "turbo-tagger:words",
        "turbo-tagger:coref",
    ]);
    assert_eq!(rows(&mut client, &all).await, one);
    assert_eq!(rows(&mut client, &whole(&["turbo-tagger:heads"])).await, one);
    server.stop().await;
}
