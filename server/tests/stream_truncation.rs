//! A text cut at the token limit is told (issue #2511). The default
//! truncation embeds a long text from its first tokens alone, and the
//! answer used to carry nothing that told that vector from the vector of
//! a whole text. Each AnalyzeStream vector and code row now carries the
//! tokens kept, the tokens the whole text gives, and whether it was cut;
//! ModelInfer counts the cut rows in its response parameters.

mod common;

use common::*;
use turbo_kserve::proto::infer_parameter::ParameterChoice;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{AnalyzeOptions, AnalyzeOutput, InputTokens};

/// Longer than the tiny bundle's max_seq of 64 tokens.
fn long() -> String {
    "the quick brown fox jumps over the lazy dog ".repeat(40)
}

fn tokens(o: &Outcome) -> InputTokens {
    match o {
        Outcome::Vectors(v) => v.tokens.expect("a vector row says what was tokenized"),
        Outcome::VectorCodes(v) => v.tokens.expect("a code row says what was tokenized"),
        other => panic!("not a vector: {other:?}"),
    }
}

#[tokio::test]
async fn a_cut_document_says_so_beside_its_vector() {
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("short", NAME, "a short text to embed", None).await;
    stream.send("long", NAME, &long(), None).await;
    let got = by_id(stream.half_close()).await;
    let short = tokens(&got["short"]);
    assert!(!short.truncated, "{short:?}");
    assert_eq!(short.seen, Some(short.kept), "{short:?}");
    assert!(short.kept > 2 && short.kept < 64, "{short:?}");
    let long = tokens(&got["long"]);
    assert!(long.truncated, "{long:?}");
    assert_eq!(long.kept, 64, "the bundle's max_seq: {long:?}");
    assert!(long.seen.is_some_and(|n| n > 64), "{long:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_cut_document_says_so_beside_its_codes() {
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    let options = AnalyzeOptions { quantization: turbo::TURBO_QUANTIZE_UBINARY, ..Default::default() };
    for (id, text) in [("short", "a short text to embed".to_owned()), ("long", long())] {
        let mut doc = stream_doc(id, NAME, &text, Some(options.clone()));
        if let Some(Kind::Document(d)) = &mut doc.kind {
            d.output = AnalyzeOutput::VectorCodes as i32;
        }
        stream.tx.send(doc).await.unwrap();
        stream.sent.push(id.into());
    }
    let got = by_id(stream.half_close()).await;
    assert!(!tokens(&got["short"]).truncated, "{:?}", got["short"]);
    let long = tokens(&got["long"]);
    assert_eq!((long.kept, long.truncated), (64, true), "{long:?}");
    s.stop().await;
}

/// A max_tokens of the request's own is the limit the row is told against.
#[tokio::test]
async fn the_requests_own_limit_is_the_one_told() {
    let (s, mut c) = serve().await;
    let mut stream = open_stream(&mut c, 0, 0, 0).await;
    stream.send("cut", NAME, &long(), Some(AnalyzeOptions { max_tokens: 16, ..Default::default() })).await;
    let got = by_id(stream.half_close()).await;
    let cut = tokens(&got["cut"]);
    assert_eq!((cut.kept, cut.truncated), (16, true), "{cut:?}");
    s.stop().await;
}

fn int_param(r: &turbo_kserve::proto::ModelInferResponse, name: &str) -> Option<i64> {
    match r.parameters.get(name)?.parameter_choice {
        Some(ParameterChoice::Int64Param(n)) => Some(n),
        _ => None,
    }
}

#[tokio::test]
async fn model_infer_counts_the_rows_it_cut() {
    let (s, _) = serve().await;
    let mut k = kserve(&s).await;
    let long = long();
    let cut = k.model_infer(texts_req(NAME, &[b"short", long.as_bytes(), b"another"])).await.unwrap().into_inner();
    assert_eq!(int_param(&cut, "truncated_rows"), Some(1), "{:?}", cut.parameters.keys());
    let whole = k.model_infer(texts_req(NAME, &[b"short"])).await.unwrap().into_inner();
    assert_eq!(int_param(&whole, "truncated_rows"), Some(0), "{:?}", whole.parameters.keys());
    s.stop().await;
}
