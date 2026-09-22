//! The model repository extension over REST: `/v2/repository/index`,
//! `/v2/repository/models/{name}/load` and `.../unload`.

mod common;

use axum::Router;
use serde_json::{json, Value};

/// A router over an engine of this test's own, so loads and unloads here do
/// not disturb the shared engine the other tests run against.
fn own_router() -> Router {
    let engine =
        common::engine_of(&[&format!("name=embed,bundle={},provider=mock,ordinal=1", common::bundle("embedding"))]);
    turbo_inferstream::http::router(engine)
}

/// The index of a router, as the array the route returns.
async fn index_of(router: &Router) -> Vec<Value> {
    let r = common::post_on(router.clone(), "/v2/repository/index", &json!({})).await;
    assert_eq!(r.status, 200, "POST /v2/repository/index: {}", r.text);
    r.json().as_array().unwrap_or_else(|| panic!("the index is not an array: {}", r.text)).clone()
}

/// The names in an index, in the order they came back.
fn names_of(index: &[Value]) -> Vec<&str> {
    index.iter().map(|m| m["name"].as_str().unwrap_or_else(|| panic!("a model without a name: {m}"))).collect()
}

#[tokio::test]
async fn repository_index_lists_every_served_model() {
    let r = common::post("/v2/repository/index", &json!({})).await;
    assert_eq!(r.status, 200, "POST /v2/repository/index: {}", r.text);
    let index = r.json();
    let index = index.as_array().expect("the index is an array");
    assert_eq!(names_of(index), ["chat", "classify", "embed", "rerank", "run", "tag"], "the index, in name order");
    for m in index {
        assert_eq!(m["version"], "1", "model {m} version");
        assert_eq!(m["state"], "READY", "model {m} state");
        assert_eq!(m["reason"], "", "a READY model has no reason: {m}");
        assert_eq!(m["provider"], "mock", "model {m} provider");
        assert_eq!(m["device"], "Mock accelerator", "model {m} device");
    }
    let embed = index.iter().find(|m| m["name"] == "embed").expect("`embed` in the index");
    assert_eq!(embed["kind"], "Embedding", "the index names the model's kind: {embed}");
    assert_eq!(embed["bundle"], common::bundle("embedding"), "the index names the bundle: {embed}");
}

#[tokio::test]
async fn load_serves_a_new_model_and_unload_takes_it_away() {
    let router = own_router();
    assert_eq!(names_of(&index_of(&router).await), ["embed"], "the test engine starts with one model");

    let loaded = common::post_on(
        router.clone(),
        "/v2/repository/models/rerank-2/load",
        &json!({"bundle": common::bundle("reranker"), "provider": "mock", "ordinal": 1}),
    )
    .await;
    assert_eq!(loaded.status, 200, "loading the reranker bundle: {}", loaded.text);
    assert_eq!(loaded.json()["name"], "rerank-2", "the load answers with the name it served: {}", loaded.text);

    let index = index_of(&router).await;
    assert_eq!(names_of(&index), ["embed", "rerank-2"], "the index does not show the loaded model");
    let loaded = index.iter().find(|m| m["name"] == "rerank-2").expect("`rerank-2` in the index");
    assert_eq!(loaded["kind"], "Reranker", "the loaded model's kind: {loaded}");
    assert_eq!(loaded["bundle"], common::bundle("reranker"), "the loaded model's bundle: {loaded}");

    let ready = common::get_on(router.clone(), "/v2/models/rerank-2/ready").await;
    assert_eq!(ready.status, 200, "the freshly loaded model is not ready: {}", ready.text);

    let r = common::post_on(
        router.clone(),
        "/v2/models/rerank-2/infer",
        &json!({
            "inputs": [
                {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["a query"]},
                {"name": "documents", "datatype": "BYTES", "shape": [3], "data": ["one", "two", "three"]},
            ]
        }),
    )
    .await;
    assert_eq!(r.status, 200, "inference on the freshly loaded model: {}", r.text);
    let body = r.json();
    assert_eq!(body["model_name"], "rerank-2", "the response names the model: {body}");
    assert_eq!(common::shape(common::output(&body, "scores")), [3], "one score per document: {body}");

    let unloaded = common::post_on(router.clone(), "/v2/repository/models/rerank-2/unload", &json!({})).await;
    assert_eq!(unloaded.status, 200, "unloading: {}", unloaded.text);
    assert_eq!(unloaded.json(), json!({"name": "rerank-2", "state": "UNAVAILABLE"}), "the unload answer");

    let ready = common::get_on(router.clone(), "/v2/models/rerank-2/ready").await;
    assert_eq!(ready.status, 404, "the unloaded model is still ready: {}", ready.text);
    assert_eq!(names_of(&index_of(&router).await), ["embed"], "the unloaded model is still in the index");
}

#[tokio::test]
async fn load_of_a_name_already_served_is_rejected_and_keeps_the_served_model() {
    let router = own_router();
    let r = common::post_on(
        router.clone(),
        "/v2/repository/models/embed/load",
        &json!({"bundle": common::bundle("reranker"), "provider": "mock", "ordinal": 1}),
    )
    .await;
    assert_eq!(r.status, 400, "loading a name that is already served: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "BAD_REQUEST", "the error object's status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("already served"),
        "the error does not say the name is taken: {body}"
    );
    let index = index_of(&router).await;
    assert_eq!(index[0]["kind"], "Embedding", "the rejected load replaced the served model: {}", index[0]);
}

#[tokio::test]
async fn load_without_a_bundle_is_a_bad_request() {
    let router = own_router();
    let empty = common::post_on(
        router.clone(),
        "/v2/repository/models/nothing/load",
        &json!({"bundle": "", "provider": "mock"}),
    )
    .await;
    assert_eq!(empty.status, 400, "an empty `bundle`: {}", empty.text);
    let body = empty.json();
    assert_eq!(body["status"], "BAD_REQUEST", "the error object's status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("`bundle` is required"),
        "the error does not name the missing field: {body}"
    );
    // A body without the key at all is refused by the body extractor with
    // the same error object, not axum's plain-text rejection.
    let missing =
        common::post_on(router.clone(), "/v2/repository/models/nothing/load", &json!({"provider": "mock"})).await;
    assert_eq!(missing.status, 400, "a body without a `bundle` key: {}", missing.text);
    let body = missing.json();
    assert_eq!(body["status"], "BAD_REQUEST", "the error object's status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("bundle"),
        "the rejection does not name the missing field: {body}"
    );
    assert_eq!(names_of(&index_of(&router).await), ["embed"], "a rejected load served something anyway");
}

#[tokio::test]
async fn load_of_a_missing_bundle_directory_reports_the_turbo_status() {
    let router = own_router();
    let r = common::post_on(
        router.clone(),
        "/v2/repository/models/gone/load",
        &json!({"bundle": format!("{}-missing", common::bundle("reranker")), "provider": "mock", "ordinal": 1}),
    )
    .await;
    assert_eq!(r.status, 404, "a bundle directory that is not there: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "TURBO_E_BUNDLE_NOT_FOUND", "the error object carries the Turbo code name: {body}");
    assert_eq!(body["field"], 0, "no option field was rejected: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("-missing"),
        "the error does not name the directory: {body}"
    );
    assert_eq!(names_of(&index_of(&router).await), ["embed"], "a failed load served something anyway");
}

#[tokio::test]
async fn unload_of_an_unknown_model_is_not_found() {
    let router = own_router();
    let r = common::post_on(router.clone(), "/v2/repository/models/nope/unload", &json!({})).await;
    assert_eq!(r.status, 404, "unloading a model that is not served: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "NOT_FOUND", "the error object's status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("no model named `nope`"),
        "the error does not name the model: {body}"
    );
}
