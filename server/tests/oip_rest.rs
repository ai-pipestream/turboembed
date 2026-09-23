//! The Open Inference Protocol v2 REST binding under `/v2`, over the six
//! mock bundles on the mock accelerator.

mod common;

use common::{floats, get, output, post, shape, strings};
use serde_json::{json, Value};
use turbo_inferstream::oip::{SERVER_NAME, SERVER_VERSION};

/// Rows of an FP32 tensor declared `[rows, width]`.
fn rows(tensor: &Value) -> Vec<Vec<f32>> {
    let s = shape(tensor);
    assert_eq!(s.len(), 2, "tensor `{}` is not two-dimensional: {s:?}", tensor["name"]);
    floats(tensor).chunks(s[1] as usize).map(<[f32]>::to_vec).collect()
}

#[tokio::test]
async fn server_metadata_names_the_server_and_the_turbo_extensions() {
    let r = get("/v2").await;
    assert_eq!(r.status, 200, "GET /v2: {}", r.text);
    let body = r.json();
    assert_eq!(body["name"], SERVER_NAME, "server name in {body}");
    assert_eq!(body["version"], SERVER_VERSION, "server version in {body}");
    assert_eq!(
        body["extensions"],
        json!(["turbo_parameters", "turbo_stream_infer", "turbo_model_repository"]),
        "extensions in {body}"
    );
}

#[tokio::test]
async fn health_live_and_ready_are_ok_while_models_are_loaded() {
    assert_eq!(get("/v2/health/live").await.status, 200, "GET /v2/health/live");
    assert_eq!(get("/v2/health/ready").await.status, 200, "GET /v2/health/ready");
    assert_eq!(get("/health").await.status, 200, "GET /health");
}

#[tokio::test]
async fn model_listing_carries_every_served_model() {
    let r = get("/v2/models").await;
    assert_eq!(r.status, 200, "GET /v2/models: {}", r.text);
    let body = r.json();
    let names: Vec<&str> =
        body["models"].as_array().expect("models array").iter().filter_map(|m| m["name"].as_str()).collect();
    assert_eq!(names, ["chat", "classify", "embed", "rerank", "run", "tag"], "models listed by /v2/models");
    let embed = body["models"]
        .as_array()
        .expect("models array")
        .iter()
        .find(|m| m["name"] == "embed")
        .expect("the embed model");
    assert_eq!(embed["outputs"][0]["name"], "embeddings", "the listing carries each model's metadata: {embed}");
}

#[tokio::test]
async fn model_metadata_names_the_tensors_of_every_kind() {
    let cases: [(&str, Value, Value); 6] = [
        (
            "embed",
            json!([{"name": "text", "datatype": "BYTES", "shape": [-1]}]),
            json!([{"name": "embeddings", "datatype": "FP32", "shape": [-1, 8]}]),
        ),
        (
            "rerank",
            json!([{"name": "query", "datatype": "BYTES", "shape": [1]}, {"name": "documents", "datatype": "BYTES", "shape": [-1]}]),
            json!([{"name": "scores", "datatype": "FP32", "shape": [-1]}, {"name": "sorted", "datatype": "INT32", "shape": [-1]}]),
        ),
        (
            "classify",
            json!([{"name": "text", "datatype": "BYTES", "shape": [-1]}]),
            json!([{"name": "scores", "datatype": "FP32", "shape": [-1, 3]}, {"name": "labels", "datatype": "BYTES", "shape": [3]}]),
        ),
        (
            "tag",
            json!([{"name": "text", "datatype": "BYTES", "shape": [-1]}]),
            json!([{"name": "spans", "datatype": "BYTES", "shape": [-1]}, {"name": "labels", "datatype": "BYTES", "shape": [3]}]),
        ),
        (
            "chat",
            json!([{"name": "prompt", "datatype": "BYTES", "shape": [1]}, {"name": "messages", "datatype": "BYTES", "shape": [-1]}]),
            json!([{"name": "text", "datatype": "BYTES", "shape": [1]}]),
        ),
        (
            "run",
            json!([{"name": "x", "datatype": "FP32", "shape": [-1, -1]}]),
            json!([{"name": "y", "datatype": "FP32", "shape": [-1, -1]}]),
        ),
    ];
    for (name, inputs, outputs) in cases {
        let r = get(&format!("/v2/models/{name}")).await;
        assert_eq!(r.status, 200, "GET /v2/models/{name}: {}", r.text);
        let body = r.json();
        assert_eq!(body["name"], name, "metadata name of `{name}`");
        assert_eq!(body["versions"], json!(["1"]), "versions of `{name}`");
        assert_eq!(body["platform"], "mock", "platform of `{name}`");
        assert_eq!(body["inputs"], inputs, "input tensors of `{name}`");
        assert_eq!(body["outputs"], outputs, "output tensors of `{name}`");
    }
}

#[tokio::test]
async fn model_metadata_properties_report_the_bundle_and_the_device() {
    let body = get("/v2/models/embed").await.json();
    let p = &body["properties"];
    for (key, want) in [
        ("model_id", "turbo/mock-embedding"),
        ("revision", "mock"),
        ("task", "embed"),
        ("kind", "Embedding"),
        ("max_seq", "16"),
        ("max_batch", "8"),
        ("served_max_seq", "16"),
        ("served_max_batch", "8"),
        ("device", "Mock accelerator"),
        ("provider", "mock"),
        ("pooling", "mean"),
        ("normalize", "l2"),
    ] {
        assert_eq!(p[key], want, "property `{key}` of the embed model: {p}");
    }
}

#[tokio::test]
async fn model_version_1_is_the_model_and_version_2_does_not_exist() {
    let one = get("/v2/models/embed/versions/1").await;
    assert_eq!(one.status, 200, "GET /v2/models/embed/versions/1: {}", one.text);
    assert_eq!(
        one.json(),
        get("/v2/models/embed").await.json(),
        "version 1 is the same metadata as the unversioned path"
    );

    let two = get("/v2/models/embed/versions/2").await;
    assert_eq!(two.status, 404, "GET /v2/models/embed/versions/2: {}", two.text);
    assert_eq!(two.json()["status"], "NOT_FOUND", "the error object of version 2: {}", two.text);
}

#[tokio::test]
async fn model_ready_answers_per_model_and_per_version() {
    assert_eq!(get("/v2/models/embed/ready").await.status, 200, "a served model is ready");
    assert_eq!(get("/v2/models/nope/ready").await.status, 404, "an unknown model is not found");
    assert_eq!(get("/v2/models/embed/versions/1/ready").await.status, 200, "version 1 of a served model is ready");
    assert_eq!(get("/v2/models/embed/versions/2/ready").await.status, 404, "version 2 does not exist");
}

#[tokio::test]
async fn infer_embedding_returns_one_l2_normalized_row_per_text() {
    let r = post(
        "/v2/models/embed/infer",
        &json!({"id": "r1", "inputs": [{"name": "text", "datatype": "BYTES", "shape": [2], "data": ["hello world", "goodbye"]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "embedding infer: {}", r.text);
    let body = r.json();
    assert_eq!(body["id"], "r1", "the response echoes the request id: {body}");
    assert_eq!(body["model_name"], "embed", "model_name in {body}");
    assert_eq!(body["model_version"], "1", "model_version in {body}");
    assert_eq!(body["outputs"].as_array().expect("outputs").len(), 1, "an embedding request has one output: {body}");
    let t = output(&body, "embeddings");
    assert_eq!(t["datatype"], "FP32", "the embeddings datatype");
    assert_eq!(shape(t), [2, 8], "the embeddings shape is [texts, dim]");
    for (i, row) in rows(t).iter().enumerate() {
        let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "row {i} is not l2 normalized (norm {norm}) although the bundle says normalize=l2"
        );
    }
    assert!(body["parameters"]["device_ms"].is_number(), "device_ms is reported: {body}");
    assert_eq!(body["parameters"]["placement"], "Host", "the mock leaves its result on the host: {body}");
}

#[tokio::test]
async fn infer_reranker_scores_every_document_and_ranks_them_best_first() {
    let r = post(
        "/v2/models/rerank/infer",
        &json!({"inputs": [
            {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["alpha beta"]},
            {"name": "documents", "datatype": "BYTES", "shape": [3], "data": ["alpha beta gamma", "zulu", "alpha"]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "rerank infer: {}", r.text);
    let body = r.json();
    let scores = output(&body, "scores");
    assert_eq!(scores["datatype"], "FP32", "the scores datatype");
    assert_eq!(shape(scores), [3], "one score per document");
    let s = floats(scores);
    let sorted = output(&body, "sorted");
    assert_eq!(sorted["datatype"], "INT32", "the sorted datatype");
    assert_eq!(shape(sorted), [3], "the ranking covers every document");
    let order: Vec<i64> =
        sorted["data"].as_array().expect("sorted data").iter().map(|v| v.as_i64().expect("an index")).collect();
    let mut by_score = order.clone();
    by_score.sort_by(|&a, &b| s[b as usize].total_cmp(&s[a as usize]));
    assert_eq!(order, by_score, "`sorted` is not the documents ranked by `scores` {s:?}");
    assert_eq!(order[0], 2, "`alpha` overlaps the query most, so it ranks first; scores {s:?}");
}

#[tokio::test]
async fn infer_reranker_honors_top_n_and_raw_scores() {
    let inputs = json!([
        {"name": "query", "datatype": "BYTES", "shape": [1], "data": ["alpha beta"]},
        {"name": "documents", "datatype": "BYTES", "shape": [3], "data": ["alpha beta gamma", "zulu", "alpha"]}]);
    let top = post("/v2/models/rerank/infer", &json!({"parameters": {"top_n": 2}, "inputs": inputs})).await;
    assert_eq!(top.status, 200, "rerank with top_n: {}", top.text);
    let body = top.json();
    assert_eq!(shape(output(&body, "scores")), [3], "top_n cuts the ranking, not the scores");
    assert_eq!(shape(output(&body, "sorted")), [2], "top_n 2 keeps two ranks");

    let raw = post("/v2/models/rerank/infer", &json!({"parameters": {"raw_scores": true}, "inputs": inputs})).await;
    assert_eq!(raw.status, 200, "rerank with raw_scores: {}", raw.text);
    let raw_scores = floats(output(&raw.json(), "scores"));
    let activated = floats(output(&top.json(), "scores"));
    assert!(
        raw_scores.iter().any(|&x| !(0.0..=1.0).contains(&x)),
        "raw_scores returned activated scores {raw_scores:?}"
    );
    assert!(
        activated.iter().all(|&x| (0.0..=1.0).contains(&x)),
        "without raw_scores the bundle's sigmoid applies: {activated:?}"
    );

    let bad = post("/v2/models/rerank/infer", &json!({"parameters": {"top_n": 9}, "inputs": inputs})).await;
    assert_eq!(bad.status, 400, "top_n past the document count: {}", bad.text);
    assert_eq!(bad.json()["field"], 4, "the error names the top_n field: {}", bad.text);
}

#[tokio::test]
async fn infer_classifier_returns_a_score_per_label_and_the_labels() {
    let r = post(
        "/v2/models/classify/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [2], "data": ["good", "bad"]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "classify infer: {}", r.text);
    let body = r.json();
    let scores = output(&body, "scores");
    assert_eq!(scores["datatype"], "FP32", "the scores datatype");
    assert_eq!(shape(scores), [2, 3], "the scores shape is [texts, labels]");
    for (i, row) in rows(scores).iter().enumerate() {
        let total: f32 = row.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "row {i} sums to {total}, not 1, although the bundle says activation=softmax"
        );
    }
    let labels = output(&body, "labels");
    assert_eq!(labels["datatype"], "BYTES", "the labels datatype");
    assert_eq!(shape(labels), [3], "the labels shape");
    assert_eq!(strings(labels), ["negative", "neutral", "positive"], "the bundle's labels, in score order");
}

#[tokio::test]
async fn infer_token_classifier_returns_one_json_object_per_span() {
    let r = post(
        "/v2/models/tag/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["Ada lives in Paris"]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "token-classify infer: {}", r.text);
    let body = r.json();
    let spans = output(&body, "spans");
    assert_eq!(spans["datatype"], "BYTES", "the spans datatype");
    let items = strings(spans);
    assert_eq!(shape(spans), [items.len() as i64], "the spans shape counts the spans");
    assert!(!items.is_empty(), "the mock tagger found no spans in `Ada lives in Paris`");
    let labels = strings(output(&body, "labels"));
    assert_eq!(labels, ["O", "PER", "LOC"], "the bundle's labels");
    for item in &items {
        let span: Value = serde_json::from_str(item).unwrap_or_else(|e| panic!("span `{item}` is not JSON: {e}"));
        assert_eq!(span["row"], 0, "the only text is row 0: {span}");
        let (start, end) =
            (span["byte_start"].as_u64().expect("byte_start"), span["byte_end"].as_u64().expect("byte_end"));
        assert!(
            start < end && end <= "Ada lives in Paris".len() as u64,
            "span bytes [{start}, {end}) do not lie in the text"
        );
        assert!(
            labels.contains(&span["label"].as_str().expect("label").to_string()),
            "span label is not one of {labels:?}: {span}"
        );
        assert!(span["score"].as_f64().expect("score") > 0.0, "span score is not positive: {span}");
    }
}

#[tokio::test]
async fn infer_generative_from_a_prompt_reports_the_finish_reason_and_token_counts() {
    let r = post(
        "/v2/models/chat/infer",
        &json!({"parameters": {"max_new_tokens": 4},
                "inputs": [{"name": "prompt", "datatype": "BYTES", "shape": [1], "data": ["hello"]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "generative infer: {}", r.text);
    let body = r.json();
    let text = output(&body, "text");
    assert_eq!(text["datatype"], "BYTES", "the generated text datatype");
    assert_eq!(shape(text), [1], "one generated text");
    assert!(!strings(text)[0].is_empty(), "the model generated nothing: {body}");
    assert_eq!(body["parameters"]["finish_reason"], "length", "4 new tokens hits the budget: {body}");
    assert_eq!(body["parameters"]["generated_tokens"], 4, "max_new_tokens 4 was honored: {body}");
    assert!(
        body["parameters"]["prompt_tokens"].as_i64().expect("prompt_tokens") > 0,
        "prompt_tokens is reported: {body}"
    );
}

#[tokio::test]
async fn infer_generative_from_message_turns() {
    let r = post(
        "/v2/models/chat/infer",
        &json!({"parameters": {"max_new_tokens": 3}, "inputs": [{"name": "messages", "datatype": "BYTES", "shape": [2], "data": [
            r#"{"role":"system","content":"be brief"}"#,
            r#"{"role":"user","content":"hello"}"#]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "generative infer from messages: {}", r.text);
    let body = r.json();
    assert_eq!(body["parameters"]["generated_tokens"], 3, "max_new_tokens 3 was honored: {body}");
    let prompt_only = post(
        "/v2/models/chat/infer",
        &json!({"parameters": {"max_new_tokens": 3}, "inputs": [{"name": "prompt", "datatype": "BYTES", "shape": [1], "data": ["hello"]}]}),
    )
    .await
    .json();
    assert_ne!(
        body["parameters"]["prompt_tokens"], prompt_only["parameters"]["prompt_tokens"],
        "the two turns did not reach the model: they tokenize to the same prompt as `hello` alone"
    );

    let bad = post(
        "/v2/models/chat/infer",
        &json!({"inputs": [{"name": "messages", "datatype": "BYTES", "shape": [1], "data": ["not json"]}]}),
    )
    .await;
    assert_eq!(bad.status, 400, "a turn that is not a role/content object: {}", bad.text);
    let body = bad.json();
    assert_eq!(body["status"], "BAD_REQUEST", "the error object's status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("messages"),
        "the message does not name the input it could not read: {body}"
    );
}

#[tokio::test]
async fn infer_generic_run_model_doubles_its_input() {
    let r = post(
        "/v2/models/run/infer",
        &json!({"inputs": [{"name": "x", "datatype": "FP32", "shape": [2, 3], "data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}]}),
    )
    .await;
    assert_eq!(r.status, 200, "generic RUN infer: {}", r.text);
    let body = r.json();
    let y = output(&body, "y");
    assert_eq!(y["datatype"], "FP32", "the output datatype");
    assert_eq!(shape(y), [2, 3], "the output keeps the input's shape");
    assert_eq!(floats(y), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0], "the mock RUN model computes y = 2x");
}

#[tokio::test]
async fn infer_outputs_filter_keeps_only_the_named_outputs() {
    let r = post(
        "/v2/models/classify/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["good"]}],
                "outputs": [{"name": "labels"}]}),
    )
    .await;
    assert_eq!(r.status, 200, "classify with an outputs filter: {}", r.text);
    let body = r.json();
    let names: Vec<&str> =
        body["outputs"].as_array().expect("outputs").iter().filter_map(|o| o["name"].as_str()).collect();
    assert_eq!(names, ["labels"], "only the requested output comes back");
}

#[tokio::test]
async fn infer_unknown_requested_output_is_400_naming_what_is_produced() {
    let r = post(
        "/v2/models/classify/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["good"]}],
                "outputs": [{"name": "logits"}]}),
    )
    .await;
    assert_eq!(r.status, 400, "an output the model does not produce: {}", r.text);
    let message = r.json()["message"].as_str().expect("message").to_string();
    assert!(
        message.contains("logits") && message.contains("scores"),
        "the message names the request and the real outputs: {message}"
    );
}

#[tokio::test]
async fn infer_bad_enum_value_is_400_naming_the_option_field() {
    let r = post(
        "/v2/models/embed/infer",
        &json!({"parameters": {"pooling": "sideways"},
                "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["hi"]}]}),
    )
    .await;
    assert_eq!(r.status, 400, "a pooling value that is not an enum member: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "BAD_REQUEST", "the error status: {body}");
    assert_eq!(body["field"], 6, "pooling is option field 6: {body}");
    assert_eq!(
        body["error"], "BAD_REQUEST (field 6): pooling `sideways` is not model, mean, cls or last",
        "the error line: {body}"
    );
}

#[tokio::test]
async fn infer_unsupported_option_is_501_naming_the_field() {
    let r = post(
        "/v2/models/embed/infer",
        &json!({"parameters": {"pooling": "cls"},
                "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["hi"]}]}),
    )
    .await;
    assert_eq!(r.status, 501, "the mock does not carry TURBO_CAP_OPT_POOLING_OVERRIDE: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "TURBO_E_UNSUPPORTED_OPTION", "the Turbo status: {body}");
    assert_eq!(body["field"], 6, "pooling is option field 6: {body}");
}

#[tokio::test]
async fn infer_over_a_capacity_limit_is_422() {
    let long = (0..19).map(|i| format!("w{i}")).collect::<Vec<_>>().join(" ");
    let r = post(
        "/v2/models/embed/infer",
        &json!({"parameters": {"truncate": "none"},
                "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": [long]}]}),
    )
    .await;
    assert_eq!(r.status, 422, "19 words plus specials do not fit max_seq 16 and nothing may be cut: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "TURBO_E_CAPACITY", "the Turbo status: {body}");
    assert!(
        body["message"].as_str().expect("a message").contains("16"),
        "the error does not name the limit that was hit: {body}"
    );
    // The mock bundle carries no core tokenizer, so this is the provider's
    // own length rule and it names no option field (`server/README.md`,
    // "Session buckets"). The bucket rule, which names field 2, is asserted
    // in `tests/engine.rs` on a bundle that does have a tokenizer.
    assert_eq!(body["field"], 0, "the provider's length rule rejects no option field: {body}");
    // The same text goes through when the caller asks for a cut.
    let cut = post(
        "/v2/models/embed/infer",
        &json!({"parameters": {"truncate": "right"},
                "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": [long]}]}),
    )
    .await;
    assert_eq!(cut.status, 200, "an explicit right truncation is served: {}", cut.text);

    let over = post(
        "/v2/models/embed/infer",
        &json!({"parameters": {"max_tokens": 99},
                "inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": ["hi"]}]}),
    )
    .await;
    assert_eq!(over.status, 422, "max_tokens past the model's max_seq: {}", over.text);
    assert_eq!(over.json()["field"], 3, "max_tokens is option field 3: {}", over.text);
}

#[tokio::test]
async fn infer_unknown_model_is_404_listing_what_is_served() {
    let r = post("/v2/models/nope/infer", &json!({"inputs": []})).await;
    assert_eq!(r.status, 404, "an unknown model: {}", r.text);
    let body = r.json();
    assert_eq!(body["status"], "NOT_FOUND", "the error status: {body}");
    let message = body["message"].as_str().expect("message");
    assert!(message.contains("embed"), "the message lists the served models: {message}");
}

#[tokio::test]
async fn infer_rejects_a_wrong_input_name_a_wrong_datatype_and_a_shape_that_does_not_match_the_data() {
    let wrong_name = post(
        "/v2/models/embed/infer",
        &json!({"inputs": [{"name": "texts", "datatype": "BYTES", "shape": [1], "data": ["hi"]}]}),
    )
    .await;
    assert_eq!(wrong_name.status, 400, "the embedding model's input is `text`: {}", wrong_name.text);
    assert!(
        wrong_name.json()["message"].as_str().expect("message").contains("texts"),
        "the message names what was sent: {}",
        wrong_name.text
    );

    let wrong_dtype = post(
        "/v2/models/embed/infer",
        &json!({"inputs": [{"name": "text", "datatype": "FP32", "shape": [1], "data": [1.0]}]}),
    )
    .await;
    assert_eq!(wrong_dtype.status, 400, "`text` must be BYTES: {}", wrong_dtype.text);

    let bad_shape = post(
        "/v2/models/embed/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [3], "data": ["a", "b"]}]}),
    )
    .await;
    assert_eq!(bad_shape.status, 400, "shape [3] with two strings: {}", bad_shape.text);

    let bad_element = post(
        "/v2/models/embed/infer",
        &json!({"inputs": [{"name": "text", "datatype": "BYTES", "shape": [1], "data": [42]}]}),
    )
    .await;
    assert_eq!(bad_element.status, 400, "a BYTES element that is not a string: {}", bad_element.text);
    assert!(
        bad_element.json()["message"].as_str().expect("message").contains("data[0]"),
        "the message names the element: {}",
        bad_element.text
    );

    let short_data = post(
        "/v2/models/run/infer",
        &json!({"inputs": [{"name": "x", "datatype": "FP32", "shape": [2, 3], "data": [1.0, 2.0, 3.0]}]}),
    )
    .await;
    assert_eq!(short_data.status, 400, "three floats do not fill a [2, 3] tensor: {}", short_data.text);
}
