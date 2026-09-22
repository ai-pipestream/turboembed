//! The OpenAI-shaped routes under `/v1`, plus `/info`.

mod common;

use common::{base64_to_f32, get, post, sse_data, sse_events};
use serde_json::{json, Value};

/// The floats of one `/v1/embeddings` item.
fn embedding(body: &Value, index: usize) -> Vec<f32> {
    body["data"][index]["embedding"]
        .as_array()
        .unwrap_or_else(|| panic!("data[{index}].embedding is not an array of floats: {body}"))
        .iter()
        .map(|v| v.as_f64().expect("a float") as f32)
        .collect()
}

#[tokio::test]
async fn v1_models_lists_every_served_model() {
    let r = get("/v1/models").await;
    assert_eq!(r.status, 200, "GET /v1/models: {}", r.text);
    let body = r.json();
    assert_eq!(body["object"], "list", "the envelope object: {body}");
    let ids: Vec<&str> = body["data"].as_array().expect("data").iter().filter_map(|m| m["id"].as_str()).collect();
    assert_eq!(ids, ["chat", "classify", "embed", "rerank", "run", "tag"], "the model ids");
    let embed = body["data"].as_array().expect("data").iter().find(|m| m["id"] == "embed").expect("the embed model");
    assert_eq!(embed["object"], "model", "each entry is a model object: {embed}");
    assert_eq!(embed["owned_by"], "mock", "the provider owns the model: {embed}");
}

#[tokio::test]
async fn info_reports_the_text_embeddings_inference_fields_of_the_first_embedding_model() {
    let r = get("/info").await;
    assert_eq!(r.status, 200, "GET /info: {}", r.text);
    let body = r.json();
    assert_eq!(body["model_id"], "turbo/mock-embedding", "the primary model is the first embedding model: {body}");
    assert_eq!(body["max_input_length"], 16, "max_input_length is the served sequence limit: {body}");
    assert_eq!(body["max_client_batch_size"], 8, "max_client_batch_size is the served batch: {body}");
    assert_eq!(body["model_type"]["embedding"]["pooling"], "mean", "model_type carries the bundle's pooling: {body}");
    let names: Vec<&str> =
        body["models"].as_array().expect("models").iter().filter_map(|m| m["name"].as_str()).collect();
    assert_eq!(names, ["chat", "classify", "embed", "rerank", "run", "tag"], "/info lists every served model");
}

#[tokio::test]
async fn embeddings_accepts_a_string_and_a_list_and_gives_the_same_vectors() {
    let one = post("/v1/embeddings", &json!({"model": "embed", "input": "hello world"})).await;
    assert_eq!(one.status, 200, "a string input: {}", one.text);
    let one = one.json();
    assert_eq!(one["object"], "list", "the envelope object: {one}");
    assert_eq!(one["model"], "embed", "the model name: {one}");
    assert_eq!(one["data"].as_array().expect("data").len(), 1, "one text gives one embedding: {one}");
    assert_eq!(one["data"][0]["object"], "embedding", "each item is an embedding object: {one}");
    assert_eq!(one["data"][0]["index"], 0, "the item index: {one}");
    assert_eq!(embedding(&one, 0).len(), 8, "the bundle's dim is 8: {one}");
    assert_eq!(one["turbo"]["dim"], 8, "the turbo block reports the width: {one}");
    assert_eq!(one["turbo"]["device"], "Mock accelerator", "the turbo block names the device: {one}");
    // The mock bundle has no core tokenizer, so no token count is claimed:
    // `usage` is null rather than a zero that would read as a measurement.
    assert!(one["usage"].is_null(), "usage is absent without a token count: {one}");

    let many = post("/v1/embeddings", &json!({"model": "embed", "input": ["hello world", "goodbye"]})).await;
    assert_eq!(many.status, 200, "a list input: {}", many.text);
    let many = many.json();
    assert_eq!(many["data"].as_array().expect("data").len(), 2, "two texts give two embeddings: {many}");
    assert_eq!(many["data"][1]["index"], 1, "the second item keeps its index: {many}");
    assert_eq!(embedding(&many, 0), embedding(&one, 0), "a text embeds the same alone and in a batch");
}

#[tokio::test]
async fn embeddings_base64_encoding_decodes_back_to_the_float_vector() {
    let floats = post("/v1/embeddings", &json!({"model": "embed", "input": ["a", "b"]})).await.json();
    let encoded =
        post("/v1/embeddings", &json!({"model": "embed", "input": ["a", "b"], "encoding_format": "base64"})).await;
    assert_eq!(encoded.status, 200, "base64 encoding_format: {}", encoded.text);
    let encoded = encoded.json();
    for i in 0..2 {
        let text = encoded["data"][i]["embedding"]
            .as_str()
            .unwrap_or_else(|| panic!("data[{i}].embedding is not a base64 string: {encoded}"));
        assert_eq!(base64_to_f32(text), embedding(&floats, i), "base64 item {i} does not decode to the float vector");
    }

    let bad = post("/v1/embeddings", &json!({"model": "embed", "input": "a", "encoding_format": "utf7"})).await;
    assert_eq!(bad.status, 400, "an encoding_format that is neither float nor base64: {}", bad.text);
}

#[tokio::test]
async fn embeddings_dimensions_follows_the_bundles_truncate_dims() {
    let ok = post("/v1/embeddings", &json!({"model": "embed", "input": "a", "dimensions": 4})).await;
    assert_eq!(ok.status, 200, "the bundle lists 4 in truncate_dims: {}", ok.text);
    let ok = ok.json();
    assert_eq!(embedding(&ok, 0).len(), 4, "dimensions 4 gives a 4-wide vector: {ok}");
    assert_eq!(ok["turbo"]["dim"], 4, "the turbo block reports the narrowed width: {ok}");

    let bad = post("/v1/embeddings", &json!({"model": "embed", "input": "a", "dimensions": 5})).await;
    assert_eq!(bad.status, 400, "5 is not one of the bundle's truncate_dims: {}", bad.text);
    let body = bad.json();
    assert_eq!(body["field"], 7, "output_dim is option field 7: {body}");
    assert_eq!(body["status"], "TURBO_E_INVALID_ARGUMENT", "the Turbo status: {body}");
}

#[tokio::test]
async fn embeddings_on_a_model_of_another_kind_is_400_naming_both_kinds() {
    let r = post("/v1/embeddings", &json!({"model": "chat", "input": "a"})).await;
    assert_eq!(r.status, 400, "the generative model is not an embedder: {}", r.text);
    let message = r.json()["message"].as_str().expect("message").to_string();
    assert!(message.contains("Generative") && message.contains("Embedding"), "the message names both kinds: {message}");
}

#[tokio::test]
async fn rerank_returns_results_best_first_with_every_score() {
    let r = post(
        "/v1/rerank",
        &json!({"model": "rerank", "query": "alpha beta", "documents": ["alpha beta gamma", "zulu", "alpha"]}),
    )
    .await;
    assert_eq!(r.status, 200, "rerank: {}", r.text);
    let body = r.json();
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 3, "every document is ranked: {body}");
    let scores: Vec<f64> = results.iter().map(|x| x["relevance_score"].as_f64().expect("a score")).collect();
    assert!(scores.windows(2).all(|w| w[0] >= w[1]), "results are not best first: {scores:?}");
    assert_eq!(results[0]["index"], 2, "`alpha` overlaps the query most: {body}");
    assert!(results[0].get("document").is_none(), "return_documents defaults off: {}", results[0]);
    let all: Vec<f64> = body["turbo"]["scores"]
        .as_array()
        .expect("turbo.scores")
        .iter()
        .map(|v| v.as_f64().expect("a score"))
        .collect();
    assert_eq!(all.len(), 3, "turbo.scores keeps every score in document order: {body}");
    assert_eq!(all[2], scores[0], "turbo.scores is in document order, not rank order: {body}");
}

#[tokio::test]
async fn rerank_honors_top_n_and_return_documents() {
    let r = post(
        "/v1/rerank",
        &json!({"model": "rerank", "query": "alpha beta", "documents": ["alpha beta gamma", "zulu", "alpha"],
                "top_n": 2, "return_documents": true}),
    )
    .await;
    assert_eq!(r.status, 200, "rerank with top_n and return_documents: {}", r.text);
    let body = r.json();
    let results = body["results"].as_array().expect("results");
    assert_eq!(results.len(), 2, "top_n 2 keeps two results: {body}");
    assert_eq!(results[0]["document"]["text"], "alpha", "return_documents echoes the ranked document: {body}");
    assert_eq!(
        body["turbo"]["scores"].as_array().expect("turbo.scores").len(),
        3,
        "top_n cuts the ranking, not the scores: {body}"
    );
}

#[tokio::test]
async fn classify_returns_labels_best_first_for_a_sequence_classifier() {
    let r = post("/v1/classify", &json!({"model": "classify", "inputs": ["good", "bad"]})).await;
    assert_eq!(r.status, 200, "classify: {}", r.text);
    let body = r.json();
    let rows = body["results"].as_array().expect("results");
    assert_eq!(rows.len(), 2, "one row per input: {body}");
    for (i, row) in rows.iter().enumerate() {
        let items = row.as_array().unwrap_or_else(|| panic!("row {i} is not a list: {body}"));
        assert_eq!(items.len(), 3, "the bundle has three labels: {body}");
        let scores: Vec<f64> = items.iter().map(|x| x["score"].as_f64().expect("a score")).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "row {i} is not best first: {scores:?}");
        let total: f64 = scores.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "row {i} sums to {total}, not 1, although the bundle says activation=softmax"
        );
        for item in items {
            let label = item["label"].as_str().expect("a label");
            assert!(
                ["negative", "neutral", "positive"].contains(&label),
                "`{label}` is not one of the bundle's labels"
            );
        }
    }
}

#[tokio::test]
async fn classify_returns_entity_spans_for_a_token_classifier() {
    let text = "Ada lives in Paris";
    let r = post("/v1/classify", &json!({"model": "tag", "inputs": [text]})).await;
    assert_eq!(r.status, 200, "token classify: {}", r.text);
    let body = r.json();
    let rows = body["results"].as_array().expect("results");
    assert_eq!(rows.len(), 1, "one row per input: {body}");
    let spans = rows[0].as_array().expect("the row's spans");
    assert!(!spans.is_empty(), "the mock tagger found no spans in `{text}`: {body}");
    for span in spans {
        let group = span["entity_group"].as_str().expect("entity_group");
        assert!(["O", "PER", "LOC"].contains(&group), "`{group}` is not one of the bundle's labels: {span}");
        let (start, end) =
            (span["start"].as_u64().expect("start") as usize, span["end"].as_u64().expect("end") as usize);
        assert_eq!(span["word"], &text[start..end], "`word` is not the text the offsets name: {span}");
        assert!(span["score"].as_f64().expect("score") > 0.0, "a span scored zero: {span}");
    }
}

#[tokio::test]
async fn classify_on_a_model_of_another_kind_is_400() {
    let r = post("/v1/classify", &json!({"model": "embed", "inputs": "a"})).await;
    assert_eq!(r.status, 400, "the embedding model is not a classifier: {}", r.text);
    assert!(
        r.json()["message"].as_str().expect("message").contains("classifier"),
        "the message says what the route needs: {}",
        r.text
    );
}

#[tokio::test]
async fn chat_completions_without_streaming_returns_one_choice_and_usage() {
    let r = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 4}),
    )
    .await;
    assert_eq!(r.status, 200, "chat completion: {}", r.text);
    let body = r.json();
    assert_eq!(body["object"], "chat.completion", "the object kind: {body}");
    assert_eq!(body["model"], "chat", "the model name: {body}");
    let choices = body["choices"].as_array().expect("choices");
    assert_eq!(choices.len(), 1, "one choice per request: {body}");
    assert_eq!(choices[0]["index"], 0, "the choice index: {body}");
    assert_eq!(choices[0]["message"]["role"], "assistant", "the reply role: {body}");
    assert!(
        !choices[0]["message"]["content"].as_str().expect("content").is_empty(),
        "the model generated nothing: {body}"
    );
    assert_eq!(choices[0]["finish_reason"], "length", "4 new tokens hits the budget: {body}");
    assert_eq!(body["usage"]["completion_tokens"], 4, "max_tokens 4 was honored: {body}");
    assert_eq!(
        body["usage"]["total_tokens"].as_i64().expect("total_tokens"),
        body["usage"]["prompt_tokens"].as_i64().expect("prompt_tokens") + 4,
        "total_tokens is the sum: {body}"
    );
}

#[tokio::test]
async fn chat_completions_streaming_sends_the_role_first_the_finish_last_and_ends_with_done() {
    let r = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3, "stream": true}),
    )
    .await;
    assert_eq!(r.status, 200, "streamed chat completion: {}", r.text);
    assert!(r.content_type.starts_with("text/event-stream"), "the response is not an SSE stream: `{}`", r.content_type);
    let data = sse_data(&r.text);
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "the stream does not end with [DONE]: {data:?}");
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("chunk `{d}` is not JSON: {e}")))
        .collect();
    assert_eq!(chunks.len(), 3, "max_tokens 3 gives three chunks: {data:?}");
    for c in &chunks {
        assert_eq!(c["object"], "chat.completion.chunk", "the object kind: {c}");
        assert_eq!(c["model"], "chat", "the model name: {c}");
    }
    assert_eq!(
        chunks[0]["choices"][0]["delta"]["role"], "assistant",
        "the first delta does not carry the role: {}",
        chunks[0]
    );
    assert!(
        chunks[1]["choices"][0]["delta"].get("role").is_none(),
        "the role is repeated after the first delta: {}",
        chunks[1]
    );
    for c in &chunks[..chunks.len() - 1] {
        assert_eq!(
            c["choices"][0]["finish_reason"],
            Value::Null,
            "a chunk before the last carries a finish_reason: {c}"
        );
        assert_eq!(c["usage"], Value::Null, "a chunk before the last carries usage: {c}");
    }
    let last = chunks.last().expect("a last chunk");
    assert_eq!(
        last["choices"][0]["finish_reason"], "length",
        "the last chunk does not carry the finish reason: {last}"
    );
    assert_eq!(last["usage"]["completion_tokens"], 3, "the last chunk does not carry usage: {last}");
    let text: String = chunks.iter().filter_map(|c| c["choices"][0]["delta"]["content"].as_str()).collect();
    assert!(!text.is_empty(), "the deltas carried no text: {data:?}");

    let whole = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3}),
    )
    .await
    .json();
    assert_eq!(
        text, whole["choices"][0]["message"]["content"],
        "the streamed deltas do not join to the unstreamed reply"
    );
}

#[tokio::test]
async fn chat_completions_reject_more_than_one_choice_naming_the_field() {
    let r = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "n": 2}),
    )
    .await;
    assert_eq!(r.status, 400, "n other than 1: {}", r.text);
    let body = r.json();
    assert_eq!(body["field"], 4, "n_sequences is generation field 4: {body}");
    assert!(body["message"].as_str().expect("message").contains("n 2"), "the message names the value: {body}");

    let one = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "n": 1, "max_tokens": 1}),
    )
    .await;
    assert_eq!(one.status, 200, "n 1 is the one choice this server serves: {}", one.text);
}

#[tokio::test]
async fn a_provider_error_during_a_stream_becomes_an_sse_error_event() {
    // The mock rejects top_k past its vocabulary when the generation is
    // created, which happens on the generation's own task: the response has
    // already started, so the error can only reach the client as an event.
    let r = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "top_k": 5000, "stream": true}),
    )
    .await;
    assert_eq!(r.status, 200, "the stream had already started when the provider failed: {}", r.text);
    assert_eq!(sse_events(&r.text), ["error"], "the stream carries no error event: {}", r.text);
    let data = sse_data(&r.text);
    assert_eq!(data.last().map(String::as_str), Some("[DONE]"), "the stream does not end with [DONE]: {data:?}");
    let error: Value =
        serde_json::from_str(&data[0]).unwrap_or_else(|e| panic!("the error event is not JSON: {e}; {}", data[0]));
    assert_eq!(error["error"]["type"], "TURBO_E_INVALID_ARGUMENT", "the event carries the Turbo status: {error}");
    assert_eq!(error["error"]["field"], 6, "top_k is generation field 6: {error}");

    // The same request without streaming fails before any bytes are written.
    let plain = post(
        "/v1/chat/completions",
        &json!({"model": "chat", "messages": [{"role": "user", "content": "hello"}], "top_k": 5000}),
    )
    .await;
    assert_eq!(plain.status, 400, "unstreamed, the same failure is a status: {}", plain.text);
    assert_eq!(plain.json()["status"], "TURBO_E_INVALID_ARGUMENT", "the error object: {}", plain.text);
}
