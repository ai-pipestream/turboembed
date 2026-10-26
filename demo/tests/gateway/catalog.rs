//! The TurboCatalogService and TurboTrainingService routes: a static model
//! distilled from the tiny bundle and streamed to its trained update,
//! listed and deleted, and the refusals of a server with no catalog.

use super::*;

#[tokio::test]
async fn a_static_model_trains_over_a_stream_and_is_listed_and_deleted() {
    let dir = scratch("training");
    let serve = Serve {
        admin: true,
        vocabulary: Some(dir.join("vocabulary")),
        training: Some(dir.join("training")),
        ..Serve::default()
    };
    let (s, d) = start_with(serve).await;
    let teachers = ok(&d, "GET", "/api/training/teachers", Value::Null).await;
    assert_eq!(teachers["teachers"][0]["name"], NAME, "{teachers}");

    let req = json!({ "name": "tiny-static", "teacher": NAME, "pca_dims": 16, "max_length": 32 });
    let updates = lines(&d, "/api/training/static", req.clone()).await;
    assert!(updates[0]["update"]["progress"]["stage"].is_number(), "{}", updates[0]);
    let model = &updates.last().unwrap()["update"]["trained"]["model"];
    assert_eq!((&model["name"], &model["dims"], &model["served"]), (&json!("tiny-static"), &json!(16), &json!(true)));
    let listed = ok(&d, "GET", "/api/training/models", Value::Null).await;
    assert_eq!(&listed["models"][0], model);

    // The name is taken now.
    stream_refused(&d, "/api/training/static", req, "FailedPrecondition").await;
    let deleted = ok(&d, "POST", "/api/training/delete", json!({ "name": "tiny-static" })).await;
    assert_eq!(deleted["unloaded"], true, "{deleted}");
    refused(&d, "POST", "/api/training/delete", json!({ "name": "tiny-static" }), "NotFound").await;
    d.stop().await;
    s.stop().await;
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn a_server_without_a_catalog_refuses_every_catalog_route() {
    let (s, d) = start(true).await;
    refused(&d, "GET", "/api/catalog/models", Value::Null, "Unimplemented").await;
    refused(&d, "POST", "/api/catalog/models", json!({ "language": "en" }), "Unimplemented").await;
    refused(&d, "GET", "/api/catalog/installed", Value::Null, "Unimplemented").await;
    let accept = json!({ "user": "alice", "licence": "CC-BY-SA-4.0" });
    refused(&d, "POST", "/api/catalog/accept", accept, "Unimplemented").await;
    stream_refused(&d, "/api/catalog/install", json!({ "user": "alice", "artifact_id": "hello" }), "Unimplemented")
        .await;
    // Training needs its directory.
    stream_refused(&d, "/api/training/static", json!({ "name": "x", "teacher": NAME }), "Unimplemented").await;
    d.stop().await;
    s.stop().await;
}
