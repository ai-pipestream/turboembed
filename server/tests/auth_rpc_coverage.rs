//! End-to-end gRPC role-separation tests across mounted state-mutating services
//! (`TurboSearchService`, `TurboCatalogService`, `TurboTrainingService`, and
//! `AnalysisService` admin calls) for `#2346`.

mod common;

use tonic::metadata::MetadataValue;
use tonic::{Code, Request};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{CapabilitiesRequest, UnloadBundleRequest};
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::{
    DeleteAliasRequest, DeleteCollectionRequest, DeleteIndexRequest, IndexDocumentRequest, ListIndexesRequest,
    ReindexRequest, SealIndexRequest, SetCollectionRequest,
};
use turbo_kserve::proto::training::turbo_catalog_service_client::TurboCatalogServiceClient;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::{
    FinishDriveSignInRequest, ListTeachersRequest, RegisterModelSourceRequest, RemoveModelSourceRequest,
    RevokeLicenceAcceptanceRequest, StartDriveSignInRequest, StreamingTrainingRequest,
};
use turbo_kserve::{AuthConfig, Serve, Server};

use common::*;

fn with_bearer<T>(mut req: Request<T>, token: &str) -> Request<T> {
    let val = MetadataValue::try_from(format!("Bearer {token}")).unwrap();
    req.metadata_mut().insert("authorization", val);
    req
}

#[tokio::test]
async fn infer_token_is_denied_on_every_mounted_state_mutating_service() {
    let dir = std::env::temp_dir().join(format!("turbo-auth-rpc-{}", std::process::id()));
    let tmp = turbo_scratch::Scratch::at(dir);
    std::fs::create_dir_all(tmp.path()).unwrap();

    let auth = AuthConfig::from_tokens(["infer-only"], ["admin-only"]).unwrap();
    let serve = Serve { admin: true, vocabulary: Some(tmp.join("vocab")), auth: Some(auth), ..Serve::default() };
    let model = ModelConfig {
        bundle: tiny().to_str().unwrap().to_string(),
        name: None,
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let server = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    server.load().await.unwrap();
    let endpoint = format!("http://{}", server.local_addr());

    let mut search = TurboSearchServiceClient::connect(endpoint.clone()).await.unwrap();
    let mut catalog = TurboCatalogServiceClient::connect(endpoint.clone()).await.unwrap();
    let mut training = TurboTrainingServiceClient::connect(endpoint.clone()).await.unwrap();
    let mut analysis = AnalysisServiceClient::connect(endpoint).await.unwrap();

    // Read-only calls succeed with the infer-only token.
    search.list_indexes(with_bearer(Request::new(ListIndexesRequest {}), "infer-only")).await.unwrap();
    training.list_teachers(with_bearer(Request::new(ListTeachersRequest {}), "infer-only")).await.unwrap();
    analysis.capabilities(with_bearer(Request::new(CapabilitiesRequest {}), "infer-only")).await.unwrap();

    // Mounted state-mutating RPCs are rejected at AuthLayer with PermissionDenied
    // before reaching the service handler (so DeleteIndex("missing") returns
    // PermissionDenied rather than NotFound).
    assert_eq!(
        search
            .delete_index(with_bearer(Request::new(DeleteIndexRequest { index_name: "missing".into() }), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search
            .delete_alias(with_bearer(Request::new(DeleteAliasRequest { alias: "a".into() }), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search
            .delete_collection(with_bearer(
                Request::new(DeleteCollectionRequest { collection_name: "c".into() }),
                "infer-only",
            ))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search
            .set_collection(with_bearer(Request::new(SetCollectionRequest::default()), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search
            .index_documents(with_bearer(
                Request::new(tokio_stream::iter(Vec::<IndexDocumentRequest>::new())),
                "infer-only",
            ))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search.reindex(with_bearer(Request::new(ReindexRequest::default()), "infer-only")).await.unwrap_err().code(),
        Code::PermissionDenied
    );
    assert_eq!(
        search
            .seal_index(with_bearer(Request::new(SealIndexRequest { index_name: "missing".into() }), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        catalog
            .register_model_source(with_bearer(Request::new(RegisterModelSourceRequest::default()), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        catalog
            .remove_model_source(with_bearer(Request::new(RemoveModelSourceRequest::default()), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        catalog
            .revoke_licence_acceptance(with_bearer(
                Request::new(RevokeLicenceAcceptanceRequest::default()),
                "infer-only",
            ))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        catalog
            .start_drive_sign_in(with_bearer(Request::new(StartDriveSignInRequest::default()), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        catalog
            .finish_drive_sign_in(with_bearer(Request::new(FinishDriveSignInRequest::default()), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        training
            .streaming_training(with_bearer(
                Request::new(tokio_stream::iter(Vec::<StreamingTrainingRequest>::new())),
                "infer-only",
            ))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );
    assert_eq!(
        analysis
            .unload_bundle(with_bearer(Request::new(UnloadBundleRequest { name: "missing".into() }), "infer-only"))
            .await
            .unwrap_err()
            .code(),
        Code::PermissionDenied
    );

    // With the admin token, DeleteIndex("missing") reaches the handler and returns NotFound.
    assert_eq!(
        search
            .delete_index(with_bearer(Request::new(DeleteIndexRequest { index_name: "missing".into() }), "admin-only"))
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );

    server.stop().await;
}
