//! The TurboSearchService routes (#1360).

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::Response;
use axum::routing::{get, post};
use serde::Deserialize;

use super::{CLIENT_STREAM_BODY_BYTES, answer, body, large};
use crate::App;
use crate::search::{
    CreateIndexRequest, DeleteAliasRequest, DeleteCollectionRequest, DeleteIndexRequest, GetCollectionRequest,
    IndexDocumentRequest, ListAliasesRequest, ListCollectionsRequest, ListIndexesRequest, ListSearchProvidersRequest,
    PersistIndexRequest, ReindexRequest, SealIndexRequest, SearchRequest, SetAliasRequest, SetCollectionRequest,
    WatchCollectionRequest,
};

pub(super) fn routes(r: Router<App>) -> Router<App> {
    r.route("/api/search/indexes", get(indexes))
        .route("/api/search/providers", get(providers))
        .route("/api/search/create", post(create))
        .route("/api/search/index", large(index, CLIENT_STREAM_BODY_BYTES))
        .route("/api/search/query", post(query))
        .route("/api/search/persist", post(persist))
        .route("/api/search/seal", post(seal))
        .route("/api/search/reindex", post(reindex))
        .route("/api/search/alias", post(set_alias).delete(delete_alias))
        .route("/api/search/aliases", get(aliases))
        .route("/api/search/delete", post(delete))
        .route("/api/search/collections", get(collections))
        .route("/api/search/collection", post(collection))
        .route("/api/search/collection/get", post(get_collection))
        .route("/api/search/collection/delete", post(delete_collection))
        .route("/api/search/watch", post(watch))
}

listing!(
    /// `GET /api/search/indexes`: ListIndexes.
    indexes: search.list_indexes(ListIndexesRequest)
);
listing!(
    /// `GET /api/search/providers`: ListSearchProviders.
    providers: search.list_search_providers(ListSearchProvidersRequest)
);
unary!(
    /// `POST /api/search/create`: CreateIndex.
    create: search.create_index(CreateIndexRequest)
);
unary!(
    /// `POST /api/search/query`: Search.
    query: search.search(SearchRequest)
);
unary!(
    /// `POST /api/search/persist`: PersistIndex, an admin call.
    persist: search.persist_index(PersistIndexRequest)
);
unary!(
    /// `POST /api/search/seal`: SealIndex.
    seal: search.seal_index(SealIndexRequest)
);
unary!(
    /// `POST /api/search/reindex`: Reindex.
    reindex: search.reindex(ReindexRequest)
);
unary!(
    /// `POST /api/search/alias`: SetAlias.
    set_alias: search.set_alias(SetAliasRequest)
);
unary!(
    /// `DELETE /api/search/alias`: DeleteAlias.
    delete_alias: search.delete_alias(DeleteAliasRequest)
);
listing!(
    /// `GET /api/search/aliases`: ListAliases.
    aliases: search.list_aliases(ListAliasesRequest)
);
unary!(
    /// `POST /api/search/delete`: DeleteIndex.
    delete: search.delete_index(DeleteIndexRequest)
);
listing!(
    /// `GET /api/search/collections`: ListCollections.
    collections: search.list_collections(ListCollectionsRequest)
);
unary!(
    /// `POST /api/search/collection`: SetCollection, a collection's
    /// baseline terms and drift threshold.
    collection: search.set_collection(SetCollectionRequest)
);
unary!(
    /// `POST /api/search/collection/get`: GetCollection.
    get_collection: search.get_collection(GetCollectionRequest)
);
unary!(
    /// `POST /api/search/collection/delete`: DeleteCollection.
    delete_collection: search.delete_collection(DeleteCollectionRequest)
);
streamed!(
    /// `POST /api/search/watch`: WatchCollection, each CollectionEvent as
    /// it comes, until the page goes away.
    watch: search.watch_collection(WatchCollectionRequest)
);

/// The body of `POST /api/search/index`: the stream's messages, the
/// first naming the index.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Index {
    documents: Vec<IndexDocumentRequest>,
}

/// `POST /api/search/index`: one IndexDocuments call with the documents,
/// answered with its IndexDocumentsResponse.
async fn index(State(app): State<App>, req: Result<Json<Index>, JsonRejection>) -> Response {
    match body(req) {
        Ok(req) => answer(app.search.clone().index_documents(tokio_stream::iter(req.documents)).await),
        Err(refusal) => *refusal,
    }
}
