//! The TurboVocabularyService routes (#1360).

use axum::Json;
use axum::Router;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::response::Response;
use axum::routing::{get, post};
use serde::Deserialize;

use super::{answer, body};
use crate::App;
use crate::vocabulary::{
    DeleteVocabularyArtifactRequest, DownloadVocabularyRequest, ImportDictionaryHeader, ImportDictionaryRequest,
    LearnVocabularyDocument, LearnVocabularyHeader, LearnVocabularyRequest, ListDictionariesRequest,
    ListDictionaryFormatsRequest, ListVocabulariesRequest, import_dictionary_request, learn_vocabulary_request,
};

/// The most bytes of an upload in one ImportDictionary frame.
const CHUNK_BYTES: usize = 64 << 10;

pub(super) fn routes(r: Router<App>) -> Router<App> {
    r.route("/api/vocabulary/formats", get(formats))
        .route("/api/vocabulary/dictionaries", get(dictionaries))
        .route("/api/vocabulary/vocabularies", get(vocabularies))
        .route("/api/vocabulary/import", post(import))
        .route("/api/vocabulary/learn", post(learn))
        .route("/api/vocabulary/download", post(download))
        .route("/api/vocabulary/delete", post(delete))
}

listing!(
    /// `GET /api/vocabulary/formats`: ListDictionaryFormats, with the
    /// store's limits.
    formats: vocabulary.list_dictionary_formats(ListDictionaryFormatsRequest)
);
listing!(
    /// `GET /api/vocabulary/dictionaries`: ListDictionaries.
    dictionaries: vocabulary.list_dictionaries(ListDictionariesRequest)
);
listing!(
    /// `GET /api/vocabulary/vocabularies`: ListVocabularies.
    vocabularies: vocabulary.list_vocabularies(ListVocabulariesRequest)
);
streamed!(
    /// `POST /api/vocabulary/download`: DownloadVocabulary, the
    /// descriptor and then the artifact's bytes, base64, frame by frame.
    download: vocabulary.download_vocabulary(DownloadVocabularyRequest)
);
unary!(
    /// `POST /api/vocabulary/delete`: DeleteVocabularyArtifact, an admin
    /// call.
    delete: vocabulary.delete_vocabulary_artifact(DeleteVocabularyArtifactRequest)
);

/// The body of `POST /api/vocabulary/import`: the header, with the
/// upload's size and SHA-256, and the upload, base64.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Import {
    header: ImportDictionaryHeader,
    #[serde(with = "crate::wire::base64")]
    data: Vec<u8>,
}

/// `POST /api/vocabulary/import`: one ImportDictionary call, the header
/// and then the upload in frames of at most 64 KiB, answered with the
/// stored dictionary's descriptor.
async fn import(State(app): State<App>, req: Result<Json<Import>, JsonRejection>) -> Response {
    use import_dictionary_request::Frame;
    let req = match body(req) {
        Ok(req) => req,
        Err(refusal) => return *refusal,
    };
    let chunks: Vec<Frame> = req.data.chunks(CHUNK_BYTES).map(|c| Frame::Chunk(c.to_vec())).collect();
    let frames: Vec<ImportDictionaryRequest> = std::iter::once(Frame::Header(req.header))
        .chain(chunks)
        .map(|f| ImportDictionaryRequest { frame: Some(f) })
        .collect();
    answer(app.vocabulary.clone().import_dictionary(tokio_stream::iter(frames)).await)
}

/// The body of `POST /api/vocabulary/learn`: the header and the
/// documents to learn from.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Learn {
    #[serde(default)]
    header: LearnVocabularyHeader,
    documents: Vec<LearnVocabularyDocument>,
}

/// `POST /api/vocabulary/learn`: one LearnVocabulary call, answered with
/// the stored vocabulary's descriptor.
async fn learn(State(app): State<App>, req: Result<Json<Learn>, JsonRejection>) -> Response {
    use learn_vocabulary_request::Frame;
    let req = match body(req) {
        Ok(req) => req,
        Err(refusal) => return *refusal,
    };
    let frames: Vec<LearnVocabularyRequest> = std::iter::once(Frame::Header(req.header))
        .chain(req.documents.into_iter().map(Frame::Document))
        .map(|f| LearnVocabularyRequest { frame: Some(f) })
        .collect();
    answer(app.vocabulary.clone().learn_vocabulary(tokio_stream::iter(frames)).await)
}
