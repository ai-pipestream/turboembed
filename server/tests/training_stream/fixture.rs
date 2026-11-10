//! What the corpus job tests share: a real server on the tiny BERT
//! bundle, the job's requests, and what the server holds afterwards.
#![allow(dead_code, unused_imports)]

pub use std::fs;

use super::common;
pub use std::path::{Path, PathBuf};
pub use std::time::Duration;

pub use tokio::sync::mpsc;
pub use tokio_stream::wrappers::ReceiverStream;
pub use tonic::Code;
pub use tonic::transport::Channel;
pub use turbo_kserve::api::{
    TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE,
    TURBO_E_UNSUPPORTED,
};
pub use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
pub use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
pub use turbo_kserve::proto::search::{
    CreateIndexRequest, IndexSpec, ListIndexesRequest, ModelIdentity, SearchRequest, SpaceSpec,
};
pub use turbo_kserve::proto::training::streaming_training_request::Part;
pub use turbo_kserve::proto::training::streaming_training_start::Job;
pub use turbo_kserve::proto::training::streaming_training_update::Update;
pub use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
pub use turbo_kserve::proto::training::*;
pub use turbo_kserve::proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
pub use turbo_kserve::proto::vocabulary::{
    LearnVocabularyDocument, LearnVocabularyHeader, LearnVocabularyRequest, ListVocabulariesRequest,
    learn_vocabulary_request::Frame,
};
pub use turbo_kserve::{Serve, Server};

pub type Client = TurboTrainingServiceClient<Channel>;

/// Documents that use two words the tiny BERT vocabulary has no entry for.
pub const DOCS: [(&str, &str); 3] = [
    ("a", "Neuroimaging data from turboembed. The second sentence is about weather."),
    ("b", "Turboembed reads neuroimaging scans."),
    ("c", "The scans were clear."),
];

pub fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-kserve-streaming-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

pub struct Dirs {
    pub admin: bool,
    pub training: bool,
    pub vocabulary: bool,
}

pub const ALL: Dirs = Dirs { admin: true, training: true, vocabulary: true };

pub async fn server(d: &Path, dirs: Dirs) -> (Server, Client) {
    let serve = Serve {
        admin: dirs.admin,
        vocabulary: dirs.vocabulary.then(|| d.join("vocabulary")),
        training: dirs.training.then(|| d.join("training")),
        upload_idle: Some(Duration::from_secs(2)),
        ..Serve::default()
    };
    let model = ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

pub fn refusal(s: &tonic::Status) -> ((Code, i32), String) {
    let get = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    ((s.code(), get("turbo-code").parse().unwrap()), get("turbo-field"))
}

pub fn vocabulary() -> CorpusVocabulary {
    CorpusVocabulary { lowercase: true, min_term_frequency: 2, ..Default::default() }
}

pub fn corpus(model: &str, index: &str) -> CorpusTraining {
    CorpusTraining {
        vocabulary: Some(vocabulary()),
        model: Some(TrainStaticModelRequest {
            name: model.into(),
            teacher: common::NAME.into(),
            pca_dims: 16,
            max_length: 32,
            ..Default::default()
        }),
        index: Some(CorpusIndex { index_name: index.into(), ..Default::default() }),
    }
}

pub fn part(p: Part) -> StreamingTrainingRequest {
    StreamingTrainingRequest { part: Some(p) }
}

pub fn start(c: CorpusTraining) -> StreamingTrainingRequest {
    part(Part::Start(StreamingTrainingStart { job: Some(Job::Corpus(c)) }))
}

pub fn document(id: &str, text: &str) -> StreamingTrainingRequest {
    part(Part::Document(CorpusDocument { doc_id: id.into(), text: text.into() }))
}

pub fn documents() -> Vec<StreamingTrainingRequest> {
    DOCS.iter().map(|(id, text)| document(id, text)).collect()
}

/// Every update of a job, the stream's error when it ends in one.
pub async fn run(c: &mut Client, requests: Vec<StreamingTrainingRequest>) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.streaming_training(tokio_stream::iter(requests)).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

pub async fn job(c: &mut Client, t: CorpusTraining) -> Result<Vec<Update>, tonic::Status> {
    run(c, std::iter::once(start(t)).chain(documents()).collect()).await
}

pub fn stages(updates: &[Update]) -> Vec<StreamingStage> {
    let mut stages: Vec<_> = updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) => Some(p.stage()),
            _ => None,
        })
        .collect();
    stages.dedup();
    stages
}

/// What the server holds: the vocabulary ids, the training directory's
/// children, and the index names.
pub async fn held(s: &Server, d: &Path) -> (Vec<String>, Vec<String>, Vec<String>) {
    let at = format!("http://{}", s.local_addr());
    let mut v = TurboVocabularyServiceClient::connect(at.clone()).await.unwrap();
    let vocabularies = v.list_vocabularies(ListVocabulariesRequest {}).await.unwrap().into_inner().vocabularies;
    let mut names: Vec<String> =
        fs::read_dir(d.join("training")).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    names.sort();
    let mut x = TurboSearchServiceClient::connect(at).await.unwrap();
    let indexes = x.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes;
    (
        vocabularies.into_iter().map(|v| v.artifact_id).collect(),
        names,
        indexes.into_iter().map(|i| i.index_name).collect(),
    )
}

/// A job whose upload stays open until the sender is dropped.
pub async fn open(
    c: &mut Client,
    t: CorpusTraining,
) -> (mpsc::Sender<StreamingTrainingRequest>, tonic::Streaming<StreamingTrainingUpdate>) {
    let (tx, rx) = mpsc::channel(8);
    tx.send(start(t)).await.unwrap();
    let mut stream = c.streaming_training(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let first = stream.message().await.unwrap().unwrap().update.unwrap();
    assert_eq!(stages(&[first]), [StreamingStage::Admitted]);
    (tx, stream)
}

/// Runs jobs until one is not refused as busy.
pub async fn when_free(c: &mut Client, t: CorpusTraining) -> Vec<Update> {
    loop {
        match job(c, t.clone()).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
            other => break other.unwrap(),
        }
    }
}
