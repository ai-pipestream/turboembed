//! What the vocabulary store tests share: a server with or without the
//! store and the admin calls, a real client, and the upload and download
//! frames.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use tonic::transport::Channel;
use tonic::{Code, Status};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
use turbo_kserve::proto::vocabulary::*;
use turbo_kserve::{Serve, Server};

pub type Client = TurboVocabularyServiceClient<Channel>;

/// A fresh store directory for one test.
pub fn dir(name: &str) -> PathBuf {
    let d = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("vocabulary-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

/// A server that is never loaded: the vocabulary service needs no model.
pub async fn start(admin: bool, vocabulary: Option<PathBuf>) -> (Server, Client) {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    let model = ModelConfig {
        bundle: bundle.to_str().unwrap().into(),
        name: None,
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start(
        "127.0.0.1:0".parse().unwrap(),
        vec![model],
        MAX,
        Serve { admin, vocabulary, ..Serve::default() },
    )
    .await
    .unwrap();
    let c = TurboVocabularyServiceClient::connect(format!("http://{}", s.local_addr()))
        .await
        .unwrap()
        .max_decoding_message_size(MAX);
    (s, c)
}

pub fn sha(bytes: &[u8]) -> String {
    turbo_maxent::sha256(bytes).to_hex()
}

/// The gRPC code and turbo-code of a refusal.
pub fn refusal(s: &Status) -> (Code, i32) {
    let md = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}")).to_str().unwrap();
    (s.code(), md.parse().unwrap())
}

pub fn header(format: &str, language: &str, bytes: &[u8]) -> ImportDictionaryHeader {
    ImportDictionaryHeader {
        format_id: format.into(),
        language: language.into(),
        name: format!("{format} sample"),
        byte_size: bytes.len() as u64,
        sha256: sha(bytes),
        has_header_row: false,
    }
}

/// The header, then `bytes` in chunks of 1000.
pub fn frames(h: ImportDictionaryHeader, bytes: &[u8]) -> Vec<ImportDictionaryRequest> {
    use import_dictionary_request::Frame;
    std::iter::once(Frame::Header(h))
        .chain(bytes.chunks(1000).map(|c| Frame::Chunk(c.to_vec())))
        .map(|f| ImportDictionaryRequest { frame: Some(f) })
        .collect()
}

pub async fn import_with(
    c: &mut Client,
    h: ImportDictionaryHeader,
    bytes: &[u8],
) -> Result<DictionaryArtifactDescriptor, Status> {
    c.import_dictionary(tokio_stream::iter(frames(h, bytes))).await.map(|r| r.into_inner())
}

pub async fn import(
    c: &mut Client,
    format: &str,
    language: &str,
    bytes: &[u8],
) -> Result<DictionaryArtifactDescriptor, Status> {
    import_with(c, header(format, language, bytes), bytes).await
}

pub fn learn_frames(h: LearnVocabularyHeader, docs: &[&str]) -> Vec<LearnVocabularyRequest> {
    use learn_vocabulary_request::Frame;
    std::iter::once(Frame::Header(h))
        .chain(docs.iter().map(|d| Frame::Document(LearnVocabularyDocument { text: d.to_string() })))
        .map(|f| LearnVocabularyRequest { frame: Some(f) })
        .collect()
}

pub async fn learn(
    c: &mut Client,
    h: LearnVocabularyHeader,
    docs: &[&str],
) -> Result<VocabularyArtifactDescriptor, Status> {
    c.learn_vocabulary(tokio_stream::iter(learn_frames(h, docs))).await.map(|r| r.into_inner())
}

/// The first frame of a download, its bytes, and the size of each data
/// frame.
pub async fn download(
    c: &mut Client,
    id: &str,
) -> Result<(vocabulary_artifact_chunk::Frame, Vec<u8>, Vec<usize>), Status> {
    use vocabulary_artifact_chunk::Frame;
    let mut s = c.download_vocabulary(DownloadVocabularyRequest { artifact_id: id.into() }).await?.into_inner();
    let head = s.message().await?.and_then(|m| m.frame).expect("a descriptor frame");
    let (mut bytes, mut sizes) = (Vec::new(), Vec::new());
    while let Some(m) = s.message().await? {
        match m.frame {
            Some(Frame::Data(d)) => {
                sizes.push(d.len());
                bytes.extend_from_slice(&d);
            }
            other => panic!("a frame after the descriptor that is not data: {other:?}"),
        }
    }
    Ok((head, bytes, sizes))
}
