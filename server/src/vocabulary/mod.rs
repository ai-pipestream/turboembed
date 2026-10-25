//! TurboVocabularyService (proto/turbo_vocabulary.proto,
//! docs/grpc-vocabulary.md): dictionary uploads validated by the analysis
//! readers, corpus vocabularies learned from a stream of documents, and
//! the store that keeps both. The calls that write to the store are admin
//! calls, refused before anything is read when the server runs without
//! --admin or without a --vocabulary-dir.

mod formats;
mod learn;
mod store;

use std::pin::Pin;
use std::sync::Arc;

use tokio_stream::Stream;
use tonic::{Request, Response, Status, Streaming};

use crate::api::{
    Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT,
    TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED,
};
use crate::kserve::Reply;
use crate::proto::vocabulary::turbo_vocabulary_service_server::TurboVocabularyService;
use crate::proto::vocabulary::*;
use crate::status::to_status;
use learn::{Learner, Options};
pub(crate) use store::Store;
use store::{Dictionary, Stored};

/// The largest dictionary upload: 256 MiB.
pub const MAX_DICTIONARY_UPLOAD_BYTES: u64 = 256 << 20;
/// The most document text one LearnVocabulary reads: 1 GiB.
pub const MAX_LEARN_BYTES: u64 = 1 << 30;
pub const MAX_NGRAM_SIZE: u32 = 5;
pub const DOWNLOAD_CHUNK_BYTES: usize = 64 << 10;
/// The longest upload name and language tag, in bytes.
const MAX_NAME: usize = 256;
const MAX_LANGUAGE: usize = 64;

pub(crate) struct Vocabulary {
    pub(crate) store: Option<Arc<Store>>,
    pub(crate) admin: bool,
}

fn fail(code: i32, message: impl Into<String>) -> Status {
    to_status(&Failure::new(code, 0, message), &[])
}

fn invalid(message: impl Into<String>) -> Status {
    fail(TURBO_E_INVALID_ARGUMENT, message)
}

fn internal(message: impl Into<String>) -> Status {
    fail(TURBO_E_INTERNAL, message)
}

fn not_found(id: &str) -> Status {
    fail(TURBO_E_BUNDLE_NOT_FOUND, format!("no artifact {id:?} in the vocabulary store"))
}

/// Refuses a name or language tag that is too long or holds a control
/// character, which the store's descriptor files cannot carry.
fn check_text(what: &str, s: &str, max: usize) -> Result<(), Status> {
    if s.len() > max || s.chars().any(char::is_control) {
        return Err(invalid(format!("{what} must be at most {max} bytes without control characters")));
    }
    Ok(())
}

fn hex_digest(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// An id derived from the parts that make an artifact what it is.
fn artifact_id(prefix: &str, parts: &[&str]) -> String {
    let digest = turbo_maxent::sha256(parts.join("\n").as_bytes()).to_hex();
    format!("{prefix}-{}", &digest[..24])
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, Status> + Send + 'static) -> Result<T, Status> {
    tokio::task::spawn_blocking(f).await.map_err(|e| internal(format!("the vocabulary task failed: {e}")))?
}

impl Vocabulary {
    /// The store, for a call that writes to it.
    fn writer(&self, what: &str) -> Result<Arc<Store>, Status> {
        if !self.admin {
            return Err(fail(
                TURBO_E_UNSUPPORTED,
                format!("{what} is admin-only: start the server with --admin to serve it"),
            ));
        }
        self.store.clone().ok_or_else(|| {
            fail(
                TURBO_E_INVALID_STATE,
                format!("{what} needs a vocabulary store: start the server with --vocabulary-dir"),
            )
        })
    }

    fn limits(&self) -> StoreLimits {
        StoreLimits {
            store: self.store.is_some(),
            admin: self.admin,
            max_dictionary_upload_bytes: MAX_DICTIONARY_UPLOAD_BYTES,
            max_learn_bytes: MAX_LEARN_BYTES,
            max_ngram_size: MAX_NGRAM_SIZE,
            download_chunk_bytes: DOWNLOAD_CHUNK_BYTES as u32,
        }
    }
}

/// The header an upload or a learner opens with.
async fn first<M, H>(s: &mut Streaming<M>, header: impl Fn(M) -> Option<H>) -> Result<H, Status> {
    let frame = s.message().await?.ok_or_else(|| invalid("the stream ended before its header"))?;
    header(frame).ok_or_else(|| invalid("the first frame must be the header"))
}

type ChunkStream = Pin<Box<dyn Stream<Item = Result<VocabularyArtifactChunk, Status>> + Send>>;

#[tonic::async_trait]
impl TurboVocabularyService for Vocabulary {
    type DownloadVocabularyStream = ChunkStream;

    async fn list_dictionary_formats(
        &self,
        _: Request<ListDictionaryFormatsRequest>,
    ) -> Reply<ListDictionaryFormatsResponse> {
        let formats = formats::FORMATS
            .iter()
            .map(|f| DictionaryFormat {
                format_id: f.id.into(),
                description: f.description.into(),
                importable: f.importable,
                needs_language: f.needs_language,
            })
            .collect();
        Ok(Response::new(ListDictionaryFormatsResponse { formats, limits: Some(self.limits()) }))
    }

    async fn list_dictionaries(&self, _: Request<ListDictionariesRequest>) -> Reply<ListDictionariesResponse> {
        let dictionaries = self.store.as_ref().map(|s| s.dictionaries()).unwrap_or_default();
        Ok(Response::new(ListDictionariesResponse { dictionaries }))
    }

    async fn list_vocabularies(&self, _: Request<ListVocabulariesRequest>) -> Reply<ListVocabulariesResponse> {
        let vocabularies = self.store.as_ref().map(|s| s.vocabularies()).unwrap_or_default();
        Ok(Response::new(ListVocabulariesResponse { vocabularies }))
    }

    async fn delete_vocabulary_artifact(
        &self,
        req: Request<DeleteVocabularyArtifactRequest>,
    ) -> Reply<DeleteVocabularyArtifactResponse> {
        let store = self.writer("DeleteVocabularyArtifact")?;
        let id = req.into_inner().artifact_id;
        let gone = {
            let id = id.clone();
            blocking(move || store.delete(&id).map_err(internal)).await?
        };
        if !gone {
            return Err(not_found(&id));
        }
        Ok(Response::new(DeleteVocabularyArtifactResponse {}))
    }

    async fn import_dictionary(
        &self,
        req: Request<Streaming<ImportDictionaryRequest>>,
    ) -> Reply<DictionaryArtifactDescriptor> {
        let store = self.writer("ImportDictionary")?;
        let mut s = req.into_inner();
        let h = first(&mut s, |m: ImportDictionaryRequest| match m.frame {
            Some(import_dictionary_request::Frame::Header(h)) => Some(h),
            _ => None,
        })
        .await?;
        let format = formats::find(&h.format_id).ok_or_else(|| invalid(format!("unknown format {:?}", h.format_id)))?;
        if !format.importable {
            return Err(fail(TURBO_E_UNSUPPORTED, format!("{} is not importable as one upload", format.id)));
        }
        check_text("name", &h.name, MAX_NAME)?;
        check_text("language", &h.language, MAX_LANGUAGE)?;
        if format.needs_language && h.language.trim().is_empty() {
            return Err(invalid(format!("{} needs a language", format.id)));
        }
        if !hex_digest(&h.sha256) {
            return Err(invalid("sha256 must be 64 lowercase hex digits"));
        }
        if h.byte_size > MAX_DICTIONARY_UPLOAD_BYTES {
            return Err(fail(
                TURBO_E_CAPACITY,
                format!(
                    "an upload of {} bytes is past max_dictionary_upload_bytes {MAX_DICTIONARY_UPLOAD_BYTES}",
                    h.byte_size
                ),
            ));
        }
        let mut bytes = Vec::with_capacity(h.byte_size.min(16 << 20) as usize);
        while let Some(m) = s.message().await? {
            match m.frame {
                Some(import_dictionary_request::Frame::Chunk(c)) => {
                    if (bytes.len() + c.len()) as u64 > h.byte_size {
                        return Err(invalid(format!("more than the header's {} bytes", h.byte_size)));
                    }
                    bytes.extend_from_slice(&c);
                }
                _ => return Err(invalid("every frame after the header must be a chunk")),
            }
        }
        if bytes.len() as u64 != h.byte_size {
            return Err(invalid(format!("{} bytes, the header says {}", bytes.len(), h.byte_size)));
        }
        let descriptor = blocking(move || {
            let sha256 = turbo_maxent::sha256(&bytes).to_hex();
            if sha256 != h.sha256 {
                return Err(invalid(format!("the upload's sha256 is {sha256}, the header says {}", h.sha256)));
            }
            let read = formats::read(format, &bytes, &h.language, h.has_header_row)
                .map_err(|e| invalid(format!("{}: {e}", format.id)))?;
            let header_row = if h.has_header_row { "header" } else { "data" };
            let artifact_id = artifact_id("dict", &[format.id, &h.language, header_row, &sha256]);
            let d = Dictionary {
                descriptor: DictionaryArtifactDescriptor {
                    artifact_id,
                    format_id: format.id.into(),
                    language: h.language,
                    name: h.name,
                    entry_count: read.entry_count,
                    byte_size: bytes.len() as u64,
                    sha256,
                },
                has_header_row: h.has_header_row,
            };
            store.put_dictionary(d, &bytes, read.image.as_deref()).map_err(internal)
        })
        .await?;
        Ok(Response::new(descriptor))
    }

    async fn learn_vocabulary(
        &self,
        req: Request<Streaming<LearnVocabularyRequest>>,
    ) -> Reply<VocabularyArtifactDescriptor> {
        let store = self.writer("LearnVocabulary")?;
        let mut s = req.into_inner();
        let h = first(&mut s, |m: LearnVocabularyRequest| match m.frame {
            Some(learn_vocabulary_request::Frame::Header(h)) => Some(h),
            _ => None,
        })
        .await?;
        check_text("language", &h.language, MAX_LANGUAGE)?;
        let options = Options::new(&h, MAX_NGRAM_SIZE).map_err(invalid)?;
        let seed = match options.seed.as_str() {
            "" => None,
            id => Some(store.dictionary(id).ok_or_else(|| not_found(id))?),
        };
        // Documents are counted off the async runtime, one at a time, in
        // the order they arrive; a full queue holds the reading back.
        let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(16);
        let counting = tokio::task::spawn_blocking(move || {
            let mut learner = Learner::new(options);
            while let Some(text) = rx.blocking_recv() {
                learner.add(&text);
            }
            learner
        });
        let mut read = 0u64;
        while let Some(m) = s.message().await? {
            let Some(learn_vocabulary_request::Frame::Document(d)) = m.frame else {
                return Err(invalid("every frame after the header must be a document"));
            };
            read += d.text.len() as u64;
            if read > MAX_LEARN_BYTES {
                return Err(fail(TURBO_E_CAPACITY, format!("past max_learn_bytes {MAX_LEARN_BYTES}")));
            }
            tx.send(d.text).await.map_err(|_| internal("the vocabulary counter stopped"))?;
        }
        drop(tx);
        let learner = counting.await.map_err(|e| internal(format!("the vocabulary counter failed: {e}")))?;
        let descriptor = blocking(move || {
            let seeds = match &seed {
                None => Vec::new(),
                Some(d) => {
                    let id = &d.descriptor.artifact_id;
                    let Some(Stored::Dictionary(_, bytes)) = store.read(id).map_err(internal)? else {
                        return Err(not_found(id));
                    };
                    let format = formats::find(&d.descriptor.format_id)
                        .ok_or_else(|| internal(format!("stored format {:?}", d.descriptor.format_id)))?;
                    formats::headwords(format, &bytes, &d.descriptor.language, d.has_header_row).map_err(internal)?
                }
            };
            let learned = learner.finish(&seeds);
            if learned.documents == 0 {
                return Err(invalid("no documents to learn from"));
            }
            let sha256 = turbo_maxent::sha256(&learned.bytes).to_hex();
            let v = VocabularyArtifactDescriptor {
                artifact_id: artifact_id("vocab", &[&sha256]),
                language: h.language,
                documents: learned.documents,
                tokens: learned.tokens,
                term_count: learned.terms,
                ngram_count: learned.ngrams,
                byte_size: learned.bytes.len() as u64,
                sha256,
                seed_dictionary_artifact_id: h.dictionary_artifact_id,
            };
            store.put_vocabulary(v, &learned.bytes).map_err(internal)
        })
        .await?;
        Ok(Response::new(descriptor))
    }

    async fn download_vocabulary(&self, req: Request<DownloadVocabularyRequest>) -> Reply<ChunkStream> {
        let id = req.into_inner().artifact_id;
        let store = self.store.clone().ok_or_else(|| not_found(&id))?;
        let stored = {
            let id = id.clone();
            blocking(move || store.read(&id).map_err(internal)).await?
        };
        use vocabulary_artifact_chunk::Frame;
        let (head, bytes) = match stored.ok_or_else(|| not_found(&id))? {
            Stored::Dictionary(d, b) => (Frame::Dictionary(d), b),
            Stored::Vocabulary(v, b) => (Frame::Vocabulary(v), b),
        };
        let frames: Vec<Result<VocabularyArtifactChunk, Status>> = std::iter::once(head)
            .chain(bytes.chunks(DOWNLOAD_CHUNK_BYTES).map(|c| Frame::Data(c.to_vec())))
            .map(|f| Ok(VocabularyArtifactChunk { frame: Some(f) }))
            .collect();
        Ok(Response::new(Box::pin(tokio_stream::iter(frames))))
    }
}
