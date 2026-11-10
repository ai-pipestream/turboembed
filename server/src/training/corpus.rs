//! A StreamingTraining corpus job (docs/grpc-training.md, Streamed
//! training): a vocabulary learned from the documents the client streams,
//! a static model distilled with its terms, and a search index of the
//! documents embedded by that model, as one transaction. Each part is
//! made as its own call makes it (LearnVocabulary, TrainStaticModel,
//! CreateIndex and IndexDocuments). The vocabulary is stored and the
//! index published in one last step; a job that fails, or whose client
//! goes away before then, removes the model it made, and its vocabulary
//! and index were never anyone else's to see.

use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Status, Streaming};

use super::stream::{Tx, UpdateStream, cancelled, internal, next, progress, update};
use super::train::{self, Slot, Watch};
use super::{TrainingService, blocking, invalid, plain, remove_static, runtime};
use crate::api::{Failure, TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_STATE};
use crate::proto::search::{
    DistanceMetric, IndexDocumentRequest, IndexSpec, ModelIdentity, NormalizationMode, SpaceSpec, VectorStorageKind,
};
use crate::proto::training::streaming_training_request::Part;
use crate::proto::training::streaming_training_update::Update;
use crate::proto::training::{
    CorpusDocument, CorpusIndex, CorpusIndexStorage, CorpusTrained, CorpusTraining, StaticModelDescriptor,
    StreamingStage, StreamingTrainingProgress, StreamingTrainingRequest, TrainStaticModelProgress,
};
use crate::proto::vocabulary::LearnVocabularyHeader;
use crate::search::SearchRegistry;
use crate::search::ingest::{commit_staged_documents, stage_document};
use crate::search::registry::{ALREADY_EXISTS_FIELD, HybridIndex, parse_storage};
use crate::vocabulary::{self, Dictionary, Learner, MAX_LANGUAGE, MAX_NGRAM_SIZE, Options, Store};

/// The most document bytes one job reads; they are held in memory, and
/// the index keeps them.
pub(crate) const MAX_CORPUS_BYTES: usize = 256 << 20;

/// The most documents one job reads.
pub(crate) const MAX_CORPUS_DOCUMENTS: usize = 1 << 20;

/// The longest document id, index name or collection name, in bytes.
const MAX_NAME: usize = 1024;

/// Documents between two progress messages of the reading.
const ANALYZED_EVERY: usize = 256;

/// Documents between two progress messages of the indexing.
const INDEXED_EVERY: usize = 64;

/// What a corpus job reaches beyond its start.
struct Corpus {
    job: train::Job,
    store: Arc<Store>,
    search: Arc<SearchRegistry>,
}

/// A start with its checks passed.
struct Plan {
    model: train::Plan,
    header: LearnVocabularyHeader,
    seed: Option<Dictionary>,
    index: CorpusIndex,
    storage: VectorStorageKind,
}

/// Removes what a job made unless it is kept: the static model, and
/// always the staging directory. It runs when the job's steps return, and
/// when they panic. The vocabulary and the index need no undoing: neither
/// is stored or published before the step that keeps the job.
struct Undo<'a> {
    corpus: &'a Corpus,
    staging: PathBuf,
    model: Option<String>,
}

impl Drop for Undo<'_> {
    fn drop(&mut self) {
        let c = self.corpus;
        if let Some(name) = self.model.take() {
            let _ = remove_static(&c.job.state, &c.job.publishing, &c.job.dir, &name);
        }
        let _ = fs::remove_dir_all(&self.staging);
    }
}

/// A distillation's steps as the job's DISTILLING_MODEL progress.
struct Distilling<'a>(&'a Tx);

impl Watch for Distilling<'_> {
    fn step(&self, p: TrainStaticModelProgress) {
        let p = StreamingTrainingProgress {
            stage: StreamingStage::DistillingModel as i32,
            distillation: Some(p),
            ..Default::default()
        };
        let _ = self.0.blocking_send(Ok(update(Update::Progress(p))));
    }

    fn gone(&self) -> bool {
        self.0.is_closed()
    }
}

/// A name the index keeps: not empty where `required`, at most
/// [`MAX_NAME`] bytes, no control character.
fn check_name(field: u32, what: &str, s: &str, required: bool) -> Result<(), Status> {
    if (required && s.is_empty()) || s.len() > MAX_NAME || s.chars().any(char::is_control) {
        let least = if required { 1 } else { 0 };
        return Err(invalid(field, format!("{what}: {least} to {MAX_NAME} bytes, no control characters")));
    }
    Ok(())
}

/// The index's spec once the model it is embedded by is known.
fn spec(ix: &CorpusIndex, storage: VectorStorageKind, model: Option<&StaticModelDescriptor>) -> IndexSpec {
    IndexSpec {
        index_name: ix.index_name.clone(),
        space: model.map(|m| SpaceSpec {
            model: Some(ModelIdentity {
                id: m.model_id.clone(),
                revision: m.revision.clone(),
                sha256: m.manifest_sha256.clone(),
            }),
            dim: m.dims,
            normalization: NormalizationMode::Unit as i32,
        }),
        metric: DistanceMetric::Cosine as i32,
        storage: storage as i32,
        quant_bits: ix.quant_bits,
        quant_calibrate: ix.quant_calibrate,
        collection_name: ix.collection_name.clone(),
        embed_bundle: model.map(|m| m.name.clone()).unwrap_or_default(),
        chunk_budget_tokens: ix.chunk_budget_tokens,
    }
}

/// The start's checks. A refusal names the part of the job it is about:
/// 1 the vocabulary, 2 the model, 3 the index.
fn plan(c: &Corpus, t: CorpusTraining) -> Result<(Plan, Options), Status> {
    let v = t.vocabulary.unwrap_or_default();
    let header = LearnVocabularyHeader {
        min_term_frequency: v.min_term_frequency,
        max_terms: v.max_terms,
        lowercase: v.lowercase,
        filter_stopwords: v.filter_stopwords,
        include_ngrams: v.include_ngrams,
        min_ngram_size: v.min_ngram_size,
        max_ngram_size: v.max_ngram_size,
        min_pmi: v.min_pmi,
        dictionary_artifact_id: v.dictionary_artifact_id,
        language: v.language,
    };
    if header.language.len() > MAX_LANGUAGE || header.language.chars().any(char::is_control) {
        return Err(invalid(1, format!("vocabulary.language: at most {MAX_LANGUAGE} bytes, no control characters")));
    }
    let options = Options::new(&header, MAX_NGRAM_SIZE).map_err(|e| invalid(1, format!("vocabulary: {e}")))?;
    let seed = match header.dictionary_artifact_id.as_str() {
        "" => None,
        id => Some(c.store.dictionary(id).ok_or_else(|| {
            plain(Failure::new(TURBO_E_BUNDLE_NOT_FOUND, 1, format!("no dictionary {id:?} in the vocabulary store")))
        })?),
    };
    let model = t.model.ok_or_else(|| invalid(2, "model: describe the static model to distil"))?;
    if !model.vocabulary_artifact_id.is_empty() {
        return Err(invalid(
            2,
            "model.vocabulary_artifact_id: the vocabulary is the one the job learns; leave it empty",
        ));
    }
    let model = train::plan(&c.job, model, Some(2))?;
    let index = t.index.ok_or_else(|| invalid(3, "index: name the index the documents are put in"))?;
    check_name(3, "index.index_name", &index.index_name, true)?;
    check_name(3, "index.collection_name", &index.collection_name, false)?;
    let storage = match CorpusIndexStorage::try_from(index.storage) {
        Ok(CorpusIndexStorage::FlatF32) => VectorStorageKind::FlatF32,
        Ok(CorpusIndexStorage::TurboQuant) => VectorStorageKind::TurboQuant,
        Err(_) => return Err(invalid(3, format!("index.storage {}: not a CorpusIndexStorage", index.storage))),
    };
    parse_storage(&spec(&index, storage, None), model.dims()).map_err(|f| plain(Failure { field: 3, ..f }))?;
    let taken = c.search.indexes.read().unwrap_or_else(|e| e.into_inner()).contains_key(&index.index_name)
        || c.search.aliases.read().unwrap_or_else(|e| e.into_inner()).contains_key(&index.index_name);
    if taken {
        let m = format!("a search index or alias named {} already exists", index.index_name);
        return Err(plain(Failure::new(TURBO_E_INVALID_STATE, 3, m)));
    }
    Ok((Plan { model, header, seed, index, storage }, options))
}

/// Checks the job, takes the slot, then reads the documents and runs the
/// transaction.
pub(super) async fn start(
    service: &TrainingService,
    t: CorpusTraining,
    mut requests: Streaming<StreamingTrainingRequest>,
) -> Result<UpdateStream, Status> {
    let dir = service.dir()?;
    let store = service.vocabulary.clone().ok_or_else(|| {
        plain(Failure::new(TURBO_E_INVALID_STATE, 1, "no vocabulary store: start the server with --vocabulary-dir"))
    })?;
    let corpus = Arc::new(Corpus {
        job: train::Job {
            state: service.state.clone(),
            vocabulary: service.vocabulary.clone(),
            dir,
            publishing: service.publishing.clone(),
        },
        store,
        search: service.search.clone(),
    });
    let c = corpus.clone();
    let (plan, options) = blocking(move || plan(&c, t)).await?;
    let slot = Slot::take(service.training.clone())?;
    let idle = service.upload_idle;
    let (tx, rx) = mpsc::channel(16);
    let _ = tx.send(Ok(progress(StreamingStage::Admitted, 0, 0))).await;
    tokio::spawn(async move {
        let out = match receive(&mut requests, &tx, idle, options).await {
            Ok((learner, documents)) => {
                let t = tx.clone();
                tokio::task::spawn_blocking(move || corpus.run(plan, learner, documents, &t))
                    .await
                    .unwrap_or_else(|e| Err(internal(e)))
            }
            Err(e) => Err(e),
        };
        // The slot is free once nothing of a failed job is left, before
        // the last message, so a client that has it can start the next.
        drop(slot);
        let _ = tx.send(out.map(|kept| update(Update::Corpus(kept)))).await;
    });
    Ok(Box::pin(ReceiverStream::new(rx)))
}

/// Every document of the upload, counted for the vocabulary as it
/// arrives, in the order it arrives; a full queue holds the reading back.
async fn receive(
    requests: &mut Streaming<StreamingTrainingRequest>,
    tx: &Tx,
    idle: Duration,
    options: Options,
) -> Result<(Learner, Vec<CorpusDocument>), Status> {
    let (to, mut from) = mpsc::channel::<CorpusDocument>(16);
    let t = tx.clone();
    let counting = tokio::task::spawn_blocking(move || {
        let (mut learner, mut documents) = (Learner::new(options), Vec::new());
        while let Some(d) = from.blocking_recv() {
            learner.add(&d.text);
            documents.push(d);
            if documents.len().is_multiple_of(ANALYZED_EVERY) {
                // Told when there is room: a client that reads no answer
                // while it uploads does not hold the counting back.
                let _ = t.try_send(Ok(progress(StreamingStage::Analyzing, documents.len() as u64, 0)));
            }
        }
        (learner, documents)
    });
    let read = async {
        let (mut ids, mut bytes) = (HashSet::new(), 0usize);
        while let Some(part) = next(requests, tx, idle).await? {
            let Part::Document(d) = part else {
                return Err(invalid(0, "a corpus job takes documents, not samples"));
            };
            let n = ids.len() + 1;
            if d.doc_id.is_empty() || d.doc_id.len() > MAX_NAME {
                return Err(invalid(0, format!("document {n}: doc_id is 1 to {MAX_NAME} bytes")));
            }
            if d.text.is_empty() {
                return Err(invalid(0, format!("document {n} ({:?}): no text", d.doc_id)));
            }
            bytes += d.text.len();
            if bytes > MAX_CORPUS_BYTES || n > MAX_CORPUS_DOCUMENTS {
                let m = format!("documents: more than {MAX_CORPUS_BYTES} bytes or {MAX_CORPUS_DOCUMENTS} documents");
                return Err(plain(Failure::new(TURBO_E_CAPACITY, 0, m)));
            }
            if !ids.insert(d.doc_id.clone()) {
                return Err(invalid(0, format!("document {n}: a second document with doc_id {:?}", d.doc_id)));
            }
            to.send(d).await.map_err(|_| internal("the vocabulary counter stopped"))?;
        }
        Ok(())
    }
    .await;
    drop(to);
    let counted = counting.await.map_err(internal)?;
    read.map(|()| counted)
}

impl Corpus {
    /// The transaction: the vocabulary, the model, then the index, which
    /// is filled before any search can see it. The last step stores the
    /// vocabulary and publishes the index as one, while the index's name
    /// is checked free of indexes and aliases; the job is kept from there.
    fn run(
        &self,
        mut plan: Plan,
        learner: Learner,
        documents: Vec<CorpusDocument>,
        tx: &Tx,
    ) -> Result<CorpusTrained, Status> {
        let send = |u| {
            let _ = tx.blocking_send(Ok(u));
        };
        let gone = || if tx.is_closed() { Err(cancelled()) } else { Ok(()) };
        let total = documents.len() as u64;
        send(progress(StreamingStage::Analyzing, total, total));
        let staging = train::staging(&self.job.dir, &plan.model.name);
        let mut undo = Undo { corpus: self, staging, model: None };

        let (v, tsv) = vocabulary::finish(&self.store, plan.seed.as_ref(), learner, &plan.header)?;
        let terms = plan.model.learned(v.artifact_id.clone(), v.sha256.clone(), &tsv, 2)?;
        send(progress(StreamingStage::LearningVocabulary, terms as u64, 0));
        gone()?;

        let trained = train::run(&self.job, &plan.model, &undo.staging, &Distilling(tx))?;
        undo.model = Some(plan.model.name.clone());
        let model = trained.model.as_ref().ok_or_else(|| runtime("the static model has no description"))?;

        let at_index = |f: Failure| plain(Failure { field: 3, ..f });
        let mut index =
            HybridIndex::empty(&self.job.state, spec(&plan.index, plan.storage, Some(model))).map_err(at_index)?;
        let mut staged = Vec::with_capacity(documents.len());
        for (i, d) in documents.into_iter().enumerate() {
            gone()?;
            let request = IndexDocumentRequest { doc_id: d.doc_id, text: d.text, ..Default::default() };
            staged.push(stage_document(&index, &self.job.state, request).map_err(plain)?);
            let done = i as u64 + 1;
            if done == total || (i + 1).is_multiple_of(INDEXED_EVERY) {
                send(progress(StreamingStage::Indexing, done, total));
            }
        }
        let mut chunk_terms = Vec::new();
        let (indexed_documents, indexed_chunks) =
            commit_staged_documents(&mut index, staged, &mut chunk_terms).map_err(plain)?;
        gone()?;
        let name = plan.index.index_name.clone();
        // A vocabulary of these bytes that the store already holds is
        // someone else's, and stays whatever becomes of this job; one
        // this job stores is stored only once the index's name is known
        // free, with nothing left to fail after it.
        let mut stored = None;
        let commit = || {
            let put = self.store.put_vocabulary(v, &tsv).map_err(|e| Failure::new(TURBO_E_INTERNAL, 0, e))?;
            stored = Some(put);
            Ok(())
        };
        self.search.publish_after(index, commit).map_err(|f| {
            // A taken name is the index's refusal; a store that fails is not.
            plain(Failure { field: if f.field == ALREADY_EXISTS_FIELD { 3 } else { f.field }, ..f })
        })?;
        let (v, created) = stored.ok_or_else(|| runtime("the vocabulary was not stored"))?;
        // Kept from here: nothing below fails.
        undo.model = None;
        drop(undo);
        let collection = if plan.index.collection_name.is_empty() { &name } else { &plan.index.collection_name };
        self.search.record_collection_update(collection, &name, &chunk_terms);
        send(progress(StreamingStage::Completed, total, total));
        Ok(CorpusTrained {
            vocabulary_artifact_id: v.artifact_id,
            vocabulary_sha256: v.sha256,
            vocabulary_created: created,
            vocabulary_terms: v.term_count,
            vocabulary_ngrams: v.ngram_count,
            model: Some(trained),
            index_name: name,
            indexed_documents: u64::from(indexed_documents),
            indexed_chunks: u64::from(indexed_chunks),
        })
    }
}
