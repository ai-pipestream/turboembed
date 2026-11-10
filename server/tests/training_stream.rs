//! A StreamingTraining corpus job end to end on the tiny BERT bundle: a
//! vocabulary learned from the streamed documents, a static model
//! distilled with its terms and served, and an index of the documents
//! that a search reads; the start's checks, each naming the part of the
//! job it refuses; the stream's own rules; and a job that fails or is
//! abandoned leaving no vocabulary, model or index, with the slot free.

mod common;

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{
    TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE,
    TURBO_E_UNSUPPORTED,
};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::search::turbo_search_service_client::TurboSearchServiceClient;
use turbo_kserve::proto::search::{
    CreateIndexRequest, IndexSpec, ListIndexesRequest, ModelIdentity, SearchRequest, SpaceSpec,
};
use turbo_kserve::proto::training::streaming_training_request::Part;
use turbo_kserve::proto::training::streaming_training_start::Job;
use turbo_kserve::proto::training::streaming_training_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
use turbo_kserve::proto::vocabulary::{
    LearnVocabularyDocument, LearnVocabularyHeader, LearnVocabularyRequest, ListVocabulariesRequest,
    learn_vocabulary_request::Frame,
};
use turbo_kserve::{Serve, Server};

type Client = TurboTrainingServiceClient<Channel>;

/// Documents that use two words the tiny BERT vocabulary has no entry for.
const DOCS: [(&str, &str); 3] = [
    ("a", "Neuroimaging data from turboembed. The second sentence is about weather."),
    ("b", "Turboembed reads neuroimaging scans."),
    ("c", "The scans were clear."),
];

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-kserve-streaming-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

struct Dirs {
    admin: bool,
    training: bool,
    vocabulary: bool,
}

const ALL: Dirs = Dirs { admin: true, training: true, vocabulary: true };

async fn server(d: &Path, dirs: Dirs) -> (Server, Client) {
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

fn refusal(s: &tonic::Status) -> ((Code, i32), String) {
    let get = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    ((s.code(), get("turbo-code").parse().unwrap()), get("turbo-field"))
}

fn vocabulary() -> CorpusVocabulary {
    CorpusVocabulary { lowercase: true, min_term_frequency: 2, ..Default::default() }
}

fn corpus(model: &str, index: &str) -> CorpusTraining {
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

fn part(p: Part) -> StreamingTrainingRequest {
    StreamingTrainingRequest { part: Some(p) }
}

fn start(c: CorpusTraining) -> StreamingTrainingRequest {
    part(Part::Start(StreamingTrainingStart { job: Some(Job::Corpus(c)) }))
}

fn document(id: &str, text: &str) -> StreamingTrainingRequest {
    part(Part::Document(CorpusDocument { doc_id: id.into(), text: text.into() }))
}

fn documents() -> Vec<StreamingTrainingRequest> {
    DOCS.iter().map(|(id, text)| document(id, text)).collect()
}

/// Every update of a job, the stream's error when it ends in one.
async fn run(c: &mut Client, requests: Vec<StreamingTrainingRequest>) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.streaming_training(tokio_stream::iter(requests)).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

async fn job(c: &mut Client, t: CorpusTraining) -> Result<Vec<Update>, tonic::Status> {
    run(c, std::iter::once(start(t)).chain(documents()).collect()).await
}

fn stages(updates: &[Update]) -> Vec<StreamingStage> {
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
async fn held(s: &Server, d: &Path) -> (Vec<String>, Vec<String>, Vec<String>) {
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

#[tokio::test]
async fn a_corpus_becomes_a_vocabulary_a_served_model_and_a_searchable_index() {
    let d = scratch("full");
    let (s, mut c) = server(&d, ALL).await;
    let updates = job(&mut c, corpus("corpus-static", "corpus-index")).await.unwrap();
    use StreamingStage::*;
    assert_eq!(
        stages(&updates),
        [Admitted, Analyzing, LearningVocabulary, DistillingModel, Indexing, Completed],
        "{updates:?}"
    );
    let progress = |stage| {
        updates.iter().rev().find_map(|u| match u {
            Update::Progress(p) if p.stage() == stage => Some(p),
            _ => None,
        })
    };
    let analyzed = progress(Analyzing).unwrap();
    assert_eq!((analyzed.completed_items, analyzed.total_items), (3, 3));
    // neuroimaging, turboembed, scans and the are seen twice.
    assert_eq!(progress(LearningVocabulary).unwrap().completed_items, 4);
    let distilled: Vec<TrainStage> = updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) => p.distillation.as_ref().map(|d| d.stage()),
            _ => None,
        })
        .collect();
    assert_eq!(distilled.first(), Some(&TrainStage::Kept), "{distilled:?}");
    assert_eq!(distilled[distilled.len() - 2..], [TrainStage::Sealed, TrainStage::Loaded], "{distilled:?}");
    let indexed = progress(Indexing).unwrap();
    assert_eq!((indexed.completed_items, indexed.total_items), (3, 3));

    let Some(Update::Corpus(kept)) = updates.last() else { panic!("{updates:?}") };
    assert!(kept.vocabulary_created);
    assert_eq!((kept.vocabulary_terms, kept.vocabulary_ngrams), (4, 0));
    let trained = kept.model.as_ref().unwrap();
    assert_eq!(trained.added_terms, ["neuroimaging", "turboembed"]);
    assert_eq!(trained.vocabulary_artifact_id, kept.vocabulary_artifact_id);
    assert_eq!(trained.vocabulary_sha256, kept.vocabulary_sha256);
    let m = trained.model.as_ref().unwrap();
    assert!(m.verified && m.served, "{m:?}");
    assert_eq!((m.name.as_str(), m.dims, m.learned_terms), ("corpus-static", 16, 2));
    assert_eq!((kept.index_name.as_str(), kept.indexed_documents), ("corpus-index", 3));
    assert!(kept.indexed_chunks >= 3, "{kept:?}");

    // All three are where their own services keep them.
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!(vocabularies, std::slice::from_ref(&kept.vocabulary_artifact_id));
    assert_eq!(models, ["corpus-static"], "no staging is left");
    assert_eq!(indexes, ["corpus-index"]);
    let mut k = common::kserve(&s).await;
    assert_eq!(common::embed_shape(&mut k, "corpus-static", &[b"neuroimaging scans"]).await, [1, 16]);

    // The index is in the model's space, every chunk has its vector, and
    // a search reads the documents.
    let mut x = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let summary = x.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes.remove(0);
    let space = summary.space.as_ref().unwrap();
    assert_eq!(space.dim, 16);
    assert_eq!(space.model.as_ref().unwrap().sha256, m.manifest_sha256);
    assert_eq!((summary.document_count, u64::from(summary.chunk_count)), (3, kept.indexed_chunks));
    assert_eq!(summary.dense_row_count, summary.chunk_count);
    let query = SearchRequest {
        index_name: "corpus-index".into(),
        natural_query: "neuroimaging".into(),
        top_k: 3,
        ..Default::default()
    };
    let hits = x.search(query).await.unwrap().into_inner().hits;
    let mut found: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    found.sort();
    assert_eq!(found, ["a", "b"], "{hits:?}");

    // Both names are taken now, and the vocabulary is the store's.
    let e = job(&mut c, corpus("corpus-static", "other")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "2".into()), "{e:?}");
    let e = job(&mut c, corpus("other", "corpus-index")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "3".into()), "{e:?}");
    let updates = job(&mut c, corpus("second-static", "second-index")).await.unwrap();
    let Some(Update::Corpus(second)) = updates.last() else { panic!("{updates:?}") };
    assert!(!second.vocabulary_created);
    assert_eq!(second.vocabulary_artifact_id, kept.vocabulary_artifact_id);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_corpus_job_needs_admin_a_training_directory_and_a_vocabulary_store() {
    let d = scratch("opt-in");
    let unsupported = (Code::Unimplemented, TURBO_E_UNSUPPORTED);
    let (s, mut c) = server(&d, Dirs { admin: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, unsupported, "{e:?}");
    assert!(e.message().contains("--admin"), "{e:?}");
    s.stop().await;
    let (s, mut c) = server(&d, Dirs { training: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, unsupported, "{e:?}");
    assert!(e.message().contains("--training-dir"), "{e:?}");
    s.stop().await;
    let (s, mut c) = server(&d, Dirs { vocabulary: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "1".into()), "{e:?}");
    assert!(e.message().contains("--vocabulary-dir"), "{e:?}");
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_bad_start_is_refused_for_the_part_it_names_before_any_work() {
    let d = scratch("checks");
    let (s, mut c) = server(&d, ALL).await;
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let not_found = (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND);
    let good = || corpus("x", "x");
    let with_vocabulary = |v: CorpusVocabulary| CorpusTraining { vocabulary: Some(v), ..good() };
    let with_model = |f: &dyn Fn(&mut TrainStaticModelRequest)| {
        let mut t = good();
        f(t.model.as_mut().unwrap());
        t
    };
    let with_index = |ix: CorpusIndex| CorpusTraining { index: Some(ix), ..good() };
    let named = |ix: CorpusIndex| CorpusIndex { index_name: "x".into(), ..ix };
    let cases = [
        (with_vocabulary(CorpusVocabulary { language: "a\nb".into(), ..vocabulary() }), invalid, "1"),
        (with_vocabulary(CorpusVocabulary { include_ngrams: true, min_ngram_size: 1, ..vocabulary() }), invalid, "1"),
        (
            with_vocabulary(CorpusVocabulary { dictionary_artifact_id: "dict-none".into(), ..vocabulary() }),
            not_found,
            "1",
        ),
        (CorpusTraining { model: None, ..good() }, invalid, "2"),
        (with_model(&|m| m.vocabulary_artifact_id = "vocab-none".into()), invalid, "2"),
        (with_model(&|m| m.name = "../escape".into()), invalid, "2"),
        (with_model(&|m| m.teacher = "nobody".into()), not_found, "2"),
        (with_model(&|m| m.pca_dims = 33), invalid, "2"),
        (CorpusTraining { index: None, ..good() }, invalid, "3"),
        (with_index(CorpusIndex::default()), invalid, "3"),
        (with_index(named(CorpusIndex { storage: 7, ..Default::default() })), invalid, "3"),
        (with_index(named(CorpusIndex { collection_name: "a\tb".into(), ..Default::default() })), invalid, "3"),
        (
            with_index(named(CorpusIndex {
                storage: CorpusIndexStorage::TurboQuant as i32,
                quant_bits: 9,
                ..Default::default()
            })),
            invalid,
            "3",
        ),
        // 12 values are no whole number of TurboQuant's blocks of 8.
        (
            CorpusTraining {
                index: Some(named(CorpusIndex {
                    storage: CorpusIndexStorage::TurboQuant as i32,
                    ..Default::default()
                })),
                ..with_model(&|m| m.pca_dims = 12)
            },
            invalid,
            "3",
        ),
    ];
    for (t, want, field) in cases {
        let e = job(&mut c, t.clone()).await.unwrap_err();
        assert_eq!(refusal(&e), (want, field.to_owned()), "{t:?}: {e:?}");
    }

    // The stream's own rules, field 0: the start first and once, a job
    // named, documents and not samples, each with an id of its own and a
    // text, and at least one.
    let stream_cases: Vec<(Vec<StreamingTrainingRequest>, &str)> = vec![
        (vec![], "before its start"),
        (documents(), "first message must be the start"),
        (vec![part(Part::Start(StreamingTrainingStart { job: None }))], "corpus job or a component job"),
        (vec![start(good()), document("a", "one"), start(good())], "second start"),
        (vec![start(good()), part(Part::Samples(b"1\tx".to_vec()))], "not samples"),
        (vec![start(good()), document("", "one")], "doc_id"),
        (vec![start(good()), document("a", "")], "no text"),
        (vec![start(good()), document("a", "one"), document("a", "two")], "second document"),
        (vec![start(good())], "no documents"),
    ];
    for (requests, why) in stream_cases {
        let e = run(&mut c, requests).await.unwrap_err();
        assert_eq!(refusal(&e), (invalid, "0".into()), "{why}: {e:?}");
        assert!(e.message().contains(why), "{why}: {e:?}");
    }
    assert_eq!(held(&s, &d).await, (vec![], vec![], vec![]));
    // Every refusal gave the slot back.
    job(&mut c, good()).await.unwrap();
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

/// A job whose upload stays open until the sender is dropped.
async fn open(
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
async fn when_free(c: &mut Client, t: CorpusTraining) -> Vec<Update> {
    loop {
        match job(c, t.clone()).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
            other => break other.unwrap(),
        }
    }
}

#[tokio::test]
async fn a_job_that_fails_at_its_last_step_removes_its_model_and_its_vocabulary() {
    let d = scratch("rollback");
    let (s, mut c) = server(&d, ALL).await;
    let at = format!("http://{}", s.local_addr());
    let mut x = TurboSearchServiceClient::connect(at.clone()).await.unwrap();
    // The index name is free at the start and taken before the job
    // publishes: the vocabulary and the model are made, then removed.
    let taken = |name: &str| CreateIndexRequest {
        spec: Some(IndexSpec {
            index_name: name.into(),
            space: Some(SpaceSpec {
                model: Some(ModelIdentity { id: "other".into(), revision: "1".into(), sha256: "0".repeat(64) }),
                dim: 8,
                normalization: 1,
            }),
            ..Default::default()
        }),
        load_from_path: String::new(),
    };
    let (tx, mut stream) = open(&mut c, corpus("rolled-back", "late")).await;
    // While it holds the slot another job is busy.
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    x.create_index(taken("late")).await.unwrap();
    for r in documents() {
        tx.send(r).await.unwrap();
    }
    drop(tx);
    let mut updates = Vec::new();
    let e = loop {
        match stream.message().await {
            Ok(Some(u)) => updates.push(u.update.unwrap()),
            Ok(None) => panic!("the job ended without its failure: {updates:?}"),
            Err(e) => break e,
        }
    };
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "3".into()), "{e:?}");
    // It got as far as the index: the model was served on the way.
    use StreamingStage::*;
    assert_eq!(stages(&updates), [Analyzing, LearningVocabulary, DistillingModel, Indexing], "{updates:?}");
    assert_eq!(held(&s, &d).await, (vec![], vec![], vec!["late".to_owned()]));
    let mut k = common::kserve(&s).await;
    let e = k.model_infer(common::texts_req("rolled-back", &[b"hello"])).await.unwrap_err();
    assert_eq!(e.code(), Code::NotFound, "{e:?}");

    // A vocabulary the store held before the job is not the job's to
    // remove: LearnVocabulary of the same documents gives the same bytes.
    let mut v = TurboVocabularyServiceClient::connect(at).await.unwrap();
    let header = LearnVocabularyHeader { lowercase: true, min_term_frequency: 2, ..Default::default() };
    let frames: Vec<_> = std::iter::once(Frame::Header(header))
        .chain(DOCS.iter().map(|(_, t)| Frame::Document(LearnVocabularyDocument { text: (*t).into() })))
        .map(|f| LearnVocabularyRequest { frame: Some(f) })
        .collect();
    let learned = v.learn_vocabulary(tokio_stream::iter(frames)).await.unwrap().into_inner().artifact_id;
    let e = {
        let (tx, mut stream) = open(&mut c, corpus("rolled-back", "later")).await;
        x.create_index(taken("later")).await.unwrap();
        for r in documents() {
            tx.send(r).await.unwrap();
        }
        drop(tx);
        loop {
            match stream.message().await {
                Ok(Some(_)) => {}
                Ok(None) => panic!("the job ended without its failure"),
                Err(e) => break e,
            }
        }
    };
    assert_eq!(refusal(&e).1, "3", "{e:?}");
    assert_eq!(held(&s, &d).await, (vec![learned.clone()], vec![], vec!["late".to_owned(), "later".to_owned()]));

    // The names the failed jobs asked for are free again.
    let updates = job(&mut c, corpus("rolled-back", "third")).await.unwrap();
    let Some(Update::Corpus(kept)) = updates.last() else { panic!("{updates:?}") };
    assert_eq!((kept.vocabulary_artifact_id.as_str(), kept.vocabulary_created), (learned.as_str(), false));
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn an_abandoned_or_idle_upload_leaves_nothing_and_frees_the_slot() {
    let d = scratch("cancel");
    let (s, mut c) = server(&d, ALL).await;
    // The client drops the answer and its upload after two documents.
    let (tx, stream) = open(&mut c, corpus("gone", "gone")).await;
    tx.send(document("a", DOCS[0].1)).await.unwrap();
    tx.send(document("b", DOCS[1].1)).await.unwrap();
    drop((tx, stream));
    let updates = when_free(&mut c, corpus("after", "after")).await;
    assert!(matches!(updates.last(), Some(Update::Corpus(_))), "{updates:?}");
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!((vocabularies.len(), models, indexes), (1, vec!["after".to_owned()], vec!["after".to_owned()]));

    // An upload that sends nothing for the idle limit is dropped.
    let (tx, mut stream) = open(&mut c, corpus("idle", "idle")).await;
    tx.send(document("a", DOCS[0].1)).await.unwrap();
    let e = loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the idle upload ended without a status"),
            Err(e) => break e,
        }
    };
    assert_eq!(refusal(&e), ((Code::OutOfRange, TURBO_E_CAPACITY), "0".into()), "{e:?}");
    drop(tx);
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!((vocabularies.len(), models, indexes), (1, vec!["after".to_owned()], vec!["after".to_owned()]));
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}
