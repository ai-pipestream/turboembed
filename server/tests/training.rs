//! TurboTrainingService end to end on the tiny BERT bundle: the teacher
//! listing, a static model distilled with a learned vocabulary's terms,
//! sealed, verified and served at once, then listed, refused again, and
//! deleted; the admin opt-in, the request checks, and a cancelled training
//! that leaves nothing behind.

mod common;

use std::fs;
use std::path::{Path, PathBuf};

use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{
    TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, TURBO_E_RUNTIME, TURBO_E_UNSUPPORTED,
};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::CapabilitiesRequest;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::training::train_static_model_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::proto::vocabulary::turbo_vocabulary_service_client::TurboVocabularyServiceClient;
use turbo_kserve::proto::vocabulary::{
    LearnVocabularyDocument, LearnVocabularyHeader, LearnVocabularyRequest, learn_vocabulary_request::Frame,
};
use turbo_kserve::{Serve, Server};

type Client = TurboTrainingServiceClient<Channel>;

const TEACHER: &str = "tiny-bert-bundle";
const BASE_VOCAB: u32 = 30522;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-kserve-training-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    fs::create_dir_all(&d).unwrap();
    d
}

fn bundle(name: &str) -> ModelConfig {
    ModelConfig {
        bundle: Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata").join(name).to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn server(d: &Path, admin: bool, training: bool) -> (Server, Client) {
    let serve = Serve {
        admin,
        vocabulary: Some(d.join("vocabulary")),
        training: training.then(|| d.join("training")),
        ..Serve::default()
    };
    let models = vec![bundle("tiny-bert-bundle"), bundle("tiny-static-bundle")];
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn refusal(s: &tonic::Status) -> (Code, i32) {
    let tc = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}"));
    (s.code(), tc.to_str().unwrap().parse().unwrap())
}

fn request(name: &str) -> TrainStaticModelRequest {
    TrainStaticModelRequest {
        name: name.into(),
        teacher: TEACHER.into(),
        pca_dims: 16,
        max_length: 32,
        ..Default::default()
    }
}

/// Every update of a training, the stream's error last when it ends in one.
async fn train(c: &mut Client, req: TrainStaticModelRequest) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.train_static_model(req).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

fn stage(u: &Update) -> Option<TrainStage> {
    match u {
        Update::Progress(p) => Some(p.stage()),
        Update::Trained(_) => None,
    }
}

/// A vocabulary learned from documents that use two words the tiny BERT
/// vocabulary has no entry for.
async fn learn(s: &Server) -> String {
    let mut v = TurboVocabularyServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let header = LearnVocabularyHeader { lowercase: true, min_term_frequency: 2, ..Default::default() };
    let docs = ["neuroimaging data from turboembed", "turboembed reads neuroimaging scans", "the scans"];
    let frames: Vec<_> = std::iter::once(Frame::Header(header))
        .chain(docs.iter().map(|t| Frame::Document(LearnVocabularyDocument { text: (*t).into() })))
        .map(|f| LearnVocabularyRequest { frame: Some(f) })
        .collect();
    v.learn_vocabulary(tokio_stream::iter(frames)).await.unwrap().into_inner().artifact_id
}

#[tokio::test]
async fn a_static_model_is_distilled_sealed_served_listed_and_deleted() {
    let d = scratch("full");
    let (s, mut c) = server(&d, true, true).await;

    let teachers = c.list_teachers(ListTeachersRequest {}).await.unwrap().into_inner().teachers;
    assert_eq!(teachers.len(), 1, "the static bundle is no teacher: {teachers:?}");
    assert_eq!((teachers[0].name.as_str(), teachers[0].hidden, teachers[0].vocab_size), (TEACHER, 32, BASE_VOCAB));

    let vocabulary = learn(&s).await;
    let req = TrainStaticModelRequest {
        vocabulary_artifact_id: vocabulary.clone(),
        terms: vec!["two words".into()],
        ..request("tiny-static-terms")
    };
    let updates = train(&mut c, req.clone()).await.unwrap();
    let stages: Vec<TrainStage> = updates.iter().filter_map(stage).collect();
    let mut order = stages.clone();
    order.dedup();
    use TrainStage::*;
    assert_eq!(order, [Admitted, Kept, Embedding, Reduced, Measured, Sealed, Loaded], "{stages:?}");
    let Some(Update::Progress(kept)) = updates.iter().find(|u| stage(u) == Some(Kept)) else { panic!() };
    assert_eq!((kept.total_items, kept.added_terms, kept.refused_terms), (BASE_VOCAB as u64, 2, 2));
    let Some(Update::Trained(t)) = updates.last() else { panic!("{updates:?}") };
    assert_eq!(t.added_terms, ["neuroimaging", "turboembed"]);
    let refused: Vec<(&str, &str)> = t.refused_terms.iter().map(|r| (r.term.as_str(), r.reason.as_str())).collect();
    assert_eq!(refused, [("scans", "PRESENT"), ("two words", "WORDS")]);
    assert_eq!(t.vocabulary_artifact_id, vocabulary);
    let m = t.model.as_ref().unwrap();
    assert!(m.verified && m.served && m.problem.is_empty(), "{m:?}");
    assert_eq!((m.dims, m.vocab_size, m.learned_terms, m.dtype.as_str()), (16, BASE_VOCAB + 2, 2, "F32"));
    assert_eq!(m.teacher.as_ref().unwrap().manifest_sha256, teachers[0].manifest_sha256);
    let dir = d.join("training/tiny-static-terms");
    for f in &m.files {
        let bytes = fs::read(dir.join(&f.path)).unwrap();
        assert_eq!((bytes.len() as u64, turbo::bundle::sha256_hex(&bytes)), (f.size, f.sha256.clone()), "{}", f.path);
    }
    assert!(fs::read_dir(d.join("training")).unwrap().count() == 1, "no staging is left");

    // Served at once: the inference protocol and the capability listing.
    let mut k = common::kserve(&s).await;
    let shape = common::embed_shape(&mut k, "tiny-static-terms", &[b"neuroimaging scans", b"hello"]).await;
    assert_eq!(shape, [2, 16]);
    let mut a = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let caps = a.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert!(caps.bundles.iter().any(|b| b.name == "tiny-static-terms"));

    let listed = c.list_static_models(ListStaticModelsRequest {}).await.unwrap().into_inner().models;
    assert_eq!(listed, std::slice::from_ref(m));

    // The name is taken, by the directory and the served model.
    let e = train(&mut c, req).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE), "{e:?}");

    // A changed byte is seen by the listing.
    let table = dir.join("weights/static.safetensors");
    let mut bytes = fs::read(&table).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 1;
    fs::write(&table, bytes).unwrap();
    let listed = c.list_static_models(ListStaticModelsRequest {}).await.unwrap().into_inner().models;
    assert!(!listed[0].verified && listed[0].problem.contains("weights/static.safetensors"), "{listed:?}");

    let del = DeleteStaticModelRequest { name: "tiny-static-terms".into() };
    assert!(c.delete_static_model(del.clone()).await.unwrap().into_inner().unloaded);
    assert!(!dir.exists());
    let e = k.model_infer(common::texts_req("tiny-static-terms", &[b"hello"])).await.unwrap_err();
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    assert!(c.list_static_models(ListStaticModelsRequest {}).await.unwrap().into_inner().models.is_empty());
    let e = c.delete_static_model(del).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn writes_need_admin_and_listing_needs_a_training_directory() {
    let d = scratch("admin");
    // A staging file a stopped server left goes at start, and a link to a
    // bundle elsewhere is not listed as a model.
    fs::create_dir_all(d.join("training")).unwrap();
    fs::write(d.join("training/.staging-left-0"), b"partial").unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(common::tiny(), d.join("training/linked")).unwrap();
    let (s, mut c) = server(&d, false, true).await;
    assert!(!d.join("training/.staging-left-0").exists());
    assert_eq!(c.list_teachers(ListTeachersRequest {}).await.unwrap().into_inner().teachers.len(), 1);
    let e = c.train_static_model(request("x")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    let e = c.delete_static_model(DeleteStaticModelRequest { name: "x".into() }).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    assert!(c.list_static_models(ListStaticModelsRequest {}).await.unwrap().into_inner().models.is_empty());
    s.stop().await;

    #[cfg(unix)]
    {
        let (s, mut c) = server(&d, true, true).await;
        let e = c.delete_static_model(DeleteStaticModelRequest { name: "linked".into() }).await.unwrap_err();
        assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
        assert!(common::tiny().join("manifest.json").is_file());
        s.stop().await;
    }

    let (s, mut c) = server(&d, true, false).await;
    let e = c.list_static_models(ListStaticModelsRequest {}).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    let e = c.train_static_model(request("x")).await.unwrap_err();
    assert!(e.message().contains("--training-dir"), "{e:?}");
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn bad_requests_are_refused_before_any_work() {
    let d = scratch("checks");
    let (s, mut c) = server(&d, true, true).await;
    let bad = |e: tonic::Status| (refusal(&e), e.metadata().get("turbo-field").map(|v| v.to_str().unwrap().to_owned()));
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let cases = [
        (request("../escape"), invalid, "1"),
        (request(".hidden"), invalid, "1"),
        (request(TEACHER), invalid, "1"),
        (
            TrainStaticModelRequest { teacher: "nobody".into(), ..request("x") },
            (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND),
            "2",
        ),
        (TrainStaticModelRequest { teacher: "tiny-static-bundle".into(), ..request("x") }, invalid, "2"),
        (TrainStaticModelRequest { pca_dims: 33, ..request("x") }, invalid, "3"),
        (TrainStaticModelRequest { sif_coefficient: 1.0, ..request("x") }, invalid, "4"),
        (TrainStaticModelRequest { dtype: 9, ..request("x") }, invalid, "5"),
        (TrainStaticModelRequest { max_length: 8193, ..request("x") }, invalid, "6"),
        (
            TrainStaticModelRequest { vocabulary_artifact_id: "vocab-none".into(), ..request("x") },
            (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND),
            "7",
        ),
    ];
    for (req, want, field) in cases {
        let e = c.train_static_model(req.clone()).await.unwrap_err();
        assert_eq!(bad(e), (want, Some(field.to_owned())), "{req:?}");
    }
    assert_eq!(fs::read_dir(d.join("training")).unwrap().count(), 0);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_cancelled_training_leaves_nothing_and_frees_its_slot() {
    let d = scratch("cancel");
    let (s, mut c) = server(&d, true, true).await;
    let mut stream = c.train_static_model(request("cancelled")).await.unwrap().into_inner();
    let first = stream.message().await.unwrap().unwrap().update.unwrap();
    assert_eq!(stage(&first), Some(TrainStage::Admitted));
    drop(stream);
    // The training stops at its next step and gives its slot back; until
    // then another is refused as busy.
    let updates = loop {
        match train(&mut c, request("after")).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(std::time::Duration::from_millis(50)).await,
            other => break other.unwrap(),
        }
    };
    assert!(matches!(updates.last(), Some(Update::Trained(_))), "{updates:?}");
    let names: Vec<String> =
        fs::read_dir(d.join("training")).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
    assert_eq!(names, ["after"]);
    let mut k = common::kserve(&s).await;
    let e = k.model_infer(common::texts_req("cancelled", &[b"hello"])).await.unwrap_err();
    assert_eq!(e.code(), Code::NotFound);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_failed_distillation_is_a_runtime_error_with_no_files() {
    let d = scratch("fail");
    // A teacher whose tokenizer file goes after it is loaded: it still
    // lists, and the distillation fails reading it, after admission.
    let teacher = d.join("teacher");
    fs::create_dir_all(teacher.join("weights")).unwrap();
    fs::create_dir_all(teacher.join("reference")).unwrap();
    let tiny = common::tiny();
    for f in ["manifest.json", "tokenizer.json", "weights/model.safetensors", "reference/reference.safetensors"] {
        fs::copy(tiny.join(f), teacher.join(f)).unwrap();
    }
    let model = ModelConfig { bundle: teacher.to_str().unwrap().into(), name: Some(TEACHER.into()), ..bundle(TEACHER) };
    let serve = Serve { admin: true, training: Some(d.join("training")), ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let mut c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    fs::remove_file(teacher.join("tokenizer.json")).unwrap();
    let e = train(&mut c, request("broken")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Internal, TURBO_E_RUNTIME), "{e:?}");
    assert!(e.message().contains("tokenizer.json"), "{e:?}");
    assert_eq!(fs::read_dir(d.join("training")).unwrap().count(), 0, "no staging or model is left");
    let mut k = common::kserve(&s).await;
    assert_eq!(k.model_infer(common::texts_req("broken", &[b"hello"])).await.unwrap_err().code(), Code::NotFound);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}
