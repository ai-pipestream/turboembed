//! TrainStaticModel and StreamingTraining run the teacher encoder on the
//! device the server loaded that teacher on, including CUDA.

#![cfg(feature = "cuda")]

mod common;

use std::fs;
use std::path::Path;

use serde_json::Value;
use tonic::transport::Channel;
use turbo_kserve::api::{Runtime, field};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::streaming_training_request::Part;
use turbo_kserve::proto::training::streaming_training_start::Job;
use turbo_kserve::proto::training::streaming_training_update::Update as StreamUpdate;
use turbo_kserve::proto::training::train_static_model_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_scratch::Scratch;

type Client = TurboTrainingServiceClient<Channel>;

fn scratch(name: &str) -> Scratch {
    let d = Scratch::at(std::env::temp_dir().join(format!("turbo-kserve-training-cuda-{}-{name}", std::process::id())));
    fs::create_dir_all(&d).unwrap();
    d
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn backend_arg(bundle: &Path) -> String {
    let m: Value = serde_json::from_slice(&fs::read(bundle.join("manifest.json")).unwrap()).unwrap();
    m["artifacts"][0]["produced_by"]["args"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|a| a.as_str()?.strip_prefix("backend="))
        .unwrap_or("")
        .to_owned()
}

async fn train(c: &mut Client, req: TrainStaticModelRequest) -> TrainedStaticModel {
    let mut stream = c.train_static_model(req).await.unwrap().into_inner();
    let mut last = None;
    while let Some(u) = stream.message().await.unwrap() {
        if let Some(Update::Trained(t)) = u.update {
            last = Some(t);
        }
    }
    last.expect("TrainedStaticModel message")
}

async fn infer_rows(k: &mut common::Kserve, model: &str, texts: &[&[u8]]) -> Vec<Vec<f32>> {
    let r = k.model_infer(common::texts_req(model, texts)).await.unwrap().into_inner();
    let dim = r.outputs[0].shape[1] as usize;
    let flat: Vec<f32> = r.raw_output_contents[0].as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
    flat.chunks_exact(dim).map(<[f32]>::to_vec).collect()
}

#[tokio::test]
async fn teacher_loaded_on_cuda_distils_and_serves_static_model_matching_cpu() {
    let rt = Runtime::create().unwrap();
    let has_cuda = (0..16).any(|i| rt.device_info(i).is_ok_and(|d| field(&d.backend) == "cuda"));
    if !has_cuda {
        assert!(
            turbo_pins::device_skip_allowed("cuda"),
            "the cuda backend lists no device; TURBO_TEST_REQUIRE_CUDA=0 lets the test skip"
        );
        eprintln!("CUDA unavailable; skipped");
        return;
    }

    let d = scratch("cuda");
    let bundle_path: String = common::tiny().to_str().unwrap().into();
    let models = vec![
        ModelConfig {
            bundle: bundle_path.clone(),
            name: Some("teacher-cuda".into()),
            device: Device::Select,
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        },
        ModelConfig {
            bundle: bundle_path,
            name: Some("teacher-cpu".into()),
            device: Device::Index(common::cpu()),
            precision: 0,
            max_batch: 0,
            max_seq: 0,
            sessions: 1,
        },
    ];
    let serve = Serve {
        admin: true,
        vocabulary: Some(d.join("vocabulary")),
        training: Some(d.join("training")),
        ..Serve::default()
    };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let mut c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();

    let terms = vec!["neuroimaging".to_owned(), "turboembed".to_owned()];
    let make_req = |name: &str, teacher: &str| TrainStaticModelRequest {
        name: name.into(),
        teacher: teacher.into(),
        pca_dims: 16,
        max_length: 32,
        terms: terms.clone(),
        ..Default::default()
    };

    let t_cuda = train(&mut c, make_req("from-cuda", "teacher-cuda")).await;
    let t_cpu = train(&mut c, make_req("from-cpu", "teacher-cpu")).await;
    assert_eq!(t_cuda.added_terms, ["neuroimaging", "turboembed"]);
    assert_eq!(t_cpu.added_terms, ["neuroimaging", "turboembed"]);
    assert_eq!(backend_arg(&d.join("training/from-cuda")), "cuda");
    assert_eq!(backend_arg(&d.join("training/from-cpu")), "cpu");

    let mut k = common::kserve(&s).await;
    let queries: [&[u8]; 4] = [
        b"neuroimaging scans from turboembed",
        b"the quick brown fox jumps over the lazy dog",
        "Café naïve RÉSUMÉ".as_bytes(),
        b"reset password in security settings",
    ];
    let (v_cuda, v_cpu) =
        (infer_rows(&mut k, "from-cuda", &queries).await, infer_rows(&mut k, "from-cpu", &queries).await);
    for (i, (a, b)) in v_cuda.iter().zip(&v_cpu).enumerate() {
        let cos = cosine(a, b);
        assert!(cos > 0.9999, "query {i}: cosine {cos}");
    }

    // StreamingTraining corpus job also runs the teacher on its loaded CUDA device.
    let start = StreamingTrainingRequest {
        part: Some(Part::Start(StreamingTrainingStart {
            job: Some(Job::Corpus(CorpusTraining {
                vocabulary: Some(CorpusVocabulary { lowercase: true, min_term_frequency: 2, ..Default::default() }),
                model: Some(TrainStaticModelRequest {
                    name: "corpus-cuda".into(),
                    teacher: "teacher-cuda".into(),
                    pca_dims: 16,
                    max_length: 32,
                    ..Default::default()
                }),
                index: Some(CorpusIndex { index_name: "idx-cuda".into(), ..Default::default() }),
            })),
        })),
    };
    let docs = ["neuroimaging scans from turboembed", "turboembed analyses neuroimaging scans"]
        .into_iter()
        .enumerate()
        .map(|(i, text)| StreamingTrainingRequest {
            part: Some(Part::Document(CorpusDocument { doc_id: format!("d{i}"), text: text.into() })),
        });
    let mut stream =
        c.streaming_training(tokio_stream::iter(std::iter::once(start).chain(docs))).await.unwrap().into_inner();
    let mut corpus_done = None;
    while let Some(u) = stream.message().await.unwrap() {
        if let Some(StreamUpdate::Corpus(done)) = u.update {
            corpus_done = Some(done);
        }
    }
    let done = corpus_done.expect("CorpusTrained message");
    assert_eq!(done.indexed_documents, 2);
    assert_eq!(backend_arg(&d.join("training/corpus-cuda")), "cuda");

    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}
