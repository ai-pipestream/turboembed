//! A StreamingTraining component job end to end: a feed-forward
//! dependency parser trained from CoNLL-U samples streamed in chunks cut
//! anywhere, each epoch streamed as progress, and its model file sent
//! back whole, the bytes the trainer gives for those samples and
//! settings; the admin opt-in and the job's checks; the stream's rules;
//! and the one training slot it shares.

mod common;

use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_BUSY, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::streaming_training_request::Part;
use turbo_kserve::proto::training::streaming_training_start::Job;
use turbo_kserve::proto::training::streaming_training_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_train_components::depparse::{
    FeatureLayout, FeedforwardDependencyModel, FeedforwardSettings, Tagset, train_feedforward_conllu,
};

type Client = TurboTrainingServiceClient<Channel>;

/// A real UD sample: Spanish sentences with multiword tokens.
const CONLLU: &str = include_str!("../../analysis/depparse/tests/data/ud-es.conllu");

/// No training directory and no vocabulary store: the job needs neither.
async fn server(admin: bool) -> (Server, Client) {
    let serve = Serve { admin, upload_idle: Some(Duration::from_secs(2)), ..Serve::default() };
    let tiny = ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![tiny], MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn parser() -> NeuralComponentTraining {
    NeuralComponentTraining {
        component: NeuralComponentKind::DependencyParser as i32,
        embedding_size: 3,
        hidden_size: 5,
        epochs: 3,
        batch_size: 16,
        ..Default::default()
    }
}

fn part(p: Part) -> StreamingTrainingRequest {
    StreamingTrainingRequest { part: Some(p) }
}

fn start(t: NeuralComponentTraining) -> StreamingTrainingRequest {
    part(Part::Start(StreamingTrainingStart { job: Some(Job::Component(t)) }))
}

fn samples(b: &[u8]) -> StreamingTrainingRequest {
    part(Part::Samples(b.to_vec()))
}

async fn run(c: &mut Client, requests: Vec<StreamingTrainingRequest>) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.streaming_training(tokio_stream::iter(requests)).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

fn refusal(s: &tonic::Status) -> ((Code, i32), String) {
    let get = |k: &str| s.metadata().get(k).unwrap_or_else(|| panic!("no {k} in {s:?}")).to_str().unwrap().to_owned();
    ((s.code(), get("turbo-code").parse().unwrap()), get("turbo-field"))
}

#[tokio::test]
async fn a_dependency_parser_is_trained_from_streamed_conllu_and_sent_back() {
    let (s, mut c) = server(true).await;
    // Chunks of 97 bytes cut lines, and the accented letters, anywhere.
    let requests: Vec<_> = std::iter::once(start(parser())).chain(CONLLU.as_bytes().chunks(97).map(samples)).collect();
    let updates = run(&mut c, requests).await.unwrap();
    let sentences: Vec<&str> = CONLLU.trim().split("\n\n").collect();

    let mut progress = Vec::new();
    let mut model = Vec::new();
    for u in &updates[..updates.len() - 1] {
        match u {
            Update::Progress(p) => {
                assert!(model.is_empty(), "progress after the model's first chunk: {p:?}");
                progress.push(p);
            }
            Update::Model(b) => model.extend_from_slice(b),
            other => panic!("{other:?}"),
        }
    }
    use StreamingStage::*;
    let stages: Vec<_> = progress.iter().map(|p| p.stage()).collect();
    assert_eq!(stages, [Admitted, Read, Epoch, Epoch, Epoch, Packaged]);
    assert_eq!(progress[1].completed_items, sentences.len() as u64);
    let epochs: Vec<_> = progress[2..5].iter().map(|p| (p.completed_items, p.total_items)).collect();
    assert_eq!(epochs, [(1, 3), (2, 3), (3, 3)]);
    assert!(progress[2..5].iter().all(|p| p.loss.is_finite() && p.loss > 0.0), "{progress:?}");
    assert!(progress[4].loss < progress[2].loss, "the loss falls: {progress:?}");

    let Some(Update::Component(trained)) = updates.last() else { panic!("{updates:?}") };
    assert_eq!(trained.component(), NeuralComponentKind::DependencyParser);
    assert_eq!((trained.samples, trained.size), (sentences.len() as u64, model.len() as u64));
    assert_eq!(trained.sha256, turbo::bundle::sha256_hex(&model));

    // The file is the trainer's own for these samples and settings, and
    // loads as a model.
    let settings = FeedforwardSettings {
        embedding_size: 3,
        hidden_size: 5,
        epochs: 3,
        batch_size: 16,
        layout: FeatureLayout::ChenManning18,
        ..Default::default()
    };
    let local = train_feedforward_conllu(&sentences, Tagset::U, &settings).unwrap();
    assert_eq!(model, local.to_bytes().unwrap());
    FeedforwardDependencyModel::from_bytes(&model).unwrap();

    // Another layout and activation give another model.
    let other = NeuralComponentTraining {
        layout: DependencyFeatureLayout::Compact14 as i32,
        activation: DependencyActivation::Relu as i32,
        ..parser()
    };
    let updates = run(&mut c, vec![start(other), samples(CONLLU.as_bytes())]).await.unwrap();
    let Some(Update::Component(second)) = updates.last() else { panic!("{updates:?}") };
    assert_ne!(second.sha256, trained.sha256);
    s.stop().await;
}

#[tokio::test]
async fn a_component_job_needs_admin_and_good_settings_and_samples() {
    let (s, mut c) = server(false).await;
    let e = run(&mut c, vec![start(parser()), samples(CONLLU.as_bytes())]).await.unwrap_err();
    assert_eq!(refusal(&e).0, (Code::Unimplemented, TURBO_E_UNSUPPORTED), "{e:?}");
    s.stop().await;

    let (s, mut c) = server(true).await;
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let cases = [
        (NeuralComponentTraining { component: 0, ..parser() }, "1"),
        (NeuralComponentTraining { component: 9, ..parser() }, "1"),
        (NeuralComponentTraining { tagset: 5, ..parser() }, "2"),
        (NeuralComponentTraining { epochs: 10_001, ..parser() }, "5"),
        (NeuralComponentTraining { layout: 4, ..parser() }, "12"),
        (NeuralComponentTraining { activation: 4, ..parser() }, "13"),
        // The trainer's own bounds.
        (NeuralComponentTraining { embedding_size: 4097, ..parser() }, "0"),
        (NeuralComponentTraining { learning_rate: -1.0, ..parser() }, "0"),
        (NeuralComponentTraining { dropout: Some(1.0), ..parser() }, "0"),
        (NeuralComponentTraining { l2: Some(-1.0), ..parser() }, "0"),
    ];
    for (t, field) in cases {
        let e = run(&mut c, vec![start(t), samples(CONLLU.as_bytes())]).await.unwrap_err();
        assert_eq!(refusal(&e), (invalid, field.to_owned()), "{t:?}: {e:?}");
    }
    let document = part(Part::Document(CorpusDocument { doc_id: "a".into(), text: "one".into() }));
    let stream_cases: Vec<(Vec<StreamingTrainingRequest>, &str)> = vec![
        (vec![start(parser())], "none were read"),
        (vec![start(parser()), samples(b"\n \n")], "none were read"),
        (vec![start(parser()), samples(b"1\tCaf\xc3")], "not UTF-8 at byte 5"),
        (vec![start(parser()), samples(b"not a sentence\n")], "sentence 1"),
        (vec![start(parser()), document], "not documents"),
        (vec![start(parser()), samples(b"1"), start(parser())], "second start"),
    ];
    for (requests, why) in stream_cases {
        let e = run(&mut c, requests).await.unwrap_err();
        assert_eq!(refusal(&e), (invalid, "0".into()), "{why}: {e:?}");
        assert!(e.message().contains(why), "{why}: {e:?}");
    }
    // Zero dropout, zero penalty and a zero seed are settings, not unset.
    let zeros = NeuralComponentTraining { dropout: Some(0.0), l2: Some(0.0), seed: Some(0), ..parser() };
    let updates = run(&mut c, vec![start(zeros), samples(CONLLU.as_bytes())]).await.unwrap();
    assert!(matches!(updates.last(), Some(Update::Component(_))), "{updates:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_component_job_shares_the_training_slot_and_frees_it_when_abandoned() {
    let (s, mut c) = server(true).await;
    let (tx, rx) = mpsc::channel(4);
    tx.send(start(parser())).await.unwrap();
    let mut stream = c.streaming_training(ReceiverStream::new(rx)).await.unwrap().into_inner();
    let first = stream.message().await.unwrap().unwrap().update.unwrap();
    assert!(matches!(first, Update::Progress(p) if p.stage() == StreamingStage::Admitted));
    // TrainComponent and another streamed job are busy while it runs.
    let e = run(&mut c, vec![start(parser()), samples(CONLLU.as_bytes())]).await.unwrap_err();
    assert_eq!(refusal(&e).0, (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    let classical = TrainComponentRequest {
        part: Some(train_component_request::Part::Start(TrainComponentStart {
            component: ComponentKind::PosTagger as i32,
            language: "en".into(),
            ..Default::default()
        })),
    };
    let e = c.train_component(tokio_stream::iter(vec![classical])).await.unwrap_err();
    assert_eq!(refusal(&e).0, (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    // The client goes away mid-upload: the slot comes back.
    tx.send(samples(&CONLLU.as_bytes()[..200])).await.unwrap();
    drop((tx, stream));
    let updates = loop {
        match run(&mut c, vec![start(parser()), samples(CONLLU.as_bytes())]).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
            other => break other.unwrap(),
        }
    };
    assert!(matches!(updates.last(), Some(Update::Component(_))), "{updates:?}");
    s.stop().await;
}
