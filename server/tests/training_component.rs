//! TrainComponent end to end: a POS tagger trained from samples streamed
//! in chunks cut through a character, its model sent back whole and
//! tagging as trained; the admin opt-in and the start's checks, a
//! trainer the component would not run as asked among them; progress in
//! its documented order; a perceptron categorizer that categorizes; one
//! training at a time; and a training whose client goes away, during the
//! upload or the training, stopping and freeing its slot; and samples
//! past the 256 MiB a training holds refused.

mod common;

use std::time::Duration;

use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::Code;
use tonic::transport::Channel;
use turbo_kserve::api::{TURBO_E_BUSY, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT, TURBO_E_UNSUPPORTED};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::training::train_component_request::Part;
use turbo_kserve::proto::training::train_component_update::Update;
use turbo_kserve::proto::training::turbo_training_service_client::TurboTrainingServiceClient;
use turbo_kserve::proto::training::*;
use turbo_kserve::{Serve, Server};
use turbo_postag::{PosModel, PosTagFormat, PosTagger};

type Client = TurboTrainingServiceClient<Channel>;

/// Word and tag samples; "Zoë" makes the chunk cut fall inside a character.
const POS: &str = "The_DT cats_NNS barked_VBD ._.\nThe_DT dogs_NNS ran_VBD ._.\nZoë_NNP ran_VBD ._.\n";

async fn server(admin: bool) -> (Server, Client) {
    // A short upload idle limit, so the abandoned upload frees the slot soon.
    let serve = Serve { admin, upload_idle: Some(Duration::from_secs(2)), ..Serve::default() };
    // A server that cannot add models needs one to start with.
    let tiny = ModelConfig {
        bundle: common::tiny().to_str().unwrap().into(),
        name: None,
        device: Device::Index(common::cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let models = if admin { vec![] } else { vec![tiny] };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, serve).await.unwrap();
    s.load().await.unwrap();
    let c = TurboTrainingServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    (s, c)
}

fn start(component: ComponentKind, language: &str, parameters: &[(&str, &str)]) -> TrainComponentRequest {
    let parameters = parameters.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
    let s = TrainComponentStart {
        component: component as i32,
        language: language.into(),
        parameters,
        name_type: "".into(),
    };
    TrainComponentRequest { part: Some(Part::Start(s)) }
}

fn samples(b: &[u8]) -> TrainComponentRequest {
    TrainComponentRequest { part: Some(Part::Samples(b.to_vec())) }
}

/// Category samples: two categories, three documents each.
const DOCS: &str = "good fine nice day\ngood nice warm sun\ngood fine sun day\n\
    bad cold rain day\nbad wet cold wind\nbad rain wind night\n";

/// Word, tag and lemma samples.
const LEMMAS: &str = "The\tDT\tthe\ncats\tNNS\tcat\nran\tVBD\trun\n\nDogs\tNNS\tdog\nbarked\tVBD\tbark\n";

/// Name samples.
const NAMES: &str = "<START:person> Ann <END> ran .\nThe dog saw <START:person> Bob <END> .\nIt rained .\n";

fn pos_start() -> TrainComponentRequest {
    start(ComponentKind::PosTagger, "en", &[("Iterations", "20"), ("Cutoff", "0")])
}

/// Every update of a call that sends `requests`, its error last when it
/// ends in one.
async fn train(c: &mut Client, requests: Vec<TrainComponentRequest>) -> Result<Vec<Update>, tonic::Status> {
    let mut stream = c.train_component(tokio_stream::iter(requests)).await?.into_inner();
    let mut out = Vec::new();
    while let Some(u) = stream.message().await? {
        out.push(u.update.unwrap());
    }
    Ok(out)
}

fn refusal(s: &tonic::Status) -> (Code, i32) {
    let tc = s.metadata().get("turbo-code").unwrap_or_else(|| panic!("no turbo-code in {s:?}"));
    (s.code(), tc.to_str().unwrap().parse().unwrap())
}

fn stage(u: &Update) -> Option<ComponentStage> {
    match u {
        Update::Progress(p) => Some(p.stage()),
        _ => None,
    }
}

#[tokio::test]
async fn a_pos_tagger_is_trained_from_streamed_samples_and_tags() {
    let (s, mut c) = server(true).await;
    let bytes = POS.as_bytes();
    let cut = POS.find('ë').unwrap() + 1;
    let updates = train(&mut c, vec![pos_start(), samples(&bytes[..cut]), samples(&bytes[cut..])]).await.unwrap();

    let stages: Vec<_> = updates.iter().filter_map(stage).collect();
    assert_eq!(stages.first(), Some(&ComponentStage::Admitted));
    assert_eq!(stages.last(), Some(&ComponentStage::Packaged));
    let read = updates.iter().find_map(|u| match u {
        Update::Progress(p) if p.stage() == ComponentStage::Read => Some(p.samples),
        _ => None,
    });
    assert_eq!(read, Some(3));
    let iterations: Vec<_> = updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) if p.stage() == ComponentStage::Iteration => Some(p),
            _ => None,
        })
        .collect();
    assert!(!iterations.is_empty() && iterations.len() <= 20, "{} iterations", iterations.len());
    assert_eq!(iterations[0].iteration, 1);
    assert!(iterations.iter().all(|p| p.events == 11 && p.measure == "Log Likelihood"), "{iterations:?}");

    let model: Vec<u8> = updates
        .iter()
        .filter_map(|u| if let Update::Model(b) = u { Some(b.as_slice()) } else { None })
        .flatten()
        .copied()
        .collect();
    let Some(Update::Trained(t)) = updates.last() else { panic!("no trained last: {updates:?}") };
    assert_eq!((t.component(), t.samples), (ComponentKind::PosTagger, 3));
    assert_eq!((t.size, t.sha256.as_str()), (model.len() as u64, turbo_maxent::sha256(&model).to_hex().as_str()));

    let tagger = PosTagger::new(PosModel::from_trained_archive_with_format(model, PosTagFormat::Penn).unwrap());
    let (mut tags, mut probs) = (Vec::new(), Vec::new());
    assert!(tagger.tag(&["The", "cats", "barked", "."], &mut tags, &mut probs).unwrap());
    let set = tagger.pos_model().tagset();
    let labels: Vec<&str> = tags.iter().map(|&t| set.label(t)).collect();
    assert_eq!(labels, ["DT", "NNS", "VBD", "."]);
    s.stop().await;
}

#[tokio::test]
async fn training_needs_admin_and_a_good_start() {
    let (s, mut c) = server(false).await;
    let e = train(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED), "{e:?}");
    assert!(e.message().contains("--admin"), "{e:?}");
    s.stop().await;

    let (s, mut c) = server(true).await;
    let invalid = |e: &tonic::Status| refusal(e) == (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let pos = POS.as_bytes();
    let mut unknown = start(ComponentKind::PosTagger, "en", &[]);
    if let Some(Part::Start(s)) = &mut unknown.part {
        s.component = 99;
    }
    let cases: Vec<(Vec<TrainComponentRequest>, &str)> = vec![
        (vec![samples(pos)], "the first message must be the start"),
        (vec![], "the stream ended before its start"),
        (vec![start(ComponentKind::Unspecified, "en", &[]), samples(pos)], "component"),
        (vec![start(ComponentKind::PosTagger, "", &[]), samples(pos)], "language"),
        (vec![start(ComponentKind::PosTagger, "en", &[("Algorithm", "NOPE")]), samples(pos)], "parameters"),
        // The POS tagger trains GIS only: a perceptron is refused, not swapped for GIS.
        (vec![start(ComponentKind::PosTagger, "en", &[("Algorithm", "PERCEPTRON")]), samples(pos)], "MAXENT only"),
        (vec![start(ComponentKind::Lemmatizer, "en", &[("Algorithm", "MAXENT_QN")]), samples(pos)], "MAXENT only"),
        // A perceptron of no iterations categorizes as NaN: refused before training.
        (
            vec![
                start(ComponentKind::DocumentCategorizer, "en", &[("Algorithm", "PERCEPTRON"), ("Iterations", "0")]),
                samples(DOCS.as_bytes()),
            ],
            "Iterations must be at least 1",
        ),
        (vec![pos_start()], "samples: none were read"),
        (vec![pos_start(), samples(b"The_DT \xff_NN\n")], "samples: not UTF-8 at byte 7"),
        (vec![pos_start(), samples(pos), pos_start()], "a second start"),
        (vec![unknown, samples(pos)], "component"),
    ];
    for (requests, says) in cases {
        let e = train(&mut c, requests).await.unwrap_err();
        assert!(invalid(&e) && e.message().contains(says), "{says}: {e:?}");
    }
    s.stop().await;
}

/// The stages a training streams, in order, its iterations as one.
fn stages(updates: &[Update]) -> Vec<ComponentStage> {
    let mut out: Vec<ComponentStage> = updates.iter().filter_map(stage).collect();
    out.dedup();
    out
}

#[tokio::test]
async fn progress_comes_in_its_documented_order() {
    use ComponentStage::*;
    let (s, mut c) = server(true).await;
    let cases =
        [(ComponentKind::Lemmatizer, LEMMAS), (ComponentKind::NameFinder, NAMES), (ComponentKind::PosTagger, POS)];
    for (kind, text) in cases {
        let go = start(kind, "en", &[("Iterations", "5"), ("Cutoff", "0")]);
        let updates = train(&mut c, vec![go, samples(text.as_bytes())]).await.unwrap();
        assert_eq!(stages(&updates), [Admitted, Read, Iteration, Packaged], "{kind:?}");
    }
    s.stop().await;
}

#[tokio::test]
async fn a_perceptron_categorizer_is_a_perceptron_and_categorizes() {
    let (s, mut c) = server(true).await;
    let params = [("Algorithm", "PERCEPTRON"), ("Iterations", "5"), ("Cutoff", "0")];
    let go = start(ComponentKind::DocumentCategorizer, "en", &params);
    let updates = train(&mut c, vec![go, samples(DOCS.as_bytes())]).await.unwrap();
    let measures: Vec<&str> = updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) if p.stage() == ComponentStage::Iteration => Some(p.measure.as_str()),
            _ => None,
        })
        .collect();
    assert!(!measures.is_empty() && measures.iter().all(|m| *m == "Training Accuracy"), "{measures:?}");
    let model: Vec<u8> = updates
        .iter()
        .filter_map(|u| if let Update::Model(b) = u { Some(b.clone()) } else { None })
        .flatten()
        .collect();
    let model = turbo_doccat::DoccatModel::from_trained_archive(model).unwrap();
    let scores = turbo_doccat::DocumentCategorizerME::new(model).categorize(&["nice", "warm", "sun"]);
    assert!(scores.len() == 2 && scores.iter().all(|p| p.is_finite()), "{scores:?}");
    s.stop().await;
}

/// A call whose requests the test sends as it goes.
async fn open(
    c: &mut Client,
    first: Vec<TrainComponentRequest>,
) -> (mpsc::Sender<TrainComponentRequest>, tonic::Streaming<TrainComponentUpdate>) {
    let (tx, rx) = mpsc::channel(4);
    for r in first {
        tx.send(r).await.unwrap();
    }
    let stream = c.train_component(ReceiverStream::new(rx)).await.unwrap().into_inner();
    (tx, stream)
}

#[tokio::test]
async fn one_training_runs_at_a_time() {
    let (s, mut c) = server(true).await;
    // Admitted, and still reading its samples: it holds the slot.
    let (tx, mut first) = open(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await;
    assert_eq!(stage(&first.message().await.unwrap().unwrap().update.unwrap()), Some(ComponentStage::Admitted));
    let e = train(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    drop(tx);
    let mut last = None;
    while let Some(u) = first.message().await.unwrap() {
        last = u.update;
    }
    assert!(matches!(last, Some(Update::Trained(_))), "{last:?}");
    let after = train(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await.unwrap();
    assert!(matches!(after.last(), Some(Update::Trained(_))));
    s.stop().await;
}

#[tokio::test]
async fn a_training_whose_client_goes_away_stops_and_frees_its_slot() {
    let (s, mut c) = server(true).await;
    // GIS with a log likelihood threshold it never meets and no end: only
    // the client going away ends it.
    let endless =
        start(ComponentKind::PosTagger, "en", &[("Iterations", "2147483647"), ("LLThreshold", "-1"), ("Cutoff", "0")]);
    let (tx, mut stream) = open(&mut c, vec![endless, samples(POS.as_bytes())]).await;
    drop(tx);
    loop {
        let u = stream.message().await.unwrap().unwrap().update.unwrap();
        if stage(&u) == Some(ComponentStage::Iteration) {
            break;
        }
    }
    drop(stream);
    let updates = loop {
        match train(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await {
            Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
            other => break other.unwrap(),
        }
    };
    assert!(matches!(updates.last(), Some(Update::Trained(_))), "{updates:?}");
    s.stop().await;
}

#[tokio::test]
async fn a_client_that_drops_the_answer_mid_upload_frees_the_slot() {
    let (s, mut c) = server(true).await;
    // Admitted and reading; the upload stays open while the answer goes.
    // The server cannot see that, so the idle limit (two seconds here)
    // ends the read and frees the slot.
    let (tx, mut stream) = open(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await;
    assert_eq!(stage(&stream.message().await.unwrap().unwrap().update.unwrap()), Some(ComponentStage::Admitted));
    drop(stream);
    let updates = tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            match train(&mut c, vec![pos_start(), samples(POS.as_bytes())]).await {
                Err(e) if e.code() == Code::Unavailable => tokio::time::sleep(Duration::from_millis(50)).await,
                other => break other.unwrap(),
            }
        }
    })
    .await
    .expect("the slot is freed while the upload is still open");
    assert!(matches!(updates.last(), Some(Update::Trained(_))), "{updates:?}");
    drop(tx);
    s.stop().await;
}

#[tokio::test]
async fn samples_past_256_mib_are_refused_and_free_the_slot() {
    let (s, mut c) = server(true).await;
    // 256 chunks of 1 MiB fill the limit; the 257th byte over is refused.
    let chunk = vec![b'a'; 1 << 20];
    let requests = std::iter::once(start(ComponentKind::DocumentCategorizer, "en", &[]))
        .chain((0..256).map(move |_| samples(&chunk)))
        .chain(std::iter::once(samples(b"a")));
    let e = async {
        let mut stream = c.train_component(tokio_stream::iter(requests)).await?.into_inner();
        while stream.message().await?.is_some() {}
        Ok::<(), tonic::Status>(())
    }
    .await
    .unwrap_err();
    assert_eq!(refusal(&e), (Code::OutOfRange, TURBO_E_CAPACITY), "{e:?}");
    assert!(e.message().contains("more than 268435456 bytes"), "{e:?}");
    // The slot is free again.
    let docs = start(ComponentKind::DocumentCategorizer, "en", &[("Cutoff", "0")]);
    let updates = train(&mut c, vec![docs, samples(DOCS.as_bytes())]).await.unwrap();
    assert!(matches!(updates.last(), Some(Update::Trained(_))), "{updates:?}");
    s.stop().await;
}
