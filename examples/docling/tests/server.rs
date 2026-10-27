//! The docling example against a real `turbo-kserve` on the CPU: the
//! server is started in process on a free port with the tiny test bundle
//! and every call crosses gRPC. The client sends only the docling
//! structure (`data/report.seed.graph`); the server analyzes, chunks,
//! embeds and answers the heat. The summary every client prints is held
//! to `data/report.server.expected.txt`, the Rust client's report to
//! `data/report.server.expected.json` (vector values and scores within
//! `report::TOLERANCE`). `BLESS=1` rewrites all three files.

use std::path::{Path, PathBuf};

use serde_json::Value;
use turbo_analysis::Layer;
use turbo_analysis_layers::{Chunks, Entities};
use turbo_example_docling::report::close;
use turbo_example_docling::server::{self, MODEL, Remote};
use turbo_example_docling::{docling::Docling, remote};
use turbo_kserve::api::Runtime;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES, Device, ModelConfig};
use turbo_kserve::proto::result::OffsetMode;
use turbo_kserve::{Serve, Server};

fn here() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn input() -> String {
    std::fs::read_to_string(here().join("data/report.docling.json")).unwrap()
}

/// The runtime index of the CPU; `Device::Select` never picks it.
fn cpu() -> u32 {
    let rt = Runtime::create().unwrap();
    (0..16).find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false)).expect("a CPU")
}

async fn start() -> Server {
    let bundle = here().join("../../testdata/tiny-bert-bundle");
    let model = ModelConfig {
        bundle: bundle.to_str().unwrap().to_owned(),
        name: Some(MODEL.to_owned()),
        device: Device::Index(cpu()),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), vec![model], DEFAULT_MAX_MESSAGE_BYTES, Serve::default())
        .await
        .unwrap();
    s.load().await.unwrap();
    s
}

async fn ask(s: &Server, offsets: OffsetMode) -> (Docling, Vec<u8>, Remote) {
    let (d, seed) = server::seed(&input()).unwrap();
    let mut client = server::connect(&format!("http://{}", s.local_addr())).await.unwrap();
    let answer = server::run(&mut client, seed.clone(), offsets).await.unwrap();
    (d, seed, answer)
}

/// `got` against the file at `path`, which `BLESS=1` rewrites first.
fn expect(path: &Path, got: &[u8]) -> Vec<u8> {
    if std::env::var_os("BLESS").is_some() {
        std::fs::write(path, got).unwrap();
    }
    std::fs::read(path).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn the_server_runs_the_docling_example() {
    let s = start().await;
    let (d, seed, answer) = ask(&s, OffsetMode::Utf8).await;
    let data = here().join("data");
    // The seed the Java and Python clients send is this one.
    assert!(expect(&data.join("report.seed.graph"), &seed) == seed, "the seed changed; rerun with BLESS=1");
    assert_eq!(answer.completed.len(), server::STEPS.len());

    let summary = remote::summary(&answer).unwrap();
    let want = expect(&data.join("report.server.expected.txt"), summary.as_bytes());
    assert_eq!(summary, String::from_utf8(want).unwrap(), "rerun with BLESS=1 to accept, then review the diff");

    let doc = server::document(&answer).unwrap();
    // Every entity is inside one docling item and is that item's text;
    // every chunk covers at least one item.
    let entities = doc.get::<Entities>().unwrap();
    assert!(entities.len() >= 3, "{}", entities.len());
    for (_, e) in entities.spans.iter() {
        let located = d.locate(e);
        let [l] = located.as_slice() else { panic!("{e:?} spans items") };
        let own = &d.text[l.item.span.range()][l.bytes.0 as usize..l.bytes.1 as usize];
        assert_eq!(own, &d.text[e.range()]);
    }
    let chunks = doc.get::<Chunks>().unwrap();
    assert!(chunks.len() > 2);
    assert!(chunks.spans.iter().all(|(_, c)| !d.locate(c).is_empty()));
    // The entity lane counts every entity once.
    let lane = answer.window.heatmap_lanes.iter().find(|l| l.id == "entities").expect("an entity lane");
    assert_eq!(lane.buckets.len(), server::HEAT_BUCKETS as usize);
    assert_eq!(lane.buckets.iter().map(|b| b.count).sum::<u32>(), entities.len() as u32);

    let got = remote::report(&d, &doc, &answer).unwrap();
    let text = serde_json::to_string_pretty(&got).unwrap() + "\n";
    let want: Value =
        serde_json::from_slice(&expect(&data.join("report.server.expected.json"), text.as_bytes())).unwrap();
    if let Err(e) = close("report", &got, &want) {
        panic!("{e}\nrerun with BLESS=1 to accept, then review the diff of data/report.server.expected.json");
    }
}

/// Every span in the answers is in the unit the request names: `Zürich`
/// is 7 bytes, and 6 UTF-16 units and code points.
#[tokio::test(flavor = "multi_thread")]
async fn offsets_follow_the_request() {
    let s = start().await;
    let width = |r: &Remote| {
        let seg = r.window.segments.iter().find(|g| g.annotations.iter().any(|a| a.layer == remote::ENTITIES)).unwrap();
        (seg.end - seg.start, seg.text.clone())
    };
    let levels =
        |r: &Remote| r.chunks.projection.as_ref().unwrap().points.iter().map(|p| (p.item, p.level)).collect::<Vec<_>>();
    let (_, _, utf8) = ask(&s, OffsetMode::Utf8).await;
    assert_eq!(width(&utf8), (7, "Zürich".to_owned()));
    for mode in [OffsetMode::Utf16, OffsetMode::CodePoint] {
        let (_, _, other) = ask(&s, mode).await;
        assert_eq!(width(&other), (6, "Zürich".to_owned()), "{mode:?}");
        assert_eq!(levels(&other), levels(&utf8), "{mode:?}");
    }
}
