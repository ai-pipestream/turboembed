use tonic::Code;
use turbo_analysis::{Document, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_kserve::config::{Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::{DocumentWindowRequest, OffsetMode, ProjectChunksRequest};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

pub async fn start() -> (Server, AnalysisServiceClient<tonic::transport::Channel>) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let config = ModelConfig {
        bundle: root.join("testdata/tiny-bert-bundle").to_string_lossy().into_owned(),
        name: Some("tiny".into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![config], 8 * 1024 * 1024, Serve::default()).await.unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    (server, client)
}

pub fn model() -> ModelRef {
    ModelRef { id: "analytic-two-row-control".into(), revision: "1".into(), sha256: String::new() }
}

pub fn graph(doc: &Document) -> Vec<u8> {
    let g = turbo_analysis::export::export(doc, codecs()).unwrap();
    let bytes = g.to_bytes().unwrap();
    let g = turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap();
    turbo_analysis::export::import(&g, codecs()).unwrap();
    bytes
}

pub fn base() -> Document {
    let mut d = Document::new("é 😀");
    let mut s = Spans::new(ViewId::SOURCE);
    s.push(Span::new(0, 2));
    s.push(Span::new(3, 7));
    d.add(&Key::DEFAULT, Sentences { spans: s }, Provenance::by("review-fixture")).unwrap();
    let mut t = Spans::new(ViewId::SOURCE);
    t.push(Span::new(0, 2));
    t.push(Span::new(3, 7));
    d.add(&Key::DEFAULT, Tokens { spans: t, sentence: None }, Provenance::by("review-fixture")).unwrap();
    let mut e = Spans::new(ViewId::SOURCE);
    e.push(Span::new(3, 7));
    d.add(
        &Key::DEFAULT,
        Entities {
            spans: e,
            tokens: vec![IxRange::new(1, 2)],
            kind: vec![Tag(0)],
            kinds: TagSet::new("entity", &["EMOJI"]),
            score: vec![0.9],
        },
        Provenance::by("review-fixture"),
    )
    .unwrap();
    d
}

pub fn window(bytes: Vec<u8>, mode: OffsetMode) -> DocumentWindowRequest {
    DocumentWindowRequest {
        graph: bytes,
        view: 0,
        offsets: mode as i32,
        page_size: 16_000,
        page: 0,
        filter_layers: Vec::new(),
        heatmap_buckets: 0,
        xray_target_view: None,
    }
}

pub fn project(bytes: Vec<u8>) -> ProjectChunksRequest {
    ProjectChunksRequest {
        graph: bytes,
        offsets: OffsetMode::Utf8 as i32,
        pca_dims: 2,
        max_stack_segments: 5,
        include_drift_curve: false,
        include_strategy_matrix: false,
        strategy_budgets: Vec::new(),
    }
}

pub fn error(status: tonic::Status, grpc: Code, code: i32) {
    assert_eq!(status.code(), grpc, "{status}");
    assert_eq!(status.metadata().get("turbo-code").unwrap().to_str().unwrap(), code.to_string());
}
