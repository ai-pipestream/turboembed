//! `AnalyzePipelineRequest.structure_seed`: a heading, a list and a table
//! sent as protobuf blocks give the same graph as the same structure sent
//! as `initial_graph` DocumentGraph bytes, in every offset unit, and a
//! malformed block is refused naming it.

use tokio_stream::StreamExt as _;
use tonic::Code;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Block, Structure};
use turbo_kserve::offsets::OffsetMode as Wire;
use turbo_kserve::progressive::seed::{SEED_ANNOTATOR, seed_document};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::{
    AnalyzePipelineRequest, DocumentGraphResponse, OffsetMode, StructureBlock, StructureBlockKind, StructureSeed,
};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};

const TEXT: &str = "# Café menu\n\n- crème brûlée\n- \u{1f600} soup\n\n| a | b |\n| ü | é |\n";

const STEPS: &[&str] = &["turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped", "turbo:structure-boundaries"];

/// One block: its kind, the text it covers (found from `from` on), its
/// level and its parent.
struct Want {
    kind: Block,
    wire: StructureBlockKind,
    text: &'static str,
    level: u8,
    parent: Option<u32>,
}

fn wants() -> Vec<Want> {
    use Block as B;
    use StructureBlockKind as K;
    let w = |kind, wire, text, level, parent| Want { kind, wire, text, level, parent };
    vec![
        w(B::Heading, K::Heading, "# Café menu", 1, None),
        w(B::List, K::List, "- crème brûlée\n- \u{1f600} soup", 1, None),
        w(B::ListItem, K::ListItem, "- crème brûlée", 1, Some(1)),
        w(B::ListItem, K::ListItem, "- \u{1f600} soup", 1, Some(1)),
        w(B::Table, K::Table, "| a | b |\n| ü | é |", 0, None),
        w(B::TableRow, K::TableRow, "| a | b |", 0, Some(4)),
        w(B::TableCell, K::TableCell, "a", 0, Some(5)),
        w(B::TableRow, K::TableRow, "| ü | é |", 0, Some(4)),
        w(B::TableCell, K::TableCell, "ü", 0, Some(7)),
    ]
}

/// Each block's byte span: found in text order, a child searched for
/// from its parent's start.
fn byte_spans() -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut from = 0usize;
    for w in wants() {
        let at = w.parent.map_or(from, |p| from.max(out[p as usize].start as usize));
        let start = at + TEXT[at..].find(w.text).unwrap();
        out.push(Span::new(start as u32, (start + w.text.len()) as u32));
        from = start;
    }
    out
}

fn to_wire(byte: u32, mode: Wire) -> u32 {
    let head = &TEXT[..byte as usize];
    match mode {
        Wire::Utf8 => byte,
        Wire::Utf16 => head.encode_utf16().count() as u32,
        Wire::CodePoint => head.chars().count() as u32,
    }
}

fn seed(mode: Wire) -> StructureSeed {
    let blocks = wants()
        .iter()
        .zip(byte_spans())
        .map(|(w, s)| StructureBlock {
            kind: w.wire as i32,
            start: to_wire(s.start, mode),
            end: to_wire(s.end, mode),
            level: u32::from(w.level),
            parent: w.parent,
        })
        .collect();
    StructureSeed { text: TEXT.to_owned(), blocks }
}

/// The document a DocumentGraph writer would send: the text and one
/// `turbo:structure` layer, the default instance.
fn written() -> Document {
    let mut spans = Spans::<Structure>::new(ViewId::SOURCE);
    for s in byte_spans() {
        spans.push(s);
    }
    let w = wants();
    let structure = Structure {
        spans,
        kind: w.iter().map(|w| w.kind).collect(),
        parent: w.iter().map(|w| w.parent.map(Ix::new)).collect(),
        depth: w.iter().map(|w| w.level).collect(),
    };
    let mut doc = Document::new(TEXT);
    doc.add(&Key::<Structure>::DEFAULT, structure, Provenance::by(SEED_ANNOTATOR)).unwrap();
    doc
}

fn graph(d: &Document) -> Vec<u8> {
    export(d, codecs()).unwrap().to_bytes().unwrap()
}

const MODES: [(Wire, OffsetMode); 3] =
    [(Wire::Utf8, OffsetMode::Utf8), (Wire::Utf16, OffsetMode::Utf16), (Wire::CodePoint, OffsetMode::CodePoint)];

#[test]
fn a_seed_builds_the_written_document_in_every_offset_unit() {
    let want = graph(&written());
    for (mode, _) in MODES {
        let doc = seed_document(&seed(mode), mode).unwrap();
        assert_eq!(graph(&doc), want, "{mode:?}");
    }
}

async fn run(
    client: &mut AnalysisServiceClient<tonic::transport::Channel>,
    req: AnalyzePipelineRequest,
) -> DocumentGraphResponse {
    let mut stream = client.analyze_document_progressive(req).await.unwrap().into_inner();
    let mut complete = None;
    while let Some(event) = stream.next().await {
        match event.unwrap().event.unwrap() {
            Event::StepFailed(f) => panic!("step {} failed: {:?}", f.step_id, f.failure),
            Event::Complete(c) => complete = Some(c),
            Event::Started(_) | Event::LayersReady(_) => {}
        }
    }
    complete.expect("the stream ends with the document")
}

async fn start() -> (Server, AnalysisServiceClient<tonic::transport::Channel>) {
    let server =
        Server::start("127.0.0.1:0".parse().unwrap(), vec![], 8 << 20, Serve { admin: true, ..Serve::default() })
            .await
            .unwrap();
    let client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr())).await.unwrap();
    (server, client)
}

#[tokio::test]
async fn a_seed_and_the_equivalent_initial_graph_give_identical_graphs() {
    let (server, mut client) = start().await;
    let steps: Vec<String> = STEPS.iter().map(|s| (*s).to_owned()).collect();
    let from_graph = run(
        &mut client,
        AnalyzePipelineRequest { initial_graph: graph(&written()), steps: steps.clone(), ..Default::default() },
    )
    .await;
    let structure = from_graph.layers.iter().find(|l| l.layer == "turbo:structure").unwrap();
    assert_eq!(structure.items, wants().len() as u32);
    for (mode, proto) in MODES {
        let req = AnalyzePipelineRequest {
            offsets: proto as i32,
            steps: steps.clone(),
            structure_seed: Some(seed(mode)),
            ..Default::default()
        };
        let from_seed = run(&mut client, req).await;
        assert_eq!(from_seed.graph, from_graph.graph, "{mode:?}");
        assert_eq!(from_seed.layers, from_graph.layers, "{mode:?}");
        assert_eq!(from_seed.source_text, TEXT);
    }
    server.stop().await;
}

/// The seed with `edit` applied to it, read in UTF-16 units, refused
/// with a message holding `want`.
fn refused(edit: impl FnOnce(&mut StructureSeed), want: &str) {
    let mut s = seed(Wire::Utf16);
    edit(&mut s);
    let Err(e) = seed_document(&s, Wire::Utf16) else { panic!("a seed refused for {want:?} was accepted") };
    assert!(e.message.contains(want), "{want:?} not in {:?}", e.message);
}

#[test]
fn malformed_blocks_are_refused_naming_the_block() {
    // Surrogate pair of the emoji: list item 3 starts "- " then the pair.
    let item = seed(Wire::Utf16).blocks[3].start;
    refused(|s| s.blocks[2].kind = 0, "block 2: kind is STRUCTURE_BLOCK_KIND_UNSPECIFIED");
    refused(|s| s.blocks[2].kind = 99, "block 2: kind 99 is not a known StructureBlockKind");
    refused(|s| s.blocks[0].level = 256, "block 0: level 256 is over 255");
    refused(|s| s.blocks[2].parent = Some(2), "block 2: parent 2 is not a block before it");
    refused(|s| s.blocks[8].end = 10_000, "structure_seed block 8");
    refused(|s| s.blocks[3].start = item + 3, "structure_seed block 3");
    refused(|s| s.blocks[3].start = s.blocks[2].start + 1, "block 3 overlaps the block before it");
    refused(|s| s.blocks[6].parent = Some(4), "block 6 overlaps the block before it");
    refused(|s| s.blocks[2].end = s.blocks[1].end + 1, "block 2 is outside its parent 1");
    refused(|s| s.blocks.swap(0, 1), "block 1 starts before the block before it");
}

#[tokio::test]
async fn the_rpc_refuses_a_bad_seed_and_a_seed_beside_text() {
    let (server, mut client) = start().await;
    let mut bad = seed(Wire::Utf8);
    bad.blocks[3].start = bad.blocks[2].start + 1;
    let beside =
        AnalyzePipelineRequest { text: TEXT.into(), structure_seed: Some(seed(Wire::Utf8)), ..Default::default() };
    let overlapping = AnalyzePipelineRequest { structure_seed: Some(bad), ..Default::default() };
    for (req, want) in [(beside, "leave text, source_bytes and initial_graph empty"), (overlapping, "overlaps")] {
        let Err(e) = client.analyze_document_progressive(req).await else { panic!("{want:?} streamed, not refused") };
        assert_eq!(e.code(), Code::InvalidArgument);
        assert!(e.message().contains(want), "{want:?} not in {:?}", e.message());
    }
    server.stop().await;
}

#[test]
fn the_utf16_offsets_are_the_ones_docs_grpc_analysis_lists() {
    let got: Vec<(u32, u32)> = seed(Wire::Utf16).blocks.iter().map(|b| (b.start, b.end)).collect();
    let docs = [(0, 11), (13, 37), (13, 27), (28, 37), (39, 58), (39, 48), (41, 42), (49, 58), (51, 52)];
    assert_eq!(got, docs);
    let item = |mode| seed(mode).blocks[3].end;
    assert_eq!((item(Wire::CodePoint), item(Wire::Utf8)), (36, 43));
}
