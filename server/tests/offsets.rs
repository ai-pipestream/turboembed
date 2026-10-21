//! OffsetMode conversion and the DocumentGraph round trip over the
//! wire: ASCII, multi-byte UTF-8, and astral-plane text across
//! Utf8, Utf16, and CodePoint modes, with Java UTF-16 indices as
//! the independent oracle, plus malformed-offset refusals.

use std::marker::PhantomData;

use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{Block, Chunks, Cut, Levels, Sentences, Structure, Unit, Vectors};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{ChunkLevelsResult, DocumentGraphRequest, LayerWireStatus};
use turbo_kserve::results::Wire;
use turbo_kserve::results::document::round_trip_document;
use turbo_kserve::results::layers::ChunkLevels;

mod common;

// Accented Latin, an astral emoji, CJK, Gothic (astral), a musical
// symbol (astral): every offset width class in one line.
const TEXT: &str = "Café \u{1f600} 北京. Gothic \u{10330} music \u{1d11e} end.";

fn model() -> ModelRef {
    ModelRef { id: "toy/embed".into(), revision: "1".into(), sha256: "0".into() }
}

fn byte_span(needle: &str) -> Span {
    let at = TEXT.find(needle).unwrap();
    Span::new(at as u32, (at + needle.len()) as u32)
}

/// Two chunks (the second discontinuous), their level tree, and
/// 2-wide embeddings, over TEXT.
fn fixture(offsets: OffsetMode) -> ChunkLevels {
    let cafe = byte_span("Café");
    let beijing = byte_span("北京");
    let gothic = byte_span("\u{10330}");
    let whole = Span::new(0, TEXT.len() as u32);
    let mut chunk_spans = Spans::<Chunks>::new(ViewId::SOURCE);
    chunk_spans.push_fragments(&[cafe]);
    chunk_spans.push_fragments(&[beijing, gothic]);
    let chunks = Chunks {
        spans: chunk_spans,
        level: vec![0, 0],
        parent: vec![None, None],
        sentences: vec![IxRange::<Sentences>::new(0, 1), IxRange::<Sentences>::new(1, 2)],
        cut: vec![Cut::DocumentStart, Cut::Sentence],
        cost: vec![0, 7],
        model_tokens: vec![3, 5],
        over_budget: vec![false, false],
    };
    let mut level_spans = Spans::<Levels>::new(ViewId::SOURCE);
    level_spans.push_fragments(&[whole]);
    level_spans.push_fragments(&[cafe]);
    level_spans.push_fragments(&[beijing, gothic]);
    let levels = Levels {
        spans: level_spans,
        unit: vec![Unit::Document, Unit::Chunk(Ix::new(0)), Unit::Chunk(Ix::new(1))],
        parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
    };
    let vectors = Vectors::<Chunks> {
        dim: 2,
        data: vec![1.0, 0.0, 0.0, 1.0],
        empty: Vec::new(),
        model: model(),
        _of: PhantomData,
    };
    ChunkLevels { text: TEXT.into(), view: ViewId::SOURCE, offsets, chunks, levels, vectors }
}

/// UTF-16 code-unit index of a byte offset, counted the Java way:
/// independently of the checkpoint table under test.
fn java_utf16(byte: u32) -> u32 {
    TEXT[..byte as usize].encode_utf16().count() as u32
}

#[test]
fn chunk_levels_round_trip_in_every_offset_mode() {
    for mode in [OffsetMode::Utf8, OffsetMode::Utf16, OffsetMode::CodePoint] {
        let want = fixture(mode);
        let msg = want.to_msg();
        let back = ChunkLevels::from_msg(&msg).unwrap();
        assert_eq!(back.to_msg(), msg, "{mode:?} message round trip");
        // The decoded layers hold the internal byte spans again.
        let frags: Vec<Span> = back.chunks.spans.fragments(Ix::new(1)).collect();
        assert_eq!(frags, [byte_span("北京"), byte_span("\u{10330}")], "{mode:?}");
        assert_eq!(back.levels.unit.len(), 3);
        assert_eq!(back.vectors.data, [1.0, 0.0, 0.0, 1.0]);
    }
    // The UTF-16 wire offsets are Java's indices, byte for byte.
    let msg = fixture(OffsetMode::Utf16).to_msg();
    let beijing = byte_span("北京");
    assert_eq!(msg.chunk_frag_start[1], java_utf16(beijing.start));
    assert_eq!(msg.chunk_frag_end[1], java_utf16(beijing.end));
    // And code points count scalar values.
    let msg = fixture(OffsetMode::CodePoint).to_msg();
    assert_eq!(msg.chunk_frag_start[1], TEXT[..beijing.start as usize].chars().count() as u32);
}

#[test]
fn malformed_wire_offsets_are_refused_by_name() {
    let mut msg = fixture(OffsetMode::Utf16).to_msg();
    // The emoji's low surrogate: one unit into the pair.
    msg.chunk_frag_start[0] = java_utf16(byte_span("\u{1f600}").start) + 1;
    let err = ChunkLevels::from_msg(&msg).err().expect("mid-surrogate refused");
    assert_eq!(err.code, turbo_kserve::api::TURBO_E_INVALID_ARGUMENT, "{}", err.message);

    let mut msg = fixture(OffsetMode::Utf8).to_msg();
    msg.chunk_frag_end[0] = TEXT.len() as u32 + 1;
    assert!(ChunkLevels::from_msg(&msg).is_err(), "out of bounds refused");
    msg.chunk_frag_start[0] = 5;
    msg.chunk_frag_end[0] = 3;
    assert!(ChunkLevels::from_msg(&msg).is_err(), "inverted refused");

    let mut msg = fixture(OffsetMode::Utf8).to_msg();
    msg.offsets = 99;
    assert!(ChunkLevels::from_msg(&msg).is_err(), "unknown mode refused");
}

fn graph_bytes() -> (Vec<u8>, DocumentGraph) {
    let fixture = fixture(OffsetMode::Utf8);
    let mut doc = Document::new(TEXT);
    let mut sentence_spans = Spans::<Sentences>::new(ViewId::SOURCE);
    sentence_spans.push(Span::new(0, byte_span("北京").end + 1));
    sentence_spans.push(Span::new(byte_span("北京").end + 2, TEXT.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sentence_spans }, Provenance::by("test:sentences")).unwrap();
    let mut structure_spans = Spans::<Structure>::new(ViewId::SOURCE);
    structure_spans.push(Span::new(0, TEXT.len() as u32));
    let structure =
        Structure { spans: structure_spans, kind: vec![Block::Section], parent: vec![None], depth: vec![1] };
    doc.add(&Key::DEFAULT, structure, Provenance::by("test:structure")).unwrap();
    doc.add(&Key::DEFAULT, fixture.chunks, Provenance::by("test:chunker")).unwrap();
    doc.add(&Key::DEFAULT, fixture.levels, Provenance::by("test:levels")).unwrap();
    let mut embed_provenance = Provenance::by("test:embed");
    embed_provenance.model = Some(model());
    doc.add(&Key::DEFAULT, fixture.vectors, embed_provenance).unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    (graph.to_bytes().unwrap(), graph)
}

#[test]
fn document_graph_round_trips_with_layer_listing() {
    let (bytes, graph) = graph_bytes();
    let out = round_trip_document(&DocumentGraphRequest { graph: bytes }).unwrap();
    assert_eq!(out.source_text, TEXT);
    let ids: Vec<(&str, LayerWireStatus)> =
        out.layers.iter().map(|l| (l.layer.as_str(), LayerWireStatus::try_from(l.status).unwrap())).collect();
    for want in ["opennlp:sentences", "turbo:chunks", "turbo:levels", "opennlp:embeddings<turbo:chunks>"] {
        assert!(ids.contains(&(want, LayerWireStatus::Ok)), "{want} in {ids:?}");
    }
    let chunks = out.layers.iter().find(|l| l.layer == "turbo:chunks").unwrap();
    assert_eq!(chunks.annotator, "test:chunker");
    assert_eq!(chunks.items, 2);
    let back = DocumentGraph::from_bytes(&out.graph).unwrap();
    assert_eq!(back, graph, "the re-exported graph is the same document");
    let doc = import(&back, &codecs()).unwrap();
    assert_eq!(doc.get::<Chunks>().unwrap().spans.len(), 2);

    let err = round_trip_document(&DocumentGraphRequest { graph: vec![1, 2, 3] }).expect_err("garbage refused");
    assert_eq!(err.code, turbo_kserve::api::TURBO_E_INVALID_ARGUMENT);
}

#[tokio::test]
async fn chunk_levels_and_document_graph_cross_grpc() {
    let (server, mut client) = common::serve_no_admin().await;

    let msg = fixture(OffsetMode::Utf16).to_msg();
    let back = client.round_trip_chunk_levels(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);

    let (bytes, _) = graph_bytes();
    let out = client.round_trip_document_graph(DocumentGraphRequest { graph: bytes }).await.unwrap().into_inner();
    assert!(out.layers.iter().any(|l| l.layer == "turbo:levels" && l.status == LayerWireStatus::Ok as i32));

    let mut bad: ChunkLevelsResult = fixture(OffsetMode::Utf16).to_msg();
    bad.chunk_frag_start[0] = java_utf16(byte_span("\u{1f600}").start) + 1;
    let status = client.round_trip_chunk_levels(bad).await.unwrap_err();
    assert_eq!(status.code(), tonic::Code::InvalidArgument, "{status}");
    server.stop().await;
}

#[test]
fn overlapping_touching_unordered_and_orphan_fragments_are_refused() {
    // Over "abcd": fragments [0,2) and [1,3) overlap.
    let mut msg = fixture(OffsetMode::Utf8).to_msg();
    msg.text = "abcd".into();
    msg.chunk_frag_start = vec![0, 1];
    msg.chunk_frag_end = vec![2, 3];
    msg.chunk_frag_at = vec![2, 2];
    let err = match ChunkLevels::from_msg(&msg) {
        Ok(_) => panic!("overlapping fragments refused, not a panic"),
        Err(e) => e,
    };
    assert_eq!(err.code, turbo_kserve::api::TURBO_E_INVALID_ARGUMENT, "{}", err.message);
    // Touching fragments are one continuous piece, not two.
    msg.chunk_frag_start = vec![0, 2];
    msg.chunk_frag_end = vec![2, 4];
    assert!(ChunkLevels::from_msg(&msg).is_err(), "touching fragments refused");
    // Unordered fragments.
    msg.chunk_frag_start = vec![2, 0];
    msg.chunk_frag_end = vec![4, 1];
    assert!(ChunkLevels::from_msg(&msg).is_err(), "unordered fragments refused");
    // The same on the levels side.
    let mut msg = fixture(OffsetMode::Utf8).to_msg();
    msg.level_frag_start = vec![0, 1, 0];
    msg.level_frag_end = vec![2, 3, 4];
    msg.level_frag_at = vec![3, 3, 3];
    assert!(ChunkLevels::from_msg(&msg).is_err(), "overlapping level fragments refused");

    // Orphan fragments: no items, but fragment columns present.
    let mut msg = fixture(OffsetMode::Utf8).to_msg();
    msg.chunk_frag_at = vec![];
    msg.chunk_frag_start = vec![u32::MAX];
    msg.chunk_frag_end = vec![u32::MAX];
    for col in [
        &mut msg.chunk_level,
        &mut msg.chunk_sentence_start,
        &mut msg.chunk_sentence_end,
        &mut msg.chunk_cut,
        &mut msg.chunk_cut_block,
        &mut msg.chunk_model_tokens,
    ] {
        col.clear();
    }
    msg.chunk_parent.clear();
    msg.chunk_cost.clear();
    msg.chunk_over_budget.clear();
    assert!(ChunkLevels::from_msg(&msg).is_err(), "orphan fragments refused");
}
