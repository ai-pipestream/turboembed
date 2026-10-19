//! The document graph export boundary (#499): a document with decoded
//! source bytes, a transformed view, abbreviation edges and a group
//! exports to a checked-in byte fixture and imports back with its node
//! identity, endpoints, spans, scores, statuses and provenance intact.
//! Refusals (versions, schemas, missing codecs, dangling endpoints,
//! malformed bytes) are typed errors.

use turbo_analysis::export::{Codecs, DocumentGraph, ExportError, NodeRef, export, import, layer_ref, node_ref};
use turbo_analysis::text::Edit;
use turbo_analysis::{
    Alignment, Analyzer, Annotator, Document, Edge, Edges, Error, Groups, Ix, Key, Layer, ModelRef, Provenance, Refs,
    Span, Status, Tag, View, ViewBuilder,
};
use turbo_analysis_layers::{AbbreviationForm, Abbreviations, abbreviation_links};

const TEXT: &str = "Café 𝄞 World Health Organization (WHO) met. The WHO met again.";
const FIXTURE: &[u8] = include_bytes!("fixtures/document-graph.bin");

fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    codecs
        .register::<Abbreviations>()
        .register::<Edges<Abbreviations, Abbreviations>>()
        .register::<Groups<Abbreviations>>();
    codecs
}

fn provenance(annotator: &'static str, config: u64, reads: Vec<turbo_analysis::KeyId>) -> Provenance {
    Provenance {
        annotator,
        version: "1",
        config,
        model: Some(ModelRef { id: "test/model".into(), revision: "r1".into(), sha256: "abc123".into() }),
        execution: None,
        reads,
        on: Vec::new(),
    }
}

/// The test document: bytes in a legacy single-byte encoding for the
/// e-acute (so source offsets and text offsets differ), a clean view
/// with the musical symbol removed, abbreviations and their links on
/// that view, one group, one skipped layer and one failed layer.
fn document() -> Document {
    let mut bytes: Vec<u8> = Vec::new();
    bytes.extend_from_slice(b"Caf");
    bytes.push(0xe9);
    bytes.extend_from_slice(" 𝄞 World Health Organization (WHO) met. The WHO met again.".as_bytes());
    let alignment = Alignment::from_edits(vec![Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }]).unwrap();
    let mut doc = Document::decoded(bytes, TEXT, alignment).unwrap();

    let symbol = TEXT.find('𝄞').unwrap();
    let mut builder = ViewBuilder::new(TEXT);
    builder.keep(symbol);
    builder.replace(symbol + "𝄞 ".len(), "");
    let (clean, alignment) = builder.finish();
    let source = doc.get::<View>().unwrap().clone();
    doc.add(
        &Key::<View>::named("clean"),
        View::derived(&source, clean.clone(), alignment),
        provenance("test:normalize", 7, vec![]),
    )
    .unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();

    let span = |needle: &str, from: usize| {
        let at = clean[from..].find(needle).unwrap() + from;
        Span::new(at as u32, (at + needle.len()) as u32)
    };
    let mut abbrev = Abbreviations::new(clean_id);
    let long = abbrev.push(span("World Health Organization", 0), AbbreviationForm::Long);
    let short = abbrev.push(span("WHO", 0), AbbreviationForm::Short);
    let mention = abbrev.push(span("WHO", span("WHO", 0).end as usize), AbbreviationForm::Mention);
    doc.add(&Key::DEFAULT, abbrev, provenance("test:abbrev", 42, vec![Key::<View>::named("clean").id()])).unwrap();

    let links =
        Edges::new(abbreviation_links(), [Edge::new(short, long, Tag(0), 0.95), Edge::new(mention, long, Tag(1), 0.5)]);
    doc.add(&Key::DEFAULT, links, provenance("test:abbrev", 42, vec![Key::<Abbreviations>::DEFAULT.id()])).unwrap();
    doc.add(&Key::DEFAULT, Groups::<Abbreviations>::new([[short, mention]]), provenance("test:groups", 3, vec![]))
        .unwrap();
    doc.skip(Key::<Abbreviations>::named("gold").id(), "no gold file".into(), Provenance::by("test:gold")).unwrap();
    doc.fail(
        Key::<Groups<Abbreviations>>::named("extra").id(),
        Error::Pipeline("grouper exploded".into()),
        Provenance::by("test:groups"),
    )
    .unwrap();
    doc
}

fn edge_rows(doc: &Document) -> Vec<(NodeRef, NodeRef, String, f64)> {
    let links = doc.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
    links
        .iter()
        .map(|(_, e)| {
            (
                node_ref(&links.from_key, e.from),
                node_ref(&links.to_key, e.to),
                links.kinds.label(e.kind).to_owned(),
                e.score,
            )
        })
        .collect()
}

#[test]
fn round_trip_through_the_fixture_bytes() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    if std::env::var_os("TURBO_UPDATE_FIXTURE").is_some() {
        std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/document-graph.bin"), &bytes).unwrap();
    }
    assert_eq!(bytes, FIXTURE, "export bytes differ from the checked-in fixture");
    let decoded = DocumentGraph::from_bytes(FIXTURE).unwrap();
    assert_eq!(decoded, graph);
    let back = import(&decoded, &codecs()).unwrap();

    // Source identity and Unicode source offsets survive.
    assert_eq!(back.bytes(), doc.bytes());
    assert_eq!(back.source(), TEXT);
    assert!(back.source_bytes().is_some());
    let e_acute = Span::new(3, 5);
    assert_eq!(back.to_source_bytes(turbo_analysis::ViewId::SOURCE, e_acute), Span::new(3, 4));

    // The transformed view and its alignment survive.
    let clean = back.get_key(&Key::<View>::named("clean")).unwrap();
    assert_eq!(clean.text(), "Café World Health Organization (WHO) met. The WHO met again.");
    let who = clean.text().find("WHO").unwrap() as u32;
    let in_source = back.to_source(clean.id(), Span::new(who, who + 3));
    assert_eq!(&TEXT[in_source.range()], "WHO");

    // Node identity, forms, endpoints, scores and groups survive.
    let abbrev = back.get::<Abbreviations>().unwrap();
    assert_eq!(abbrev.form, [AbbreviationForm::Long, AbbreviationForm::Short, AbbreviationForm::Mention]);
    assert_eq!(node_ref(&Key::<Abbreviations>::DEFAULT, Ix::new(1)).to_string(), "turbo:abbreviations#1");
    assert_eq!(edge_rows(&back), edge_rows(&doc));
    let groups = back.get::<Groups<Abbreviations>>().unwrap();
    assert_eq!(groups.get(Ix::new(0)), [Ix::new(1), Ix::new(2)]);
    assert_eq!(groups.of_key.id(), Key::<Abbreviations>::DEFAULT.id());

    // Provenance and statuses survive, including unresolved layers.
    let p = back.provenance(Key::<Edges<Abbreviations, Abbreviations>>::DEFAULT.id()).unwrap();
    assert_eq!((p.annotator, p.version, p.config), ("test:abbrev", "1", 42));
    assert_eq!(p.model.as_ref().map(|m| m.id.as_str()), Some("test/model"));
    assert_eq!(p.reads, [Key::<Abbreviations>::DEFAULT.id()]);
    assert_eq!(back.status(Key::<Abbreviations>::named("gold").id()), Some(&Status::Skipped("no gold file".into())));
    assert_eq!(
        back.status(Key::<Groups<Abbreviations>>::named("extra").id()),
        Some(&Status::Failed(Error::Pipeline("grouper exploded".into())))
    );
    assert_eq!(
        back.made_on(Key::<Edges<Abbreviations, Abbreviations>>::DEFAULT.id()),
        [Key::<Abbreviations>::DEFAULT.id()]
    );

    // Re-exporting the import is byte stable.
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), FIXTURE);
}

#[test]
fn unknown_versions_and_schemas_are_refused() {
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.version = 2;
    assert_eq!(import(&graph, &codecs()).err(), Some(ExportError::UnsupportedVersion { found: 2, supported: 1 }));

    let mut graph = export(&document(), &codecs()).unwrap();
    graph.layers[0].schema = 9;
    assert_eq!(
        import(&graph, &codecs()).err(),
        Some(ExportError::IncompatibleSchema { layer: "turbo:abbreviations".into(), exported: 9, supported: 1 })
    );

    let mut bytes = graph.to_bytes().unwrap();
    bytes[4] = 2;
    bytes[5] = 0;
    assert_eq!(DocumentGraph::from_bytes(&bytes), Err(ExportError::UnsupportedVersion { found: 2, supported: 1 }));
}

#[test]
fn a_layer_without_a_codec_is_refused_both_ways() {
    let mut partial = Codecs::new();
    partial.register::<Abbreviations>();
    assert_eq!(
        export(&document(), &partial),
        Err(ExportError::NoCodec { layer: "turbo:edges<turbo:abbreviations, turbo:abbreviations>".into() })
    );
    let graph = export(&document(), &codecs()).unwrap();
    assert!(matches!(import(&graph, &partial), Err(ExportError::NoCodec { .. })));
}

#[test]
fn dangling_endpoints_and_references_are_refused() {
    // Remove the abbreviations layer: the edges' endpoints dangle.
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.layers.retain(|l| l.layer != "turbo:abbreviations" || !l.instance.is_empty());
    match import(&graph, &codecs()) {
        Err(ExportError::DanglingReference { reference, .. }) => {
            assert_eq!(reference, layer_ref(&Key::<Abbreviations>::DEFAULT));
        }
        Ok(_) => panic!("a graph without its endpoint layer imported"),
        Err(other) => panic!("a graph without its endpoint layer failed as {other}"),
    }

    // Shrink the abbreviations layer to one item: endpoints 1 and 2
    // fall past its end.
    let mut small = Document::new("WHO");
    let mut abbrev = Abbreviations::new(turbo_analysis::ViewId::SOURCE);
    abbrev.push(Span::new(0, 3), AbbreviationForm::Short);
    small.add(&Key::DEFAULT, abbrev, Provenance::by("test:abbrev")).unwrap();
    let small_graph = export(&small, &codecs()).unwrap();
    let mut graph = export(&document(), &codecs()).unwrap();
    let at = graph.layers.iter().position(|l| l.layer == "turbo:abbreviations" && l.instance.is_empty()).unwrap();
    graph.layers[at] = small_graph.layers.into_iter().next().unwrap();
    match import(&graph, &codecs()) {
        Err(ExportError::DanglingEndpoint { endpoint, .. }) => {
            assert_eq!(endpoint, node_ref(&Key::<Abbreviations>::DEFAULT, Ix::new(1)));
        }
        Ok(_) => panic!("a graph with endpoints past its layer imported"),
        Err(other) => panic!("a graph with endpoints past its layer failed as {other}"),
    }
}

#[test]
fn view_alignments_are_checked_against_both_texts() {
    // An alignment whose edits point outside both texts.
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.views[1].edits = vec![Edit { parent: Span::new(500, 600), child: Span::new(500, 500) }];
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::InvalidLayer { .. })));

    // Identity claimed for a view whose text differs from its parent.
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.views[1].edits = Vec::new();
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::InvalidLayer { .. })));

    // The first view record is the source view, whatever its status.
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.views[0].status = turbo_analysis::export::StatusRecord::Skipped("not here".into());
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::InvalidSource(_))));
}

#[test]
fn relations_on_named_instances_round_trip() {
    // Annotators build relations with the default key even when the
    // step is bound to a named instance; the document resolves the
    // endpoints through provenance, and the export must name the
    // resolved instance, not the empty one.
    let mut doc = Document::new("World Health Organization (WHO) met. The WHO met again.");
    let text = doc.source().to_owned();
    let span = |needle: &str, from: usize| {
        let at = text[from..].find(needle).unwrap() + from;
        Span::new(at as u32, (at + needle.len()) as u32)
    };
    let mut abbrev = Abbreviations::new(turbo_analysis::ViewId::SOURCE);
    let long = abbrev.push(span("World Health Organization", 0), AbbreviationForm::Long);
    let short = abbrev.push(span("WHO", 0), AbbreviationForm::Short);
    let mention = abbrev.push(span("WHO", span("WHO", 0).end as usize), AbbreviationForm::Mention);
    let gold = Key::<Abbreviations>::named("gold");
    doc.add(&gold, abbrev, Provenance::by("test:abbrev")).unwrap();
    let reads_gold = provenance("test:abbrev", 1, vec![gold.id()]);
    let links =
        Edges::new(abbreviation_links(), [Edge::new(short, long, Tag(0), 1.0), Edge::new(mention, long, Tag(1), 1.0)]);
    doc.add(&Key::DEFAULT, links, reads_gold.clone()).unwrap();
    doc.add(&Key::DEFAULT, Groups::<Abbreviations>::new([[short, mention]]), reads_gold).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&graph, &codecs()).unwrap();
    assert!(back.get::<Abbreviations>().is_none(), "no default instance was exported");
    assert!(back.get_key(&gold).is_some());
    let links = back.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
    assert_eq!(links.from_key.name, "gold");
    assert_eq!(node_ref(&links.from_key, Ix::new(1)).to_string(), "turbo:abbreviations@gold#1");
    assert_eq!(back.get::<Groups<Abbreviations>>().unwrap().of_key.name, "gold");
    // The in-memory original still carries the empty key the annotator
    // built it with; the imported copy names the resolved instance.
    // Endpoints, kinds and scores are the same rows.
    let rows =
        |d: &Document| edge_rows(d).iter().map(|(a, b, k, s)| (a.index, b.index, k.clone(), *s)).collect::<Vec<_>>();
    assert_eq!(rows(&back), rows(&doc));
}

#[test]
fn provenance_references_to_missing_instances_are_refused() {
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.layers[1]
        .provenance
        .reads
        .push(turbo_analysis::export::LayerRef { layer: "turbo:abbreviations".into(), instance: "ghost".into() });
    match import(&graph, &codecs()) {
        Err(ExportError::DanglingReference { reference, .. }) => assert_eq!(reference.instance, "ghost"),
        Ok(_) => panic!("a graph with a ghost provenance reference imported"),
        Err(other) => panic!("a graph with a ghost provenance reference failed as {other}"),
    }

    // An empty relation layer still names its endpoint layer: with
    // that layer gone it is a dangling reference, not an invalid layer.
    let mut doc = Document::new("WHO");
    let mut abbrev = Abbreviations::new(turbo_analysis::ViewId::SOURCE);
    abbrev.push(Span::new(0, 3), AbbreviationForm::Short);
    doc.add(&Key::DEFAULT, abbrev, Provenance::by("test:abbrev")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Edges::<Abbreviations, Abbreviations>::new(abbreviation_links(), []),
        Provenance::by("test:abbrev"),
    )
    .unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    graph.layers.retain(|l| l.layer != "turbo:abbreviations");
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::DanglingReference { .. })));
}

struct GroupAll;

impl Annotator for GroupAll {
    const ID: &'static str = "test:group-all";
    type Reads = (Abbreviations,);
    type Writes = (Groups<Abbreviations>,);

    fn annotate(&self, (abbrev,): Refs<'_, Self>) -> Result<(Groups<Abbreviations>,), Error> {
        let all: Vec<Ix<Abbreviations>> = (0..abbrev.spans.len() as u32).map(Ix::new).collect();
        Ok((Groups::new([all]),))
    }
}

#[test]
fn a_failed_step_whose_required_input_is_absent_round_trips() {
    // The pipeline is told the documents arrive with abbreviations and
    // this one does not have them, so the step is recorded failed with
    // the missing input in its error and its provenance reads. That
    // unresolved output must cross the boundary like any other.
    let pipeline = Analyzer::builder().given(Key::<Abbreviations>::DEFAULT).add(GroupAll).build().unwrap();
    let mut docs = [Document::new("WHO")];
    let _ = pipeline.run(&mut docs);
    let doc = docs.into_iter().next().unwrap();
    let groups_key = Key::<Groups<Abbreviations>>::DEFAULT.id();
    assert_eq!(doc.status(groups_key), Some(&Status::Failed(Error::Missing(Key::<Abbreviations>::DEFAULT.id()))));
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.status(groups_key), Some(&Status::Failed(Error::Missing(Key::<Abbreviations>::DEFAULT.id()))));
    assert_eq!(back.provenance(groups_key).unwrap().reads, [Key::<Abbreviations>::DEFAULT.id()]);
    assert!(back.get::<Abbreviations>().is_none());
}

fn edges_payload(labels: usize) -> Vec<u8> {
    use turbo_analysis::export::Writer;
    let mut w = Writer::new();
    for _ in 0..2 {
        w.str("turbo:abbreviations").unwrap();
        w.str("").unwrap();
    }
    w.str("test:kinds").unwrap();
    w.usize(labels).unwrap();
    for i in 0..labels {
        w.str(&format!("label-{i}")).unwrap();
    }
    w.usize(1).unwrap();
    w.u32(1);
    w.u32(0);
    w.u32(0);
    w.f64(1.0);
    w.finish()
}

#[test]
fn tag_sets_longer_than_a_u16_can_index_are_refused_on_import() {
    let mut graph = export(&document(), &codecs()).unwrap();
    let at = graph.layers.iter().position(|l| l.layer.starts_with("turbo:edges")).unwrap();
    graph.layers[at].body = Some(edges_payload(65_537));
    graph.layers[at].len = 1;
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("tag set"), "{detail}"),
        Ok(_) => panic!("a tag set of 65,537 labels imported"),
        Err(other) => panic!("a tag set of 65,537 labels failed as {other}"),
    }
    // Exactly 65,536 labels still indexes, last label included.
    graph.layers[at].body = Some(edges_payload(65_536));
    let back = import(&graph, &codecs()).unwrap();
    let links = back.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
    assert_eq!(links.kinds.tag("label-65535"), Some(Tag(65_535)));
    assert_eq!(links.kinds.label(Tag(65_535)), "label-65535");
}

#[test]
fn source_view_provenance_is_fixed_by_the_format() {
    let mut graph = export(&document(), &codecs()).unwrap();
    graph.views[0].provenance.annotator = "source:decoder".into();
    graph.views[0].provenance.config = 42;
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::InvalidSource(_))));
}

#[test]
fn malformed_bytes_are_refused() {
    assert!(matches!(DocumentGraph::from_bytes(&[]), Err(ExportError::Malformed(_))));
    assert!(matches!(DocumentGraph::from_bytes(b"nope"), Err(ExportError::Malformed(_))));
    assert!(matches!(DocumentGraph::from_bytes(&FIXTURE[..FIXTURE.len() - 1]), Err(ExportError::Malformed(_))));
    let mut trailing = FIXTURE.to_vec();
    trailing.push(0);
    assert!(matches!(DocumentGraph::from_bytes(&trailing), Err(ExportError::Malformed(_))));
}

#[test]
fn truncated_and_mutated_bytes_never_panic() {
    for cut in 0..FIXTURE.len() {
        assert!(DocumentGraph::from_bytes(&FIXTURE[..cut]).is_err(), "a graph cut at {cut} bytes decoded");
    }
    for at in (0..FIXTURE.len()).step_by(7) {
        let mut mutated = FIXTURE.to_vec();
        mutated[at] ^= 0xff;
        if let Ok(graph) = DocumentGraph::from_bytes(&mutated) {
            // A mutation inside a text or payload field may still
            // decode; importing it must refuse or accept, never panic.
            let _ = import(&graph, &codecs());
        }
    }
}

#[test]
fn node_ref_names_the_instance_it_was_made_on() {
    assert_eq!(layer_ref(&Key::<Abbreviations>::named("gold")).to_string(), "turbo:abbreviations@gold");
    assert_eq!(Abbreviations::ID, "turbo:abbreviations");
}

/// The record of the first layer whose name starts with `prefix`, with
/// its body's last `tail` bytes replaced.
fn with_tail(prefix: &str, tail: &[u8]) -> DocumentGraph {
    let mut graph = export(&document(), &codecs()).unwrap();
    let at = graph.layers.iter().position(|l| l.layer.starts_with(prefix)).unwrap();
    let body = graph.layers[at].body.as_mut().unwrap();
    let n = body.len();
    body[n - tail.len()..].copy_from_slice(tail);
    graph
}

fn refused_as_malformed(graph: &DocumentGraph, what: &str) {
    match import(graph, &codecs()) {
        Err(ExportError::Malformed(_)) => {}
        Ok(_) => panic!("{what} imported"),
        Err(other) => panic!("{what} failed as {other}"),
    }
}

#[test]
fn a_group_payload_with_members_out_of_order_or_repeated_is_refused() {
    // The one group is members 1 and 2: its last eight bytes.
    let le = |a: u32, b: u32| [a.to_le_bytes(), b.to_le_bytes()].concat();
    import(&with_tail("turbo:groups", &le(1, 2)), &codecs()).unwrap();
    refused_as_malformed(&with_tail("turbo:groups", &le(2, 1)), "a group with members out of order");
    refused_as_malformed(&with_tail("turbo:groups", &le(1, 1)), "a group with a repeated member");
}

#[test]
fn an_edges_payload_with_rows_out_of_order_is_refused() {
    // The two rows, 20 bytes each, are (1, 0, kind 0) then (2, 0, kind 1).
    let row = |from: u32, kind: u32, score: f64| {
        [from.to_le_bytes(), 0u32.to_le_bytes(), kind.to_le_bytes()].concat().into_iter().chain(score.to_le_bytes())
    };
    let swapped: Vec<u8> = row(2, 1, 0.5).chain(row(1, 0, 0.95)).collect();
    refused_as_malformed(&with_tail("turbo:edges", &swapped), "edges out of (from, to, kind) order");
}
