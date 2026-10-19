//! Abbreviation definitions and their links across the document
//! graph boundary, registered through this crate's own entry point:
//! they round-trip through bytes on source and derived views and
//! under named instances, and corrupted payloads are refused.

use turbo_abbrev::export::codecs;
use turbo_abbrev::{Abbreviations, abbreviation_links};
use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Alignment, Document, Edge, Edges, Ix, Key, Provenance, Span, Tag, View, ViewBuilder, ViewId};
use turbo_analysis_layers::AbbreviationForm;

const TEXT: &str = "World Health Organization (WHO) met. WHO left.";

fn abbreviations(view: ViewId) -> Abbreviations {
    let mut a = Abbreviations::new(view);
    a.push(Span::new(0, 25), AbbreviationForm::Long);
    a.push(Span::new(27, 30), AbbreviationForm::Short);
    a.push(Span::new(37, 40), AbbreviationForm::Mention);
    a
}

fn links() -> Edges<Abbreviations, Abbreviations> {
    Edges::new(
        abbreviation_links(),
        [Edge::new(Ix::new(1), Ix::new(0), Tag(0), 1.0), Edge::new(Ix::new(2), Ix::new(0), Tag(1), 1.0)],
    )
}

#[test]
fn abbreviations_and_links_round_trip() {
    let mut doc = Document::new(TEXT);
    doc.add(&Key::DEFAULT, abbreviations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, links(), Provenance::by("test")).unwrap();

    // Only this crate's registrations: no shared layers registry.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let a = back.get::<Abbreviations>().unwrap();
    assert_eq!(a.form(Ix::new(1)), AbbreviationForm::Short);
    let links = back.get::<Edges<Abbreviations, Abbreviations>>().unwrap();
    assert_eq!(a.long_form(links, Ix::new(2)), Some(Ix::new(0)));
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn abbreviations_on_a_derived_view_round_trip() {
    let mut doc = Document::decoded(
        b"Caf\xe9 WHO.".to_vec(),
        "Café WHO.",
        Alignment::from_edits(vec![turbo_analysis::text::Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }])
            .unwrap(),
    )
    .unwrap();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(5);
    builder.replace(6, "");
    let (clean, alignment) = builder.finish();
    doc.add(&Key::<View>::named("clean"), View::derived(&source, clean, alignment), Provenance::by("test")).unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    let mut a = Abbreviations::new(clean_id);
    a.push(Span::new(5, 8), AbbreviationForm::Mention);
    doc.add(&Key::DEFAULT, a, Provenance::by("test")).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let a = back.get::<Abbreviations>().unwrap();
    assert_eq!(a.spans.view(), back.get_key(&Key::<View>::named("clean")).unwrap().id());
    assert_eq!(&back.text(a.spans.view())[5..8], "WHO");
}

/// The span column encoding, written out longhand so corrupted
/// payloads can be built without an export context.
fn write_spans(w: &mut Writer, spans: &[Span]) {
    w.str("").unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

fn splice(graph: &mut DocumentGraph, layer_prefix: &str, body: Vec<u8>) {
    let at = graph.layers.iter().position(|l| l.layer.starts_with(layer_prefix)).unwrap();
    graph.layers[at].body = Some(body);
}

fn edges_body(instance: &str, to: u32) -> Vec<u8> {
    let mut w = Writer::new();
    for _ in 0..2 {
        w.str("turbo:abbreviations").unwrap();
        w.str(instance).unwrap();
    }
    w.str("abbreviation-links").unwrap();
    w.usize(2).unwrap();
    w.str("definition").unwrap();
    w.str("mention").unwrap();
    w.usize(1).unwrap();
    w.u32(1);
    w.u32(to);
    w.u32(0);
    w.f64(1.0);
    w.finish()
}

#[test]
fn corrupted_abbreviation_payloads_are_refused() {
    let mut doc = Document::new(TEXT);
    doc.add(&Key::DEFAULT, abbreviations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    doc.add(&Key::DEFAULT, links(), Provenance::by("test")).unwrap();
    let base = export(&doc, &codecs()).unwrap();

    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(0, 3)]);
    w.u8(3);
    let mut graph = base.clone();
    splice(&mut graph, "turbo:abbreviations", w.finish());
    assert!(import(&graph, &codecs()).is_err(), "an unknown form tag imported");

    let mut graph = base.clone();
    splice(&mut graph, "turbo:edges", edges_body("", 9));
    assert!(import(&graph, &codecs()).is_err(), "a link past the abbreviations imported");

    let mut graph = base.clone();
    splice(&mut graph, "turbo:abbreviations", vec![0]);
    assert!(import(&graph, &codecs()).is_err(), "a truncated payload imported");
}

#[test]
fn keyed_default_link_target_must_not_rebind_through_provenance() {
    let mut doc = Document::new(TEXT);
    doc.add(&Key::DEFAULT, abbreviations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    doc.add(&Key::named("gold"), abbreviations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Abbreviations>::named("gold").id()];
    doc.add(&Key::DEFAULT, links(), provenance).unwrap();

    // The valid document, its links resolved to the gold instance,
    // round-trips.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // The same links with the endpoint instance emptied would be
    // checked against the default abbreviations and then rebound
    // to gold, so they are refused.
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "turbo:edges", edges_body("", 0));
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("gold"), "{detail}"),
        Ok(_) => panic!("an empty link endpoint rebound through provenance imported"),
        Err(other) => panic!("an empty link endpoint failed as {other}"),
    }
}
