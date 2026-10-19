//! Resolved locations across the document graph boundary: full
//! gazetteer entries round-trip through bytes on source and
//! derived views, and payloads with coordinates, boxes, country
//! codes, populations or confidences the entry and layer checks
//! refuse are themselves refused.

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::{Alignment, Document, Ix, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_geo::export::register_codecs;
use turbo_geo::{AttributeValue, EntryParts, FEATURE_CLASS_CITY, GazetteerEntry, GeoBoundingBox, GeoPoint, Locations};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn entry(name: &str, lat: f64, lon: f64) -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "1".into(),
        name: name.into(),
        alternate_names: vec!["Alt".into()],
        location: GeoPoint::new(lat, lon).unwrap(),
        bounding_box: Some(GeoBoundingBox::new(lon - 1.0, lat - 1.0, lon + 1.0, lat + 1.0).unwrap()),
        country_code: Some("DE".into()),
        containment: vec!["Europe".into(), "Germany".into()],
        population: 3_600_000,
        feature_class: Some(FEATURE_CLASS_CITY.into()),
        attributes: vec![("wikidata".into(), AttributeValue::new("Q64", "test", "").unwrap())],
    })
    .unwrap()
}

fn bare_entry() -> GazetteerEntry {
    GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "2".into(),
        name: "Nowhere".into(),
        alternate_names: vec![],
        location: GeoPoint::new(0.0, 0.0).unwrap(),
        bounding_box: None,
        country_code: None,
        containment: vec![],
        population: 0,
        feature_class: None,
        attributes: vec![],
    })
    .unwrap()
}

fn locations(view: ViewId) -> Locations {
    let mut spans = Spans::new(view);
    spans.push(Span::new(0, 6));
    spans.push(Span::new(11, 18));
    Locations { spans, entry: vec![entry("Berlin", 52.52, 13.405), bare_entry()], confidence: vec![0.9, 0.25] }
}

#[test]
fn locations_round_trip() {
    let mut doc = Document::new("Berlin and Nowhere");
    doc.add(&Key::DEFAULT, locations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    doc.add(&Key::named("gold"), locations(ViewId::SOURCE), Provenance::by("test")).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let got = back.get::<Locations>().unwrap();
    let (_, e, c) = got.resolution(Ix::new(0));
    assert_eq!(e.name(), "Berlin");
    assert_eq!(e.country_code(), Some("DE"));
    assert_eq!(e.population(), 3_600_000);
    assert_eq!(e.attribute("wikidata").unwrap().value(), "Q64");
    assert_eq!(c, 0.9);
    assert_eq!(got.entry[1], bare_entry());
    assert!(back.get_key(&Key::<Locations>::named("gold")).is_some());
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn locations_on_a_derived_view_round_trip() {
    let mut doc = Document::decoded(
        b"Caf\xe9 Berlin.".to_vec(),
        "Café Berlin.",
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
    let mut spans = Spans::new(clean_id);
    spans.push(Span::new(5, 11));
    doc.add(
        &Key::DEFAULT,
        Locations { spans, entry: vec![entry("Berlin", 52.52, 13.405)], confidence: vec![1.0] },
        Provenance::by("test"),
    )
    .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let got = back.get::<Locations>().unwrap();
    assert_eq!(got.spans.view(), back.get_key(&Key::<View>::named("clean")).unwrap().id());
    assert_eq!(&back.text(got.spans.view())[5..11], "Berlin");
}

/// One entry on the wire, with the pieces a refusal case varies.
fn entry_body(lat: f64, lon: f64, country: Option<&str>, population: u64, attributes: &[&str]) -> Vec<u8> {
    let mut w = Writer::new();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.usize(1).unwrap();
    w.span(Span::new(0, 6));
    w.usize(1).unwrap();
    w.str("test").unwrap();
    w.str("1").unwrap();
    w.str("Berlin").unwrap();
    w.usize(0).unwrap();
    w.f64(lat);
    w.f64(lon);
    w.u8(0);
    match country {
        None => w.u8(0),
        Some(c) => {
            w.u8(1);
            w.str(c).unwrap();
        }
    }
    w.usize(0).unwrap();
    w.u64(population);
    w.u8(0);
    w.usize(attributes.len()).unwrap();
    for key in attributes {
        w.str(key).unwrap();
        w.str("v").unwrap();
        w.str("test").unwrap();
        w.str("").unwrap();
    }
    w.usize(1).unwrap();
    w.f64(0.5);
    w.finish()
}

fn splice(graph: &mut DocumentGraph, body: Vec<u8>) {
    let at = graph.layers.iter().position(|l| l.layer == "turbo:locations").unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = 1;
}

#[test]
fn corrupted_locations_are_refused() {
    let mut doc = Document::new("Berlin and Nowhere");
    doc.add(&Key::DEFAULT, locations(ViewId::SOURCE), Provenance::by("test")).unwrap();
    let base = export(&doc, &codecs()).unwrap();

    for (what, payload) in [
        ("a latitude past the pole", entry_body(91.0, 0.0, None, 0, &[])),
        ("a NaN longitude", entry_body(0.0, f64::NAN, None, 0, &[])),
        ("a lowercase country code", entry_body(0.0, 0.0, Some("de"), 0, &[])),
        ("a negative population", entry_body(0.0, 0.0, None, (-1_i64) as u64, &[])),
        ("a repeated attribute key", entry_body(0.0, 0.0, None, 0, &["k", "k"])),
        ("truncated", vec![0]),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, payload);
        assert!(import(&graph, &codecs()).is_err(), "{what} imported");
    }

    // A confidence the wire can spell but the layer refuses, and a
    // span ending inside a character of the derived text.
    let mut payload = entry_body(0.0, 0.0, None, 0, &[]);
    let at = payload.len() - 8;
    payload[at..].copy_from_slice(&f64::NAN.to_le_bytes());
    let mut graph = base.clone();
    splice(&mut graph, payload);
    assert!(import(&graph, &codecs()).is_err(), "a NaN confidence imported");

    let mut graph = base;
    splice(&mut graph, entry_body(0.0, 0.0, None, 0, &[]));
    // The valid control: the same hand-built body imports.
    assert!(import(&graph, &codecs()).is_ok(), "the hand-built control body was refused");
}

#[test]
fn a_location_span_splitting_a_character_is_refused() {
    let mut doc = Document::new("é Berlin");
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(3, 9));
    doc.add(
        &Key::DEFAULT,
        Locations { spans, entry: vec![bare_entry()], confidence: vec![0.5] },
        Provenance::by("test"),
    )
    .unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut payload = entry_body(0.0, 0.0, None, 0, &[]);
    // The span column starts the body: view name, item count,
    // fragment count, then start and end as u32s. Ending the span
    // one byte into the two-byte first character splits it.
    payload[16..20].copy_from_slice(&1u32.to_le_bytes());
    splice(&mut graph, payload);
    assert!(import(&graph, &codecs()).is_err(), "a span splitting a character imported");
}
