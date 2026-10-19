//! The document graph round trip of sketches, duplicate groups and
//! links, and boilerplate flags, and the refusal of truncated, oversized
//! and out-of-range payloads.

use turbo_analysis::export::{Codecs, DocumentGraph, ExportError, export, import};
use turbo_analysis::*;
use turbo_analysis_layers::{Boilerplate, Structure};
use turbo_group::{BoilerplateDetector, NearDuplicates, Sketcher, Sketches};
use turbo_structure::{Syntax, detect};

fn codecs() -> Codecs {
    let mut c = turbo_analysis_layers::export::codecs();
    turbo_group::export::register_codecs(&mut c);
    c
}

fn document() -> Document {
    let text = "A paragraph that repeats itself here.\n\n\
                A paragraph that repeats itself here.\n\n\
                A paragraph that repeats itself there.\n\nSomething else.";
    let mut docs = [Document::new(text)];
    docs[0].add(&Key::DEFAULT, detect(text, Syntax::Markdown, ViewId::SOURCE), Provenance::by("t")).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(Sketcher::<Structure>::new(2))
        .add(NearDuplicates::<Structure>::default())
        .add(BoilerplateDetector::<Structure>::default())
        .build()
        .unwrap();
    assert_eq!(pipeline.run(&mut docs), [Ok(())]);
    let [doc] = docs;
    doc
}

#[test]
fn the_layers_round_trip_through_bytes() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), bytes, "a second export is byte equal");

    let (a, b) = (doc.get::<Sketches<Structure>>().unwrap(), back.get::<Sketches<Structure>>().unwrap());
    assert_eq!((a.shingle, &a.words, &a.exact, &a.shape), (b.shingle, &b.words, &b.exact, &b.shape));
    assert_eq!((&a.simhash, &a.minhash), (&b.simhash, &b.minhash));

    let (a, b) =
        (doc.get::<Edges<Structure, Structure>>().unwrap(), back.get::<Edges<Structure, Structure>>().unwrap());
    let rows = |e: &Edges<Structure, Structure>| -> Vec<_> {
        e.iter().map(|(_, r)| (r.from, r.to, e.kinds.label(r.kind).to_owned(), r.score)).collect()
    };
    assert!(!rows(a).is_empty());
    assert_eq!(rows(a), rows(b));
    assert_eq!(b.kinds.label(Tag(2)), "version-edit");

    let (a, b) = (doc.get::<Groups<Structure>>().unwrap(), back.get::<Groups<Structure>>().unwrap());
    assert_eq!((&a.members, &a.end), (&b.members, &b.end));
    assert!(!a.members.is_empty());

    let (a, b) = (doc.get::<Boilerplate<Structure>>().unwrap(), back.get::<Boilerplate<Structure>>().unwrap());
    assert_eq!((&a.kind, &a.score), (&b.kind, &b.score));
    assert!(a.flagged().next().is_some(), "the repeated paragraph is a template");
}

/// The graph with the body of the layer named `layer` replaced.
fn with_body(layer: &str, f: impl FnOnce(&mut Vec<u8>)) -> DocumentGraph {
    let mut graph = export(&document(), &codecs()).unwrap();
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    f(graph.layers[at].body.as_mut().unwrap());
    graph
}

fn refused(graph: &DocumentGraph, what: &str) {
    match import(graph, &codecs()) {
        Err(ExportError::Malformed(_) | ExportError::InvalidLayer { .. }) => {}
        Ok(_) => panic!("{what} imported"),
        Err(other) => panic!("{what} failed as {other}"),
    }
}

const LAYERS: [&str; 2] = ["turbo:sketches<turbo:structure>", "turbo:boilerplate<turbo:structure>"];

#[test]
fn truncated_and_extended_payloads_are_refused() {
    for layer in LAYERS {
        let full = with_body(layer, |_| {}).layers.iter().find(|l| l.layer == layer).unwrap().body.clone().unwrap();
        for cut in 0..full.len() {
            refused(&with_body(layer, |b| b.truncate(cut)), &format!("{layer} cut at {cut}"));
        }
        refused(&with_body(layer, |b| b.push(0)), &format!("{layer} with a trailing byte"));
    }
}

#[test]
fn oversized_counts_are_refused_without_reading_past_the_payload() {
    // The sketch count follows the one-byte shingle width; the boilerplate
    // count opens its payload.
    for (layer, at) in [(LAYERS[0], 1), (LAYERS[1], 0)] {
        for count in [u32::MAX, 1 << 30, 5] {
            refused(
                &with_body(layer, |b| b[at..at + 4].copy_from_slice(&count.to_le_bytes())),
                &format!("{layer} claiming {count} items"),
            );
        }
    }
}

#[test]
fn values_outside_their_range_are_refused() {
    // A shingle of no words.
    refused(&with_body(LAYERS[0], |b| b[0] = 0), "a sketch shingle of 0");
    // An item with no words but MinHash values: the first item's words.
    refused(&with_body(LAYERS[0], |b| b[5..9].copy_from_slice(&0u32.to_le_bytes())), "an empty item with shingles");
    // A boilerplate kind code past the last kind, and a score past 1.
    refused(&with_body(LAYERS[1], |b| b[4] = 6), "a boilerplate kind code of 6");
    refused(
        &with_body(LAYERS[1], |b| {
            let n = b.len();
            b[n - 4..].copy_from_slice(&1.5f32.to_le_bytes());
        }),
        "a boilerplate score of 1.5",
    );
    refused(
        &with_body(LAYERS[1], |b| {
            let n = b.len();
            b[n - 4..].copy_from_slice(&f32::NAN.to_le_bytes());
        }),
        "a boilerplate score of NaN",
    );
}

#[test]
fn a_duplicate_edge_into_a_block_that_is_not_there_is_refused() {
    // The last edge row's target, past the blocks.
    let graph = with_body("turbo:edges<turbo:structure, turbo:structure>", |b| {
        let n = b.len();
        b[n - 16..n - 12].copy_from_slice(&999u32.to_le_bytes());
    });
    assert!(matches!(import(&graph, &codecs()), Err(ExportError::DanglingEndpoint { .. })));
}
