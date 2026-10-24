//! The borrowed graph directory, selective import, and graph
//! diff/merge of #1199, exercised against the checked-in graph
//! fixture and hand-built record graphs.

use turbo_analysis::export::{
    Codecs, DocumentGraph, DocumentGraphRef, ExportError, LayerRecord, LayerRef, ProvenanceRecord, SourceRecord,
    StatusRecord, ViewRecord, import,
};
use turbo_analysis::{Edges, Groups, Key, Layer};
use turbo_analysis_layers::Abbreviations;

const FIXTURE: &[u8] = include_bytes!("fixtures/document-graph.bin");

fn codecs() -> Codecs {
    let mut codecs = Codecs::new();
    codecs
        .register::<Abbreviations>()
        .register::<Edges<Abbreviations, Abbreviations>>()
        .register::<Groups<Abbreviations>>();
    codecs
}

#[test]
fn directory_lists_the_fixture_without_decoding_bodies() {
    let owned = DocumentGraph::from_bytes(FIXTURE).unwrap();
    let dir = DocumentGraphRef::parse(FIXTURE).unwrap();
    assert_eq!(dir.source_text(), owned.source.text);
    assert_eq!(dir.source_bytes(), owned.source.bytes.as_slice());
    assert_eq!(dir.views().len(), owned.views.len());
    for (entry, record) in dir.views().iter().zip(&owned.views) {
        assert_eq!(entry.instance, record.instance);
        assert_eq!(entry.parent.as_deref(), record.parent.as_deref());
        assert_eq!(entry.text, record.text.as_deref());
    }
    assert_eq!(dir.layers().len(), owned.layers.len());
    for (entry, record) in dir.layers().iter().zip(&owned.layers) {
        assert_eq!(entry.layer, record.layer);
        assert_eq!(entry.instance, record.instance);
        assert_eq!(entry.schema, record.schema);
        assert_eq!(entry.len, record.len);
        assert_eq!(entry.producer, record.provenance.annotator);
        assert_eq!(entry.config_hash, record.provenance.config);
        assert_eq!(dir.body(entry), record.body.as_deref());
    }
}

#[test]
fn directory_refuses_bad_payloads() {
    assert!(DocumentGraphRef::parse(b"not a graph").is_err());
    let mut truncated = FIXTURE.to_vec();
    truncated.truncate(FIXTURE.len() / 2);
    assert!(DocumentGraphRef::parse(&truncated).is_err());
    for cut in (0..FIXTURE.len()).step_by(97) {
        let _ = DocumentGraphRef::parse(&FIXTURE[..cut]);
    }
    let mut trailing = FIXTURE.to_vec();
    trailing.push(0);
    assert!(DocumentGraphRef::parse(&trailing).is_err());
}

#[test]
fn selective_import_restores_only_the_asked_layers() {
    let dir = DocumentGraphRef::parse(FIXTURE).unwrap();
    let abbrev = dir.layers().iter().find(|e| e.layer.contains("abbreviations")).expect("fixture abbreviations");
    let selected = [(abbrev.layer, abbrev.instance)];
    let graph = dir.selected_graph(&selected).unwrap();
    assert_eq!(graph.layers.len(), 1);
    assert_eq!(graph.views.len(), dir.views().len());

    let doc = dir.import_selected(&codecs(), &selected).unwrap();
    assert!(doc.get::<Abbreviations>().is_some());
    assert!(doc.get::<Edges<Abbreviations, Abbreviations>>().is_none());
    assert!(doc.get::<Groups<Abbreviations>>().is_none());

    let full = import(&DocumentGraph::from_bytes(FIXTURE).unwrap(), &codecs()).unwrap();
    assert_eq!(doc.get::<Abbreviations>().unwrap().len(), full.get::<Abbreviations>().unwrap().len());

    let missing = dir.selected_graph(&[("turbo:nope", "")]);
    assert!(matches!(missing, Err(ExportError::DanglingReference { .. })));
    // An empty selection still imports the views.
    let views_only = dir.import_selected(&codecs(), &[]).unwrap();
    assert!(views_only.get_key(&Key::<turbo_analysis::View>::named("clean")).is_some());
}

fn provenance(annotator: &str, config: u64) -> ProvenanceRecord {
    ProvenanceRecord {
        annotator: annotator.into(),
        version: "1".into(),
        config,
        model: None,
        execution: None,
        reads: vec![],
        on: vec![],
    }
}

fn layer(name: &str, instance: &str, len: u32, body: Option<&[u8]>, config: u64) -> LayerRecord {
    LayerRecord {
        layer: name.into(),
        instance: instance.into(),
        schema: 1,
        len,
        status: StatusRecord::Complete,
        provenance: provenance("test:ann", config),
        body: body.map(<[u8]>::to_vec),
    }
}

fn graph(layers: Vec<LayerRecord>) -> DocumentGraph {
    DocumentGraph {
        version: 1,
        source: SourceRecord { bytes: b"hello".to_vec(), text: "hello".into(), alignment: None },
        views: vec![ViewRecord {
            instance: String::new(),
            status: StatusRecord::Complete,
            provenance: provenance("turbo:source", 0),
            parent: None,
            text: None,
            edits: vec![],
        }],
        layers,
    }
}

fn key(name: &str, instance: &str) -> LayerRef {
    LayerRef { layer: name.into(), instance: instance.into() }
}

#[test]
fn diff_reports_added_removed_and_modified_layers() {
    let left = graph(vec![layer("a", "", 2, Some(b"x"), 1), layer("b", "", 1, Some(b"y"), 1)]);
    let right = graph(vec![layer("a", "", 3, Some(b"z"), 2), layer("c", "alt", 1, None, 1)]);
    let diff = left.diff(&right);
    assert!(diff.source_matches);
    assert_eq!(diff.added_layers, vec![key("c", "alt")]);
    assert_eq!(diff.removed_layers, vec![key("b", "")]);
    assert_eq!(diff.modified_layers.len(), 1);
    let m = &diff.modified_layers[0];
    assert_eq!(m.layer, key("a", ""));
    assert!(m.provenance_changed && m.data_changed && !m.status_changed);
    assert_eq!((m.left_len, m.right_len), (2, 3));
    assert_eq!(left.diff(&left).modified_layers.len(), 0);
    assert!(left.diff(&left).added_layers.is_empty());
}

#[test]
fn merge_folds_disjoint_layers_and_refuses_conflicts() {
    let mut left = graph(vec![layer("a", "", 2, Some(b"x"), 1)]);
    let disjoint = graph(vec![layer("b", "", 1, Some(b"y"), 1)]);
    left.merge_layers(disjoint, false).unwrap();
    assert_eq!(left.layers.len(), 2);
    assert!(left.to_bytes().is_ok());

    // Same instance, different body: refused without overwrite and
    // the graph is unchanged; accepted with it.
    let conflicting = graph(vec![layer("a", "", 2, Some(b"z"), 1)]);
    let before = left.clone();
    assert!(matches!(left.merge_layers(conflicting.clone(), false), Err(ExportError::DuplicateLayer { .. })));
    assert_eq!(left, before);
    left.merge_layers(conflicting, true).unwrap();
    assert_eq!(left.layers.iter().find(|l| l.layer == "a").unwrap().body.as_deref(), Some(b"z".as_slice()));

    // Identical records merge cleanly without overwrite.
    let same = graph(vec![layer("a", "", 2, Some(b"z"), 1)]);
    left.merge_layers(same, false).unwrap();

    // A different source is refused even with overwrite.
    let mut other_source = graph(vec![]);
    other_source.source.text = "other".into();
    assert!(matches!(left.merge_layers(other_source, true), Err(ExportError::InvalidSource(_))));
}
