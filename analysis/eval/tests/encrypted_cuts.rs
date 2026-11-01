#[path = "encrypted/common.rs"]
mod common;
use common::*;
use std::fs;
use turbo_analysis::{Document, IxRange, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Boundaries, Chunks, Cut, Sentences};
use turbo_eval::artifact::EvaluationArtifactStore;
use turbo_eval::artifact::encrypted::new_artifact_key;
use turbo_eval::cuts::{Dump, Header, Writer, compare_dumps};

fn header() -> Header {
    Header {
        set: "private-set-name".into(),
        sha256: "ab".repeat(32),
        backend: "cpu".into(),
        device: "cpu".into(),
        precision: "exact".into(),
        max: 100,
        target: 50,
        size_weight: 1000,
        overhead: 0,
        chunker: vec![],
    }
}
fn document() -> Document {
    let mut doc = Document::new("秘密 café 🦀.");
    let end = doc.source().len() as u32;
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, end));
    doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: spans.clone() }, Provenance::by("fixture")).unwrap();
    doc.add(
        &Key::<Boundaries>::DEFAULT,
        Boundaries { cost: vec![0], reason: vec![Cut::Sentence] },
        Provenance::by("fixture"),
    )
    .unwrap();
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, end));
    doc.add(
        &Key::<Chunks>::DEFAULT,
        Chunks {
            spans,
            level: vec![0],
            parent: vec![None],
            sentences: vec![IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart],
            cost: vec![0],
            model_tokens: vec![2],
            over_budget: vec![false],
            prompt_overhead: 0,
        },
        Provenance::by("fixture"),
    )
    .unwrap();
    doc
}

#[test]
fn explicitly_keyed_cut_exports_encrypt_metadata_and_preserve_comparison() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let a = new_artifact_key().unwrap();
    let b = new_artifact_key().unwrap();
    let header = header();
    let document = document();
    // The legacy entry point still requires set-derived filenames. It does
    // not silently change names, store plaintext, or bypass adapter policy.
    assert!(Writer::create_in(&store, &header).is_err());
    for key in [&a, &b] {
        let mut writer = Writer::create_at(&store, key, &header).unwrap();
        writer.document("private-document-id", &document).unwrap();
        let (publication, count) = writer.publish().unwrap();
        assert_eq!(publication.key, *key);
        assert_eq!(count, 1);
        let bytes = fs::read(fixture.physical(1, key)).unwrap();
        for private in [document.source().as_bytes(), header.set.as_bytes(), b"private-document-id"] {
            assert!(!bytes.windows(private.len()).any(|w| w == private));
        }
    }
    let a_dump = Dump::read_at(&store, &a, &header.set).unwrap();
    let b_dump = Dump::read_at(&store, &b, &header.set).unwrap();
    assert_eq!(a_dump.docs[0].1.source(), document.source());
    assert_eq!(a_dump.docs[0].0, "private-document-id");
    assert_eq!(a_dump.header, header);
    assert!(compare_dumps(&a_dump, &b_dump).unwrap().is_identical());
    assert!(Dump::read_at(&store, &a, "wrong expected set").is_err());
    assert!(Dump::read_from(&store, &a).is_err());
    assert_eq!(store.list().unwrap().len(), 2);
}

#[test]
fn keyed_cut_decoding_errors_never_reveal_private_identifiers() {
    let fixture = Fixture::new();
    let store = fixture.store(1);
    let key = key(1);
    let header = header();
    let mut writer = Writer::create_at(&store, &key, &header).unwrap();
    // This graph lacks required sentence/chunk layers: the storage framing
    // is authentic, but the consumer must refuse it without leaking the ID.
    writer.document("private-document-id", &Document::new("sensitive source")).unwrap();
    writer.publish().unwrap();
    let error = Dump::read_at(&store, &key, &header.set).err().unwrap();
    assert_eq!(error, "cut artifact decoding failed");
    assert!(!error.contains("private-document-id"));
    assert!(!error.contains("sensitive source"));
}
