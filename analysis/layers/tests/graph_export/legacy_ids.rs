//! A graph written while the shared layers had their legacy ids
//! (`fixtures/legacy-layer-ids.tgrf`, the export of [`document`] from
//! that build) reads as the same document today: every layer, every
//! cross-layer reference and every provenance read resolves under the
//! current ids, and exporting it again writes the current ids only.

use super::*;
use turbo_analysis::Layer;
use turbo_analysis::export::DocumentGraphRef;
use turbo_analysis::ids::{self, legacy_ids};

const LEGACY: &[u8] = include_bytes!("../fixtures/legacy-layer-ids.tgrf");

fn holds_legacy_id(bytes: &[u8]) -> bool {
    legacy_ids().any(|(old, _)| bytes.windows(old.len()).any(|w| w == old.as_bytes()))
}

#[test]
fn a_graph_with_legacy_ids_imports_as_the_document_it_was_written_from() {
    assert!(holds_legacy_id(LEGACY), "the fixture should carry the legacy ids");
    let graph = DocumentGraph::from_bytes(LEGACY).unwrap();
    assert!(graph.layers.iter().any(|l| l.layer == ids::TOKENS));
    assert!(graph.layers.iter().any(|l| l.layer == "turbo:dependencies<turbo:words>"));
    assert!(graph.layers.iter().any(|l| l.layer == "turbo:categories<turbo:sentences>"));
    let doc = import(&graph, &codecs()).unwrap();
    assert_eq!(doc.get::<Tokens>().unwrap().len(), 6);
    assert_eq!(doc.get::<PhraseChunks>().unwrap().len(), 1);

    // Exporting it again is byte for byte today's export of the same document.
    let again = export(&doc, &codecs()).unwrap().to_bytes().unwrap();
    let today = export(&document(), &codecs()).unwrap().to_bytes().unwrap();
    assert_eq!(again, today);
    assert!(!holds_legacy_id(&today));
}

#[test]
fn the_borrowed_directory_reads_legacy_ids_as_current_ones() {
    let dir = DocumentGraphRef::parse(LEGACY).unwrap();
    assert!(dir.layers().iter().all(|e| !holds_legacy_id(e.layer.as_bytes())));
    let tokens = dir.layer(ids::TOKENS, "").expect("tokens under the current id");
    assert_eq!(tokens.len, 6);
    let legacy_name = legacy_ids().find(|(_, new)| *new == ids::TOKENS).unwrap().0;
    assert_eq!(dir.layer(&legacy_name, ""), Some(tokens), "a lookup by the legacy id finds the same entry");
    let doc = dir.import_selected(&codecs(), &[(&legacy_name, ""), (ids::SENTENCES, "")]).unwrap();
    assert_eq!(doc.get::<Sentences>().unwrap().len(), 2);
}
