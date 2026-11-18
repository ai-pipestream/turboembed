//! A sink item for a document whose graph may not leave (#2403): with a
//! typed projection and a search projection configured, the item still
//! carries no digest of the source text or of the graph, on either way an
//! item is made (from the document, or from graph bytes).

use prost::Message;
use turbo_analysis::export::graph_content_digest;
use turbo_analysis::{Analyzer, Document, ViewId, ViewTextDigest};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

use super::document::{SinkTyped, exported_item, graph_item};
use crate::proto::result::OffsetMode;
use crate::proto::sink::DocumentSinkItem;
use crate::results::disclosure::Disclosure;
use crate::results::document::{codecs, export_document_response};

const EMAIL: &str = "alice.planted@example.com";
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };
/// Both projections on the masked view.
const ON_MASKED: Option<SinkTyped> = Some(SinkTyped { view: 1, offsets: OffsetMode::Utf8 });

/// A document that is only the address, with its findings and masked view.
fn masked() -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(EMAIL.to_owned())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

fn graph_bytes(doc: &Document) -> Vec<u8> {
    turbo_analysis::export::export(doc, codecs()).unwrap().to_bytes().unwrap()
}

/// The item made from the document, and the one made from its graph bytes.
fn items(doc: &Document, disclosure: &Disclosure) -> [(&'static str, DocumentSinkItem); 2] {
    let exported = export_document_response(doc, &Disclosure::FULL).unwrap();
    let from_doc = exported_item("d", exported, doc, ON_MASKED, ON_MASKED, disclosure, None).unwrap();
    let from_graph = graph_item("d", graph_bytes(doc), ON_MASKED, ON_MASKED, disclosure, None).unwrap();
    [("from the document", from_doc), ("from graph bytes", from_graph)]
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn holds(item: &DocumentSinkItem, needle: &[u8]) -> bool {
    item.encode_to_vec().windows(needle.len()).any(|w| w == needle)
}

#[test]
fn an_item_with_its_graph_withheld_carries_no_digest_of_the_source() {
    let doc = masked();
    let text = ViewTextDigest::of_text(doc.text(ViewId::SOURCE)).as_bytes().to_vec();
    let graph = graph_content_digest(&graph_bytes(&doc)).as_bytes().to_vec();
    for disclosure in [REWRITTEN, Disclosure::FINDINGS] {
        for (path, item) in items(&doc, &disclosure) {
            assert!(item.graph.is_empty(), "{path}: the graph is withheld");
            assert!(item.typed.is_some(), "{path}: the typed projection is carried");
            assert_eq!(item.search_projection.is_some(), disclosure.rewritten_text, "{path}");
            for (what, digest) in [("source text", &text), ("graph bytes", &graph)] {
                assert!(!holds(&item, digest), "{path}, {disclosure:?}: the item carries the SHA-256 of the {what}");
                assert!(!holds(&item, hex(digest).as_bytes()), "{path}, {disclosure:?}: hex SHA-256 of the {what}");
            }
            assert!(!holds(&item, EMAIL.as_bytes()), "{path}: the value itself left");
        }
    }
}

/// With the full text allowed the same item names its source by digest.
#[test]
fn an_item_that_carries_its_graph_keeps_the_digests() {
    let doc = masked();
    let text = ViewTextDigest::of_text(doc.text(ViewId::SOURCE)).as_bytes().to_vec();
    for (path, item) in items(&doc, &Disclosure::FULL) {
        assert!(!item.graph.is_empty(), "{path}");
        assert!(holds(&item, &text), "{path}: the source text digest is carried");
    }
}
