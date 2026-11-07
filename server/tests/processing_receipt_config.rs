//! The receipt's configuration identity (issue #1908): one digest over
//! what was asked of the document, never over its text, so the same
//! operation key reused under another configuration is detectable.

use turbo_analysis::{Document, Key, ModelRef, PROCESSING_CONFIG_DIGEST_RULE, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Pii, Sentences};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::results::typed::document_to_typed_bound;

const TEXT: &str = "One sentence here.";

fn sentences(text: &str) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    Sentences { spans }
}

fn pii_provenance(config: u64, version: &'static str) -> Provenance {
    Provenance {
        config,
        version,
        reads: vec![Key::<Sentences>::DEFAULT.id()],
        model: Some(ModelRef { id: "m".into(), revision: "r1".into(), sha256: "ab".into() }),
        ..Provenance::by("test:pii")
    }
}

/// Sentences then PII, in that order or the other, over `text`.
fn doc(text: &str, config: u64, version: &'static str, pii_first: bool) -> Document {
    let mut d = Document::new(text);
    let add_s = |d: &mut Document| d.add(&Key::DEFAULT, sentences(text), Provenance::by("test:sent")).unwrap();
    let add_p =
        |d: &mut Document| d.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), pii_provenance(config, version)).unwrap();
    if pii_first {
        add_p(&mut d);
        add_s(&mut d);
    } else {
        add_s(&mut d);
        add_p(&mut d);
    }
    d
}

fn hex(d: &Document) -> String {
    d.processing_config_digest().to_hex()
}

#[test]
fn the_config_digest_is_pinned() {
    assert_eq!(PROCESSING_CONFIG_DIGEST_RULE, "turbo.processing-config.v1");
    assert_eq!(hex(&doc(TEXT, 7, "1", false)), GOLDEN);
}

/// Recomputed outside Rust from the documented canonical input.
const GOLDEN: &str = "d73ef90db335cbff89311eee1f38782dd545fb45ba94df889c6500f72374cbbc";

#[test]
fn config_and_version_change_it_and_text_and_order_do_not() {
    let base = hex(&doc(TEXT, 7, "1", false));
    assert_ne!(base, hex(&doc(TEXT, 8, "1", false)), "a config hash change");
    assert_ne!(base, hex(&doc(TEXT, 7, "2", false)), "a version change");
    assert_eq!(base, hex(&doc("Another, longer sentence.", 7, "1", false)), "the text takes no part");
    assert_eq!(base, hex(&doc(TEXT, 7, "1", true)), "insertion order takes no part");
    let mut d = Document::new(TEXT);
    d.add(&Key::DEFAULT, sentences(TEXT), Provenance::by("test:sent")).unwrap();
    let mut p = pii_provenance(7, "1");
    p.model.as_mut().unwrap().revision = "r2".into();
    d.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), p).unwrap();
    assert_ne!(base, hex(&d), "a model revision change");
    let mut d = Document::new(TEXT);
    d.add(&Key::DEFAULT, sentences(TEXT), Provenance::by("test:sent")).unwrap();
    d.add(&Key::DEFAULT, Pii::new(ViewId::SOURCE), Provenance { reads: Vec::new(), ..pii_provenance(7, "1") }).unwrap();
    assert_ne!(base, hex(&d), "a change in what the layer reads");
}

#[test]
fn the_receipt_carries_the_config_digest_beside_the_input() {
    let (a, b) = (doc(TEXT, 7, "1", false), doc("Different text entirely.", 7, "1", false));
    let ta = document_to_typed_bound("a", &a, ViewId::SOURCE, OffsetMode::Utf8, None, "k").unwrap();
    let tb = document_to_typed_bound("b", &b, ViewId::SOURCE, OffsetMode::Utf8, None, "k").unwrap();
    let (ra, rb) = (ta.receipt.unwrap(), tb.receipt.unwrap());
    assert_eq!(ra.config_digest, a.processing_config_digest().as_bytes().to_vec());
    assert_eq!(ra.config_digest, rb.config_digest, "same configuration, different input");
    assert_ne!(ra.input, rb.input);
}
