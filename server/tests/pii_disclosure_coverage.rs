//! A PII rewrite view leaves under `rewritten_text` only when it rewrote
//! every finding the document holds (#1932). The masked view reads one
//! `turbo:pii` instance; a second instance (a later detector, a model
//! head) whose findings the view never saw stays in clear in the view's
//! text, so that view's text is withheld like the source's: the typed
//! result lists the findings with no text, and a surface that answers with
//! the view's text refuses. An empty second instance, or a second rewrite
//! that read it, leaves the view releasable.

use prost::Message;
use turbo_analysis::{Analyzer, Document, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::{Pii, PiiEvidence, PiiKind, PiiMention};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::{InspectDocumentGraphRequest, TypedDocumentResult};
use turbo_kserve::results::disclosure::{Disclosure, PiiOutputPolicy};
use turbo_kserve::results::document::codecs;
use turbo_kserve::results::typed::{document_to_typed_with, inspect_document_graph_under};
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const EMAIL: &str = "alice.planted@example.com";
/// A value the built-in detector does not find: only the second instance
/// holds it.
const LATE: &str = "ZQPLANTED77";
const MASKED: ViewId = ViewId(1);
const REWRITTEN: Disclosure = Disclosure { rewritten_text: true, full_text: false, detail: false };

fn text() -> String {
    format!("Write to {EMAIL} with code {LATE} now")
}

/// The text scanned by the built-in detector and masked from its findings.
fn masked() -> Document {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(text())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    docs.pop().unwrap()
}

/// A second `turbo:pii` instance named `name` on the source, holding the
/// late value when `found`, else empty.
fn add_late(doc: &mut Document, name: &'static str, found: bool) {
    let mut pii = Pii::new(ViewId::SOURCE);
    if found {
        let at = text().find(LATE).unwrap() as u32;
        let detector = pii.detector("test:late", "1").unwrap();
        pii.push(PiiMention {
            span: Span::new(at, at + LATE.len() as u32),
            kind: PiiKind::GithubToken,
            detector,
            evidence: &[PiiEvidence::Shape],
            score: None,
            normalized: LATE,
        });
    }
    doc.add(&Key::<Pii>::named(name), pii, Provenance::by("test:late-pii")).unwrap();
}

fn holds(m: &TypedDocumentResult, needle: &str) -> bool {
    m.encode_to_vec().windows(needle.len()).any(|w| w == needle.as_bytes())
}

fn typed(doc: &Document, d: &Disclosure) -> TypedDocumentResult {
    document_to_typed_with("d", doc, MASKED, OffsetMode::Utf8, d).unwrap()
}

#[test]
fn a_rewrite_view_that_missed_a_finding_is_not_released_as_rewritten_text() {
    let mut doc = masked();
    assert!(!doc.text(MASKED).contains(EMAIL), "the built-in finding is masked");
    assert!(doc.text(MASKED).contains(LATE), "the late value was never rewritten");
    // Positive control: before the late finding exists the view leaves.
    let before = typed(&doc, &REWRITTEN);
    assert_eq!(before.raw_text, doc.text(MASKED));
    add_late(&mut doc, "late", true);
    assert!(!REWRITTEN.view_text(&doc, MASKED), "a view that missed a finding is releasable");
    assert!(REWRITTEN.require_view_text(&doc, MASKED, "test").is_err());
    let after = typed(&doc, &REWRITTEN);
    assert!(after.raw_text.is_empty(), "the view's text is withheld");
    assert!(!holds(&after, LATE), "the late finding's value leaves in the rewritten text");
    assert!(!holds(&after, EMAIL));
    let late = after.layers.iter().find(|l| l.id == "turbo:pii" && l.instance == "late").expect("listed");
    assert_eq!(late.annotations.len(), 1, "the late finding is still reported");
    // Explicit full text is the caller's choice and still returns both.
    let full = typed(&doc, &Disclosure { full_text: true, ..REWRITTEN });
    assert!(holds(&full, LATE));
}

#[test]
fn an_empty_second_instance_leaves_the_rewrite_releasable() {
    let mut doc = masked();
    add_late(&mut doc, "late", false);
    assert!(REWRITTEN.view_text(&doc, MASKED));
    let shown = typed(&doc, &REWRITTEN);
    assert_eq!(shown.raw_text, doc.text(MASKED));
    assert!(!holds(&shown, EMAIL));
}

#[test]
fn inspect_refuses_to_restore_a_missed_finding_from_graph_bytes() {
    let mut doc = masked();
    add_late(&mut doc, "late", true);
    let graph = turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let request = InspectDocumentGraphRequest {
        graph,
        view: u32::from(MASKED.0),
        pii_output: Some(REWRITTEN.to_options()),
        ..Default::default()
    };
    let policy = PiiOutputPolicy { full_text: false, detail: false, ..PiiOutputPolicy::ALL };
    let shown = inspect_document_graph_under(&request, policy).unwrap();
    assert!(shown.raw_text.is_empty());
    assert!(!holds(&shown, LATE), "the imported graph's rewrite releases the missed value");
    assert!(!holds(&shown, EMAIL));
}

/// From review 6726971395 on #1955: a second real detector (its id is
/// `turbo-pii:pii` and it reads a view) is a finding, never a rewrite's
/// own output, so the masked view that never read it stays withheld.
#[test]
fn late_real_detector_must_prevent_rewritten_text_release() {
    use turbo_analysis::Bound;
    use turbo_analysis_layers::CustomKinds;
    use turbo_pii::DenyList;
    let doc = masked();
    let kinds = CustomKinds::from_names(["secret"]).unwrap();
    let kind = kinds.find("secret").unwrap();
    let detector = PiiDetector::new(vec![])
        .with_custom_kinds(kinds)
        .with_deny_list(DenyList::from_terms(kind, [LATE]).unwrap())
        .unwrap();
    let analyzer =
        Analyzer::builder().step(Bound::new(detector).write(0, "late").write(1, "late_candidates")).build().unwrap();
    let mut docs = vec![doc];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    assert_eq!(doc.get_key(&Key::<Pii>::named("late")).unwrap().mentions().count(), 1);
    assert!(doc.text(MASKED).contains(LATE));
    assert!(!REWRITTEN.view_text(&doc, MASKED), "a real detector's findings are taken for rewrite labels");
    let shown = typed(&doc, &REWRITTEN);
    assert!(!holds(&shown, LATE), "the late real detector's value leaves in the rewritten text");
}
