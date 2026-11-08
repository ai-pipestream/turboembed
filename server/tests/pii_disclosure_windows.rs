//! The window surfaces under the PII disclosure decision (#1904 part 2):
//! `InspectDocumentWindow` (window text, x-ray parent and child text,
//! summary details, heatmaps) and `ProjectChunks` (previews, term stacks)
//! carry view text, so a document holding PII is refused unless the text
//! of every view the answer shows is disclosed. The x-ray target follows
//! the same rule as the main view. A refusal names the option, never the
//! value.

use turbo_analysis::{Analyzer, Document, ViewId};
use turbo_kserve::api::{Failure, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE};
use turbo_kserve::inspect::inspect_document_window_under;
use turbo_kserve::inspect::project::project_chunks_under;
use turbo_kserve::proto::result::{DocumentWindowRequest, OffsetMode, PiiOutputOptions, ProjectChunksRequest};
use turbo_kserve::results::disclosure::PiiOutputPolicy;
use turbo_kserve::results::document::codecs;
use turbo_pii::PiiDetector;
use turbo_pii::rewrite::MaskedView;

const PLANTED: &str = "alice.planted@example.com";
const MASKED: u32 = 1;

fn pii_graph() -> Vec<u8> {
    let analyzer =
        Analyzer::builder().add(PiiDetector::cursor()).step(MaskedView::by_kind('*').unwrap().bound()).build().unwrap();
    let mut docs = vec![Document::new(format!("Write to {PLANTED} before noon."))];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let doc = docs.pop().unwrap();
    assert!(!doc.text(ViewId(MASKED as u16)).contains(PLANTED), "view 1 is the masked view");
    turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap()
}

fn options(full_text: bool, rewritten_text: bool) -> Option<PiiOutputOptions> {
    Some(PiiOutputOptions { full_text, rewritten_text, ..Default::default() })
}

fn window(graph: &[u8], view: u32, xray: Option<u32>, pii_output: Option<PiiOutputOptions>) -> DocumentWindowRequest {
    DocumentWindowRequest {
        graph: graph.to_vec(),
        view,
        offsets: OffsetMode::Utf8 as i32,
        page_size: 16_000,
        page: 0,
        filter_layers: Vec::new(),
        heatmap_buckets: 4,
        xray_target_view: xray,
        pii_output,
    }
}

fn project(graph: &[u8], pii_output: Option<PiiOutputOptions>) -> ProjectChunksRequest {
    ProjectChunksRequest {
        graph: graph.to_vec(),
        offsets: OffsetMode::Utf8 as i32,
        pca_dims: 2,
        max_stack_segments: 5,
        include_drift_curve: false,
        include_strategy_matrix: false,
        strategy_budgets: Vec::new(),
        pii_output,
    }
}

/// `e` is the disclosure refusal and does not quote the value.
fn refused(e: Failure, what: &str) {
    assert_eq!(e.code, TURBO_E_INVALID_STATE, "{what}: {}", e.message);
    assert!(e.message.contains("pii_output.full_text"), "{what}: {}", e.message);
    assert!(!e.message.contains(PLANTED) && !e.message.contains("planted"), "{what}: {}", e.message);
}

fn holds(bytes: &[u8]) -> bool {
    bytes.windows(PLANTED.len()).any(|w| w == PLANTED.as_bytes())
}

#[test]
fn a_window_of_a_pii_document_needs_its_view_text_disclosed() {
    use prost::Message;
    let g = pii_graph();
    let all = PiiOutputPolicy::ALL;
    // Findings only (the default) refuses every view: a window is text.
    for view in [0, MASKED] {
        refused(inspect_document_window_under(&window(&g, view, None, None), all).unwrap_err(), "default");
    }
    // rewritten_text shows the masked view, and only that view.
    let masked = inspect_document_window_under(&window(&g, MASKED, None, options(false, true)), all).unwrap();
    assert!(!holds(&masked.encode_to_vec()), "the masked window holds the value");
    refused(inspect_document_window_under(&window(&g, 0, None, options(false, true)), all).unwrap_err(), "source");
    // full_text shows the source: positive control for the leak check.
    let full = inspect_document_window_under(&window(&g, 0, None, options(true, false)), all).unwrap();
    assert!(holds(&full.encode_to_vec()), "positive control: the source window holds the value");
    // The policy caps the option as everywhere else.
    let none = PiiOutputPolicy::NONE;
    let e = inspect_document_window_under(&window(&g, 0, None, options(true, false)), none).unwrap_err();
    assert!(e.message.contains("full_text") && !e.message.contains(PLANTED), "{}", e.message);
}

#[test]
fn an_xray_target_follows_the_same_rule_as_the_main_view() {
    let g = pii_graph();
    let all = PiiOutputPolicy::ALL;
    // The masked view is disclosed, but an x-ray onto the source would show
    // the source text as its child text: refused.
    let e = inspect_document_window_under(&window(&g, MASKED, Some(0), options(false, true)), all).unwrap_err();
    refused(e, "x-ray onto the source");
    let e = inspect_document_window_under(&window(&g, 0, Some(MASKED), options(false, true)), all).unwrap_err();
    refused(e, "x-ray from the source");
    // An out-of-range target is the caller's mistake, said as for `view`,
    // whatever is disclosed.
    for opts in [None, options(true, false)] {
        let e = inspect_document_window_under(&window(&g, MASKED, Some(9), opts), all).unwrap_err();
        assert_eq!(e.code, TURBO_E_INVALID_ARGUMENT, "{}", e.message);
        assert!(e.message.contains("xray_target_view 9 is out of range"), "{}", e.message);
    }
    // With full_text both ends may leave.
    let out = inspect_document_window_under(&window(&g, MASKED, Some(0), options(true, false)), all).unwrap();
    // Positive control: the runs spell the source text out, one character
    // a run, so the value is in them even where no one field holds it.
    let xray = out.xray.expect("the x-ray is computed");
    let child: String = xray.runs.iter().map(|r| r.child_text.as_str()).collect();
    assert!(child.contains(PLANTED), "{child}");
}

#[test]
fn project_chunks_of_a_pii_document_needs_its_view_text_disclosed() {
    let g = pii_graph();
    let all = PiiOutputPolicy::ALL;
    refused(project_chunks_under(&project(&g, None), all).unwrap_err(), "default");
    refused(project_chunks_under(&project(&g, options(false, true)), all).unwrap_err(), "rewritten source");
    // With the text disclosed the gate passes; any later refusal is about
    // the graph (it holds no chunks), not about PII.
    if let Err(e) = project_chunks_under(&project(&g, options(true, false)), all) {
        assert_ne!(e.code, TURBO_E_INVALID_STATE, "{}", e.message);
    }
}

#[test]
fn a_document_without_pii_answers_as_before() {
    let doc = Document::new("No personal data here.");
    let g = turbo_analysis::export::export(&doc, codecs()).unwrap().to_bytes().unwrap();
    inspect_document_window_under(&window(&g, 0, None, None), PiiOutputPolicy::NONE).unwrap();
}
