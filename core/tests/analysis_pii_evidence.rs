//! C ABI tests for PII evidence facts, detector identity, scores and
//! detail-gated candidates (`#1941`).

use std::ffi::c_char;
use std::path::PathBuf;
use std::process::Command;

use turbo::{
    TURBO_ANALYSIS_LAYER_PII, TURBO_ANALYZER_PII_DETAIL, TURBO_ANALYZER_PII_KIND_IDS, TURBO_PII_ALLOW_RULE_TERM,
    TURBO_PII_ALLOW_SCOPE_KIND, TURBO_PII_CHECKSUM_LUHN, TURBO_PII_CONFIDENCE_FORMAT,
    TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM, TURBO_PII_DISPOSITION_ALLOWED_BY_RULE, TURBO_PII_DISPOSITION_KEPT,
    TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_EVIDENCE_CHECKSUM, TURBO_PII_EVIDENCE_DENY_HIT,
    TURBO_PII_EVIDENCE_LABEL, TURBO_PII_EVIDENCE_RULE, TURBO_PII_KIND_CUSTOM_BASE, TURBO_PII_REJECT_LABEL_REQUIRED,
    TURBO_PII_SCORE_PROBABILITY, turbo_analysis_result, turbo_analysis_result_pii_candidates,
    turbo_analysis_result_pii_facts, turbo_analysis_result_pii_spans, turbo_analysis_result_release,
    turbo_analysis_result_spans, turbo_analysis_result_strings, turbo_analyzer, turbo_analyzer_analyze_graph,
    turbo_analyzer_create, turbo_analyzer_desc, turbo_analyzer_detect_pii, turbo_analyzer_inspect_graph,
    turbo_analyzer_release, turbo_error, turbo_pii_candidate, turbo_pii_fact, turbo_pii_span, turbo_span, turbo_text,
};
use turbo_analysis::{Codecs, Document, Key, Provenance, Span, ViewId, export};
use turbo_analysis_layers::export::register_codecs;
use turbo_analysis_layers::{
    ChecksumAlgorithm, CustomKinds, Pii, PiiCandidate, PiiCandidates, PiiDisposition, PiiEvidence, PiiKind, PiiMention,
    PiiScore, PiiScoreKind, RejectReason,
};
use turbo_pii::{CustomExtractor, Extractor, PiiDetector, PiiSink, Scanner};

struct EmpExtractor {
    kinds: [PiiKind; 1],
}

impl CustomExtractor for EmpExtractor {
    fn name(&self) -> &'static str {
        "acme:emp-id"
    }
    fn kinds(&self) -> &[PiiKind] {
        &self.kinds
    }
    fn version(&self) -> &str {
        "2.1.0"
    }
    fn scan(&self, text: &str, sink: &mut PiiSink<'_>) {
        let model = sink.model("acme:scorer", "1.0").unwrap();
        let score = PiiScore { value: 0.9375, kind: PiiScoreKind::Probability, model, calibrated: true };
        for (at, hit) in text.match_indices("E-42") {
            let span = Span::new(at as u32, (at + hit.len()) as u32);
            let facts = [PiiEvidence::Rule { rule: 7 }];
            sink.try_add_candidate(span, self.kinds[0], &facts, Some(score), |s| s.push_str(hit)).unwrap();
        }
    }
}

const SOURCE_TEXT: &str = "PAN 4532-0151-1283-0366, employee E-42, and unlabelled 046 454 286.";

fn encode_doc(doc: &Document) -> Vec<u8> {
    let mut codecs = Codecs::new();
    register_codecs(&mut codecs);
    export(doc, &codecs).unwrap().to_bytes().unwrap()
}

/// Build a document holding a Luhn-passed card with a label fact, a custom
/// extractor mention with its detector id and a calibrated probability score,
/// and a `turbo:pii-candidates` layer with kept and label-rejected candidates.
fn build_fixture_graph() -> (Document, Vec<u8>) {
    let mut custom = CustomKinds::new();
    let emp_kind = custom.add("employee-id").unwrap();
    let detector = PiiDetector::new(vec![Scanner::all(Extractor::Cursor), Scanner::all(Extractor::CaIdentity)])
        .with_custom_kinds(custom)
        .label_boosts(PiiKind::Card)
        .with_custom(EmpExtractor { kinds: [emp_kind] })
        .unwrap();
    let (pii, candidates) = detector.layer_with_candidates(SOURCE_TEXT, ViewId::SOURCE);
    let mut doc = Document::new(SOURCE_TEXT);
    doc.add(&Key::<Pii>::DEFAULT, pii, Provenance::by("turbo:pii")).unwrap();
    let mut cand_prov = Provenance::by("turbo:pii");
    cand_prov.on.push(Key::<Pii>::DEFAULT.id());
    doc.add(&Key::<PiiCandidates>::DEFAULT, candidates, cand_prov).unwrap();
    let wire = encode_doc(&doc);
    (doc, wire)
}

/// Candidate layer `"audit"` bound via provenance to two-mention PII layer `"actual"` (with
/// `u32::MAX` rule and deny-list IDs), alongside an unrelated one-mention PII layer named `"audit"`.
fn build_bound_parent_and_max_u32_graph() -> Vec<u8> {
    let mut doc = Document::new("a@b.co c@d.co");
    let mut actual = Pii::new(ViewId::SOURCE);
    let detector = actual.detector("custom:det", "1").unwrap();
    let (kind, score) = (PiiKind::Email, None);
    for (span, normalized, ev) in [
        (Span::new(0, 6), "a@b.co", PiiEvidence::Rule { rule: u32::MAX }),
        (Span::new(7, 13), "c@d.co", PiiEvidence::DenyHit { list: u32::MAX, term: 1 }),
    ] {
        actual.push(PiiMention { span, kind, detector, evidence: &[ev], score, normalized });
    }
    doc.add(&Key::<Pii>::named("actual"), actual, Provenance::by("test:pii")).unwrap();
    let mut unrelated = Pii::new(ViewId::SOURCE);
    let detector = unrelated.detector("custom:other", "1").unwrap();
    let (span, normalized) = (Span::new(0, 6), "a@b.co");
    unrelated.push(PiiMention { span, kind, detector, evidence: &[PiiEvidence::Shape], score, normalized });
    doc.add(&Key::<Pii>::named("audit"), unrelated, Provenance::by("test:pii")).unwrap();
    let mut cands = PiiCandidates::new(ViewId::SOURCE);
    let detector = cands.detectors.add("custom:det", "1").unwrap();
    let (span, disposition) = (Span::new(7, 13), PiiDisposition::Kept { mention: turbo_analysis::Ix::new(1) });
    let rule_max = [PiiEvidence::Rule { rule: u32::MAX }];
    cands.push(PiiCandidate { span, kind, detector, evidence: &rule_max, score, disposition });
    let mut prov = Provenance::by("test:pii");
    prov.on.push(Key::<Pii>::named("actual").id());
    doc.add(&Key::<PiiCandidates>::named("audit"), cands, prov).unwrap();
    encode_doc(&doc)
}

fn c_text(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr().cast::<c_char>(), len: s.len() as u64 }
}

fn zeroed_error() -> turbo_error {
    let mut err: turbo_error = unsafe { std::mem::zeroed() };
    err.struct_size = size_of::<turbo_error>() as u32;
    err
}

struct BorrowedPii<'a> {
    spans: &'a [turbo_span],
    pii_spans: &'a [turbo_pii_span],
    facts: &'a [turbo_pii_fact],
    candidates: &'a [turbo_pii_candidate],
    pool: &'a [u8],
}

unsafe fn slice_from<'a, T>(ptr: *const T, count: usize) -> &'a [T] {
    if count == 0 { &[] } else { unsafe { std::slice::from_raw_parts(ptr, count) } }
}

impl<'a> BorrowedPii<'a> {
    unsafe fn from_result(r: *const turbo_analysis_result) -> Self {
        unsafe {
            let mut err = zeroed_error();
            let (mut spans_ptr, mut span_count) = (std::ptr::null(), 0u32);
            assert_eq!(turbo_analysis_result_spans(r, &mut spans_ptr, &mut span_count, &mut err), turbo::status::OK);
            let (mut pii_ptr, mut pii_count) = (std::ptr::null(), 0u32);
            assert_eq!(turbo_analysis_result_pii_spans(r, &mut pii_ptr, &mut pii_count, &mut err), turbo::status::OK);
            let (mut facts_ptr, mut fact_count) = (std::ptr::null(), 0u32);
            assert_eq!(
                turbo_analysis_result_pii_facts(r, &mut facts_ptr, &mut fact_count, &mut err),
                turbo::status::OK
            );
            let (mut cand_ptr, mut cand_count) = (std::ptr::null(), 0u32);
            assert_eq!(
                turbo_analysis_result_pii_candidates(r, &mut cand_ptr, &mut cand_count, &mut err),
                turbo::status::OK
            );
            let (mut pool_ptr, mut pool_len) = (std::ptr::null::<c_char>(), 0u64);
            assert_eq!(turbo_analysis_result_strings(r, &mut pool_ptr, &mut pool_len, &mut err), turbo::status::OK);
            Self {
                spans: slice_from(spans_ptr, span_count as usize),
                pii_spans: slice_from(pii_ptr, pii_count as usize),
                facts: slice_from(facts_ptr, fact_count as usize),
                candidates: slice_from(cand_ptr, cand_count as usize),
                pool: slice_from(pool_ptr.cast::<u8>(), pool_len as usize),
            }
        }
    }

    fn str(&self, offset: u32, len: u32) -> &'a str {
        let start = offset as usize;
        std::str::from_utf8(&self.pool[start..start + len as usize]).unwrap()
    }
}

const FIXTURE_GRAPH_HEX: &str = concat!(
    "5447524603004300000050414e20343533322d303135312d313238332d303336362c20656d706c6f79656520452d34322c20616e6420756e",
    "6c6162656c6c65642030343620343534203238362e4300000050414e20343533322d303135312d313238332d303336362c20656d706c6f79",
    "656520452d34322c20616e6420756e6c6162656c6c65642030343620343534203238362e000100000000000000000c000000747572626f3a",
    "736f75726365010000003000000000000000000000000000000000000000014300000050414e20343533322d303135312d313238332d3033",
    "36362c20656d706c6f79656520452d34322c20616e6420756e6c6162656c6c65642030343620343534203238362e00000000020000000900",
    "0000747572626f3a706969000000000100020000000009000000747572626f3a706969010000003000000000000000000000000000000000",
    "0000010b0100000000000002000000010000000400000017000000010000002200000026000000010000000b000000656d706c6f7965652d",
    "69640400000010000000747572626f2d7069693a637572736f7205000000302e312e3015000000747572626f2d7069693a63612d6964656e",
    "7469747905000000302e312e300b00000061636d653a656d702d696405000000322e312e300b00000061636d653a73636f72657203000000",
    "312e300200000008000000000001000200000000000200030000000201010300000000030000000300000005070000000200000002000000",
    "030000000200000000010000703f0103000102000000100000003435333230313531313238333033363604000000452d3432140000007475",
    "72626f3a7069692d63616e64696461746573000000000100030000000009000000747572626f3a7069690100000030000000000000000000",
    "00000000000100000009000000747572626f3a70696900000000011501000000000000030000000100000004000000170000000100000022",
    "00000026000000010000003700000042000000010000000b000000656d706c6f7965652d69640400000010000000747572626f2d7069693a",
    "637572736f7205000000302e312e3015000000747572626f2d7069693a63612d6964656e7469747905000000302e312e300b00000061636d",
    "653a656d702d696405000000322e312e300b00000061636d653a73636f72657203000000312e300300000008000000000001001c00000003",
    "0000000000020001000400000002010103000000000300000003000000050700000002010103000000020000000300000004000000030000",
    "0000010000703f010300010003000000010000000001010000000401"
);

#[test]
fn c_abi_pii_evidence_detector_score_and_detail_candidates_match_rust_layer() {
    let (doc, wire) = build_fixture_graph();
    let hex: String = wire.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, FIXTURE_GRAPH_HEX, "keep AnalyzerConformance.java PII_EVIDENCE_GRAPH_HEX in sync");
    let rust_pii = doc.get::<Pii>().unwrap();
    let rust_cand = doc.get::<PiiCandidates>().unwrap();
    assert_eq!(rust_pii.spans.len(), 2);
    assert_eq!(rust_cand.spans.len(), 3);
    assert_eq!(
        rust_pii.get(turbo_analysis::Ix::new(0)).evidence,
        &[
            PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true },
            PiiEvidence::Label { span: Span::new(0, 3), phrase: 3 },
        ]
    );
    assert_eq!(
        rust_cand.get(turbo_analysis::Ix::new(2)).disposition,
        PiiDisposition::RejectedByRule { reason: RejectReason::LabelRequired }
    );

    unsafe {
        let mut err = zeroed_error();
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.flags = TURBO_ANALYZER_PII_KIND_IDS;
        let mut default_analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut default_analyzer, &mut err), turbo::status::OK);

        let mut r: *mut turbo_analysis_result = std::ptr::null_mut();
        let inspect = |a: *mut turbo_analyzer, g: &[u8], out: &mut *mut turbo_analysis_result, e: &mut turbo_error| {
            turbo_analyzer_inspect_graph(a, g.as_ptr(), g.len() as u64, out, e)
        };
        assert_eq!(inspect(default_analyzer, &wire, &mut r, &mut err), turbo::status::OK);
        let borrowed = BorrowedPii::from_result(r);
        assert_eq!((borrowed.spans.len(), borrowed.pii_spans.len(), borrowed.candidates.len()), (2, 2, 0));

        // No planted value and no context label text appears in the string pool.
        for forbidden in ["4532-0151-1283-0366", "4532015112830366", "E-42", "046 454 286", "046454286", "PAN"] {
            assert!(!borrowed.pool.windows(forbidden.len()).any(|w| w == forbidden.as_bytes()));
        }

        // Row 0: Luhn-passed card with label fact.
        let (card_span, card_pii) = (&borrowed.spans[0], &borrowed.pii_spans[0]);
        assert_eq!(card_span.layer_kind, TURBO_ANALYSIS_LAYER_PII);
        assert_eq!(card_span.classification, TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM);
        assert_eq!(card_span.token_start, u32::from(PiiKind::Card.builtin_id().unwrap()));
        assert_eq!((card_pii.span_index, card_pii.score_kind), (0, 0));
        assert_eq!(borrowed.str(card_pii.detector_id_offset, card_pii.detector_id_len), "turbo-pii:cursor");
        let card_ver = borrowed.str(card_pii.detector_version_offset, card_pii.detector_version_len);
        assert_eq!(card_ver, turbo_pii::BUILTIN_VERSION);
        assert!(card_pii.score_value.is_nan());
        let card_facts = &borrowed.facts[card_pii.fact_start as usize..card_pii.fact_end as usize];
        assert_eq!(
            card_facts[0],
            turbo_pii_fact { kind: TURBO_PII_EVIDENCE_CHECKSUM, a: TURBO_PII_CHECKSUM_LUHN, b: 1, c: 0 }
        );
        assert_eq!(card_facts[1], turbo_pii_fact { kind: TURBO_PII_EVIDENCE_LABEL, a: 0, b: 3, c: 3 });

        // Row 1: custom extractor with detector ("acme:emp-id", "2.1.0"), Rule(7) fact, and calibrated score.
        let (emp_span, emp_pii) = (&borrowed.spans[1], &borrowed.pii_spans[1]);
        assert_eq!(emp_span.classification, TURBO_PII_CONFIDENCE_FORMAT);
        assert_eq!(emp_span.token_start, TURBO_PII_KIND_CUSTOM_BASE);
        assert_eq!(borrowed.str(emp_span.label_offset, emp_span.label_len), "employee-id");
        assert_eq!(emp_pii.span_index, 1);
        assert_eq!(borrowed.str(emp_pii.detector_id_offset, emp_pii.detector_id_len), "acme:emp-id");
        assert_eq!(borrowed.str(emp_pii.detector_version_offset, emp_pii.detector_version_len), "2.1.0");
        assert_eq!(emp_pii.score_kind, TURBO_PII_SCORE_PROBABILITY);
        assert_eq!((emp_pii.score_calibrated, emp_pii.score_value), (1, 0.9375));
        assert_eq!(borrowed.str(emp_pii.score_model_id_offset, emp_pii.score_model_id_len), "acme:scorer");
        assert_eq!(borrowed.str(emp_pii.score_model_version_offset, emp_pii.score_model_version_len), "1.0");
        let emp_facts = &borrowed.facts[emp_pii.fact_start as usize..emp_pii.fact_end as usize];
        assert_eq!(emp_facts, &[turbo_pii_fact { kind: TURBO_PII_EVIDENCE_RULE, a: 7, b: 0, c: 0 }]);

        turbo_analysis_result_release(r);
        turbo_analyzer_release(default_analyzer);

        // Now with TURBO_ANALYZER_PII_DETAIL enabled: candidates are exposed.
        desc.flags = TURBO_ANALYZER_PII_KIND_IDS | TURBO_ANALYZER_PII_DETAIL;
        let mut detail_analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(&desc, &mut detail_analyzer, &mut err), turbo::status::OK);
        assert_eq!(inspect(detail_analyzer, &wire, &mut r, &mut err), turbo::status::OK);
        let detailed = BorrowedPii::from_result(r);
        let c = detailed.candidates;
        assert_eq!(c.len(), 3);
        assert_eq!((c[0].disposition, c[0].disposition_a), (TURBO_PII_DISPOSITION_KEPT, 0));
        assert_eq!((c[1].disposition, c[1].disposition_a), (TURBO_PII_DISPOSITION_KEPT, 1));
        assert_eq!((c[1].kind_code, c[1].score_kind), (TURBO_PII_KIND_CUSTOM_BASE, TURBO_PII_SCORE_PROBABILITY));
        assert_eq!(c[1].score_value, 0.9375);
        assert_eq!(
            (c[2].disposition, c[2].disposition_a),
            (TURBO_PII_DISPOSITION_REJECTED_BY_RULE, TURBO_PII_REJECT_LABEL_REQUIRED)
        );
        assert_eq!(c[2].kind_code, u32::from(PiiKind::CaSin.builtin_id().unwrap()));

        // And detect_pii + analyze_graph with TURBO_ANALYZER_PII_DETAIL emit candidates too.
        turbo_analysis_result_release(r);
        let direct_text = "IMEI: 490154203237518 and unlabelled 046 454 286.";
        assert_eq!(
            turbo_analyzer_detect_pii(detail_analyzer, c_text(direct_text), &mut r, &mut err),
            turbo::status::OK
        );
        let direct = BorrowedPii::from_result(r);
        assert!(!direct.pii_spans.is_empty() && direct.pii_spans.len() == direct.spans.len());
        let has_rejected = direct.candidates.iter().any(|x| {
            x.disposition == TURBO_PII_DISPOSITION_REJECTED_BY_RULE
                && x.disposition_a == TURBO_PII_REJECT_LABEL_REQUIRED
        });
        assert!(has_rejected);
        turbo_analysis_result_release(r);

        assert_eq!(
            turbo_analyzer_analyze_graph(detail_analyzer, c_text(direct_text), &mut r, &mut err),
            turbo::status::OK
        );
        let analyzed = BorrowedPii::from_result(r);
        assert!(!analyzed.pii_spans.is_empty() && !analyzed.candidates.is_empty());
        turbo_analysis_result_release(r);

        // Candidate layer "audit" bound to PII layer "actual" resolves against "actual" (2 mentions)
        // even when an unrelated 1-mention PII layer is also named "audit", and retains u32::MAX IDs.
        let bound_wire = build_bound_parent_and_max_u32_graph();
        assert_eq!(inspect(detail_analyzer, &bound_wire, &mut r, &mut err), turbo::status::OK);
        let bound = BorrowedPii::from_result(r);
        assert_eq!(bound.candidates.len(), 1);
        assert_eq!(bound.str(bound.candidates[0].instance_offset, bound.candidates[0].instance_len), "audit");
        assert_eq!(bound.str(bound.candidates[0].pii_instance_offset, bound.candidates[0].pii_instance_len), "actual");
        assert_eq!(
            (bound.candidates[0].disposition, bound.candidates[0].disposition_a),
            (TURBO_PII_DISPOSITION_KEPT, 1)
        );
        assert_eq!(bound.facts[0], turbo_pii_fact { kind: TURBO_PII_EVIDENCE_RULE, a: u32::MAX, b: 0, c: 0 });
        assert_eq!(bound.facts[1], turbo_pii_fact { kind: TURBO_PII_EVIDENCE_DENY_HIT, a: u32::MAX, b: 1, c: 0 });
        turbo_analysis_result_release(r);
        turbo_analyzer_release(detail_analyzer);
    }

    // Keep unused constants referenced so the test also locks their export types.
    let _ = (TURBO_PII_DISPOSITION_ALLOWED_BY_RULE, TURBO_PII_ALLOW_SCOPE_KIND, TURBO_PII_ALLOW_RULE_TERM);
}

#[test]
fn c_program_reads_evidence_detector_score_and_candidates_from_header() {
    let (_, wire) = build_fixture_graph();
    let cdylib = build_cdylib();
    let lib_dir = cdylib.parent().unwrap();

    let mut tmp = std::env::temp_dir();
    tmp.push(format!("turbo_c_pii_evidence_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    let graph_path = tmp.join("fixture.tgrf");
    std::fs::write(&graph_path, &wire).unwrap();

    let c_src = tmp.join("pii_evidence.c");
    let bin_path = tmp.join("pii_evidence_bin");
    let include_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().join("include");
    std::fs::write(&c_src, C_PROGRAM).unwrap();

    let cc_status = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-o"])
        .arg(&bin_path)
        .arg(&c_src)
        .arg(format!("-I{}", include_dir.display()))
        .arg(format!("-L{}", lib_dir.display()))
        .arg("-lturbo")
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .status()
        .expect("failed to invoke cc");
    assert!(cc_status.success(), "C compilation failed");
    let run_status = Command::new(&bin_path).arg(&graph_path).status().expect("failed to run C binary");
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(run_status.success(), "C PII evidence binary exited with status {run_status}");
}

fn build_cdylib() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().unwrap();
    let status = Command::new(env!("CARGO"))
        .current_dir(workspace_root)
        .args(["build", "-p", "turbo", "--lib"])
        .status()
        .expect("failed to invoke cargo build -p turbo --lib");
    assert!(status.success(), "cargo build -p turbo --lib failed");
    let debug_dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    for name in ["libturbo.so", "libturbo.dylib", "turbo.dll"] {
        let candidate = debug_dir.join(name);
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate built turbo cdylib in {}", debug_dir.display());
}

const C_PROGRAM: &str = r#"
#include "turbo/turbo.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int pool_eq(const char *pool, uint32_t off, uint32_t len, const char *want) {
    size_t n = strlen(want);
    return len == n && memcmp(pool + off, want, n) == 0;
}

static int pool_contains(const char *pool, uint64_t pool_len, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0 || pool_len < n) return 0;
    for (uint64_t i = 0; i + n <= pool_len; i++) {
        if (memcmp(pool + i, needle, n) == 0) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    FILE *f = fopen(argv[1], "rb");
    if (!f) return 3;
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    uint8_t *graph = (uint8_t *)malloc((size_t)fsize);
    if (!graph || fread(graph, 1, (size_t)fsize, f) != (size_t)fsize) return 4;
    fclose(f);

    turbo_error err = { .struct_size = sizeof(turbo_error) };
    turbo_analyzer_desc desc = {
        .struct_size = sizeof(turbo_analyzer_desc),
        .flags = TURBO_ANALYZER_PII_KIND_IDS | TURBO_ANALYZER_PII_DETAIL,
    };
    turbo_analyzer *a = NULL;
    turbo_analysis_result *r = NULL;
    if (turbo_analyzer_create(&desc, &a, &err) != TURBO_OK) return 10;
    if (turbo_analyzer_inspect_graph(a, graph, (uint64_t)fsize, &r, &err) != TURBO_OK) return 11;

    const turbo_span *spans = NULL;
    const turbo_pii_span *pii_spans = NULL;
    const turbo_pii_fact *facts = NULL;
    const turbo_pii_candidate *cands = NULL;
    const char *pool = NULL;
    uint32_t span_count = 0, pii_count = 0, fact_count = 0, cand_count = 0;
    uint64_t pool_bytes = 0;

    if (turbo_analysis_result_spans(r, &spans, &span_count, &err) != TURBO_OK) return 12;
    if (turbo_analysis_result_pii_spans(r, &pii_spans, &pii_count, &err) != TURBO_OK) return 13;
    if (turbo_analysis_result_pii_facts(r, &facts, &fact_count, &err) != TURBO_OK) return 14;
    if (turbo_analysis_result_pii_candidates(r, &cands, &cand_count, &err) != TURBO_OK) return 15;
    if (turbo_analysis_result_strings(r, &pool, &pool_bytes, &err) != TURBO_OK) return 16;

    if (span_count != 2 || pii_count != 2 || cand_count != 3) return 20;
    if (pool_contains(pool, pool_bytes, "4532") || pool_contains(pool, pool_bytes, "E-42") ||
        pool_contains(pool, pool_bytes, "PAN")) return 21;

    if (spans[0].token_start != TURBO_PII_KIND_CARD) return 30;
    if (spans[0].classification != TURBO_PII_CONFIDENCE_LABELLED_CHECKSUM) return 31;
    if (!pool_eq(pool, pii_spans[0].detector_id_offset, pii_spans[0].detector_id_len, "turbo-pii:cursor")) return 32;
    if (pii_spans[0].fact_end - pii_spans[0].fact_start != 2) return 33;
    const turbo_pii_fact *f0 = &facts[pii_spans[0].fact_start];
    if (f0[0].kind != TURBO_PII_EVIDENCE_CHECKSUM || f0[0].a != TURBO_PII_CHECKSUM_LUHN || f0[0].b != 1) return 34;
    if (f0[1].kind != TURBO_PII_EVIDENCE_LABEL || f0[1].a != 0 || f0[1].b != 3 || f0[1].c != 3) return 35;
    if (pii_spans[0].score_kind != 0 || !isnan(pii_spans[0].score_value)) return 36;

    if (spans[1].token_start != TURBO_PII_KIND_CUSTOM_BASE) return 40;
    if (!pool_eq(pool, pii_spans[1].detector_id_offset, pii_spans[1].detector_id_len, "acme:emp-id")) return 41;
    if (!pool_eq(pool, pii_spans[1].detector_version_offset, pii_spans[1].detector_version_len, "2.1.0")) return 42;
    const turbo_pii_fact *f1 = &facts[pii_spans[1].fact_start];
    if (f1[0].kind != TURBO_PII_EVIDENCE_RULE || f1[0].a != 7) return 43;
    if (pii_spans[1].score_kind != TURBO_PII_SCORE_PROBABILITY || pii_spans[1].score_calibrated != 1) return 44;
    if (fabsf(pii_spans[1].score_value - 0.9375f) > 1e-6f) return 45;
    if (!pool_eq(pool, pii_spans[1].score_model_id_offset, pii_spans[1].score_model_id_len, "acme:scorer")) return 46;

    if (!pool_eq(pool, cands[0].pii_instance_offset, cands[0].pii_instance_len, "")) return 49;
    if (cands[2].kind_code != TURBO_PII_KIND_CA_SIN) return 50;
    if (cands[2].disposition != TURBO_PII_DISPOSITION_REJECTED_BY_RULE ||
        cands[2].disposition_a != TURBO_PII_REJECT_LABEL_REQUIRED) return 51;

    turbo_analysis_result_release(r);
    turbo_analyzer_release(a);
    free(graph);
    return 0;
}
"#;
