//! A PII row's numeric kind over the C ABI (#1940): with
//! `TURBO_ANALYZER_PII_KIND_IDS` every PII row carries its kind under
//! `TURBO_SPAN_PII_KIND`, the registry id for a built-in kind and
//! `TURBO_PII_KIND_CUSTOM_BASE` plus the custom-table index for a custom
//! one, on the analyze path and the inspect path alike. Without the flag
//! the rows are the ones a caller always saw.

use turbo::*;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Key, Provenance, Span, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{ChecksumAlgorithm, CustomKinds, Pii, PiiEvidence, PiiKind, PiiMention};

fn t(s: &str) -> turbo_text {
    turbo_text { ptr: s.as_ptr().cast(), len: s.len() as u64 }
}

fn new_err() -> turbo_error {
    turbo_error {
        struct_size: size_of::<turbo_error>() as u32,
        code: -1,
        field: 0,
        message: [0; TURBO_ERROR_MESSAGE_LEN],
    }
}

/// An analyzer with no bundle and `flags`.
fn analyzer(flags: u32) -> *mut turbo_analyzer {
    let desc = turbo_analyzer_desc {
        struct_size: size_of::<turbo_analyzer_desc>() as u32,
        flags,
        language: t(""),
        sent_model_path: t(""),
        tok_model_path: t(""),
        pos_model_path: t(""),
        lemma_model_path: t(""),
        ner_model_path: t(""),
        bundle_path: t(""),
    };
    let mut out: *mut turbo_analyzer = std::ptr::null_mut();
    let mut err = new_err();
    assert_eq!(unsafe { turbo_analyzer_create(&desc, &mut out, &mut err) }, status::OK);
    out
}

/// One PII row as a caller reads it: the row and its label.
struct Row {
    span: turbo_span,
    label: String,
}

/// The PII rows of `r`, then `r` released.
fn pii_rows(r: *mut turbo_analysis_result) -> Vec<Row> {
    let mut spans_ptr: *const turbo_span = std::ptr::null();
    let mut count = 0u32;
    let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
    let mut pool_len = 0u64;
    let mut err = new_err();
    unsafe {
        assert_eq!(turbo_analysis_result_spans(r, &mut spans_ptr, &mut count, &mut err), status::OK);
        assert_eq!(turbo_analysis_result_strings(r, &mut pool_ptr, &mut pool_len, &mut err), status::OK);
        let spans = if count == 0 { &[][..] } else { std::slice::from_raw_parts(spans_ptr, count as usize) };
        let pool = std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize);
        let rows = spans
            .iter()
            .filter(|s| s.layer_kind == TURBO_ANALYSIS_LAYER_PII)
            .map(|s| {
                let at = s.label_offset as usize;
                let label = std::str::from_utf8(&pool[at..at + s.label_len as usize]).unwrap().to_owned();
                Row { span: *s, label }
            })
            .collect();
        turbo_analysis_result_release(r);
        rows
    }
}

fn detect(a: *mut turbo_analyzer, text: &str) -> Vec<Row> {
    let mut r: *mut turbo_analysis_result = std::ptr::null_mut();
    let mut err = new_err();
    assert_eq!(unsafe { turbo_analyzer_detect_pii(a, t(text), &mut r, &mut err) }, status::OK);
    pii_rows(r)
}

fn analyze(a: *mut turbo_analyzer, text: &str) -> Vec<Row> {
    let mut r: *mut turbo_analysis_result = std::ptr::null_mut();
    let mut err = new_err();
    assert_eq!(unsafe { turbo_analyzer_analyze_graph(a, t(text), &mut r, &mut err) }, status::OK);
    pii_rows(r)
}

fn inspect(a: *mut turbo_analyzer, graph: &[u8]) -> Vec<Row> {
    let mut r: *mut turbo_analysis_result = std::ptr::null_mut();
    let mut err = new_err();
    let rc = unsafe { turbo_analyzer_inspect_graph(a, graph.as_ptr(), graph.len() as u64, &mut r, &mut err) };
    assert_eq!(rc, status::OK);
    pii_rows(r)
}

/// The numeric kind of a row that carries one.
fn kind(row: &Row) -> u32 {
    assert_eq!(row.span.flags & (TURBO_SPAN_PII_KIND | TURBO_SPAN_TOKEN_RANGE), TURBO_SPAN_PII_KIND, "{}", row.label);
    assert_eq!(row.span.token_end, 0, "{}", row.label);
    row.span.token_start
}

/// The row with any numeric kind taken off, as a caller without the flag
/// reads it, and its probability as bits so that NaN compares equal.
fn without_kind(row: &Row) -> (turbo_span, u64, &str) {
    let mut span = row.span;
    if span.flags & TURBO_SPAN_PII_KIND != 0 {
        span.flags &= !TURBO_SPAN_PII_KIND;
        span.token_start = 0;
    }
    let prob = span.prob.to_bits();
    span.prob = 0.0;
    (span, prob, &row.label)
}

/// A TGRF graph of one mention of every built-in kind, in registry order,
/// and then one of each of two custom kinds.
fn registry_graph() -> (Vec<u8>, Vec<PiiKind>, CustomKinds) {
    let text = "x".repeat(PiiKind::TABLE.len() + 2);
    let custom = CustomKinds::from_names(["account-number", "employee-id"]).unwrap();
    let mut pii = Pii::with_custom_kinds(ViewId::SOURCE, custom.clone());
    let mut kinds: Vec<PiiKind> = PiiKind::builtins().collect();
    kinds.push(custom.find("account-number").unwrap());
    kinds.push(custom.find("employee-id").unwrap());
    let detector = pii.detector("test", "1").unwrap();
    let luhn = PiiEvidence::Checksum { algorithm: ChecksumAlgorithm::Luhn, passed: true };
    let evidence: [&[PiiEvidence]; 2] = [&[PiiEvidence::Shape], &[luhn]];
    for (i, &kind) in kinds.iter().enumerate() {
        let span = Span::new(i as u32, i as u32 + 1);
        pii.push(PiiMention { span, kind, detector, evidence: evidence[i % 2], score: None, normalized: "v" });
    }
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, pii, Provenance::by("test:pii")).unwrap();
    (export(&doc, &codecs()).unwrap().to_bytes().unwrap(), kinds, custom)
}

#[test]
fn every_registry_kind_reads_back_as_its_id_on_an_inspected_graph() {
    let (graph, kinds, custom) = registry_graph();
    let plain = analyzer(0);
    let ids = analyzer(TURBO_ANALYZER_PII_KIND_IDS);
    let before = inspect(plain, &graph);
    let rows = inspect(ids, &graph);
    assert_eq!(rows.len(), kinds.len());
    for (row, &kind_of) in rows.iter().zip(&kinds) {
        match kind_of.builtin_id() {
            Some(id) => {
                assert_eq!(kind(row), u32::from(id), "{}", row.label);
                assert_eq!(Some(row.label.as_str()), kind_of.builtin_name());
            }
            None => {
                assert!(kind(row) >= TURBO_PII_KIND_CUSTOM_BASE, "{}: never a built-in id", row.label);
                assert!(PiiKind::TABLE.iter().all(|r| u32::from(r.id) != kind(row)));
                let index = (kind(row) - TURBO_PII_KIND_CUSTOM_BASE) as usize;
                assert_eq!(custom.names().nth(index), Some(row.label.as_str()), "the index names the row");
                assert_eq!(custom.from_code(kind(row)), Ok(kind_of), "the codec reads the same kind");
            }
        }
    }
    let customs: Vec<u32> = rows[PiiKind::TABLE.len()..].iter().map(kind).collect();
    assert_eq!(customs, [TURBO_PII_KIND_CUSTOM_BASE, TURBO_PII_KIND_CUSTOM_BASE + 1]);
    // Without the flag no row carries a kind, and with it nothing else moves.
    assert_eq!(before.len(), rows.len());
    for (b, row) in before.iter().zip(&rows) {
        assert_eq!(b.span.flags & TURBO_SPAN_PII_KIND, 0);
        assert_eq!(b.span.token_start, 0);
        assert_eq!(without_kind(b), without_kind(row));
    }
    unsafe {
        turbo_analyzer_release(plain);
        turbo_analyzer_release(ids);
    }
}

#[test]
fn analyzed_rows_carry_the_registry_id_of_their_label() {
    let text = "Mail alice@example.com, card 4111 1111 1111 1111, IBAN DE89 3704 0044 0532 0130 00, \
                from 192.168.1.20.";
    let plain = analyzer(TURBO_ANALYZER_PII_VALUES);
    let ids = analyzer(TURBO_ANALYZER_PII_VALUES | TURBO_ANALYZER_PII_KIND_IDS);
    for (path, before, rows) in [
        ("detect_pii", detect(plain, text), detect(ids, text)),
        ("analyze_graph", analyze(plain, text), analyze(ids, text)),
    ] {
        let mut seen: Vec<u32> = rows.iter().map(kind).collect();
        seen.sort_unstable();
        seen.dedup();
        assert!(seen.len() >= 4, "{path}: kinds {seen:?}");
        for row in &rows {
            let id = PiiKind::parse(&row.label).and_then(PiiKind::builtin_id).expect("a built-in kind");
            assert_eq!(kind(row), u32::from(id), "{path}: {}", row.label);
        }
        assert_eq!(before.len(), rows.len(), "{path}");
        for (b, row) in before.iter().zip(&rows) {
            assert_eq!(without_kind(b), without_kind(row), "{path}: label and classification unchanged");
        }
    }
    unsafe {
        turbo_analyzer_release(plain);
        turbo_analyzer_release(ids);
    }
}
