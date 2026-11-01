//! Integration tests for the in-process analysis C ABI (`turbo_analyzer`
//! and `turbo_analysis_result`) and `turbo_last_error_message`.

use std::ffi::CStr;

use turbo::*;

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

unsafe fn read_spans(r: *const turbo_analysis_result) -> (Vec<turbo_span>, Vec<String>) {
    let mut spans_ptr: *const turbo_span = std::ptr::null();
    let mut count: u32 = 0;
    let mut err = new_err();
    assert_eq!(unsafe { turbo_analysis_result_spans(r, &mut spans_ptr, &mut count, &mut err) }, status::OK);

    let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
    let mut pool_len: u64 = 0;
    assert_eq!(unsafe { turbo_analysis_result_strings(r, &mut pool_ptr, &mut pool_len, &mut err) }, status::OK);

    let spans =
        if count == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(spans_ptr, count as usize) }.to_vec() };
    let pool = if pool_len == 0 {
        &[][..]
    } else {
        unsafe { std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize) }
    };
    let labels = spans
        .iter()
        .map(|s| {
            let start = s.label_offset as usize;
            let end = start + s.label_len as usize;
            std::str::from_utf8(&pool[start..end]).unwrap().to_owned()
        })
        .collect();
    (spans, labels)
}

#[test]
fn analyzer_pipeline_and_utf16_emoji_offsets_round_trip() {
    unsafe {
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        let mut err = new_err();
        assert_eq!(turbo_analyzer_create(std::ptr::null(), &mut analyzer, &mut err), status::OK);
        assert!(!analyzer.is_null());

        let text = "Hello 🚀 Café 東京! Dr. John Smith visited New York and emailed alice@example.com.";
        let utf16_units: Vec<u16> = text.encode_utf16().collect();

        // 1. Sentence detection
        let mut sent_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_detect_sentences(analyzer, t(text), &mut sent_res, &mut err), status::OK);
        let (sent_spans, sent_labels) = read_spans(sent_res);
        assert_eq!(sent_spans.len(), 2);
        assert!(sent_labels.iter().all(String::is_empty), "covered text is not copied as a label");
        for sp in &sent_spans {
            assert_eq!(sp.layer_kind, TURBO_ANALYSIS_LAYER_SENTENCES);
            assert_eq!(sp.flags, TURBO_SPAN_VIEW_RANGE | TURBO_SPAN_SOURCE_RANGE);
            assert!(sp.prob.is_nan());
            let covered = &text[sp.source_byte_start as usize..sp.source_byte_end as usize];
            let from_utf16 =
                String::from_utf16(&utf16_units[sp.source_utf16_start as usize..sp.source_utf16_end as usize]).unwrap();
            assert_eq!(covered, from_utf16);
            assert_eq!(sp.view_byte_start, sp.source_byte_start);
            assert_eq!(sp.view_byte_end, sp.source_byte_end);
        }
        turbo_analysis_result_release(sent_res);

        // 2. Tokenization with astral emoji & CJK UTF-16 offset verification
        let mut tok_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_tokenize(analyzer, t(text), &mut tok_res, &mut err), status::OK);
        let (tok_spans, tok_labels) = read_spans(tok_res);
        assert!(tok_labels.iter().all(String::is_empty), "covered text is not copied as a label");
        let covered: Vec<&str> =
            tok_spans.iter().map(|sp| &text[sp.source_byte_start as usize..sp.source_byte_end as usize]).collect();
        assert!(covered.iter().any(|l| l == &"Café"));
        assert!(covered.iter().any(|l| l == &"John"));
        for (idx, sp) in tok_spans.iter().enumerate() {
            assert_eq!(sp.layer_kind, TURBO_ANALYSIS_LAYER_TOKENS);
            assert_eq!(sp.item_index, idx as u32);
            assert_eq!((sp.token_start, sp.token_end), (idx as u32, idx as u32 + 1));
            let from_utf16 =
                String::from_utf16(&utf16_units[sp.source_utf16_start as usize..sp.source_utf16_end as usize]).unwrap();
            assert_eq!(covered[idx], from_utf16);
        }
        turbo_analysis_result_release(tok_res);

        // 3. Unconfigured models are unsupported, and stale outputs are cleared.
        let words = [t("The"), t("rivers"), t("went"), t("fast")];
        let mut pos_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_pos_tag(analyzer, words.as_ptr(), words.len() as u32, &mut pos_res, &mut err),
            status::UNSUPPORTED_TASK
        );
        assert!(pos_res.is_null(), "stale result pointer survives an unsupported call");

        let tags = [t("DET"), t("NOUN"), t("VERB"), t("ADV")];
        let mut lemma_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_lemmatize(
                analyzer,
                words.as_ptr(),
                tags.as_ptr(),
                words.len() as u32,
                &mut lemma_res,
                &mut err,
            ),
            status::UNSUPPORTED_TASK
        );
        assert!(lemma_res.is_null());

        // 4. Unconfigured NER never degrades into dictionary or pattern output;
        // PII remains available but only through its separate entry point and layer.
        let ner_tokens = [t("John"), t("Smith"), t("visited"), t("New"), t("York")];
        let mut ner_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_find_tokens(analyzer, ner_tokens.as_ptr(), ner_tokens.len() as u32, &mut ner_res, &mut err),
            status::UNSUPPORTED_TASK
        );
        assert!(ner_res.is_null());
        let mut text_ner_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(turbo_analyzer_find_text(analyzer, t(text), &mut text_ner_res, &mut err), status::UNSUPPORTED_TASK);
        assert!(text_ner_res.is_null());

        let mut pii_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_detect_pii(analyzer, t(text), &mut pii_res, &mut err), status::OK);
        let (pii_spans, pii_labels) = read_spans(pii_res);
        assert!(pii_spans.iter().all(|s| s.layer_kind == TURBO_ANALYSIS_LAYER_PII));
        assert!(pii_labels.iter().any(|l| l == "email"));
        assert!(pii_spans.iter().zip(&pii_labels).any(|(s, l)| {
            l == "email" && &text[s.source_byte_start as usize..s.source_byte_end as usize] == "alice@example.com"
        }));
        turbo_analysis_result_release(pii_res);

        // 5. Full binary DocumentGraph ("TGRF") export and inspection
        let mut graph_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_analyze_graph(analyzer, t(text), &mut graph_res, &mut err), status::OK);
        let mut graph_ptr: *const u8 = std::ptr::null();
        let mut graph_len: u64 = 0;
        assert_eq!(turbo_analysis_result_graph(graph_res, &mut graph_ptr, &mut graph_len, &mut err), status::OK);
        assert!(graph_len > 8);
        let graph_slice = std::slice::from_raw_parts(graph_ptr, graph_len as usize);
        assert_eq!(&graph_slice[..4], b"TGRF");

        let mut inspect_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_inspect_graph(analyzer, graph_ptr, graph_len, &mut inspect_res, &mut err),
            status::OK
        );
        let (inspect_spans, inspect_labels) = read_spans(inspect_res);
        assert!(!inspect_spans.is_empty());
        assert!(inspect_spans.iter().any(|s| s.layer_kind == TURBO_ANALYSIS_LAYER_SENTENCES));
        assert!(inspect_spans.iter().any(|s| s.layer_kind == TURBO_ANALYSIS_LAYER_TOKENS));
        assert!(inspect_labels.iter().any(|l| l == "email"));
        turbo_analysis_result_release(inspect_res);
        turbo_analysis_result_release(graph_res);

        // Release idempotency & NULL safety
        turbo_analysis_result_release(std::ptr::null_mut());
        turbo_analyzer_release(analyzer);
        turbo_analyzer_release(std::ptr::null_mut());
    }
}

#[test]
fn last_error_message_tracks_thread_local_status() {
    unsafe {
        let mut out: *mut turbo_analyzer = std::ptr::null_mut();
        let mut bad_desc: turbo_analyzer_desc = std::mem::zeroed();
        bad_desc.struct_size = 4;
        let code = turbo_analyzer_create(&bad_desc, &mut out, std::ptr::null_mut());
        assert_eq!(code, status::INVALID_STRUCT_SIZE);
        let msg = CStr::from_ptr(turbo_last_error_message()).to_str().unwrap();
        assert!(msg.contains("turbo_analyzer_desc.struct_size"), "got: {msg}");

        // Successful call clears last error message to ""
        assert_eq!(turbo_analyzer_create(std::ptr::null(), &mut out, std::ptr::null_mut()), status::OK);
        let cleared = CStr::from_ptr(turbo_last_error_message()).to_str().unwrap();
        assert_eq!(cleared, "");
        turbo_analyzer_release(out);
    }
}

#[test]
fn retained_handles_remain_usable_and_null_accessors_clear_stale_outputs() {
    unsafe {
        let mut err = new_err();
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(std::ptr::null(), &mut analyzer, &mut err), status::OK);
        assert!(analyzer == turbo_analyzer_retain(analyzer));
        turbo_analyzer_release(analyzer);

        let mut result: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_tokenize(analyzer, t("Retain me."), &mut result, &mut err), status::OK);
        assert!(result == turbo_analysis_result_retain(result));
        turbo_analysis_result_release(result);
        let (spans, _) = read_spans(result);
        assert_eq!(spans.len(), 3);
        turbo_analysis_result_release(result);
        turbo_analyzer_release(analyzer);

        assert!(turbo_analyzer_retain(std::ptr::null_mut()).is_null());
        assert!(turbo_analysis_result_retain(std::ptr::null_mut()).is_null());
        let mut spans_ptr: *const turbo_span = std::ptr::dangling();
        let mut count = 77u32;
        assert_eq!(
            turbo_analysis_result_spans(std::ptr::null(), &mut spans_ptr, &mut count, &mut err),
            status::INVALID_HANDLE
        );
        assert!(spans_ptr.is_null() && count == 0);

        let mut pool_ptr: *const std::ffi::c_char = std::ptr::dangling();
        let mut pool_len = 77u64;
        assert_eq!(
            turbo_analysis_result_strings(std::ptr::null(), &mut pool_ptr, &mut pool_len, &mut err),
            status::INVALID_HANDLE
        );
        assert!(pool_ptr.is_null() && pool_len == 0);

        let mut graph_ptr: *const u8 = std::ptr::dangling();
        let mut graph_len = 77u64;
        assert_eq!(
            turbo_analysis_result_graph(std::ptr::null(), &mut graph_ptr, &mut graph_len, &mut err),
            status::INVALID_HANDLE
        );
        assert!(graph_ptr.is_null() && graph_len == 0);
    }
}

#[test]
fn retired_model_paths_are_refused_by_field_and_the_earlier_size_names_no_bundle() {
    unsafe {
        let mut err = new_err();
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        // Each retired field, set to any non-empty value, is refused by
        // its number, whatever it points at.
        type Field = fn(&mut turbo_analyzer_desc) -> &mut turbo_text;
        let retired: [(Field, u32); 5] = [
            (|d| &mut d.sent_model_path, 2),
            (|d| &mut d.tok_model_path, 3),
            (|d| &mut d.pos_model_path, 4),
            (|d| &mut d.lemma_model_path, 5),
            (|d| &mut d.ner_model_path, 6),
        ];
        for (field, number) in retired {
            let mut desc: turbo_analyzer_desc = std::mem::zeroed();
            desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
            *field(&mut desc) = t("en-sent.bin");
            assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::INVALID_ARGUMENT);
            assert_eq!(err.field, number);
            let msg = CStr::from_ptr(err.message.as_ptr()).to_str().unwrap();
            assert!(msg.contains("is retired") && msg.contains("bundle_path"), "field {number}: {msg}");
            assert!(analyzer.is_null());
        }

        // The size before bundle_path was appended is still accepted, and
        // then names no bundle: what lies past it is not read.
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = TURBO_ANALYZER_DESC_SIZE_V1 as u32;
        desc.bundle_path = t("testdata/does-not-exist");
        assert_eq!(TURBO_ANALYZER_DESC_SIZE_V1, 104);
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::OK);
        let words = [t("The"), t("river")];
        let mut res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_pos_tag(analyzer, words.as_ptr(), 2, &mut res, &mut err), status::UNSUPPORTED_TASK);
        let msg = CStr::from_ptr(err.message.as_ptr()).to_str().unwrap();
        assert!(msg.contains("bundle_path"), "got: {msg}");
        turbo_analyzer_release(analyzer);

        // Any other size, and a flag, are refused.
        desc.struct_size = 112;
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::INVALID_STRUCT_SIZE);
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.bundle_path = t("");
        desc.flags = 1;
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::INVALID_ARGUMENT);
    }
}
