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
        for (sp, label) in sent_spans.iter().zip(&sent_labels) {
            assert_eq!(&text[sp.byte_start as usize..sp.byte_end as usize], label);
            let from_utf16 = String::from_utf16(&utf16_units[sp.utf16_start as usize..sp.utf16_end as usize]).unwrap();
            assert_eq!(&from_utf16, label);
        }
        turbo_analysis_result_release(sent_res);

        // 2. Tokenization with astral emoji & CJK UTF-16 offset verification
        let mut tok_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_tokenize(analyzer, t(text), &mut tok_res, &mut err), status::OK);
        let (tok_spans, tok_labels) = read_spans(tok_res);
        assert!(tok_labels.iter().any(|l| l == "Café"));
        assert!(tok_labels.iter().any(|l| l == "John"));
        for (sp, label) in tok_spans.iter().zip(&tok_labels) {
            assert_eq!(&text[sp.byte_start as usize..sp.byte_end as usize], label);
            let from_utf16 = String::from_utf16(&utf16_units[sp.utf16_start as usize..sp.utf16_end as usize]).unwrap();
            assert_eq!(&from_utf16, label);
        }
        turbo_analysis_result_release(tok_res);

        // 3. POS tagging & lemmatization
        let words = [t("The"), t("rivers"), t("went"), t("fast")];
        let mut pos_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_pos_tag(analyzer, words.as_ptr(), words.len() as u32, &mut pos_res, &mut err),
            status::OK
        );
        let (pos_spans, pos_labels) = read_spans(pos_res);
        assert_eq!(pos_spans.len(), 4);
        assert!(pos_spans.iter().all(|s| s.prob > 0.0));
        turbo_analysis_result_release(pos_res);

        let tags = [t("DET"), t("NOUN"), t("VERB"), t("ADV")];
        let mut lemma_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_lemmatize(
                analyzer,
                words.as_ptr(),
                tags.as_ptr(),
                words.len() as u32,
                &mut lemma_res,
                &mut err,
            ),
            status::OK
        );
        let (_, lemma_labels) = read_spans(lemma_res);
        assert_eq!(lemma_labels[0], "the");
        assert_eq!(lemma_labels[1], "river");
        assert_eq!(lemma_labels[2], "go");
        let _ = pos_labels;
        turbo_analysis_result_release(lemma_res);

        // 4. Token-level and text-level NER / PII finding
        let ner_tokens = [t("John"), t("Smith"), t("visited"), t("New"), t("York")];
        let mut ner_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_find_tokens(analyzer, ner_tokens.as_ptr(), ner_tokens.len() as u32, &mut ner_res, &mut err),
            status::OK
        );
        let (ner_spans, ner_labels) = read_spans(ner_res);
        assert!(
            ner_spans.iter().zip(&ner_labels).any(|(s, l)| s.token_start == 0 && s.token_end == 2 && l == "person")
        );
        assert!(
            ner_spans.iter().zip(&ner_labels).any(|(s, l)| s.token_start == 3 && s.token_end == 5 && l == "location")
        );
        turbo_analysis_result_release(ner_res);

        let mut text_ner_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_find_text(analyzer, t(text), &mut text_ner_res, &mut err), status::OK);
        let (text_ner_spans, text_ner_labels) = read_spans(text_ner_res);
        assert!(text_ner_labels.iter().any(|l| l == "person"));
        assert!(text_ner_labels.iter().any(|l| l == "location"));
        assert!(text_ner_labels.iter().any(|l| l == "email"));
        for sp in &text_ner_spans {
            let covered_utf8 = &text[sp.byte_start as usize..sp.byte_end as usize];
            let covered_utf16 =
                String::from_utf16(&utf16_units[sp.utf16_start as usize..sp.utf16_end as usize]).unwrap();
            assert_eq!(covered_utf8, covered_utf16);
        }
        turbo_analysis_result_release(text_ner_res);

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
            turbo_analyzer_inspect_graph(analyzer, t(text), graph_ptr, graph_len, &mut inspect_res, &mut err),
            status::OK
        );
        let (inspect_spans, inspect_labels) = read_spans(inspect_res);
        assert!(!inspect_spans.is_empty());
        assert!(inspect_labels.iter().any(|l| l == "sentence"));
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
