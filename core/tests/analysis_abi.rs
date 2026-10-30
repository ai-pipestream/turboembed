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
fn configured_dictionary_lemmas_are_exact_and_a_miss_has_no_value() {
    let mut path = std::env::temp_dir();
    path.push(format!("turbo-abi-lemma-{}.tsv", std::process::id()));
    std::fs::write(
        &path,
        "went\tVERB\twalk\nwent\tVERB\tgo\textra\tfields\nthe\tDET\tcanonical-the\nThe\tDET\tunreachable\n",
    )
    .unwrap();
    let path_text = path.to_str().unwrap().to_owned();

    unsafe {
        let mut desc: turbo_analyzer_desc = std::mem::zeroed();
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.lemma_model_path = t(&path_text);
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        let mut err = new_err();
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::OK);

        let words = [t("went"), t("The"), t("zorped")];
        let tags = [t("VERB"), t("DET"), t("NOUN")];
        let mut result: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(
            turbo_analyzer_lemmatize(analyzer, words.as_ptr(), tags.as_ptr(), 3, &mut result, &mut err),
            status::OK
        );
        let (spans, labels) = read_spans(result);
        assert_eq!(labels, ["go", "canonical-the", ""]);
        assert_ne!(spans[0].flags & TURBO_SPAN_LABEL, 0);
        assert_eq!(spans[2].flags & TURBO_SPAN_LABEL, 0, "a dictionary miss gets no synthetic label");

        let mut pool_ptr: *const std::ffi::c_char = std::ptr::null();
        let mut pool_len: u64 = 0;
        assert_eq!(turbo_analysis_result_strings(result, &mut pool_ptr, &mut pool_len, &mut err), status::OK);
        let pool = std::slice::from_raw_parts(pool_ptr.cast::<u8>(), pool_len as usize);
        let values: Vec<Option<&str>> = spans
            .iter()
            .map(|span| {
                if span.flags & TURBO_SPAN_VALUE == 0 {
                    None
                } else {
                    let start = span.value_offset as usize;
                    Some(std::str::from_utf8(&pool[start..start + span.value_len as usize]).unwrap())
                }
            })
            .collect();
        assert_eq!(values, [None, None, None], "lemmas use label; value is reserved for normalized payloads");
        assert!(spans.iter().all(|span| span.flags & TURBO_SPAN_TOKEN_RANGE != 0));
        turbo_analysis_result_release(result);
        turbo_analyzer_release(analyzer);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn chunk_text_and_arrow_c_data_interface_export_round_trip() {
    unsafe {
        let mut err = new_err();
        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(std::ptr::null(), &mut analyzer, &mut err), status::OK);

        let text = "First sentence has five words. Second sentence also has six words here. Third sentence ends it.";
        let mut chunk_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_chunk_text(analyzer, t(text), 6, 12, &mut chunk_res, &mut err), status::OK);
        let (chunk_spans, chunk_labels) = read_spans(chunk_res);
        assert!(chunk_spans.len() >= 2);
        assert_eq!(chunk_labels[0], "document_start");
        for sp in &chunk_spans {
            assert_eq!(sp.layer_kind, TURBO_ANALYSIS_LAYER_CHUNKS);
            assert!(sp.classification > 0 && sp.classification <= 12);
            assert_eq!(sp.classification, sp.token_end - sp.token_start);
        }

        // Export chunks to Arrow C Data Interface (ArrowSchema + ArrowArray)
        let mut schema: ArrowSchema = std::mem::zeroed();
        let mut array: ArrowArray = std::mem::zeroed();
        assert_eq!(
            turbo_analysis_result_to_arrow(chunk_res, TURBO_ANALYSIS_LAYER_CHUNKS, &mut schema, &mut array, &mut err),
            status::OK
        );
        assert_eq!(CStr::from_ptr(schema.format).to_str().unwrap(), "+s");
        assert_eq!(schema.n_children, 24);
        assert_eq!(array.length, chunk_spans.len() as i64);
        assert_eq!(array.n_children, 24);
        assert_eq!(array.n_buffers, 1);

        // Verify child column 0 (layer_kind: uint32 "I") and child column 21 (label: utf8 "u")
        let children_schemas = std::slice::from_raw_parts(schema.children, schema.n_children as usize);
        let children_arrays = std::slice::from_raw_parts(array.children, array.n_children as usize);
        assert_eq!(CStr::from_ptr((*children_schemas[0]).name).to_str().unwrap(), "layer_kind");
        assert_eq!(CStr::from_ptr((*children_schemas[0]).format).to_str().unwrap(), "I");
        let lk_bufs = std::slice::from_raw_parts((*children_arrays[0]).buffers, 2);
        let lk_vals = std::slice::from_raw_parts(lk_bufs[1].cast::<u32>(), chunk_spans.len());
        assert!(lk_vals.iter().all(|&v| v == TURBO_ANALYSIS_LAYER_CHUNKS));

        assert_eq!(CStr::from_ptr((*children_schemas[21]).name).to_str().unwrap(), "label");
        assert_eq!(CStr::from_ptr((*children_schemas[21]).format).to_str().unwrap(), "u");
        let lbl_bufs = std::slice::from_raw_parts((*children_arrays[21]).buffers, 3);
        let lbl_offs = std::slice::from_raw_parts(lbl_bufs[1].cast::<i32>(), chunk_spans.len() + 1);
        let lbl_data = std::slice::from_raw_parts(lbl_bufs[2].cast::<u8>(), lbl_offs[chunk_spans.len()] as usize);
        let first_lbl = std::str::from_utf8(&lbl_data[lbl_offs[0] as usize..lbl_offs[1] as usize]).unwrap();
        assert_eq!(first_lbl, "document_start");

        // Release idempotency
        (schema.release.unwrap())(&mut schema);
        assert!(schema.release.is_none());
        (array.release.unwrap())(&mut array);
        assert!(array.release.is_none());

        // Export via ArrowArrayStream
        let mut stream: ArrowArrayStream = std::mem::zeroed();
        assert_eq!(
            turbo_analysis_result_to_arrow_stream(chunk_res, TURBO_ANALYSIS_LAYER_CHUNKS, &mut stream, &mut err),
            status::OK
        );
        let mut stream_schema: ArrowSchema = std::mem::zeroed();
        assert_eq!((stream.get_schema.unwrap())(&mut stream, &mut stream_schema), 0);
        assert_eq!(CStr::from_ptr(stream_schema.format).to_str().unwrap(), "+s");
        (stream_schema.release.unwrap())(&mut stream_schema);

        let mut batch1: ArrowArray = std::mem::zeroed();
        assert_eq!((stream.get_next.unwrap())(&mut stream, &mut batch1), 0);
        assert!(batch1.release.is_some());
        assert_eq!(batch1.length, chunk_spans.len() as i64);
        (batch1.release.unwrap())(&mut batch1);

        let mut batch2: ArrowArray = std::mem::zeroed();
        assert_eq!((stream.get_next.unwrap())(&mut stream, &mut batch2), 0);
        assert!(batch2.release.is_none(), "end of stream must have release == NULL");
        (stream.release.unwrap())(&mut stream);
        assert!(stream.release.is_none());

        turbo_analysis_result_release(chunk_res);
        turbo_analyzer_release(analyzer);
    }
}
