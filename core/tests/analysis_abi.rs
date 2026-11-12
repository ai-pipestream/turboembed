//! Integration tests for the in-process analysis C ABI (`turbo_analyzer`
//! and `turbo_analysis_result`) and `turbo_last_error_message`.

mod cdylib;

use std::ffi::CStr;
use turbo_scratch::Scratch;

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

        let mut morph_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_morphology(analyzer, words.as_ptr(), words.len() as u32, &mut morph_res, &mut err),
            status::UNSUPPORTED_TASK
        );
        assert!(morph_res.is_null());

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

        // Any other size, and an unknown flag, are refused.
        desc.struct_size = 112;
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::INVALID_STRUCT_SIZE);
        desc.struct_size = size_of::<turbo_analyzer_desc>() as u32;
        desc.bundle_path = t("");
        desc.flags = 1 << 31;
        assert_eq!(turbo_analyzer_create(&desc, &mut analyzer, &mut err), status::INVALID_ARGUMENT);
    }
}

#[repr(C)]
struct LegacyAnalyzerDescV1 {
    struct_size: u32,
    flags: u32,
    language: turbo_text,
    sent_model_path: turbo_text,
    tok_model_path: turbo_text,
    pos_model_path: turbo_text,
    lemma_model_path: turbo_text,
    ner_model_path: turbo_text,
}

const _: () = assert!(size_of::<LegacyAnalyzerDescV1>() == TURBO_ANALYZER_DESC_SIZE_V1);
const _: () = assert!(align_of::<LegacyAnalyzerDescV1>() == align_of::<turbo_analyzer_desc>());

#[test]
fn exact_legacy_sized_allocation_and_short_invalid_size_controls_work() {
    use std::alloc::{Layout, alloc_zeroed, dealloc};

    unsafe {
        let mut err = new_err();
        let layout = Layout::from_size_align(TURBO_ANALYZER_DESC_SIZE_V1, align_of::<turbo_analyzer_desc>()).unwrap();
        let raw = alloc_zeroed(layout).cast::<LegacyAnalyzerDescV1>();
        assert!(!raw.is_null());
        (*raw).struct_size = TURBO_ANALYZER_DESC_SIZE_V1 as u32;
        (*raw).language = t("en");

        let mut analyzer: *mut turbo_analyzer = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_create(raw.cast::<turbo_analyzer_desc>(), &mut analyzer, &mut err), status::OK);
        assert!(!analyzer.is_null());

        let sample = "Dr. Smith arrived. She waved.";
        let mut sent_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_detect_sentences(analyzer, t(sample), &mut sent_res, &mut err), status::OK);
        let (sent_spans, _) = read_spans(sent_res);
        assert_eq!(sent_spans.len(), 2);
        turbo_analysis_result_release(sent_res);

        let mut tok_res: *mut turbo_analysis_result = std::ptr::null_mut();
        assert_eq!(turbo_analyzer_tokenize(analyzer, t(sample), &mut tok_res, &mut err), status::OK);
        let (tok_spans, _) = read_spans(tok_res);
        assert_eq!(tok_spans.len(), 8);
        turbo_analysis_result_release(tok_res);

        let words = [t("Dr."), t("Smith")];
        let mut pos_res: *mut turbo_analysis_result = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_pos_tag(analyzer, words.as_ptr(), words.len() as u32, &mut pos_res, &mut err),
            status::UNSUPPORTED_TASK
        );
        assert!(pos_res.is_null());
        turbo_analyzer_release(analyzer);

        // Retired model fields on an exact 104-byte allocation are still refused by field number.
        (*raw).ner_model_path = t("en-ner.bin");
        let mut rejected: *mut turbo_analyzer = std::ptr::dangling_mut();
        assert_eq!(
            turbo_analyzer_create(raw.cast::<turbo_analyzer_desc>(), &mut rejected, &mut err),
            status::INVALID_ARGUMENT
        );
        assert!(rejected.is_null());
        assert_eq!(err.field, 6);
        dealloc(raw.cast::<u8>(), layout);

        // A 4-byte allocation holding only an unsupported struct_size is refused
        // before reading flags or any other field.
        let short_layout = Layout::from_size_align(size_of::<u32>(), align_of::<turbo_analyzer_desc>()).unwrap();
        let short = alloc_zeroed(short_layout).cast::<u32>();
        for bad_size in [0u32, 4, 96, 112, 128] {
            *short = bad_size;
            let mut out: *mut turbo_analyzer = std::ptr::dangling_mut();
            assert_eq!(
                turbo_analyzer_create(short.cast::<turbo_analyzer_desc>(), &mut out, &mut err),
                status::INVALID_STRUCT_SIZE
            );
            assert!(out.is_null());
        }
        dealloc(short.cast::<u8>(), short_layout);
    }
}

#[cfg(not(miri))]
const FROZEN_LEGACY_C_CALLER: &str = r#"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define TURBO_OK 0
#define TURBO_E_INVALID_ARGUMENT 256
#define TURBO_E_INVALID_STRUCT_SIZE 257
#define TURBO_E_UNSUPPORTED_TASK 514
#define TURBO_ERROR_MESSAGE_LEN 496
#define TURBO_ANALYSIS_LAYER_SENTENCES 1
#define TURBO_ANALYSIS_LAYER_TOKENS 2
typedef struct turbo_analyzer turbo_analyzer;
typedef struct turbo_analysis_result turbo_analysis_result;
typedef struct turbo_text { const char *ptr; uint64_t len; } turbo_text;
typedef struct turbo_error {
    uint32_t struct_size; int32_t code; uint32_t field; char message[TURBO_ERROR_MESSAGE_LEN];
} turbo_error;
/* Frozen legacy turbo_analyzer_desc declaration from before bundle_path. */
typedef struct turbo_analyzer_desc {
    uint32_t struct_size, flags;
    turbo_text language, sent_model_path, tok_model_path, pos_model_path, lemma_model_path, ner_model_path;
} turbo_analyzer_desc;
_Static_assert(sizeof(turbo_analyzer_desc) == 8 + 6 * sizeof(turbo_text), "frozen v1 descriptor size");
_Static_assert(offsetof(turbo_analyzer_desc, ner_model_path) + sizeof(turbo_text) == sizeof(turbo_analyzer_desc), "v1 end");
typedef struct turbo_span {
    uint32_t layer_kind, flags, item_index, fragment_index, view_id, classification;
    uint32_t view_byte_start, view_byte_end, view_utf16_start, view_utf16_end, view_codepoint_start, view_codepoint_end;
    uint32_t source_byte_start, source_byte_end, source_utf16_start, source_utf16_end;
    uint32_t source_codepoint_start, source_codepoint_end, token_start, token_end;
    uint32_t layer_offset, layer_len, instance_offset, instance_len, label_offset, label_len, value_offset, value_len;
    double prob;
} turbo_span;
int32_t turbo_analyzer_create(const turbo_analyzer_desc *desc, turbo_analyzer **out, turbo_error *err);
void turbo_analyzer_release(turbo_analyzer *a);
int32_t turbo_analyzer_detect_sentences(turbo_analyzer *a, turbo_text t, turbo_analysis_result **out, turbo_error *err);
int32_t turbo_analyzer_tokenize(turbo_analyzer *a, turbo_text t, turbo_analysis_result **out, turbo_error *err);
int32_t turbo_analyzer_pos_tag(turbo_analyzer *a, const turbo_text *toks, uint32_t n, turbo_analysis_result **out, turbo_error *err);
int32_t turbo_analysis_result_spans(const turbo_analysis_result *r, const turbo_span **spans, uint32_t *count, turbo_error *err);
void turbo_analysis_result_release(turbo_analysis_result *r);
static turbo_text T(const char *s) { turbo_text t = { s, (uint64_t)strlen(s) }; return t; }
int main(void) {
    turbo_error err = { sizeof(turbo_error), 0, 0, { 0 } };
    turbo_analyzer_desc *desc = (turbo_analyzer_desc *)calloc(1, sizeof(turbo_analyzer_desc));
    if (!desc) return 2;
    desc->struct_size = (uint32_t)sizeof(turbo_analyzer_desc);
    desc->language = T("en");
    turbo_analyzer *a = NULL;
    int32_t rc = turbo_analyzer_create(desc, &a, &err);
    free(desc);
    if (rc != TURBO_OK || !a) { printf("create failed: %d %s\n", rc, err.message); return 1; }
    const char *sample = "Dr. Smith arrived. She waved.";
    turbo_analysis_result *sent = NULL, *tok = NULL, *pos = NULL;
    const turbo_span *spans = NULL;
    uint32_t sent_count = 0, tok_count = 0;
    if (turbo_analyzer_detect_sentences(a, T(sample), &sent, &err) != TURBO_OK) return 1;
    if (turbo_analysis_result_spans(sent, &spans, &sent_count, &err) != TURBO_OK) return 1;
    if (sent_count != 2 || spans[0].layer_kind != TURBO_ANALYSIS_LAYER_SENTENCES) return 1;
    turbo_analysis_result_release(sent);
    if (turbo_analyzer_tokenize(a, T(sample), &tok, &err) != TURBO_OK) return 1;
    if (turbo_analysis_result_spans(tok, &spans, &tok_count, &err) != TURBO_OK) return 1;
    if (tok_count != 8 || spans[0].layer_kind != TURBO_ANALYSIS_LAYER_TOKENS) return 1;
    turbo_analysis_result_release(tok);
    turbo_text word = T("Dr.");
    if (turbo_analyzer_pos_tag(a, &word, 1, &pos, &err) != TURBO_E_UNSUPPORTED_TASK || pos != NULL) return 1;
    turbo_analyzer_release(a);
    turbo_analyzer_desc retired;
    memset(&retired, 0, sizeof(retired));
    retired.struct_size = (uint32_t)sizeof(turbo_analyzer_desc);
    retired.lemma_model_path = T("en-lemma.bin");
    if (turbo_analyzer_create(&retired, &a, &err) != TURBO_E_INVALID_ARGUMENT || err.field != 5) return 1;
    _Alignas(turbo_analyzer_desc) uint32_t bad_size = 112;
    if (turbo_analyzer_create((const turbo_analyzer_desc *)&bad_size, &a, &err) != TURBO_E_INVALID_STRUCT_SIZE) return 1;
    printf("size=%zu sentences=%u tokens=%u\n", sizeof(turbo_analyzer_desc), sent_count, tok_count);
    return 0;
}
"#;

#[cfg(not(miri))]
#[test]
fn c_caller_compiled_with_frozen_legacy_descriptor_links_and_runs_rule_tasks() {
    use std::process::Command;

    let lib_dir = cdylib::build();

    let scratch = Scratch::at(std::env::temp_dir().join(format!("turbo-legacy-desc-c-{}", std::process::id())));
    std::fs::create_dir_all(&scratch).unwrap();
    let (src, bin) = (scratch.join("legacy_caller.c"), scratch.join("legacy_caller"));
    std::fs::write(&src, FROZEN_LEGACY_C_CALLER).unwrap();
    let cc = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-o"])
        .args([&bin, &src])
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-lturbo")
        .output()
        .unwrap();
    assert!(cc.status.success(), "cc failed:\n{}", String::from_utf8_lossy(&cc.stderr));
    let run = Command::new(&bin).output().unwrap();
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stdout));
    let stdout = String::from_utf8(run.stdout).unwrap().replace("\r\n", "\n");
    assert_eq!(stdout, format!("size={TURBO_ANALYZER_DESC_SIZE_V1} sentences=2 tokens=8\n"));
    std::fs::remove_dir_all(&scratch).unwrap();
}
