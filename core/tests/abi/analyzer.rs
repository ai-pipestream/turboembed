//! A C program linked against libturbo that calls every analyzer and
//! analysis result function: with no bundle, where the rule tasks run and
//! the head tasks are refused, and with the sealed bundle of heads in
//! testdata, where every task writes its rows and a graph goes out and
//! back. Then a layer range write, and turbo_last_error_message after a
//! failing call and a succeeding one.

use std::process::Command;

use crate::common::*;
use crate::{compile, run};

const PROGRAM: &str = r#"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <turbo/turbo.h>

static turbo_text T(const char *s) { turbo_text t = { s, strlen(s) }; return t; }
static turbo_error err = { sizeof(turbo_error) };

/* A result as one line: each row's layer kind, flags, token range, source
 * bytes and label (- for none), its value and classification when it has
 * them, and p for a probability in 0..=1 or nan for none. */
static int show(const char *task, int32_t rc, const turbo_analysis_result *r) {
    printf("%s %s", task, turbo_status_name(rc));
    if (rc != TURBO_OK) {
        printf(" out %s\n", r ? "set" : "NULL");
        return r != NULL;
    }
    const turbo_span *spans = NULL;
    uint32_t n = 0;
    const char *pool = NULL;
    uint64_t bytes = 0;
    if (turbo_analysis_result_spans(r, &spans, &n, &err) || turbo_analysis_result_strings(r, &pool, &bytes, &err)) {
        printf(" read %s\n", err.message);
        return 1;
    }
    printf(" rows %u", n);
    for (uint32_t i = 0; i < n; i++) {
        const turbo_span *s = &spans[i];
        int has = (s->flags & TURBO_SPAN_LABEL) != 0;
        printf(" [%u %u %u-%u %u-%u %.*s", s->layer_kind, s->flags, s->token_start, s->token_end,
               s->source_byte_start, s->source_byte_end, has ? (int)s->label_len : 1,
               has ? pool + s->label_offset : "-");
        if (s->flags & TURBO_SPAN_VALUE) printf(" =%.*s", (int)s->value_len, pool + s->value_offset);
        if (s->classification) printf(" c%u", s->classification);
        printf(" %s]", isnan(s->prob) ? "nan" : s->prob >= 0.0 && s->prob <= 1.0 ? "p" : "bad");
    }
    printf("\n");
    return 0;
}

/* How many rows of each layer kind a result has. */
static int kinds(const char *what, const turbo_analysis_result *r) {
    const turbo_span *spans = NULL;
    uint32_t n = 0, of[TURBO_ANALYSIS_LAYER_MORPHOLOGY + 1] = { 0 };
    if (turbo_analysis_result_spans(r, &spans, &n, &err)) { printf("%s %s\n", what, err.message); return 1; }
    for (uint32_t i = 0; i < n; i++) {
        if (spans[i].layer_kind > TURBO_ANALYSIS_LAYER_MORPHOLOGY) {
            printf("%s kind %u\n", what, spans[i].layer_kind);
            return 1;
        }
        of[spans[i].layer_kind]++;
    }
    printf("%s rows %u:", what, n);
    for (uint32_t k = TURBO_ANALYSIS_LAYER_SENTENCES; k <= TURBO_ANALYSIS_LAYER_MORPHOLOGY; k++) printf(" %u", of[k]);
    printf("\n");
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    int32_t rc;
    turbo_analysis_result *r = NULL;
    turbo_analyzer *plain = NULL, *heads = NULL;
    /* An out pointer a failing call must clear. */
    turbo_analysis_result *const stale = (turbo_analysis_result *)(void *)&rc;

    if (turbo_analyzer_create(NULL, &plain, &err)) { printf("create %s\n", err.message); return 1; }
    printf("retain same %d, NULL %d\n", turbo_analyzer_retain(plain) == plain, turbo_analyzer_retain(NULL) == NULL);
    turbo_analyzer_release(plain);
    turbo_analyzer_release(NULL);
    const char *text = "Dr. Smith arrived. She waved.";
    rc = turbo_analyzer_detect_sentences(plain, T(text), &r, &err);
    if (show("sentences", rc, r)) return 1;
    turbo_analysis_result_release(r);
    rc = turbo_analyzer_tokenize(plain, T(text), &r, &err);
    if (show("tokens", rc, r)) return 1;
    turbo_analysis_result_release(r);

    /* No bundle: every head task is refused, by name, but no input is no rows. */
    turbo_text words[3] = { T("Ann"), T("met"), T("Bob") };
    turbo_text tags[3] = { T("NOUN"), T("VERB"), T("NOUN") };
    r = stale;
    rc = turbo_analyzer_pos_tag(plain, words, 3, &r, &err);
    if (show("pos_tag", rc, r)) return 1;
    printf("last error names bundle_path %d, field %u\n", strstr(turbo_last_error_message(), "bundle_path") != NULL,
           err.field);
    rc = turbo_analyzer_pos_tag(plain, NULL, 0, &r, &err);
    if (show("pos_tag of none", rc, r)) return 1;
    turbo_analysis_result_release(r);
    printf("last error empty %d\n", turbo_last_error_message()[0] == '\0');
    r = stale;
    rc = turbo_analyzer_morphology(plain, words, 3, &r, &err);
    if (show("morphology", rc, r)) return 1;
    r = stale;
    rc = turbo_analyzer_lemmatize(plain, words, tags, 3, &r, &err);
    if (show("lemmatize", rc, r)) return 1;
    rc = turbo_analyzer_lemmatize(plain, NULL, NULL, 0, &r, &err);
    if (show("lemmatize of none", rc, r)) return 1;
    turbo_analysis_result_release(r);
    r = stale;
    rc = turbo_analyzer_find_tokens(plain, words, 3, &r, &err);
    if (show("find_tokens", rc, r)) return 1;
    r = stale;
    rc = turbo_analyzer_find_text(plain, T(text), &r, &err);
    if (show("find_text", rc, r)) return 1;
    rc = turbo_analyzer_find_text(plain, T(""), &r, &err);
    if (show("find_text of none", rc, r)) return 1;
    turbo_analysis_result_release(r);
    rc = turbo_analyzer_detect_pii(plain, T("Mail alice@example.com today."), &r, &err);
    if (show("detect_pii", rc, r)) return 1;
    turbo_analysis_result_release(r);

    /* The descriptor's rules. */
    turbo_analyzer_desc desc;
    memset(&desc, 0, sizeof desc);
    desc.struct_size = sizeof desc;
    desc.flags = 1;
    rc = turbo_analyzer_create(&desc, &heads, &err);
    printf("flags %s field %u out %s\n", turbo_status_name(rc), err.field, heads ? "set" : "NULL");
    desc.flags = 0;
    desc.pos_model_path = T("en-pos.bin");
    rc = turbo_analyzer_create(&desc, &heads, &err);
    printf("retired %s field %u\n", turbo_status_name(rc), err.field);
    desc.pos_model_path = T("");
    desc.bundle_path = T("/nonexistent/bundle");
    rc = turbo_analyzer_create(&desc, &heads, &err);
    printf("missing bundle %s field %u\n", turbo_status_name(rc), err.field);
    desc.bundle_path = T(argv[1]);
    if (turbo_analyzer_create(&desc, &heads, &err)) { printf("heads %s\n", err.message); return 1; }

    /* The bundle's heads. */
    turbo_text sent[6] = { T("Ann"), T("met"), T("Bob"), T("in"), T("Paris"), T(".") };
    turbo_analysis_result *pos = NULL;
    rc = turbo_analyzer_pos_tag(heads, sent, 6, &pos, &err);
    if (show("heads pos_tag", rc, pos)) return 1;
    const turbo_span *ps = NULL;
    uint32_t pn = 0;
    const char *pool = NULL;
    uint64_t pool_bytes = 0;
    if (turbo_analysis_result_spans(pos, &ps, &pn, &err) ||
        turbo_analysis_result_strings(pos, &pool, &pool_bytes, &err)) {
        printf("pos rows %s\n", err.message); return 1;
    }
    turbo_text sent_tags[6];
    for (uint32_t i = 0; i < pn && i < 6; i++) {
        turbo_text t = { pool + ps[i].label_offset, ps[i].label_len };
        sent_tags[i] = t;
    }
    rc = turbo_analyzer_lemmatize(heads, sent, sent_tags, 6, &r, &err);
    if (show("heads lemmatize", rc, r)) return 1;
    turbo_analysis_result_release(r);
    sent_tags[2] = T("XYZ");
    r = stale;
    rc = turbo_analyzer_lemmatize(heads, sent, sent_tags, 6, &r, &err);
    if (show("heads lemmatize XYZ", rc, r)) return 1;
    printf("last error names tags[2] %d\n", strstr(turbo_last_error_message(), "tags[2]") != NULL);
    const uint8_t *data = NULL;
    uint64_t data_bytes = 0;
    if (turbo_analysis_result_graph(pos, &data, &data_bytes, &err)) { printf("pos graph %s\n", err.message); return 1; }
    printf("pos_tag graph %llu bytes, NULL %d\n", (unsigned long long)data_bytes, data == NULL);
    turbo_analysis_result_release(pos);
    rc = turbo_analyzer_morphology(heads, sent, 6, &r, &err);
    if (show("heads morphology", rc, r)) return 1;
    turbo_analysis_result_release(r);
    rc = turbo_analyzer_find_tokens(heads, sent, 6, &r, &err);
    if (show("heads find_tokens", rc, r)) return 1;
    turbo_analysis_result_release(r);
    const char *story = "Ann met Bob in Paris . Bob slept .";
    rc = turbo_analyzer_find_text(heads, T(story), &r, &err);
    if (show("heads find_text", rc, r)) return 1;
    turbo_analysis_result_release(r);

    /* A graph out with every head layer, and back through the analyzer
     * with no bundle. */
    turbo_analysis_result *graph = NULL, *back = NULL;
    if (turbo_analyzer_analyze_graph(heads, T(story), &graph, &err)) { printf("graph %s\n", err.message); return 1; }
    turbo_analyzer_release(heads);
    if (kinds("graph", graph)) return 1;
    printf("result retain same %d, NULL %d\n", turbo_analysis_result_retain(graph) == graph,
           turbo_analysis_result_retain(NULL) == NULL);
    turbo_analysis_result_release(graph);
    if (turbo_analysis_result_graph(graph, &data, &data_bytes, &err)) { printf("bytes %s\n", err.message); return 1; }
    printf("graph magic %.4s\n", (const char *)data);
    if (turbo_analyzer_inspect_graph(plain, data, data_bytes, &back, &err)) {
        printf("inspect %s\n", err.message); return 1;
    }
    if (kinds("inspected", back)) return 1;
    const uint8_t *again = NULL;
    uint64_t again_bytes = 0;
    if (turbo_analysis_result_graph(back, &again, &again_bytes, &err)) { printf("again %s\n", err.message); return 1; }
    printf("inspected graph same %d\n", again_bytes == data_bytes && memcmp(again, data, data_bytes) == 0);
    turbo_analysis_result_release(back);
    turbo_analysis_result_release(graph);
    turbo_analysis_result_release(NULL);
    r = stale;
    rc = turbo_analyzer_inspect_graph(plain, NULL, 0, &r, &err);
    if (show("inspect NULL", rc, r)) return 1;
    static const uint8_t junk[8] = { 'T', 'G', 'R', 'F', 0, 0, 0, 0 };
    r = stale;
    rc = turbo_analyzer_inspect_graph(plain, junk, sizeof junk, &r, &err);
    if (show("inspect junk", rc, r)) return 1;
    const turbo_span *none = (const turbo_span *)(const void *)junk;
    uint32_t none_count = 7;
    rc = turbo_analysis_result_spans(NULL, &none, &none_count, &err);
    printf("spans of NULL %s, spans NULL %d, count %u\n", turbo_status_name(rc), none == NULL, none_count);
    turbo_analyzer_release(plain);

    /* Each range from each of the tiny encoder's two layers. */
    turbo_runtime *rt = NULL;
    turbo_context *ctx = NULL;
    turbo_model *m = NULL;
    turbo_tokenizer *tok = NULL;
    turbo_session *s = NULL;
    turbo_result *res = NULL;
    if (turbo_runtime_create(NULL, &rt, &err) || turbo_context_create(rt, 0, &ctx, &err) ||
        turbo_model_load(ctx, T(argv[2]), &m, &err) || turbo_tokenizer_create(rt, T(argv[2]), &tok, &err)) {
        printf("model %s\n", err.message); return 1;
    }
    turbo_context_release(ctx);
    turbo_runtime_release(rt);
    turbo_text two[2] = { T("The quick brown fox."), T("reset a password") };
    int32_t ids[2][32], mask[2][32];
    uint32_t lens[2] = { 0, 0 };
    if (turbo_tokenizer_encode(tok, two, 2, NULL, &ids[0][0], &mask[0][0], NULL, 32, lens, &err)) {
        printf("encode %s\n", err.message); return 1;
    }
    turbo_tokenizer_release(tok);
    turbo_token_batch tb = { sizeof(turbo_token_batch), 2, lens[0] > lens[1] ? lens[0] : lens[1], 32,
                             &ids[0][0], &mask[0][0], NULL };
    turbo_session_desc sd = { sizeof(turbo_session_desc), 2, 0, TURBO_PRECISION_EXACT, TURBO_AUTOTUNE_OFF, 0, 0, 4 };
    if (turbo_session_create(m, &sd, &s, &err)) { printf("session %s\n", err.message); return 1; }
    turbo_model_release(m);
    turbo_token_range ranges[2] = { { 0, 0, lens[0] }, { 1, 1, 3 } };
    float plain_rows[2][32], layer_rows[4][32];
    if (turbo_embed_write_ranges(s, &tb, ranges, 2, NULL, &err) || turbo_session_run(s, &res, &err) ||
        turbo_result_read(res, plain_rows, sizeof plain_rows, NULL, &err)) {
        printf("ranges %s\n", err.message); return 1;
    }
    turbo_result_release(res);
    turbo_result_info ri = { sizeof(turbo_result_info) };
    if (turbo_embed_write_layer_ranges(s, &tb, ranges, 2, 2, NULL, &err) || turbo_session_run(s, &res, &err) ||
        turbo_result_get_info(res, &ri, &err) || turbo_result_read(res, layer_rows, sizeof layer_rows, NULL, &err)) {
        printf("layer ranges %s\n", err.message); return 1;
    }
    turbo_result_release(res);
    printf("layer ranges batch %u dim %u, last block the plain write %d, first block another %d\n", ri.batch, ri.dim,
           memcmp(layer_rows[2], plain_rows, sizeof plain_rows) == 0,
           memcmp(layer_rows[0], plain_rows, sizeof plain_rows) != 0);
    rc = turbo_embed_write_layer_ranges(s, &tb, ranges, 2, 3, NULL, &err);
    printf("layers 3 %s field %u, last error names 1 to 2 %d\n", turbo_status_name(rc), err.field,
           strstr(turbo_last_error_message(), "1 to 2") != NULL);
    turbo_session_release(s);
    return 0;
}
"#;

// Flags: 3 view and source range, 7 and a token range, 27 view, source,
// label and value, 44 token range, label and probability, 12 token range
// and label, 47 every range, label and probability. The heads are seeded,
// not trained: their tags are arbitrary, and their lemma trees keep the
// word, drop a final s, or turn met into meet (tests/analysis_bundle.rs).
const WANT: &str = "\
retain same 1, NULL 1
sentences TURBO_OK rows 2 [1 3 0-0 0-18 - nan] [1 3 0-0 19-29 - nan]
tokens TURBO_OK rows 8 [2 7 0-1 0-2 - nan] [2 7 1-2 2-3 - nan] [2 7 2-3 4-9 - nan] [2 7 3-4 10-17 - nan] \
[2 7 4-5 17-18 - nan] [2 7 5-6 19-22 - nan] [2 7 6-7 23-28 - nan] [2 7 7-8 28-29 - nan]
pos_tag TURBO_E_UNSUPPORTED_TASK out NULL
last error names bundle_path 1, field 0
pos_tag of none TURBO_OK rows 0
last error empty 1
morphology TURBO_E_UNSUPPORTED_TASK out NULL
lemmatize TURBO_E_UNSUPPORTED_TASK out NULL
lemmatize of none TURBO_OK rows 0
find_tokens TURBO_E_UNSUPPORTED_TASK out NULL
find_text TURBO_E_UNSUPPORTED_TASK out NULL
find_text of none TURBO_OK rows 0
detect_pii TURBO_OK rows 1 [6 27 0-0 5-22 email =alice@example.com c1 nan]
flags TURBO_E_INVALID_ARGUMENT field 0 out NULL
retired TURBO_E_INVALID_ARGUMENT field 4
missing bundle TURBO_E_BUNDLE_NOT_FOUND field 7
heads pos_tag TURBO_OK rows 6 [3 44 0-1 0-0 PUNCT p] [3 44 1-2 0-0 DET p] [3 44 2-3 0-0 ADJ p] \
[3 44 3-4 0-0 DET p] [3 44 4-5 0-0 ADJ p] [3 44 5-6 0-0 ADJ p]
heads lemmatize TURBO_OK rows 6 [4 12 0-1 0-0 Ann nan] [4 12 1-2 0-0 meet nan] [4 12 2-3 0-0 Bob nan] \
[4 12 3-4 0-0 in nan] [4 12 4-5 0-0 Pari nan] [4 12 5-6 0-0 . nan]
heads lemmatize XYZ TURBO_E_INVALID_ARGUMENT out NULL
last error names tags[2] 1
pos_tag graph 0 bytes, NULL 1
heads morphology TURBO_OK rows 6 [8 12 0-1 0-0 Tense=Past|VerbForm=Fin nan] \
[8 12 1-2 0-0 Tense=Past|VerbForm=Fin nan] [8 12 2-3 0-0 Tense=Past|VerbForm=Fin nan] [8 12 3-4 0-0 _ nan] \
[8 12 4-5 0-0 _ nan] [8 12 5-6 0-0 _ nan]
heads find_tokens TURBO_OK rows 1 [5 44 2-3 0-0 PER p]
heads find_text TURBO_OK rows 1 [5 47 2-3 8-11 PER p]
graph rows 48: 2 9 9 9 1 0 9 9
result retain same 1, NULL 1
graph magic TGRF
inspected rows 48: 2 9 9 9 1 0 9 9
inspected graph same 1
inspect NULL TURBO_E_INVALID_ARGUMENT out NULL
inspect junk TURBO_E_INVALID_ARGUMENT out NULL
spans of NULL TURBO_E_INVALID_HANDLE, spans NULL 1, count 0
layer ranges batch 4 dim 32, last block the plain write 1, first block another 1
layers 3 TURBO_E_INVALID_ARGUMENT field 9, last error names 1 to 2 1
";

#[test]
fn a_c_program_calls_every_analyzer_function() {
    let (d, program) = compile("analyzer", PROGRAM);
    let heads = turbo_pins::require(testdata().join("tiny-bert-heads-bundle"));
    let out = run(Command::new(&program).arg(&heads).arg(tiny_bundle()));
    assert_eq!(out, WANT);
    std::fs::remove_dir_all(d).unwrap();
}
