//! A C program that asks for int8 and ubinary vectors through
//! turbo_embed_options fields 7 and 8, for pooled rows and for ranges, at
//! the model's dim and at a declared narrower width, and computes the
//! codes it expects from the f32 run itself, with turbo.h's formulas.

use std::process::Command;

use crate::common::*;
use crate::{compile, run};

const PROGRAM: &str = r#"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <turbo/turbo.h>

#define ROWS 3
#define DIM 8

static turbo_text T(const char *s) { turbo_text t = { s, strlen(s) }; return t; }

static turbo_session *s;
static turbo_token_batch tb;
static turbo_token_range ranges[ROWS];
static turbo_error err = { sizeof(turbo_error) };

/* One write and run: the result's info and its bytes in out. */
static int embed(int use_ranges, const turbo_embed_options *o, turbo_result_info *ri, void *out) {
    int32_t rc = use_ranges ? turbo_embed_write_ranges(s, &tb, ranges, ROWS, o, &err)
                            : turbo_embed_write_tokens(s, &tb, o, &err);
    turbo_result *res = NULL;
    if (rc || turbo_session_run(s, &res, &err)) { printf("embed %s\n", err.message); return 1; }
    ri->struct_size = sizeof(turbo_result_info);
    if (turbo_result_get_info(res, ri, &err) || turbo_result_read(res, out, ri->bytes, NULL, &err)) {
        printf("read %s\n", err.message); return 1;
    }
    turbo_result_release(res);
    return 0;
}

static int check(int use_ranges, uint32_t width) {
    float v[ROWS][DIM], lohi[2 * DIM];
    int8_t codes[ROWS][DIM], want8[ROWS][DIM];
    uint8_t bits[ROWS][DIM / 8], wantb[ROWS][DIM / 8];
    turbo_result_info ri;
    turbo_embed_options o = { sizeof(turbo_embed_options), 0, 0, 0, TURBO_NORMALIZE_L2, 0, width, 0, NULL };
    if (embed(use_ranges, &o, &ri, v)) return 1;
    printf("%s %u: f32 dtype %u dim %u bytes %llu;", use_ranges ? "ranges" : "pooled", width, ri.dtype, ri.dim,
           (unsigned long long)ri.bytes);
    /* The rows are packed at width; the ranges are the rows' own extremes. */
    const float *x = &v[0][0];
    for (uint32_t d = 0; d < width; d++) {
        lohi[d] = lohi[width + d] = x[d];
        for (int r = 1; r < ROWS; r++) {
            float e = x[r * width + d];
            if (e < lohi[d]) lohi[d] = e;
            if (e > lohi[width + d]) lohi[width + d] = e;
        }
    }
    int8_t *w8 = &want8[0][0];
    uint8_t *wb = &wantb[0][0];
    memset(wantb, 0, sizeof wantb);
    for (int r = 0; r < ROWS; r++)
        for (uint32_t d = 0; d < width; d++) {
            float lo = lohi[d], step = (lohi[width + d] - lo) / 255.0f;
            if (step == 0.0f) step = 1.0f;
            float q = floorf((x[r * width + d] - lo) / step);
            q = q < 0.0f ? 0.0f : q > 255.0f ? 255.0f : q;
            w8[r * width + d] = (int8_t)((int)q - 128);
            if (x[r * width + d] > 0.0f) wb[r * ((width + 7) / 8) + d / 8] |= (uint8_t)(0x80 >> (d % 8));
        }
    o.quantization = TURBO_QUANTIZE_INT8;
    o.int8_ranges = lohi;
    if (embed(use_ranges, &o, &ri, codes)) return 1;
    printf(" int8 dtype %u dim %u bytes %llu same %d;", ri.dtype, ri.dim, (unsigned long long)ri.bytes,
           memcmp(codes, want8, ri.bytes) == 0);
    o.quantization = TURBO_QUANTIZE_UBINARY;
    o.int8_ranges = NULL;
    if (embed(use_ranges, &o, &ri, bits)) return 1;
    printf(" ubinary dtype %u dim %u bytes %llu same %d\n", ri.dtype, ri.dim, (unsigned long long)ri.bytes,
           memcmp(bits, wantb, ri.bytes) == 0);
    return 0;
}

int main(int argc, char **argv) {
    turbo_runtime *rt = NULL;
    turbo_context *ctx = NULL;
    turbo_model *m = NULL;
    turbo_tokenizer *tok = NULL;
    if (argc != 2) return 2;
    if (turbo_runtime_create(NULL, &rt, &err) || turbo_context_create(rt, 0, &ctx, &err) ||
        turbo_model_load(ctx, T(argv[1]), &m, &err) || turbo_tokenizer_create(rt, T(argv[1]), &tok, &err)) {
        printf("load %s\n", err.message); return 1;
    }
    turbo_session_desc sd = { sizeof(turbo_session_desc), ROWS, 0, TURBO_PRECISION_EXACT, 0, 0, 0, ROWS };
    if (turbo_session_create(m, &sd, &s, &err)) { printf("session %s\n", err.message); return 1; }
    turbo_text texts[ROWS] = { T("The quick brown fox jumps over the lazy dog."), T("reset a password"),
                               T("a b c d") };
    static int32_t ids[ROWS][64], mask[ROWS][64];
    uint32_t len[ROWS];
    if (turbo_tokenizer_encode(tok, texts, ROWS, NULL, &ids[0][0], &mask[0][0], NULL, 64, len, &err)) {
        printf("encode %s\n", err.message); return 1;
    }
    uint32_t seq = 0;
    for (int r = 0; r < ROWS; r++) {
        if (len[r] > seq) seq = len[r];
        turbo_token_range g = { (uint32_t)r, 1, len[r] - 1 };
        ranges[r] = g;
    }
    turbo_token_batch b = { sizeof(turbo_token_batch), ROWS, seq, 64, &ids[0][0], &mask[0][0], NULL };
    tb = b;
    if (check(0, DIM) || check(0, 4) || check(1, DIM) || check(1, 4)) return 1;
    turbo_embed_options bad = { sizeof(turbo_embed_options), 0, 0, 0, 0, 0, 0, TURBO_QUANTIZE_INT8, NULL };
    int32_t rc = turbo_embed_write_tokens(s, &tb, &bad, &err);
    printf("int8 without ranges %s field %u\n", turbo_status_name(rc), err.field);
    bad.quantization = TURBO_QUANTIZE_UBINARY;
    bad.output_dim = 3;
    rc = turbo_embed_write_tokens(s, &tb, &bad, &err);
    printf("undeclared width %s field %u\n", turbo_status_name(rc), err.field);
    turbo_session_release(s);
    turbo_tokenizer_release(tok);
    turbo_model_release(m);
    turbo_context_release(ctx);
    turbo_runtime_release(rt);
    return 0;
}
"#;

#[test]
fn a_c_program_gets_the_int8_and_ubinary_codes_it_computes_itself() {
    let (d, program) = compile("quantized", PROGRAM);
    let mut model = Fixture::model("c-quantized-model");
    model.manifest["embed"]["output_dims"] = serde_json::json!([4]);
    model.write();
    let out = run(Command::new(&program).arg(&model.dir));
    let line = |what: &str, w: u32| {
        format!(
            "{what} {w}: f32 dtype 12 dim {w} bytes {}; int8 dtype 6 dim {w} bytes {} same 1; \
             ubinary dtype 7 dim {w} bytes 3 same 1\n",
            3 * w * 4,
            3 * w,
        )
    };
    let want = [line("pooled", 8), line("pooled", 4), line("ranges", 8), line("ranges", 4)].concat()
        + "int8 without ranges TURBO_E_INVALID_ARGUMENT field 8\n\
           undeclared width TURBO_E_UNSUPPORTED_OPTION field 6\n";
    assert_eq!(out, want);
    std::fs::remove_dir_all(d).unwrap();
}
