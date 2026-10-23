//! A C program linked against libturbo, from the runtime to the embedding.

use std::path::Path;
use std::process::Command;

use crate::common::*;
use crate::{include, run, scratch};
use turbo::*;

const PROGRAM: &str = r#"
#include <stdio.h>
#include <string.h>
#include <turbo/turbo.h>

static turbo_text T(const char *s) { turbo_text t = { s, strlen(s) }; return t; }

int main(int argc, char **argv) {
    turbo_error err = { sizeof(turbo_error) };
    turbo_runtime *rt = NULL;
    turbo_tokenizer *tok = NULL;
    int32_t rc;
    if (argc != 5) return 2;
    if (turbo_runtime_create(NULL, &rt, &err)) { printf("runtime %s\n", err.message); return 1; }
    uint32_t n = 0, pick = 99;
    turbo_device_info info = { sizeof(turbo_device_info) };
    turbo_capability cap = { sizeof(turbo_capability) };
    if (turbo_runtime_device_count(rt, &n, &err)) { printf("count %s\n", err.message); return 1; }
    if (turbo_runtime_device_info(rt, 0, &info, &err)) { printf("info %s\n", err.message); return 1; }
    if (turbo_runtime_capability(rt, 0, TURBO_TASK_EMBED, TURBO_PRECISION_MODEL, &cap, &err)) {
        printf("capability %s\n", err.message); return 1;
    }
    printf("devices %u, device 0 is %s kind %u, embed status %u\n", n, info.backend, info.kind, cap.status);
    rc = turbo_runtime_select(rt, TURBO_TASK_EMBED, &pick, NULL, 0, &err);
    printf("select %s %u\n", turbo_status_name(rc), pick);
    turbo_context *ctx = NULL;
    turbo_buffer *buf = NULL, *wrapped = NULL;
    uint32_t dev = 99;
    if (turbo_context_create(rt, 0, &ctx, &err)) { printf("context %s\n", err.message); return 1; }
    if (turbo_context_device(ctx, &dev, &err)) { printf("context device %s\n", err.message); return 1; }
    turbo_buffer_desc bd = { sizeof(turbo_buffer_desc), TURBO_PLACE_HOST, TURBO_DTYPE_F32, 2, { 3, 4 }, 0 };
    if (turbo_buffer_alloc(ctx, &bd, &buf, &err)) { printf("alloc %s\n", err.message); return 1; }
    turbo_buffer_desc got = { sizeof(turbo_buffer_desc) };
    void *host = NULL;
    if (turbo_buffer_get_desc(buf, &got, &err) || turbo_buffer_host_ptr(buf, &host, &err)) {
        printf("buffer %s\n", err.message); return 1;
    }
    printf("context on device %u, buffer %llu bytes, aligned %d\n", dev, (unsigned long long)got.bytes,
           (int)((uintptr_t)host % 64 == 0));
    int32_t stack[8] = { 0 };
    turbo_buffer_desc sd = { sizeof(turbo_buffer_desc), TURBO_PLACE_HOST, TURBO_DTYPE_I32, 1, { 6, 0 }, 24 };
    turbo_native_handle nh = { sizeof(turbo_native_handle), TURBO_HANDLE_HOST_PTR, (uint64_t)(uintptr_t)stack, 0, 8 };
    if (turbo_buffer_import(ctx, &sd, &nh, &wrapped, &err)) { printf("import %s\n", err.message); return 1; }
    if (turbo_buffer_host_ptr(wrapped, &host, &err)) { printf("import host %s\n", err.message); return 1; }
    ((int32_t *)host)[1] = 7;
    turbo_native_handle ex = { sizeof(turbo_native_handle) };
    if (turbo_buffer_export(wrapped, TURBO_HANDLE_HOST_PTR, &ex, &err)) { printf("export %s\n", err.message); return 1; }
    printf("import at +%d, stack[3] %d, export same %d\n", (int)((char *)host - (char *)stack), stack[3],
           (int)(ex.handle == (uint64_t)(uintptr_t)host && ex.offset == 0));
    bd.placement = TURBO_PLACE_DEVICE;
    rc = turbo_buffer_alloc(ctx, &bd, &buf, &err);
    printf("device placement %s\n", turbo_status_name(rc));
    turbo_model *model = NULL, *tiny = NULL;
    if (turbo_model_load(ctx, T(argv[3]), &model, &err)) { printf("model %s\n", err.message); return 1; }
    if (turbo_model_load(ctx, T(argv[4]), &tiny, &err)) { printf("tiny model %s\n", err.message); return 1; }
    turbo_context_release(ctx); /* the buffers and the model keep it */
    turbo_buffer_release(wrapped);
    turbo_buffer_release(buf);
    turbo_buffer_release(NULL);
    turbo_context_release(NULL);
    rc = turbo_tokenizer_create(rt, T("/nonexistent/bundle"), &tok, &err);
    printf("missing %s %d\n", turbo_status_name(rc), err.code);
    if (turbo_tokenizer_create(rt, T(argv[1]), &tok, &err)) { printf("create %s\n", err.message); return 1; }
    turbo_tokenizer *tinytok = NULL;
    if (turbo_tokenizer_create(rt, T(argv[4]), &tinytok, &err)) { printf("tiny tok %s\n", err.message); return 1; }
    turbo_runtime_release(rt); /* the tokenizer and the model keep what they need */
    turbo_model_info mi = { sizeof(turbo_model_info) };
    if (turbo_model_get_info(model, &mi, &err)) { printf("model info %s\n", err.message); return 1; }
    printf("model task %u dim %u pooling %u normalize %u max_seq %u max_batch %u dtype %u\n", mi.task, mi.dim,
           mi.pooling, mi.normalize, mi.max_seq, mi.max_batch, mi.dtype);
    printf("%s revision %s\nartifact %s\n", mi.model_id, mi.revision, mi.artifact_sha256);
    printf("output_dims %u:", mi.output_dims_count);
    for (uint32_t i = 0; i < TURBO_OUTPUT_DIMS_MAX; i++) printf(" %u", mi.output_dims[i]);
    printf("\n");
    turbo_model_release(model);
    turbo_model_release(NULL);
    /* The text comes in a file: argv is not UTF-8 everywhere. */
    static char input[1024];
    FILE *in = fopen(argv[2], "rb");
    if (!in) { printf("text %s\n", argv[2]); return 1; }
    size_t input_len = fread(input, 1, sizeof(input) - 1, in);
    fclose(in);
    turbo_text text = { input, input_len };
    int32_t ids[64], mask[64];
    uint32_t len = 0;
    if (turbo_tokenizer_encode(tok, &text, 1, NULL, ids, mask, NULL, 64, &len, &err)) {
        printf("encode %s\n", err.message); return 1;
    }
    for (uint32_t i = 0; i < len; i++) printf("%d%s", ids[i], i + 1 < len ? " " : "\n");
    uint32_t spans[64][2];
    int32_t sids[64];
    if (turbo_tokenizer_encode_spans(tok, &text, 1, NULL, sids, mask, NULL, &spans[0][0], 64, &len, &err)) {
        printf("encode spans %s\n", err.message); return 1;
    }
    printf("spans same ids %d:", memcmp(ids, sids, len * sizeof(int32_t)) == 0);
    for (uint32_t i = 0; i < len; i++) printf(" %u-%u", spans[i][0], spans[i][1]);
    printf(", padding %u-%u\n", spans[len][0], spans[len][1]);
    char decoded[256];
    uint64_t dlen = 0;
    if (turbo_tokenizer_decode(tok, ids, len, 1, decoded, sizeof decoded, &dlen, &err)) {
        printf("decode %s\n", err.message); return 1;
    }
    char small[4] = "abc";
    rc = turbo_tokenizer_decode(tok, ids, len, 1, small, sizeof small, &dlen, &err);
    printf("decoded %s, %s needs %llu, left %s\n", decoded, turbo_status_name(rc), (unsigned long long)dlen, small);
    turbo_tokenizer_release(tok);
    turbo_tokenizer_release(NULL);

    turbo_session *s = NULL;
    turbo_session_desc sd2 = { sizeof(turbo_session_desc), 4, 0, TURBO_PRECISION_EXACT };
    if (turbo_session_create(tiny, &sd2, &s, &err)) { printf("session %s\n", err.message); return 1; }
    turbo_model_release(tiny); /* the session keeps it */
    turbo_session_info si = { sizeof(turbo_session_info) };
    if (turbo_session_get_info(s, &si, &err)) { printf("session info %s\n", err.message); return 1; }
    printf("session max_batch %u max_seq %u precision %u compute %u\n", si.max_batch, si.max_seq, si.precision,
           si.compute_dtype);
    turbo_result *res = NULL;
    rc = turbo_session_run(s, &res, &err);
    printf("run before write %s\n", turbo_status_name(rc));
    turbo_text two[2] = { T("The quick brown fox jumps over the lazy dog."), T("reset a password") };
    turbo_embed_options eo = { sizeof(turbo_embed_options), 0, 0, TURBO_PROMPT_QUERY, 0, 0, 0 };
    if (turbo_embed_write_text(s, two, 2, &eo, &err)) { printf("write %s\n", err.message); return 1; }
    if (turbo_session_run(s, &res, &err)) { printf("run %s\n", err.message); return 1; }
    rc = turbo_embed_write_text(s, two, 1, NULL, &err);
    printf("write while held %s\n", turbo_status_name(rc));
    float vec[2][32];
    uint64_t got_bytes = 0;
    if (turbo_result_read(res, vec, sizeof(vec), &got_bytes, &err)) { printf("read %s\n", err.message); return 1; }
    turbo_result_info ri = { sizeof(turbo_result_info) };
    if (turbo_result_get_info(res, &ri, &err)) { printf("result info %s\n", err.message); return 1; }
    printf("result task %u batch %u dim %u dtype %u compute %u placement %u bytes %llu read %llu\n", ri.task,
           ri.batch, ri.dim, ri.dtype, ri.compute_dtype, ri.placement, (unsigned long long)ri.bytes,
           (unsigned long long)got_bytes);
    printf("h2d %llu d2h %llu host_allocs %llu device_allocs %llu backend %s stages %u:",
           (unsigned long long)ri.h2d_bytes, (unsigned long long)ri.d2h_bytes, (unsigned long long)ri.host_allocs,
           (unsigned long long)ri.device_allocs, ri.backend, ri.stage_count);
    for (uint32_t i = 0; i < ri.stage_count; i++) printf(" %u", ri.stage[i]);
    printf("\n");
    for (int r = 0; r < 2; r++) {
        double norm = 0;
        for (int i = 0; i < 32; i++) norm += (double)vec[r][i] * vec[r][i];
        printf("row %d: %.4f %.4f %.4f %.4f norm %.6f\n", r, vec[r][0], vec[r][1], vec[r][2], vec[r][3], norm);
    }
    turbo_result_release(res);

    /* The same two texts, tokenized by the caller and written as tokens. */
    int32_t tids[2][64], tmask[2][64];
    uint32_t tlen[2] = { 0, 0 };
    turbo_encode_options qo = { sizeof(turbo_encode_options), 0, 0, 0, TURBO_PROMPT_QUERY };
    if (turbo_tokenizer_encode(tinytok, two, 2, &qo, &tids[0][0], &tmask[0][0], NULL, 64, tlen, &err)) {
        printf("encode two %s\n", err.message); return 1;
    }
    turbo_tokenizer_release(tinytok);
    uint32_t seq = tlen[0] > tlen[1] ? tlen[0] : tlen[1];
    turbo_token_batch tb = { sizeof(turbo_token_batch), 2, seq, 64, &tids[0][0], &tmask[0][0], NULL };
    printf("tokens %u and %u, seq %u, row_stride 64\n", tlen[0], tlen[1], seq);
    if (turbo_embed_write_tokens(s, &tb, NULL, &err)) { printf("write tokens %s\n", err.message); return 1; }
    if (turbo_session_run(s, &res, &err)) { printf("run tokens %s\n", err.message); return 1; }
    float tvec[2][32];
    if (turbo_result_read(res, tvec, sizeof(tvec), NULL, &err)) { printf("read tokens %s\n", err.message); return 1; }
    for (int r = 0; r < 2; r++)
        printf("tokens row %d: %.4f %.4f %.4f %.4f\n", r, tvec[r][0], tvec[r][1], tvec[r][2], tvec[r][3]);
    printf("tokens same as text %d\n", (int)(memcmp(tvec, vec, sizeof(vec)) == 0));
    turbo_buffer *out = NULL;
    void *where = NULL;
    if (turbo_result_buffer(res, &out, &err) || turbo_buffer_host_ptr(out, &where, &err)) {
        printf("result buffer %s\n", err.message); return 1;
    }
    turbo_result_release(res);
    turbo_session_release(s); /* the buffer keeps the result, and the result the session */
    printf("buffer same %d\n", (int)(memcmp(where, tvec, sizeof(tvec)) == 0));
    turbo_buffer_release(out);
    turbo_result_release(NULL);
    turbo_session_release(NULL);
    turbo_runtime_release(NULL);
    printf("%s\n", turbo_version());
    return 0;
}
"#;

#[test]
fn a_c_program_loads_a_model_tokenizes_and_embeds() {
    // target/<profile>/deps/abi-* -> target/<profile>/libturbo.{so,dylib}
    let lib_dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    // `cargo test` builds the rlib the tests link, not the cdylib; build
    // it here, with the same profile and target directory, so the program
    // links the library as it is now.
    let profile = lib_dir.file_name().unwrap().to_str().unwrap();
    let profile = match profile {
        "debug" => "dev",
        p => p,
    };
    run(Command::new(env!("CARGO"))
        .args(["build", "--lib", "--profile", profile, "--manifest-path"])
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
        .arg("--target-dir")
        .arg(lib_dir.parent().unwrap()));
    let lib = format!("{}turbo{}", std::env::consts::DLL_PREFIX, std::env::consts::DLL_SUFFIX);
    assert!(lib_dir.join(&lib).exists(), "{lib} is not in {}", lib_dir.display());
    let d = scratch("program");
    std::fs::write(d.join("p.c"), PROGRAM).unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(d.join("p"))
        .arg(d.join("p.c"))
        .arg("-I")
        .arg(include())
        .arg("-L")
        .arg(&lib_dir)
        .arg(format!("-Wl,-rpath,{}", lib_dir.display()))
        .arg("-lturbo"));

    let f = Fixture::standard("c-program");
    f.write();
    let mut model = Fixture::model("c-program-model");
    model.manifest["embed"]["output_dims"] = serde_json::json!([4, 2]);
    model.write();
    let text = "Café naïve RÉSUMÉ, 东京 🙂";
    let tiny = tiny_bundle();
    std::fs::write(d.join("text.txt"), text).unwrap();
    let out = run(Command::new(d.join("p")).arg(&f.dir).arg(d.join("text.txt")).arg(&model.dir).arg(&tiny));
    // The same two vectors through the library from Rust.
    let l = Loaded::load(&tiny).unwrap();
    let s = Session::create(l.m, Some(&session_desc(4, 0, TURBO_PRECISION_EXACT))).unwrap();
    let o = turbo_embed_options { prompt_role: TURBO_PROMPT_QUERY, ..embed_options() };
    let two = ["The quick brown fox jumps over the lazy dog.", "reset a password"];
    let rows = s.embed(&two, Some(&o)).unwrap();
    let tok = Tok::create(&tiny).unwrap();
    let q = options(0, 0, 0, TURBO_PROMPT_QUERY);
    let lens: Vec<usize> = two.iter().map(|t| tok.row(t, Some(&q)).unwrap().len()).collect();
    let tokens: String = rows
        .iter()
        .enumerate()
        .map(|(r, v)| format!("tokens row {r}: {:.4} {:.4} {:.4} {:.4}\n", v[0], v[1], v[2], v[3]))
        .collect();
    let vectors: String = rows
        .iter()
        .enumerate()
        .map(|(r, v)| {
            let norm: f64 = v.iter().map(|x| *x as f64 * *x as f64).sum();
            format!("row {r}: {:.4} {:.4} {:.4} {:.4} norm {norm:.6}\n", v[0], v[1], v[2], v[3])
        })
        .collect();
    let ids: Vec<String> = upstream_ids(&upstream(), text).iter().map(i32::to_string).collect();
    let up_ids: Vec<u32> = upstream().encode(text, true).unwrap().get_ids().to_vec();
    let decoded = upstream().decode(&up_ids, true).unwrap();
    let spans: String =
        upstream().encode(text, true).unwrap().get_offsets().iter().map(|(a, b)| format!(" {a}-{b}")).collect();
    let want = format!(
        "devices 1, device 0 is cpu kind 1, embed status 1\nselect TURBO_E_DEVICE_NOT_FOUND 99\n\
         context on device 0, buffer 48 bytes, aligned 1\nimport at +8, stack[3] 7, export same 1\n\
         device placement TURBO_E_UNSUPPORTED\n\
         missing TURBO_E_BUNDLE_NOT_FOUND {}\n\
         model task 1 dim 8 pooling 1 normalize 2 max_seq 256 max_batch 64 dtype 12\n\
         sentence-transformers/all-MiniLM-L6-v2 revision 3\nartifact {}\n\
         output_dims 2: 2 4 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n{}\nspans same ids 1:{spans}, padding 0-0\n\
         decoded {decoded}, TURBO_E_CAPACITY needs {}, left abc\n\
         session max_batch 4 max_seq 64 precision 2 compute 12\nrun before write TURBO_E_INVALID_STATE\n\
         write while held TURBO_E_BUSY\n\
         result task 1 batch 2 dim 32 dtype 12 compute 12 placement 1 bytes 256 read 256\n\
         h2d 0 d2h 256 host_allocs 0 device_allocs 0 backend cpu stages 7: 1 0 1 1 1 1 0\n\
         {vectors}tokens {} and {}, seq {}, row_stride 64\n{tokens}tokens same as text 1\n\
         buffer same 1\n0.1.0 cpu\n",
        turbo::status::BUNDLE_NOT_FOUND,
        model.sha256("weights/model.safetensors"),
        ids.join(" "),
        decoded.len(),
        lens[0],
        lens[1],
        lens[0].max(lens[1]),
    );
    assert_eq!(out, want);
    std::fs::remove_dir_all(d).unwrap();
}
