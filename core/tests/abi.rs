//! The header against the library: turbo.h compiles standalone as C11 and
//! C++17, the Rust mirrors of its structs have the C compiler's layout, and
//! a C program linked against libturbo makes a context and buffers on the
//! CPU, loads a model there, and gets the upstream ids.

mod common;

use std::mem::offset_of;
use std::path::{Path, PathBuf};
use std::process::Command;

use common::*;
use turbo::*;

fn include() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../include")
}

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-abi-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn run(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("{cmd:?}: {e} (a C compiler is required)"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn the_header_compiles_standalone() {
    let d = scratch("standalone");
    let c = d.join("h.c");
    std::fs::write(&c, "#include <turbo/turbo.h>\n#include <turbo/turbo_backend.h>\n").unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-fsyntax-only", "-I"])
        .arg(include())
        .arg(&c));
    run(Command::new("c++")
        .args(["-std=c++17", "-Wall", "-Wextra", "-Werror", "-pedantic", "-fsyntax-only", "-x", "c++", "-I"])
        .arg(include())
        .arg(&c));
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn struct_layouts_match_the_header() {
    let fields: &[(&str, &str, usize)] = &[
        ("turbo_text", "ptr", offset_of!(turbo_text, ptr)),
        ("turbo_text", "len", offset_of!(turbo_text, len)),
        ("turbo_error", "code", offset_of!(turbo_error, code)),
        ("turbo_error", "field", offset_of!(turbo_error, field)),
        ("turbo_error", "message", offset_of!(turbo_error, message)),
        ("turbo_runtime_desc", "reserved", offset_of!(turbo_runtime_desc, reserved)),
        ("turbo_runtime_desc", "log", offset_of!(turbo_runtime_desc, log)),
        ("turbo_runtime_desc", "log_user_data", offset_of!(turbo_runtime_desc, log_user_data)),
        ("turbo_tokenizer_info", "vocab_size", offset_of!(turbo_tokenizer_info, vocab_size)),
        ("turbo_tokenizer_info", "max_seq", offset_of!(turbo_tokenizer_info, max_seq)),
        ("turbo_tokenizer_info", "specials_per_sequence", offset_of!(turbo_tokenizer_info, specials_per_sequence)),
        ("turbo_tokenizer_info", "pad_id", offset_of!(turbo_tokenizer_info, pad_id)),
        ("turbo_tokenizer_info", "bos_id", offset_of!(turbo_tokenizer_info, bos_id)),
        ("turbo_tokenizer_info", "eos_id", offset_of!(turbo_tokenizer_info, eos_id)),
        ("turbo_tokenizer_info", "unk_id", offset_of!(turbo_tokenizer_info, unk_id)),
        ("turbo_tokenizer_info", "kind", offset_of!(turbo_tokenizer_info, kind)),
        ("turbo_tokenizer_info", "sha256", offset_of!(turbo_tokenizer_info, sha256)),
        ("turbo_tokenizer_info", "manifest_sha256", offset_of!(turbo_tokenizer_info, manifest_sha256)),
        ("turbo_encode_options", "omit_special_tokens", offset_of!(turbo_encode_options, omit_special_tokens)),
        ("turbo_encode_options", "truncate", offset_of!(turbo_encode_options, truncate)),
        ("turbo_encode_options", "max_tokens", offset_of!(turbo_encode_options, max_tokens)),
        ("turbo_encode_options", "prompt_role", offset_of!(turbo_encode_options, prompt_role)),
        ("turbo_model_info", "task", offset_of!(turbo_model_info, task)),
        ("turbo_model_info", "dim", offset_of!(turbo_model_info, dim)),
        ("turbo_model_info", "pooling", offset_of!(turbo_model_info, pooling)),
        ("turbo_model_info", "normalize", offset_of!(turbo_model_info, normalize)),
        ("turbo_model_info", "max_seq", offset_of!(turbo_model_info, max_seq)),
        ("turbo_model_info", "max_batch", offset_of!(turbo_model_info, max_batch)),
        ("turbo_model_info", "dtype", offset_of!(turbo_model_info, dtype)),
        ("turbo_model_info", "model_id", offset_of!(turbo_model_info, model_id)),
        ("turbo_model_info", "revision", offset_of!(turbo_model_info, revision)),
        ("turbo_model_info", "manifest_sha256", offset_of!(turbo_model_info, manifest_sha256)),
        ("turbo_model_info", "artifact_sha256", offset_of!(turbo_model_info, artifact_sha256)),
        ("turbo_model_info", "tokenizer_sha256", offset_of!(turbo_model_info, tokenizer_sha256)),
        ("turbo_model_info", "prefix_query", offset_of!(turbo_model_info, prefix_query)),
        ("turbo_model_info", "prefix_document", offset_of!(turbo_model_info, prefix_document)),
    ];
    use turbo::backend::{turbo_backend_model, turbo_backend_tensor};
    let model_fields: &[(&str, &str, usize)] = &[
        ("turbo_backend", "model_load", offset_of!(turbo_backend, model_load)),
        ("turbo_backend", "model_release", offset_of!(turbo_backend, model_release)),
        ("turbo_backend_tensor", "name", offset_of!(turbo_backend_tensor, name)),
        ("turbo_backend_tensor", "data", offset_of!(turbo_backend_tensor, data)),
        ("turbo_backend_tensor", "shape", offset_of!(turbo_backend_tensor, shape)),
        ("turbo_backend_tensor", "ndim", offset_of!(turbo_backend_tensor, ndim)),
        ("turbo_backend_tensor", "dtype", offset_of!(turbo_backend_tensor, dtype)),
        ("turbo_backend_tensor", "bytes", offset_of!(turbo_backend_tensor, bytes)),
        ("turbo_backend_model", "family", offset_of!(turbo_backend_model, family)),
        ("turbo_backend_model", "dtype", offset_of!(turbo_backend_model, dtype)),
        ("turbo_backend_model", "layers", offset_of!(turbo_backend_model, layers)),
        ("turbo_backend_model", "hidden", offset_of!(turbo_backend_model, hidden)),
        ("turbo_backend_model", "heads", offset_of!(turbo_backend_model, heads)),
        ("turbo_backend_model", "intermediate", offset_of!(turbo_backend_model, intermediate)),
        ("turbo_backend_model", "vocab_size", offset_of!(turbo_backend_model, vocab_size)),
        ("turbo_backend_model", "max_positions", offset_of!(turbo_backend_model, max_positions)),
        ("turbo_backend_model", "token_types", offset_of!(turbo_backend_model, token_types)),
        ("turbo_backend_model", "layer_norm_eps", offset_of!(turbo_backend_model, layer_norm_eps)),
        ("turbo_backend_model", "tensor_count", offset_of!(turbo_backend_model, tensor_count)),
        ("turbo_backend_model", "reserved", offset_of!(turbo_backend_model, reserved)),
        ("turbo_backend_model", "tensors", offset_of!(turbo_backend_model, tensors)),
    ];
    use turbo::backend::turbo_backend;
    let fields = [
        fields,
        model_fields,
        &[
            ("turbo_device_info", "kind", offset_of!(turbo_device_info, kind)),
            ("turbo_device_info", "ordinal", offset_of!(turbo_device_info, ordinal)),
            ("turbo_device_info", "unified_memory", offset_of!(turbo_device_info, unified_memory)),
            ("turbo_device_info", "memory_total", offset_of!(turbo_device_info, memory_total)),
            ("turbo_device_info", "memory_free", offset_of!(turbo_device_info, memory_free)),
            ("turbo_device_info", "arch", offset_of!(turbo_device_info, arch)),
            ("turbo_device_info", "name", offset_of!(turbo_device_info, name)),
            ("turbo_device_info", "vendor", offset_of!(turbo_device_info, vendor)),
            ("turbo_device_info", "backend", offset_of!(turbo_device_info, backend)),
            ("turbo_device_info", "runtime_version", offset_of!(turbo_device_info, runtime_version)),
            ("turbo_device_info", "driver_version", offset_of!(turbo_device_info, driver_version)),
            ("turbo_capability", "status", offset_of!(turbo_capability, status)),
            ("turbo_capability", "dtype", offset_of!(turbo_capability, dtype)),
            ("turbo_capability", "options_honored", offset_of!(turbo_capability, options_honored)),
            ("turbo_capability", "cosine_floor", offset_of!(turbo_capability, cosine_floor)),
            ("turbo_capability", "speed_ratio", offset_of!(turbo_capability, speed_ratio)),
            ("turbo_capability", "benchmark", offset_of!(turbo_capability, benchmark)),
            ("turbo_capability", "reason", offset_of!(turbo_capability, reason)),
            ("turbo_backend", "name", offset_of!(turbo_backend, name)),
            ("turbo_backend", "runtime_version", offset_of!(turbo_backend, runtime_version)),
            ("turbo_backend", "device_count", offset_of!(turbo_backend, device_count)),
            ("turbo_backend", "device_info", offset_of!(turbo_backend, device_info)),
            ("turbo_backend", "capability", offset_of!(turbo_backend, capability)),
            ("turbo_backend", "context_create", offset_of!(turbo_backend, context_create)),
            ("turbo_backend", "context_release", offset_of!(turbo_backend, context_release)),
            ("turbo_backend", "buffer_alloc", offset_of!(turbo_backend, buffer_alloc)),
            ("turbo_backend", "buffer_import", offset_of!(turbo_backend, buffer_import)),
            ("turbo_backend", "buffer_release", offset_of!(turbo_backend, buffer_release)),
            ("turbo_backend", "buffer_export", offset_of!(turbo_backend, buffer_export)),
            ("turbo_native_handle", "kind", offset_of!(turbo_native_handle, kind)),
            ("turbo_native_handle", "handle", offset_of!(turbo_native_handle, handle)),
            ("turbo_native_handle", "aux", offset_of!(turbo_native_handle, aux)),
            ("turbo_native_handle", "offset", offset_of!(turbo_native_handle, offset)),
            ("turbo_buffer_desc", "placement", offset_of!(turbo_buffer_desc, placement)),
            ("turbo_buffer_desc", "dtype", offset_of!(turbo_buffer_desc, dtype)),
            ("turbo_buffer_desc", "ndim", offset_of!(turbo_buffer_desc, ndim)),
            ("turbo_buffer_desc", "shape", offset_of!(turbo_buffer_desc, shape)),
            ("turbo_buffer_desc", "bytes", offset_of!(turbo_buffer_desc, bytes)),
        ],
    ]
    .concat();
    let sizes: &[(&str, usize)] = &[
        ("turbo_device_info", size_of::<turbo_device_info>()),
        ("turbo_capability", size_of::<turbo_capability>()),
        ("turbo_backend", size_of::<turbo_backend>()),
        ("turbo_text", size_of::<turbo_text>()),
        ("turbo_error", size_of::<turbo_error>()),
        ("turbo_runtime_desc", size_of::<turbo_runtime_desc>()),
        ("turbo_tokenizer_info", size_of::<turbo_tokenizer_info>()),
        ("turbo_encode_options", size_of::<turbo_encode_options>()),
        ("turbo_native_handle", size_of::<turbo_native_handle>()),
        ("turbo_buffer_desc", size_of::<turbo_buffer_desc>()),
        ("turbo_model_info", size_of::<turbo_model_info>()),
        ("turbo_backend_tensor", size_of::<turbo_backend_tensor>()),
        ("turbo_backend_model", size_of::<turbo_backend_model>()),
    ];
    let mut src = String::from(
        "#include <stddef.h>\n#include <stdio.h>\n#include <turbo/turbo.h>\n#include <turbo/turbo_backend.h>\nint main(void) {\n",
    );
    for (s, f, _) in &fields {
        src += &format!("  printf(\"{s}.{f} %zu\\n\", offsetof({s}, {f}));\n");
    }
    for (s, _) in sizes {
        src += &format!("  printf(\"{s} %zu\\n\", sizeof({s}));\n");
    }
    src += "  return 0;\n}\n";
    let d = scratch("layout");
    std::fs::write(d.join("layout.c"), src).unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(d.join("layout"))
        .arg(d.join("layout.c"))
        .arg("-I")
        .arg(include()));
    let out = run(&mut Command::new(d.join("layout")));
    let mut want = String::new();
    for (s, f, o) in &fields {
        want += &format!("{s}.{f} {o}\n");
    }
    for (s, n) in sizes {
        want += &format!("{s} {n}\n");
    }
    assert_eq!(out, want);
    std::fs::remove_dir_all(d).unwrap();
}

#[test]
fn mirrored_constants_match_the_header() {
    use turbo::status::*;
    // Every constant the Rust side mirrors belongs here.
    let constants: &[(&str, i64)] = &[
        ("TURBO_ERROR_MESSAGE_LEN", TURBO_ERROR_MESSAGE_LEN as i64),
        ("TURBO_TRUNCATE_MODEL", TURBO_TRUNCATE_MODEL.into()),
        ("TURBO_TRUNCATE_NONE", TURBO_TRUNCATE_NONE.into()),
        ("TURBO_TRUNCATE_RIGHT", TURBO_TRUNCATE_RIGHT.into()),
        ("TURBO_TRUNCATE_LEFT", TURBO_TRUNCATE_LEFT.into()),
        ("TURBO_PROMPT_NONE", TURBO_PROMPT_NONE.into()),
        ("TURBO_PROMPT_QUERY", TURBO_PROMPT_QUERY.into()),
        ("TURBO_PROMPT_DOCUMENT", TURBO_PROMPT_DOCUMENT.into()),
        ("TURBO_NORMALIZE_MODEL", TURBO_NORMALIZE_MODEL.into()),
        ("TURBO_NORMALIZE_NONE", TURBO_NORMALIZE_NONE.into()),
        ("TURBO_NORMALIZE_L2", TURBO_NORMALIZE_L2.into()),
        ("TURBO_POOLING_MODEL", TURBO_POOLING_MODEL.into()),
        ("TURBO_POOLING_MEAN", TURBO_POOLING_MEAN.into()),
        ("TURBO_POOLING_CLS", TURBO_POOLING_CLS.into()),
        ("TURBO_POOLING_LAST", TURBO_POOLING_LAST.into()),
        ("TURBO_BERT_EMBEDDING_TENSORS", turbo::backend::TURBO_BERT_EMBEDDING_TENSORS.into()),
        ("TURBO_BERT_LAYER_TENSORS", turbo::backend::TURBO_BERT_LAYER_TENSORS.into()),
        ("TURBO_FAMILY_BERT", turbo::backend::TURBO_FAMILY_BERT.into()),
        ("TURBO_TASK_EMBED", TURBO_TASK_EMBED.into()),
        ("TURBO_DEVICE_CPU", TURBO_DEVICE_CPU.into()),
        ("TURBO_DEVICE_GPU", TURBO_DEVICE_GPU.into()),
        ("TURBO_DEVICE_IGPU", TURBO_DEVICE_IGPU.into()),
        ("TURBO_DEVICE_NPU", TURBO_DEVICE_NPU.into()),
        ("TURBO_DTYPE_I32", TURBO_DTYPE_I32.into()),
        ("TURBO_DTYPE_F16", TURBO_DTYPE_F16.into()),
        ("TURBO_DTYPE_BF16", TURBO_DTYPE_BF16.into()),
        ("TURBO_DTYPE_F32", TURBO_DTYPE_F32.into()),
        ("TURBO_PLACE_HOST", TURBO_PLACE_HOST.into()),
        ("TURBO_PLACE_PINNED", TURBO_PLACE_PINNED.into()),
        ("TURBO_PLACE_DEVICE", TURBO_PLACE_DEVICE.into()),
        ("TURBO_PLACE_SHARED", TURBO_PLACE_SHARED.into()),
        ("TURBO_HANDLE_HOST_PTR", TURBO_HANDLE_HOST_PTR.into()),
        ("TURBO_HANDLE_CUDA_PTR", TURBO_HANDLE_CUDA_PTR.into()),
        ("TURBO_HANDLE_CL_MEM", TURBO_HANDLE_CL_MEM.into()),
        ("TURBO_HANDLE_ZE_USM", TURBO_HANDLE_ZE_USM.into()),
        ("TURBO_HANDLE_MTL_BUFFER", TURBO_HANDLE_MTL_BUFFER.into()),
        ("TURBO_HANDLE_DMABUF_FD", TURBO_HANDLE_DMABUF_FD.into()),
        ("TURBO_PRECISION_MODEL", TURBO_PRECISION_MODEL.into()),
        ("TURBO_PRECISION_FASTEST", TURBO_PRECISION_FASTEST.into()),
        ("TURBO_PRECISION_EXACT", TURBO_PRECISION_EXACT.into()),
        ("TURBO_CAP_UNSUPPORTED", turbo::backend::TURBO_CAP_UNSUPPORTED.into()),
        ("TURBO_CAP_EXPERIMENTAL", turbo::backend::TURBO_CAP_EXPERIMENTAL.into()),
        ("TURBO_CAP_SUPPORTED", turbo::backend::TURBO_CAP_SUPPORTED.into()),
        ("TURBO_OK", OK.into()),
        ("TURBO_E_INVALID_ARGUMENT", INVALID_ARGUMENT.into()),
        ("TURBO_E_INVALID_STRUCT_SIZE", INVALID_STRUCT_SIZE.into()),
        ("TURBO_E_INVALID_UTF8", INVALID_UTF8.into()),
        ("TURBO_E_INVALID_HANDLE", INVALID_HANDLE.into()),
        ("TURBO_E_INVALID_SHAPE", INVALID_SHAPE.into()),
        ("TURBO_E_INVALID_STATE", INVALID_STATE.into()),
        ("TURBO_E_INVALID_ENUM", INVALID_ENUM.into()),
        ("TURBO_E_UNSUPPORTED", UNSUPPORTED.into()),
        ("TURBO_E_UNSUPPORTED_OPTION", UNSUPPORTED_OPTION.into()),
        ("TURBO_E_UNSUPPORTED_TASK", UNSUPPORTED_TASK.into()),
        ("TURBO_E_OUT_OF_MEMORY", OUT_OF_MEMORY.into()),
        ("TURBO_E_BUSY", BUSY.into()),
        ("TURBO_E_CAPACITY", CAPACITY.into()),
        ("TURBO_E_DEVICE_NOT_FOUND", DEVICE_NOT_FOUND.into()),
        ("TURBO_E_DEVICE_UNAVAILABLE", DEVICE_UNAVAILABLE.into()),
        ("TURBO_E_RUNTIME", RUNTIME.into()),
        ("TURBO_E_BUNDLE_NOT_FOUND", BUNDLE_NOT_FOUND.into()),
        ("TURBO_E_BUNDLE_INVALID", BUNDLE_INVALID.into()),
        ("TURBO_E_BUNDLE_INTEGRITY", BUNDLE_INTEGRITY.into()),
        ("TURBO_E_BUNDLE_NO_ARTIFACT", BUNDLE_NO_ARTIFACT.into()),
        ("TURBO_E_INTERNAL", INTERNAL.into()),
        ("TURBO_E_PANIC", PANIC.into()),
    ];
    // Where the core puts each BERT tensor is where the header says.
    let roles =
        turbo::model::BERT_EMBEDDING_ROLES.iter().enumerate().chain(turbo::model::BERT_LAYER_ROLES.iter().enumerate());
    let roles: Vec<(String, i64)> = roles
        .map(|(i, &r)| (format!("TURBO_BERT_{}", turbo::manifest::role_name(r).to_uppercase()), i as i64))
        .collect();
    let constants: Vec<(&str, i64)> =
        constants.iter().copied().chain(roles.iter().map(|(n, i)| (n.as_str(), *i))).collect();
    let mut src = String::from(
        "#include <stdio.h>\n#include <turbo/turbo.h>\n#include <turbo/turbo_backend.h>\nint main(void) {\n",
    );
    for (name, _) in &constants {
        src += &format!("  printf(\"{name} %lld\\n\", (long long)({name}));\n");
    }
    src += "  return 0;\n}\n";
    let d = scratch("constants");
    std::fs::write(d.join("constants.c"), src).unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Werror", "-o"])
        .arg(d.join("constants"))
        .arg(d.join("constants.c"))
        .arg("-I")
        .arg(include()));
    let out = run(&mut Command::new(d.join("constants")));
    let want: String = constants.iter().map(|(name, v)| format!("{name} {v}\n")).collect();
    assert_eq!(out, want);
    for (name, v) in constants.iter().filter(|(n, _)| n.starts_with("TURBO_OK") || n.starts_with("TURBO_E_")) {
        assert_eq!(turbo_status_name_str(*v as i32), *name);
    }
    std::fs::remove_dir_all(d).unwrap();
}

fn turbo_status_name_str(code: i32) -> String {
    unsafe { std::ffi::CStr::from_ptr(turbo_status_name(code)) }.to_str().unwrap().to_owned()
}

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
    if (argc != 4) return 2;
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
    turbo_model *model = NULL;
    if (turbo_model_load(ctx, T(argv[3]), &model, &err)) { printf("model %s\n", err.message); return 1; }
    turbo_context_release(ctx); /* the buffers and the model keep it */
    turbo_buffer_release(wrapped);
    turbo_buffer_release(buf);
    turbo_buffer_release(NULL);
    turbo_context_release(NULL);
    rc = turbo_tokenizer_create(rt, T("/nonexistent/bundle"), &tok, &err);
    printf("missing %s %d\n", turbo_status_name(rc), err.code);
    if (turbo_tokenizer_create(rt, T(argv[1]), &tok, &err)) { printf("create %s\n", err.message); return 1; }
    turbo_runtime_release(rt); /* the tokenizer and the model keep what they need */
    turbo_model_info mi = { sizeof(turbo_model_info) };
    if (turbo_model_get_info(model, &mi, &err)) { printf("model info %s\n", err.message); return 1; }
    printf("model task %u dim %u pooling %u normalize %u max_seq %u max_batch %u dtype %u\n", mi.task, mi.dim,
           mi.pooling, mi.normalize, mi.max_seq, mi.max_batch, mi.dtype);
    printf("%s revision %s\nartifact %s\n", mi.model_id, mi.revision, mi.artifact_sha256);
    turbo_model_release(model);
    turbo_model_release(NULL);
    turbo_text text = T(argv[2]);
    int32_t ids[64], mask[64];
    uint32_t len = 0;
    if (turbo_tokenizer_encode(tok, &text, 1, NULL, ids, mask, NULL, 64, &len, &err)) {
        printf("encode %s\n", err.message); return 1;
    }
    for (uint32_t i = 0; i < len; i++) printf("%d%s", ids[i], i + 1 < len ? " " : "\n");
    turbo_tokenizer_release(tok);
    turbo_tokenizer_release(NULL);
    turbo_runtime_release(NULL);
    printf("%s\n", turbo_version());
    return 0;
}
"#;

#[test]
fn a_c_program_loads_a_model_and_gets_the_upstream_ids() {
    // target/<profile>/deps/abi-* -> target/<profile>/libturbo.so
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
    assert!(lib_dir.join("libturbo.so").exists(), "libturbo.so is not in {}", lib_dir.display());
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
    let model = Fixture::model("c-program-model");
    model.write();
    let text = "Café naïve RÉSUMÉ, 东京 🙂";
    let out = run(Command::new(d.join("p")).arg(&f.dir).arg(text).arg(&model.dir));
    let ids: Vec<String> = upstream_ids(&upstream(), text).iter().map(i32::to_string).collect();
    let want = format!(
        "devices 1, device 0 is cpu kind 1, embed status 0\nselect TURBO_E_DEVICE_NOT_FOUND 99\n\
         context on device 0, buffer 48 bytes, aligned 1\nimport at +8, stack[3] 7, export same 1\n\
         device placement TURBO_E_UNSUPPORTED\n\
         missing TURBO_E_BUNDLE_NOT_FOUND {}\n\
         model task 1 dim 8 pooling 1 normalize 2 max_seq 256 max_batch 64 dtype 12\n\
         sentence-transformers/all-MiniLM-L6-v2 revision 3\nartifact {}\n{}\n0.1.0 cpu\n",
        turbo::status::BUNDLE_NOT_FOUND,
        model.sha256("weights/model.safetensors"),
        ids.join(" ")
    );
    assert_eq!(out, want);
    std::fs::remove_dir_all(d).unwrap();
}
