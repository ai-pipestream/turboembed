//! turbo.h compiled standalone, and the Rust mirrors of its structs laid out as the C compiler lays them out.

use std::mem::offset_of;
use std::process::Command;

use crate::{include, run, scratch};
use turbo::*;

#[test]
fn the_header_compiles_standalone() {
    let d = scratch("standalone");
    let c = d.join("h.c");
    std::fs::write(&c, "#include <turbo/turbo.h>\n#include <turbo/turbo_backend.h>\n").unwrap();
    run(Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Wpadded", "-Werror", "-pedantic", "-fsyntax-only", "-I"])
        .arg(include())
        .arg(&c));
    run(Command::new("c++")
        .args([
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Wpadded",
            "-Werror",
            "-pedantic",
            "-fsyntax-only",
            "-x",
            "c++",
            "-I",
        ])
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
        ("turbo_model_info", "output_dims_count", offset_of!(turbo_model_info, output_dims_count)),
        ("turbo_model_info", "output_dims", offset_of!(turbo_model_info, output_dims)),
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
        ("turbo_backend_model", "position_offset", offset_of!(turbo_backend_model, position_offset)),
        ("turbo_backend_model", "tensors", offset_of!(turbo_backend_model, tensors)),
        ("turbo_backend_model", "format", offset_of!(turbo_backend_model, format)),
        ("turbo_backend_model", "graph_input", offset_of!(turbo_backend_model, graph_input)),
        ("turbo_backend_model", "graph_output", offset_of!(turbo_backend_model, graph_output)),
        ("turbo_backend_model", "compute_dtype", offset_of!(turbo_backend_model, compute_dtype)),
        ("turbo_backend_model", "fixed_seq", offset_of!(turbo_backend_model, fixed_seq)),
        ("turbo_backend_model", "fixed_batch", offset_of!(turbo_backend_model, fixed_batch)),
        ("turbo_backend_model", "artifact", offset_of!(turbo_backend_model, artifact)),
        ("turbo_backend_model", "artifact_bytes", offset_of!(turbo_backend_model, artifact_bytes)),
        ("turbo_backend_model", "artifact2", offset_of!(turbo_backend_model, artifact2)),
        ("turbo_backend_model", "artifact2_bytes", offset_of!(turbo_backend_model, artifact2_bytes)),
        ("turbo_backend_model", "head", offset_of!(turbo_backend_model, head)),
        ("turbo_backend_model", "head_out", offset_of!(turbo_backend_model, head_out)),
        ("turbo_backend_model", "lookahead", offset_of!(turbo_backend_model, lookahead)),
        ("turbo_backend", "formats", offset_of!(turbo_backend, formats)),
        ("turbo_backend", "reserved2", offset_of!(turbo_backend, reserved2)),
    ];
    use turbo::backend::{
        turbo_backend_embed_rows, turbo_backend_run, turbo_backend_session_desc, turbo_backend_tuning,
    };
    let session_fields: &[(&str, &str, usize)] = &[
        ("turbo_session_desc", "max_batch", offset_of!(turbo_session_desc, max_batch)),
        ("turbo_session_desc", "max_seq", offset_of!(turbo_session_desc, max_seq)),
        ("turbo_session_desc", "precision", offset_of!(turbo_session_desc, precision)),
        ("turbo_embed_options", "truncate", offset_of!(turbo_embed_options, truncate)),
        ("turbo_embed_options", "max_tokens", offset_of!(turbo_embed_options, max_tokens)),
        ("turbo_embed_options", "prompt_role", offset_of!(turbo_embed_options, prompt_role)),
        ("turbo_embed_options", "normalize", offset_of!(turbo_embed_options, normalize)),
        ("turbo_embed_options", "pooling", offset_of!(turbo_embed_options, pooling)),
        ("turbo_embed_options", "output_dim", offset_of!(turbo_embed_options, output_dim)),
        ("turbo_token_batch", "batch", offset_of!(turbo_token_batch, batch)),
        ("turbo_token_batch", "seq", offset_of!(turbo_token_batch, seq)),
        ("turbo_token_batch", "row_stride", offset_of!(turbo_token_batch, row_stride)),
        ("turbo_token_batch", "ids", offset_of!(turbo_token_batch, ids)),
        ("turbo_token_batch", "mask", offset_of!(turbo_token_batch, mask)),
        ("turbo_token_batch", "types", offset_of!(turbo_token_batch, types)),
        ("turbo_session_info", "max_batch", offset_of!(turbo_session_info, max_batch)),
        ("turbo_session_info", "max_seq", offset_of!(turbo_session_info, max_seq)),
        ("turbo_session_info", "precision", offset_of!(turbo_session_info, precision)),
        ("turbo_session_info", "compute_dtype", offset_of!(turbo_session_info, compute_dtype)),
        ("turbo_session_info", "reserved", offset_of!(turbo_session_info, reserved)),
        ("turbo_session_info", "tuned", offset_of!(turbo_session_info, tuned)),
        ("turbo_session_info", "tune_ms", offset_of!(turbo_session_info, tune_ms)),
        ("turbo_session_info", "choices", offset_of!(turbo_session_info, choices)),
        ("turbo_session_desc", "tuning", offset_of!(turbo_session_desc, tuning)),
        ("turbo_session_desc", "tuning_budget_ms", offset_of!(turbo_session_desc, tuning_budget_ms)),
        ("turbo_session_desc", "cpu_threads", offset_of!(turbo_session_desc, cpu_threads)),
        ("turbo_session_desc", "max_ranges", offset_of!(turbo_session_desc, max_ranges)),
        ("turbo_token_range", "row", offset_of!(turbo_token_range, row)),
        ("turbo_token_range", "start", offset_of!(turbo_token_range, start)),
        ("turbo_token_range", "end", offset_of!(turbo_token_range, end)),
        ("turbo_backend", "session_create_tuned", offset_of!(turbo_backend, session_create_tuned)),
        ("turbo_backend_tuning", "mode", offset_of!(turbo_backend_tuning, mode)),
        ("turbo_backend_tuning", "budget_ms", offset_of!(turbo_backend_tuning, budget_ms)),
        ("turbo_backend_tuning", "numerics_allowed", offset_of!(turbo_backend_tuning, numerics_allowed)),
        ("turbo_backend_tuning", "cached", offset_of!(turbo_backend_tuning, cached)),
        ("turbo_backend_tuning", "tuned", offset_of!(turbo_backend_tuning, tuned)),
        ("turbo_backend_tuning", "tune_ms", offset_of!(turbo_backend_tuning, tune_ms)),
        ("turbo_backend_tuning", "choices", offset_of!(turbo_backend_tuning, choices)),
        ("turbo_backend_tuning", "timings", offset_of!(turbo_backend_tuning, timings)),
        ("turbo_backend_tuning", "timings_len", offset_of!(turbo_backend_tuning, timings_len)),
        ("turbo_backend_tuning", "numerics_used", offset_of!(turbo_backend_tuning, numerics_used)),
        ("turbo_result_info", "task", offset_of!(turbo_result_info, task)),
        ("turbo_result_info", "batch", offset_of!(turbo_result_info, batch)),
        ("turbo_result_info", "dim", offset_of!(turbo_result_info, dim)),
        ("turbo_result_info", "dtype", offset_of!(turbo_result_info, dtype)),
        ("turbo_result_info", "compute_dtype", offset_of!(turbo_result_info, compute_dtype)),
        ("turbo_result_info", "placement", offset_of!(turbo_result_info, placement)),
        ("turbo_result_info", "device", offset_of!(turbo_result_info, device)),
        ("turbo_result_info", "bytes", offset_of!(turbo_result_info, bytes)),
        ("turbo_result_info", "h2d_bytes", offset_of!(turbo_result_info, h2d_bytes)),
        ("turbo_result_info", "d2h_bytes", offset_of!(turbo_result_info, d2h_bytes)),
        ("turbo_result_info", "host_allocs", offset_of!(turbo_result_info, host_allocs)),
        ("turbo_result_info", "device_allocs", offset_of!(turbo_result_info, device_allocs)),
        ("turbo_result_info", "stage_count", offset_of!(turbo_result_info, stage_count)),
        ("turbo_result_info", "reserved", offset_of!(turbo_result_info, reserved)),
        ("turbo_result_info", "stage", offset_of!(turbo_result_info, stage)),
        ("turbo_result_info", "backend", offset_of!(turbo_result_info, backend)),
        ("turbo_result_info", "arch", offset_of!(turbo_result_info, arch)),
        ("turbo_result_info", "runtime_version", offset_of!(turbo_result_info, runtime_version)),
        ("turbo_result_info", "manifest_sha256", offset_of!(turbo_result_info, manifest_sha256)),
        ("turbo_result_info", "artifact_sha256", offset_of!(turbo_result_info, artifact_sha256)),
        ("turbo_result_info", "tokenizer_sha256", offset_of!(turbo_result_info, tokenizer_sha256)),
        ("turbo_backend", "session_create", offset_of!(turbo_backend, session_create)),
        ("turbo_backend", "session_release", offset_of!(turbo_backend, session_release)),
        ("turbo_backend", "embed_write", offset_of!(turbo_backend, embed_write)),
        ("turbo_backend", "session_run", offset_of!(turbo_backend, session_run)),
        ("turbo_backend", "buffer_read", offset_of!(turbo_backend, buffer_read)),
        ("turbo_backend_embed_rows", "batch", offset_of!(turbo_backend_embed_rows, batch)),
        ("turbo_backend_embed_rows", "seq", offset_of!(turbo_backend_embed_rows, seq)),
        ("turbo_backend_embed_rows", "row_stride", offset_of!(turbo_backend_embed_rows, row_stride)),
        ("turbo_backend_embed_rows", "ids", offset_of!(turbo_backend_embed_rows, ids)),
        ("turbo_backend_embed_rows", "mask", offset_of!(turbo_backend_embed_rows, mask)),
        ("turbo_backend_embed_rows", "types", offset_of!(turbo_backend_embed_rows, types)),
        ("turbo_backend_embed_rows", "pooling", offset_of!(turbo_backend_embed_rows, pooling)),
        ("turbo_backend_embed_rows", "normalize", offset_of!(turbo_backend_embed_rows, normalize)),
        ("turbo_backend_embed_rows", "output_dim", offset_of!(turbo_backend_embed_rows, output_dim)),
        ("turbo_backend_embed_rows", "head", offset_of!(turbo_backend_embed_rows, head)),
        ("turbo_backend_run", "placement", offset_of!(turbo_backend_run, placement)),
        ("turbo_backend_run", "output", offset_of!(turbo_backend_run, output)),
        ("turbo_backend_run", "host", offset_of!(turbo_backend_run, host)),
        ("turbo_backend_run", "h2d_bytes", offset_of!(turbo_backend_run, h2d_bytes)),
        ("turbo_backend_run", "d2h_bytes", offset_of!(turbo_backend_run, d2h_bytes)),
        ("turbo_backend_run", "host_allocs", offset_of!(turbo_backend_run, host_allocs)),
        ("turbo_backend_run", "device_allocs", offset_of!(turbo_backend_run, device_allocs)),
        ("turbo_backend_run", "stage", offset_of!(turbo_backend_run, stage)),
        ("turbo_backend", "session_open", offset_of!(turbo_backend, session_open)),
        ("turbo_backend", "embed_write_ranges", offset_of!(turbo_backend, embed_write_ranges)),
        ("turbo_backend", "token_head", offset_of!(turbo_backend, token_head)),
        ("turbo_backend", "reserved3", offset_of!(turbo_backend, reserved3)),
        ("turbo_backend_session_desc", "task", offset_of!(turbo_backend_session_desc, task)),
        ("turbo_backend_session_desc", "max_batch", offset_of!(turbo_backend_session_desc, max_batch)),
        ("turbo_backend_session_desc", "max_seq", offset_of!(turbo_backend_session_desc, max_seq)),
        ("turbo_backend_session_desc", "precision", offset_of!(turbo_backend_session_desc, precision)),
        ("turbo_backend_session_desc", "cpu_threads", offset_of!(turbo_backend_session_desc, cpu_threads)),
        ("turbo_backend_session_desc", "max_ranges", offset_of!(turbo_backend_session_desc, max_ranges)),
        ("turbo_backend_session_desc", "reserved", offset_of!(turbo_backend_session_desc, reserved)),
    ];
    use turbo::backend::turbo_backend;
    let fields = [
        fields,
        model_fields,
        session_fields,
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
        ("turbo_session_desc", size_of::<turbo_session_desc>()),
        ("turbo_embed_options", size_of::<turbo_embed_options>()),
        ("turbo_token_batch", size_of::<turbo_token_batch>()),
        ("turbo_session_info", size_of::<turbo_session_info>()),
        ("turbo_result_info", size_of::<turbo_result_info>()),
        ("turbo_backend_embed_rows", size_of::<turbo_backend_embed_rows>()),
        ("turbo_backend_run", size_of::<turbo_backend_run>()),
        ("turbo_backend_tuning", size_of::<turbo_backend_tuning>()),
        ("turbo_token_range", size_of::<turbo_token_range>()),
        ("turbo_backend_session_desc", size_of::<turbo_backend_session_desc>()),
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

/// The layout the tuning ABI was proposed with: the table's new function
/// after reserved2, and the grown structs' sizes, old sizes kept.
#[test]
fn the_tuning_layout_is_the_one_proposed() {
    use turbo::backend::{turbo_backend, turbo_backend_tuning};
    assert_eq!(offset_of!(turbo_backend, session_create_tuned), 160);
    // The table and the session description have grown since: what they
    // were then is the size before the next group of fields.
    assert_eq!(offset_of!(turbo_backend, session_open), 168);
    assert_eq!(TURBO_SESSION_DESC_SIZE_V2, 24);
    assert_eq!(size_of::<turbo_session_info>(), 1056);
    assert_eq!(offset_of!(turbo_session_info, choices), 32);
    assert_eq!(offset_of!(turbo_backend_tuning, cached), 16);
    assert_eq!(offset_of!(turbo_backend_tuning, choices), 32);
    assert_eq!(offset_of!(turbo_backend_tuning, timings), 1056);
    assert_eq!(offset_of!(turbo_backend_tuning, numerics_used), 1068);
    assert_eq!(size_of::<turbo_backend_tuning>(), 1072);
    assert_eq!((TURBO_SESSION_DESC_SIZE_V1, TURBO_SESSION_INFO_SIZE_V1), (16, 24));
}

/// The layout late pooling and the session thread count were added with:
/// the table's two functions after session_create_tuned, the session
/// description's two fields after tuning_budget_ms, and the older sizes
/// still accepted.
#[test]
fn the_late_pooling_layout_is_the_one_proposed() {
    use turbo::backend::{turbo_backend, turbo_backend_session_desc};
    assert_eq!(offset_of!(turbo_backend, session_open), 168);
    assert_eq!(offset_of!(turbo_backend, embed_write_ranges), 176);
    assert_eq!(offset_of!(turbo_backend, token_head), 184);
    assert_eq!(offset_of!(turbo_session_desc, cpu_threads), 24);
    assert_eq!(offset_of!(turbo_session_desc, max_ranges), 28);
    assert_eq!(size_of::<turbo_session_desc>(), 32);
    assert_eq!(size_of::<turbo_token_range>(), 12);
    assert_eq!(size_of::<turbo_backend_session_desc>(), 32);
    assert_eq!((TURBO_SESSION_DESC_SIZE_V1, TURBO_SESSION_DESC_SIZE_V2), (16, 24));
}

/// The layout token heads were added with: the table's flag after
/// embed_write_ranges, the model description's head after artifact2_bytes,
/// and the rows' head in place of their reserved field.
#[test]
fn the_token_head_layout_is_the_one_proposed() {
    use turbo::backend::{turbo_backend, turbo_backend_embed_rows, turbo_backend_model};
    assert_eq!(offset_of!(turbo_backend, reserved3), 188);
    assert_eq!(size_of::<turbo_backend>(), 192);
    assert_eq!(offset_of!(turbo_backend_model, head), 120);
    assert_eq!(offset_of!(turbo_backend_model, head_out), 128);
    assert_eq!(size_of::<turbo_backend_model>(), 136);
    assert_eq!(offset_of!(turbo_backend_embed_rows, head), 52);
    assert_eq!(size_of::<turbo_backend_embed_rows>(), 56);
}
