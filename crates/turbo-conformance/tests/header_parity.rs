//! Group `header parity` (Rust only): the committed C header is the artifact
//! consumers compile against, so a C compiler must agree with `turbo-abi`
//! about the size and alignment of every public struct, about
//! `TURBO_ABI_VERSION`, and about every status and enumeration constant
//! (PLAN.md section 4.3).
//!
//! The check shells out to `cc`. If no C compiler is installed the test
//! prints why it did nothing and passes, so the suite still runs on hosts
//! without a toolchain.

use std::mem::{align_of, size_of};
use std::path::{Path, PathBuf};
use std::process::Command;

use turbo_abi::*;

/// Every `#[repr(C)]` struct in `turbo-abi` that the header defines, with the
/// size and alignment Rust computes for it. A struct in the header that is
/// missing here fails the test: the suite must be told about new ABI types.
fn rust_layouts() -> Vec<(&'static str, usize, usize)> {
    macro_rules! layouts {
        ($($t:ty),* $(,)?) => {
            vec![$((stringify!($t), size_of::<$t>(), align_of::<$t>())),*]
        };
    }
    layouts![
        turbo_text,
        turbo_kv,
        turbo_error,
        turbo_runtime_desc,
        turbo_device_selector,
        turbo_device_info,
        turbo_capability,
        turbo_context_desc,
        turbo_buffer_desc,
        turbo_native_handle,
        turbo_model_desc,
        turbo_model_info,
        turbo_tensor_info,
        turbo_session_desc,
        turbo_embed_options,
        turbo_rerank_options,
        turbo_classify_options,
        turbo_run_options,
        turbo_token_batch,
        turbo_session_stats,
        turbo_result_info,
        turbo_span,
        turbo_message,
        turbo_logit_bias,
        turbo_generate_desc,
        turbo_generation_chunk,
        turbo_encode_options,
        turbo_tokenizer_info,
        turbo_chunk_desc,
        turbo_chunk,
    ]
}

fn include_dir() -> PathBuf {
    turbo_conformance::repo_root().join("include")
}

fn header_path() -> PathBuf {
    include_dir().join("turbo").join("turbo_types.h")
}

/// Names of the structs the header actually defines (a body, not a forward
/// declaration of an opaque handle).
fn header_structs(text: &str) -> Vec<String> {
    text.lines()
        .filter_map(|line| {
            let line = line.trim();
            let rest = line.strip_prefix("typedef struct ")?;
            let name = rest.strip_suffix(" {")?;
            Some(name.to_string())
        })
        .collect()
}

fn cc() -> Option<String> {
    let candidate = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    match Command::new(&candidate).arg("--version").output() {
        Ok(out) if out.status.success() => Some(candidate),
        _ => None,
    }
}

/// Compile and run `source`, returning its stdout.
fn compile_and_run(cc: &str, dir: &Path, source: &str) -> Result<String, String> {
    let src = dir.join("probe.c");
    let exe = dir.join("probe");
    std::fs::write(&src, source).map_err(|e| format!("writing {}: {e}", src.display()))?;
    let out = Command::new(cc)
        .arg("-std=c11")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-I")
        .arg(include_dir())
        .arg("-o")
        .arg(&exe)
        .arg(&src)
        .output()
        .map_err(|e| format!("running {cc}: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "{cc} failed:\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let run = Command::new(&exe).output().map_err(|e| format!("running {}: {e}", exe.display()))?;
    if !run.status.success() {
        return Err(format!("{} exited with {}", exe.display(), run.status));
    }
    Ok(String::from_utf8_lossy(&run.stdout).into_owned())
}

#[test]
fn header_parity_struct_layouts_match_a_c_compiler() {
    let header = header_path();
    let text = std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("reading {}: {e}", header.display()));
    let layouts = rust_layouts();

    // Every struct the header defines must be covered by the table above.
    for name in header_structs(&text) {
        assert!(
            layouts.iter().any(|(n, _, _)| *n == name),
            "{} defines `{name}`, which the conformance suite does not check; add it to rust_layouts()",
            header.display()
        );
    }

    let Some(cc) = cc() else {
        println!(
            "header_parity_struct_layouts_match_a_c_compiler: no C compiler found (set CC); \
             the Rust-side table was still checked against {}",
            header.display()
        );
        return;
    };

    let mut source = String::from("#include \"turbo/turbo_types.h\"\n#include <stdio.h>\n\nint main(void) {\n");
    for (name, _, _) in &layouts {
        source.push_str(&format!("    printf(\"{name} %zu %zu\\n\", sizeof({name}), _Alignof({name}));\n"));
    }
    source.push_str("    return 0;\n}\n");

    let dir = tempfile::tempdir().expect("temp dir");
    let stdout = match compile_and_run(&cc, dir.path(), &source) {
        Ok(s) => s,
        Err(e) => panic!("the committed header does not compile as C11: {e}"),
    };

    let mut seen = 0;
    for line in stdout.lines() {
        let mut parts = line.split_whitespace();
        let name = parts.next().expect("a struct name");
        let size: usize = parts.next().expect("a size").parse().expect("a number");
        let align: usize = parts.next().expect("an alignment").parse().expect("a number");
        let (_, rust_size, rust_align) =
            layouts.iter().find(|(n, _, _)| *n == name).unwrap_or_else(|| panic!("unexpected line {line:?}"));
        assert_eq!(size, *rust_size, "sizeof({name}): C says {size}, Rust says {rust_size}");
        assert_eq!(align, *rust_align, "_Alignof({name}): C says {align}, Rust says {rust_align}");
        seen += 1;
    }
    assert_eq!(seen, layouts.len(), "the probe did not report every struct");
    println!("header parity: {seen} structs match between {cc} and turbo-abi");
}

#[test]
fn header_parity_constants_match_a_c_compiler() {
    let Some(cc) = cc() else {
        println!("header_parity_constants_match_a_c_compiler: no C compiler found (set CC)");
        return;
    };
    // A representative constant from every family, including the ones whose
    // value the suite asserts elsewhere.
    let constants: &[(&str, i64)] = &[
        ("TURBO_ABI_VERSION", TURBO_ABI_VERSION as i64),
        ("TURBO_ERROR_MESSAGE_LEN", TURBO_ERROR_MESSAGE_LEN as i64),
        ("TURBO_MAX_RANK", TURBO_MAX_RANK as i64),
        ("TURBO_STAGE_COUNT", TURBO_STAGE_COUNT as i64),
        ("TURBO_OK", TURBO_OK as i64),
        ("TURBO_E_INVALID_ARGUMENT", TURBO_E_INVALID_ARGUMENT as i64),
        ("TURBO_E_INVALID_STRUCT_SIZE", TURBO_E_INVALID_STRUCT_SIZE as i64),
        ("TURBO_E_INVALID_UTF8", TURBO_E_INVALID_UTF8 as i64),
        ("TURBO_E_INVALID_HANDLE", TURBO_E_INVALID_HANDLE as i64),
        ("TURBO_E_INVALID_SHAPE", TURBO_E_INVALID_SHAPE as i64),
        ("TURBO_E_INVALID_STATE", TURBO_E_INVALID_STATE as i64),
        ("TURBO_E_INVALID_ENUM", TURBO_E_INVALID_ENUM as i64),
        ("TURBO_E_UNSUPPORTED", TURBO_E_UNSUPPORTED as i64),
        ("TURBO_E_UNSUPPORTED_OPTION", TURBO_E_UNSUPPORTED_OPTION as i64),
        ("TURBO_E_UNSUPPORTED_TASK", TURBO_E_UNSUPPORTED_TASK as i64),
        ("TURBO_E_NOT_IMPLEMENTED", TURBO_E_NOT_IMPLEMENTED as i64),
        ("TURBO_E_BUSY", TURBO_E_BUSY as i64),
        ("TURBO_E_CAPACITY", TURBO_E_CAPACITY as i64),
        ("TURBO_E_DEVICE_NOT_FOUND", TURBO_E_DEVICE_NOT_FOUND as i64),
        ("TURBO_E_BUNDLE_NOT_FOUND", TURBO_E_BUNDLE_NOT_FOUND as i64),
        ("TURBO_E_BUNDLE_INVALID", TURBO_E_BUNDLE_INVALID as i64),
        ("TURBO_E_BUNDLE_INTEGRITY", TURBO_E_BUNDLE_INTEGRITY as i64),
        ("TURBO_E_BUNDLE_NO_ARTIFACT", TURBO_E_BUNDLE_NO_ARTIFACT as i64),
        ("TURBO_E_PANIC", TURBO_E_PANIC as i64),
        ("TURBO_DEVICE_CPU", TURBO_DEVICE_CPU as i64),
        ("TURBO_DEVICE_ACCEL", TURBO_DEVICE_ACCEL as i64),
        ("TURBO_TASK_EMBED", TURBO_TASK_EMBED as i64),
        ("TURBO_TASK_CHUNK", TURBO_TASK_CHUNK as i64),
        ("TURBO_MODALITY_TEXT", TURBO_MODALITY_TEXT as i64),
        ("TURBO_MODEL_GENERIC", TURBO_MODEL_GENERIC as i64),
        ("TURBO_DTYPE_F32", TURBO_DTYPE_F32 as i64),
        ("TURBO_DTYPE_BYTES", TURBO_DTYPE_BYTES as i64),
        ("TURBO_PLACE_HOST", TURBO_PLACE_HOST as i64),
        ("TURBO_SELECT_AUTO", TURBO_SELECT_AUTO as i64),
        ("TURBO_SELECT_EXPLICIT", TURBO_SELECT_EXPLICIT as i64),
        ("TURBO_TRUNCATE_LEFT", TURBO_TRUNCATE_LEFT as i64),
        ("TURBO_PROMPT_DOCUMENT", TURBO_PROMPT_DOCUMENT as i64),
        ("TURBO_NORMALIZE_L2", TURBO_NORMALIZE_L2 as i64),
        ("TURBO_POOLING_LAST", TURBO_POOLING_LAST as i64),
        ("TURBO_OUTPUT_I8", TURBO_OUTPUT_I8 as i64),
        ("TURBO_AGGREGATE_MAX", TURBO_AGGREGATE_MAX as i64),
        ("TURBO_FINISH_CANCELLED", TURBO_FINISH_CANCELLED as i64),
        ("TURBO_STRUCTURED_GRAMMAR", TURBO_STRUCTURED_GRAMMAR as i64),
        ("TURBO_STREAM_STOP", TURBO_STREAM_STOP as i64),
        ("TURBO_HANDLE_DMABUF_FD", TURBO_HANDLE_DMABUF_FD as i64),
        ("TURBO_IO_OUTPUT", TURBO_IO_OUTPUT as i64),
        ("TURBO_STAGE_FUSED", TURBO_STAGE_FUSED as i64),
        ("TURBO_CAP_SUPPORTED", TURBO_CAP_SUPPORTED as i64),
    ];
    let bits: &[(&str, u64)] = &[
        ("TURBO_CAP_ASYNC", TURBO_CAP_ASYNC),
        ("TURBO_CAP_HOST_PTR_IMPORT", TURBO_CAP_HOST_PTR_IMPORT),
        ("TURBO_CAP_DEVICE_RESULT", TURBO_CAP_DEVICE_RESULT),
        ("TURBO_CAP_DETERMINISTIC", TURBO_CAP_DETERMINISTIC),
        ("TURBO_CAP_OPT_TRUNCATE", TURBO_CAP_OPT_TRUNCATE),
        ("TURBO_CAP_OPT_MAX_TOKENS", TURBO_CAP_OPT_MAX_TOKENS),
        ("TURBO_CAP_OPT_PROMPT_ROLE", TURBO_CAP_OPT_PROMPT_ROLE),
        ("TURBO_CAP_OPT_NORMALIZE", TURBO_CAP_OPT_NORMALIZE),
        ("TURBO_CAP_OPT_POOLING_OVERRIDE", TURBO_CAP_OPT_POOLING_OVERRIDE),
        ("TURBO_CAP_OPT_OUTPUT_DIM", TURBO_CAP_OPT_OUTPUT_DIM),
        ("TURBO_CAP_OPT_OUTPUT_DTYPE", TURBO_CAP_OPT_OUTPUT_DTYPE),
        ("TURBO_CAP_OPT_TOP_N", TURBO_CAP_OPT_TOP_N),
        ("TURBO_CAP_OPT_AGGREGATION", TURBO_CAP_OPT_AGGREGATION),
        // The generation bits (1 << 32 and above) are checked separately:
        // the committed header defines them as `int` shifts, which no C
        // compiler can evaluate. See
        // `header_parity_sixty_four_bit_capability_bits_are_usable_from_c`.
    ];

    let mut source = String::from("#include \"turbo/turbo_types.h\"\n#include <stdio.h>\n\nint main(void) {\n");
    for (name, _) in constants {
        source.push_str(&format!("    printf(\"{name} %lld\\n\", (long long){name});\n"));
    }
    for (name, _) in bits {
        source.push_str(&format!("    printf(\"{name} %llu\\n\", (unsigned long long){name});\n"));
    }
    source.push_str("    return 0;\n}\n");

    let dir = tempfile::tempdir().expect("temp dir");
    let stdout = match compile_and_run(&cc, dir.path(), &source) {
        Ok(s) => s,
        Err(e) => panic!("the committed header does not compile as C11: {e}"),
    };
    let mut lines = stdout.lines();
    for (name, expected) in constants {
        let line = lines.next().unwrap_or_else(|| panic!("no output for {name}"));
        let value: i64 = line.split_whitespace().nth(1).expect("a value").parse().expect("a number");
        assert_eq!(value, *expected, "{name}: the header says {value}, turbo-abi says {expected}");
    }
    for (name, expected) in bits {
        let line = lines.next().unwrap_or_else(|| panic!("no output for {name}"));
        let value: u64 = line.split_whitespace().nth(1).expect("a value").parse().expect("a number");
        assert_eq!(value, *expected, "{name}: the header says {value:#x}, turbo-abi says {expected:#x}");
    }
    println!("header parity: {} constants match", constants.len() + bits.len());
}

#[test]
fn header_parity_the_public_header_compiles_as_cpp() {
    let Some(cc) = cc() else {
        println!("header_parity_the_public_header_compiles_as_cpp: no C compiler found (set CC)");
        return;
    };
    // The header set is consumed from C++ too (PLAN.md section 8); a C++
    // compiler must accept it with no warnings.
    let cxx = std::env::var("CXX").unwrap_or_else(|_| "c++".to_string());
    if Command::new(&cxx).arg("--version").output().map(|o| !o.status.success()).unwrap_or(true) {
        println!("header_parity_the_public_header_compiles_as_cpp: no C++ compiler found (set CXX)");
        return;
    }
    let dir = tempfile::tempdir().expect("temp dir");
    let src = dir.path().join("probe.cpp");
    std::fs::write(&src, "#include \"turbo/turbo.h\"\nint main() { return (int)turbo_abi_version(); }\n")
        .expect("write");
    let out = Command::new(&cxx)
        .arg("-std=c++17")
        .arg("-Wall")
        .arg("-Wextra")
        .arg("-Werror")
        .arg("-fsyntax-only")
        .arg("-I")
        .arg(include_dir())
        .arg(&src)
        .output()
        .expect("running the C++ compiler");
    assert!(out.status.success(), "{cxx} rejected the public header:\n{}", String::from_utf8_lossy(&out.stderr));
    let _ = cc;
}

#[test]
fn header_parity_every_exported_symbol_is_declared() {
    // The header is the contract: every `turbo_*` entry point the library
    // exports must be declared in it, and nothing else may be.
    let header = include_dir().join("turbo").join("turbo.h");
    let text = std::fs::read_to_string(&header).unwrap_or_else(|e| panic!("reading {}: {e}", header.display()));
    let declared: Vec<&str> = text
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|w| w.starts_with("turbo_") && !w.is_empty())
        .collect();
    for symbol in [
        "turbo_abi_version",
        "turbo_version",
        "turbo_status_name",
        "turbo_runtime_create",
        "turbo_runtime_release",
        "turbo_runtime_load_provider",
        "turbo_runtime_device_count",
        "turbo_runtime_device_info",
        "turbo_runtime_select_device",
        "turbo_runtime_capability",
        "turbo_can_run",
        "turbo_context_create",
        "turbo_context_release",
        "turbo_context_device",
        "turbo_buffer_alloc",
        "turbo_buffer_import",
        "turbo_buffer_release",
        "turbo_buffer_get_desc",
        "turbo_buffer_host_ptr",
        "turbo_buffer_export",
        "turbo_model_load",
        "turbo_model_release",
        "turbo_model_get_info",
        "turbo_model_io_info",
        "turbo_model_label",
        "turbo_session_create",
        "turbo_session_release",
        "turbo_session_write_text",
        "turbo_session_write_tokens",
        "turbo_session_write_pairs",
        "turbo_session_write_text_classify",
        "turbo_session_bind",
        "turbo_session_run",
        "turbo_session_get_stats",
        "turbo_result_get_info",
        "turbo_result_output_info",
        "turbo_result_buffer",
        "turbo_result_read",
        "turbo_result_spans",
        "turbo_result_release",
        "turbo_generation_create",
        "turbo_generation_prompt",
        "turbo_generation_prompt_tokens",
        "turbo_generation_step",
        "turbo_generation_cancel",
        "turbo_generation_release",
        "turbo_generate",
    ] {
        assert!(
            declared.contains(&symbol),
            "{} does not declare `{symbol}`, which the library exports",
            header.display()
        );
    }
}

#[test]
fn header_parity_sixty_four_bit_capability_bits_are_usable_from_c() {
    // Every capability bit is a `uint64_t` in `turbo-abi`, and the header
    // must define it so a C caller gets the same value. cbindgen emits
    // `#define TURBO_CAP_OPT_GEN_STRUCTURED (1 << 32)`, whose type is `int`:
    // the shift is undefined behaviour, `-Werror=shift-count-overflow`
    // rejects it, and without that flag the value is not 2^32. A `ULL`
    // suffix (or a `(uint64_t)1 << 32` form) is what the ABI needs.
    let Some(cc) = cc() else {
        println!("header_parity_sixty_four_bit_capability_bits: no C compiler found (set CC)");
        return;
    };
    let bits: &[(&str, u64)] = &[
        ("TURBO_CAP_OPT_GEN_STRUCTURED", TURBO_CAP_OPT_GEN_STRUCTURED),
        ("TURBO_CAP_OPT_GEN_JSON_SCHEMA", TURBO_CAP_OPT_GEN_JSON_SCHEMA),
        ("TURBO_CAP_OPT_GEN_TOOLS", TURBO_CAP_OPT_GEN_TOOLS),
        ("TURBO_CAP_OPT_GEN_N", TURBO_CAP_OPT_GEN_N),
        ("TURBO_CAP_OPT_GEN_LOGIT_BIAS", TURBO_CAP_OPT_GEN_LOGIT_BIAS),
        ("TURBO_CAP_OPT_GEN_PENALTIES", TURBO_CAP_OPT_GEN_PENALTIES),
        ("TURBO_CAP_OPT_GEN_LOGPROBS", TURBO_CAP_OPT_GEN_LOGPROBS),
        ("TURBO_CAP_OPT_GEN_STOP_STRINGS", TURBO_CAP_OPT_GEN_STOP_STRINGS),
        ("TURBO_CAP_OPT_GEN_SEED", TURBO_CAP_OPT_GEN_SEED),
    ];
    let mut source = String::from("#include \"turbo/turbo_types.h\"\n#include <stdio.h>\n\nint main(void) {\n");
    for (name, _) in bits {
        source.push_str(&format!("    printf(\"{name} %llu\\n\", (unsigned long long){name});\n"));
    }
    source.push_str("    return 0;\n}\n");
    let dir = tempfile::tempdir().expect("temp dir");
    let stdout = match compile_and_run(&cc, dir.path(), &source) {
        Ok(s) => s,
        Err(e) => panic!("a C caller cannot use the 64-bit capability bits: {e}"),
    };
    for ((name, expected), line) in bits.iter().zip(stdout.lines()) {
        let value: u64 = line.split_whitespace().nth(1).expect("a value").parse().expect("a number");
        assert_eq!(value, *expected, "{name}: the header says {value:#x}, turbo-abi says {expected:#x}");
    }
}
