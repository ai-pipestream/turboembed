//! The header against the library: turbo.h compiles standalone as C11 and
//! C++17, the Rust mirrors of its structs have the C compiler's layout, and
//! a C program linked against libturbo gets the upstream ids.

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
    ];
    use turbo::backend::turbo_backend;
    let fields = [
        fields,
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
        ("TURBO_TASK_EMBED", TURBO_TASK_EMBED.into()),
        ("TURBO_DEVICE_CPU", TURBO_DEVICE_CPU.into()),
        ("TURBO_DEVICE_GPU", TURBO_DEVICE_GPU.into()),
        ("TURBO_DEVICE_IGPU", TURBO_DEVICE_IGPU.into()),
        ("TURBO_DEVICE_NPU", TURBO_DEVICE_NPU.into()),
        ("TURBO_DTYPE_I32", TURBO_DTYPE_I32.into()),
        ("TURBO_DTYPE_F16", TURBO_DTYPE_F16.into()),
        ("TURBO_DTYPE_BF16", TURBO_DTYPE_BF16.into()),
        ("TURBO_DTYPE_F32", TURBO_DTYPE_F32.into()),
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
    let mut src = String::from("#include <stdio.h>\n#include <turbo/turbo.h>\nint main(void) {\n");
    for (name, _) in constants {
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
    if (argc != 3) return 2;
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
    rc = turbo_tokenizer_create(rt, T("/nonexistent/bundle"), &tok, &err);
    printf("missing %s %d\n", turbo_status_name(rc), err.code);
    if (turbo_tokenizer_create(rt, T(argv[1]), &tok, &err)) { printf("create %s\n", err.message); return 1; }
    turbo_runtime_release(rt); /* the tokenizer keeps what it needs */
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
fn a_c_program_gets_the_upstream_ids() {
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
    let text = "Café naïve RÉSUMÉ, 东京 🙂";
    let out = run(Command::new(d.join("p")).arg(&f.dir).arg(text));
    let ids: Vec<String> = upstream_ids(&upstream(), text).iter().map(i32::to_string).collect();
    let want = format!(
        "devices 1, device 0 is cpu kind 1, embed status 0\nselect TURBO_E_DEVICE_NOT_FOUND 99\nmissing TURBO_E_BUNDLE_NOT_FOUND {}\n{}\n0.1.0 cpu\n",
        turbo::status::BUNDLE_NOT_FOUND,
        ids.join(" ")
    );
    assert_eq!(out, want);
    std::fs::remove_dir_all(d).unwrap();
}
