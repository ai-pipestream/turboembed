//! Compile native/turborerank into the Rust crate.
//!
//! Detects nvcc + cuda_runtime.h and, unless TURBORERANK_DISABLE_CUDA=1,
//! builds the CUDA MiniLM CE. Detects OpenVINO + Level Zero unless
//! TURBORERANK_DISABLE_OPENVINO=1. On macOS, unless
//! TURBORERANK_DISABLE_METAL=1, compiles metal_api.mm (MTL shared +
//! first-party Metal MiniLM CE).

use std::path::{Path, PathBuf};
use std::process::Command;

fn cuda_enabled() -> bool {
    if std::env::var_os("TURBORERANK_DISABLE_CUDA").is_some() {
        return false;
    }
    if Command::new("nvcc").arg("--version").output().is_err() {
        return false;
    }
    Path::new("/usr/include/cuda_runtime.h").exists()
        || Path::new("/usr/local/cuda/include/cuda_runtime.h").exists()
}

struct OpenVinoPaths {
    include_dirs: Vec<String>,
    lib_dirs: Vec<String>,
}

fn pkg_config_libs(name: &str) -> Option<(Vec<String>, Vec<String>)> {
    let out = Command::new("pkg-config")
        .args(["--cflags", "--libs", name])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut includes = Vec::new();
    let mut libs = Vec::new();
    for tok in text.split_whitespace() {
        if let Some(dir) = tok.strip_prefix("-I") {
            includes.push(dir.to_string());
        } else if let Some(dir) = tok.strip_prefix("-L") {
            libs.push(dir.to_string());
        }
    }
    if includes.is_empty() && libs.is_empty() {
        return None;
    }
    Some((includes, libs))
}

fn find_openvino() -> Option<OpenVinoPaths> {
    if let Some((include_dirs, lib_dirs)) = pkg_config_libs("openvino") {
        return Some(OpenVinoPaths {
            include_dirs,
            lib_dirs,
        });
    }
    let mut roots = Vec::new();
    for key in ["OPENVINO_DIR", "OpenVINO_DIR", "INTEL_OPENVINO_DIR"] {
        if let Ok(val) = std::env::var(key) {
            if !val.is_empty() {
                roots.push(PathBuf::from(val));
            }
        }
    }
    for candidate in [
        "/work/opt/openvino_genai",
        "/work/opt/openvino_genai_ubuntu26_2026.3.1.0_x86_64",
        "/opt/intel/openvino",
        "/opt/intel/openvino_2026",
        "/opt/intel/openvino_2025",
        "/usr",
    ] {
        roots.push(PathBuf::from(candidate));
    }
    for root in roots {
        for base in [&root, &root.join("runtime")] {
            let include = base.join("include");
            let lib = [
                base.join("lib/intel64"),
                base.join("lib64"),
                base.join("lib"),
            ]
            .into_iter()
            .find(|p| p.join("libopenvino.so").exists());
            if include.join("openvino/openvino.hpp").exists() {
                if let Some(lib_dir) = lib {
                    return Some(OpenVinoPaths {
                        include_dirs: vec![include.display().to_string()],
                        lib_dirs: vec![lib_dir.display().to_string()],
                    });
                }
            }
        }
    }
    None
}

fn level_zero_present() -> bool {
    Path::new("/usr/include/level_zero/ze_api.h").exists()
        && (Path::new("/usr/lib/x86_64-linux-gnu/libze_loader.so").exists()
            || Path::new("/usr/lib/x86_64-linux-gnu/libze_loader.so.1").exists())
}

fn openvino_enabled() -> Option<OpenVinoPaths> {
    if std::env::var_os("TURBORERANK_DISABLE_OPENVINO").is_some() {
        return None;
    }
    find_openvino()
}

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let root = root.canonicalize().unwrap_or(root);
    let enable_cuda = cuda_enabled();
    let ov = openvino_enabled();
    let enable_l0 = ov.is_some() && level_zero_present();

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let enable_metal =
        target_os == "macos" && std::env::var_os("TURBORERANK_DISABLE_METAL").is_none();

    let sources = [
        "native/turborerank/src/alloc.cpp",
        "native/turborerank/src/pack.cpp",
        "native/turborerank/src/wordpiece.cpp",
        "native/wordpiece/vocab_load.cpp",
        "native/wordpiece/encode.cpp",
        "native/turborerank/src/safetensors.cpp",
        "native/turborerank/src/bert_cpu.cpp",
        "native/turborerank/src/cuda_api.cpp",
        "native/turborerank/src/ov_api.cpp",
        "native/turborerank/src/metal_api.cpp",
        "native/turborerank/src/engine.cpp",
        "native/turbo_buffer/src/arena.cpp",
        "native/turbo_buffer/src/cuda.cpp",
        "native/turbo_buffer/src/ze.cpp",
        "native/turbo_buffer/src/metal.cpp",
    ];
    for rel in sources {
        println!("cargo:rerun-if-changed={}", root.join(rel).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/bert_cuda.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/cuda_api.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/ov_api.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/metal_api.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/metal_api.mm").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/turborerank.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/turbo_buffer.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/internal.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/metal.mm").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/reranker.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/internal.hpp").display()
    );
    println!("cargo:rerun-if-env-changed=TURBORERANK_DISABLE_CUDA");
    println!("cargo:rerun-if-env-changed=TURBORERANK_DISABLE_OPENVINO");
    println!("cargo:rerun-if-env-changed=TURBORERANK_DISABLE_METAL");
    println!("cargo:rustc-check-cfg=cfg(turborerank_metal)");
    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    println!("cargo:rustc-check-cfg=cfg(turborerank_cuda)");
    println!("cargo:rustc-check-cfg=cfg(turborerank_openvino)");

    println!(
        "cargo:rustc-env=TURBORERANK_WORKSPACE_ROOT={}",
        root.display()
    );
    println!("cargo:rustc-env=INFERSTREAM_ROOT={}", root.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .include(root.join("include"))
        .include(root.join("native/wordpiece"))
        .include(root.join("native/turborerank/src"))
        .include(root.join("native/turbo_buffer/src"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
        .define(
            "TURBORERANK_WORKSPACE_ROOT",
            format!("\"{}\"", escape_c_string(&root.to_string_lossy())).as_str(),
        );
    if enable_cuda {
        build.define("TURBORERANK_CUDA", "1");
        build.define("TURBO_BUFFER_CUDA", "1");
    }
    if let Some(ref ov) = ov {
        build.define("TURBORERANK_OPENVINO", "1");
        for dir in &ov.include_dirs {
            build.include(dir);
        }
        if enable_l0 {
            build.define("TURBORERANK_LEVEL_ZERO", "1");
            build.define("TURBO_BUFFER_ZE", "1");
            build.include("/usr/include");
        }
    }
    if enable_metal {
        build.define("TURBORERANK_METAL", "1");
        build.define("TURBO_BUFFER_METAL", "1");
        build.flag("-fobjc-arc");
        build.file(root.join("native/turborerank/src/metal_api.mm"));
        build.file(root.join("native/turbo_buffer/src/metal.mm"));
    }
    for rel in sources {
        build.file(root.join(rel));
    }
    if std::env::var_os("CXX").is_none() && target_os == "linux" {
        build.compiler("g++");
    }
    if target_os == "macos" {
        println!("cargo:rustc-link-lib=c++");
        println!("cargo:rustc-link-lib=m");
        if enable_metal {
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
            println!("cargo:rustc-cfg=turborerank_metal");
        }
    } else {
        link_libstdcxx();
    }
    build.compile("turborerank_native");

    if enable_cuda {
        println!("cargo:rustc-cfg=turborerank_cuda");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cublasLt");
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true)
            .cpp(true)
            .std("c++17")
            .include(root.join("include"))
            .include(root.join("native/turborerank/src"))
            .include(root.join("native/turbo_buffer/src"))
            .define("TURBORERANK_CUDA", "1")
            .define(
                "TURBORERANK_WORKSPACE_ROOT",
                format!("\"{}\"", escape_c_string(&root.to_string_lossy())).as_str(),
            )
            .file(root.join("native/turborerank/src/bert_cuda.cu"))
            .flag("-O2")
            .flag("-arch=native")
            .flag("-allow-unsupported-compiler")
            .flag("-ccbin=g++-13")
            .flag_if_supported("--expt-relaxed-constexpr");
        nvcc.compile("turborerank_bert_cuda");
    }

    if let Some(ov) = ov {
        println!("cargo:rustc-cfg=turborerank_openvino");
        for dir in &ov.lib_dirs {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
        println!("cargo:rustc-link-lib=dylib=openvino");
        if enable_l0 {
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-lib=dylib=ze_loader");
        }
    }
}

fn escape_c_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn link_libstdcxx() {
    if let Ok(out) = Command::new("g++")
        .args(["-print-file-name=libstdc++.so"])
        .output()
    {
        if let Ok(path) = String::from_utf8(out.stdout) {
            if let Some(dir) = std::path::Path::new(path.trim()).parent() {
                if dir.join("libstdc++.so").exists() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
            }
        }
    }
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=m");
}
