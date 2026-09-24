//! Build script for the CUDA provider.
//!
//! Compiles the device kernels with `nvcc` (architectures from `TURBO_CUDA_ARCHS`,
//! default `87;89` for Jetson Orin and Ada, plus PTX for forward compatibility),
//! the in-tree WordPiece encoder and utf8proc with the C/C++ compiler, and
//! links the CUDA runtime. The ONNX Runtime library itself comes from the
//! `ort` crate's prebuilt CUDA bundle; the CUDA 13 user-space libraries it
//! opens at startup are fetched by `scripts/fetch-cuda-runtime.sh`.
//!
//! Every failure here is fatal and names the missing piece; the provider is
//! never built silently without its device path.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..").canonicalize().expect("repository root");
    let out = PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    println!("cargo:rerun-if-env-changed=TURBO_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-changed=src/kernels.cu");
    println!("cargo:rerun-if-changed=build.rs");
    for f in ["vocab_load.cpp", "encode.cpp", "vocab.hpp", "include/wordpiece.h"] {
        println!("cargo:rerun-if-changed={}", root.join("native/wordpiece").join(f).display());
    }

    // --- CUDA toolkit ---------------------------------------------------
    let cuda_home = std::env::var("CUDA_PATH")
        .map(PathBuf::from)
        .ok()
        .or_else(|| Some(PathBuf::from("/usr/local/cuda")).filter(|p| p.join("bin/nvcc").exists()))
        .or_else(|| Some(PathBuf::from("/usr")).filter(|p| p.join("bin/nvcc").exists()));
    let Some(cuda_home) = cuda_home else {
        panic!("nvcc not found: set CUDA_PATH to a CUDA toolkit (needs nvcc and cuda_runtime.h)");
    };
    let nvcc = cuda_home.join("bin/nvcc");
    let include = [cuda_home.join("include"), PathBuf::from("/usr/include")]
        .into_iter()
        .find(|p| p.join("cuda_runtime.h").exists())
        .unwrap_or_else(|| panic!("cuda_runtime.h not found under {} or /usr/include", cuda_home.display()));

    // --- kernels ----------------------------------------------------------
    let archs = std::env::var("TURBO_CUDA_ARCHS").unwrap_or_else(|_| "87;89".to_string());
    let kernel_obj = out.join("kernels.o");
    let mut cmd = Command::new(&nvcc);
    cmd.arg("-c")
        .arg(manifest.join("src/kernels.cu"))
        .arg("-o")
        .arg(&kernel_obj)
        .arg("-O3")
        .arg("--compiler-options")
        .arg("-fPIC")
        .arg("-std=c++17")
        .arg("-I")
        .arg(&include);
    let mut last = String::new();
    for a in archs.split(';').map(str::trim).filter(|a| !a.is_empty()) {
        cmd.arg(format!("-gencode=arch=compute_{a},code=sm_{a}"));
        last = a.to_string();
    }
    if !last.is_empty() {
        cmd.arg(format!("-gencode=arch=compute_{last},code=compute_{last}"));
    }
    let status = cmd.status().unwrap_or_else(|e| panic!("run {}: {e}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed to compile src/kernels.cu ({status})");
    }
    let kernel_lib = out.join("libturbo_cuda_kernels.a");
    let ar = Command::new("ar").arg("crs").arg(&kernel_lib).arg(&kernel_obj).status().expect("run ar");
    if !ar.success() {
        panic!("ar failed to archive the CUDA kernels");
    }
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=turbo_cuda_kernels");

    // --- WordPiece + utf8proc ---------------------------------------------
    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(root.join("native/wordpiece/vocab_load.cpp"))
        .file(root.join("native/wordpiece/encode.cpp"))
        .include(root.join("native/wordpiece"))
        .include(root.join("native/wordpiece/include"))
        .include(root.join("third_party"))
        .include(root.join("third_party/utf8proc"))
        .define("UTF8PROC_STATIC", None)
        .flag_if_supported("-fno-strict-aliasing")
        .warnings(false)
        .compile("turbo_cuda_wordpiece");
    cc::Build::new()
        .file(root.join("third_party/utf8proc/utf8proc.c"))
        .include(root.join("third_party/utf8proc"))
        .define("UTF8PROC_STATIC", None)
        .warnings(false)
        .compile("turbo_cuda_utf8proc");

    // --- CUDA runtime -----------------------------------------------------
    let lib_dir = [
        cuda_home.join("lib64"),
        cuda_home.join("lib"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/lib/aarch64-linux-gnu"),
    ]
    .into_iter()
    .find(|p: &PathBuf| p.join("libcudart.so").exists())
    .unwrap_or_else(|| panic!("libcudart.so not found under {}", cuda_home.display()));
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    let _ = Path::new("/");
}
