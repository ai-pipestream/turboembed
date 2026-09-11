//! Compile-time cfg + SYCL/Level Zero/oneMKL link for in-process llama.cpp.
//!
//! llama-cpp-sys-2 forwards `GGML_*` into CMake and produces `libggml-sycl.a`.
//! That static archive does not pull `libsycl` / oneMKL / `ze_loader` into the
//! final rustc link (archives have no `DT_NEEDED`). This build.rs re-emits
//! those libraries so the Intel binary actually links SYCL instead of dying
//! at rust-lld with `sycl::_V1::*` undefined.
//!
//! Device images inside ggml-sycl objects still need `icpx -fsycl` as the
//! rustc linker — `scripts/build-intel.sh` / `scripts/icpx-rust-linker.sh`.
//! No python3.

use std::env;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_SYCL");
    println!("cargo:rerun-if-env-changed=CMAKE_C_COMPILER");
    println!("cargo:rerun-if-env-changed=CMAKE_CXX_COMPILER");
    println!("cargo:rerun-if-env-changed=ONEAPI_ROOT");
    println!("cargo:rerun-if-env-changed=CMPLR_ROOT");
    println!("cargo:rerun-if-env-changed=MKLROOT");
    println!("cargo:rerun-if-env-changed=TBBROOT");
    println!("cargo:rerun-if-env-changed=DNNLROOT");
    println!("cargo:rustc-check-cfg=cfg(llama_sycl)");

    let sycl = env::var("GGML_SYCL")
        .map(|v| matches!(v.as_str(), "ON" | "On" | "on" | "1" | "true"))
        .unwrap_or(false);
    if sycl {
        println!("cargo:rustc-cfg=llama_sycl");
        link_sycl();
    }
}

fn link_sycl() {
    let compiler_lib = require_dir_with(
        "libsycl.so",
        &[
            env_join("CMPLR_ROOT", "lib"),
            env_join("ONEAPI_ROOT", "compiler/latest/lib"),
            env_join("ONEAPI_ROOT", "compiler/2025.3/lib"),
            Some(PathBuf::from("/opt/intel/oneapi/compiler/latest/lib")),
            Some(PathBuf::from("/opt/intel/oneapi/compiler/2025.3/lib")),
        ],
        extra_compiler_libs(),
    );

    search(&compiler_lib);
    // Host SYCL runtime. Device-image extraction is icpx -fsycl (linker wrapper).
    dylib("sycl");
    dylib("OpenCL");
    // Intel compiler helper libs (needed when rust-lld is the driver).
    for lib in ["svml", "irng", "imf", "intlc"] {
        if compiler_lib.join(format!("lib{lib}.so")).exists() {
            dylib(lib);
        }
    }

    if let Some(ze) = first_existing(&[
        Some(PathBuf::from("/usr/lib/x86_64-linux-gnu/libze_loader.so")),
        env_join("ONEAPI_ROOT", "lib/libze_loader.so"),
    ]) {
        if let Some(dir) = ze.parent() {
            search(dir);
        }
    }
    dylib("ze_loader");

    if let Some(mkl_lib) = first_dir(&[
        env_join("MKLROOT", "lib"),
        env_join("ONEAPI_ROOT", "mkl/latest/lib"),
        env_join("ONEAPI_ROOT", "mkl/2025.3/lib"),
        Some(PathBuf::from("/opt/intel/oneapi/mkl/latest/lib")),
        Some(PathBuf::from("/opt/intel/oneapi/mkl/2025.3/lib")),
    ]) {
        search(&mkl_lib);
        // Matches ggml-sycl CMake: MKL::MKL_SYCL::BLAS, dynamic, tbb_thread, ilp64.
        for lib in [
            "mkl_sycl_blas",
            "mkl_sycl",
            "mkl_intel_ilp64",
            "mkl_tbb_thread",
            "mkl_core",
        ] {
            if mkl_lib.join(format!("lib{lib}.so")).exists()
                || mkl_lib.join(format!("lib{lib}.a")).exists()
            {
                dylib(lib);
            }
        }
    }

    if let Some(tbb_lib) = first_dir(&[
        env_join("TBBROOT", "lib"),
        env_join("ONEAPI_ROOT", "tbb/latest/lib"),
        env_join("ONEAPI_ROOT", "tbb/2022.3/lib"),
        Some(PathBuf::from("/opt/intel/oneapi/tbb/latest/lib")),
        Some(PathBuf::from("/opt/intel/oneapi/tbb/2022.3/lib")),
    ]) {
        search(&tbb_lib);
        dylib("tbb");
    }

    if let Some(dnnl_lib) = first_dir(&[
        env_join("DNNLROOT", "lib"),
        env_join("ONEAPI_ROOT", "dnnl/latest/lib"),
        env_join("ONEAPI_ROOT", "dnnl/2025.3/lib"),
        Some(PathBuf::from("/opt/intel/oneapi/dnnl/latest/lib")),
        Some(PathBuf::from("/opt/intel/oneapi/dnnl/2025.3/lib")),
    ]) {
        if dnnl_lib.join("libdnnl.so").exists() {
            search(&dnnl_lib);
            dylib("dnnl");
        }
    }

    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=m");
    println!("cargo:rustc-link-lib=dylib=dl");
}

fn extra_compiler_libs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    let root = Path::new("/opt/intel/oneapi/compiler");
    if let Ok(rd) = std::fs::read_dir(root) {
        for entry in rd.flatten() {
            let lib = entry.path().join("lib");
            if lib.join("libsycl.so").exists() {
                out.push(lib);
            }
        }
    }
    out
}

fn require_dir_with(file: &str, candidates: &[Option<PathBuf>], extra: Vec<PathBuf>) -> PathBuf {
    for c in candidates.iter().flatten().chain(extra.iter()) {
        if c.join(file).exists() {
            return c.clone();
        }
    }
    panic!(
        "GGML_SYCL=ON but {file} was not found. Source /opt/intel/oneapi/setvars.sh \
         and build with scripts/build-intel.sh (icx/icpx, no python3)."
    );
}

fn first_dir(candidates: &[Option<PathBuf>]) -> Option<PathBuf> {
    candidates.iter().flatten().find(|p| p.is_dir()).cloned()
}

fn first_existing(candidates: &[Option<PathBuf>]) -> Option<PathBuf> {
    candidates
        .iter()
        .flatten()
        .find(|p| p.exists())
        .cloned()
}

fn env_join(var: &str, rel: &str) -> Option<PathBuf> {
    env::var_os(var).map(|root| PathBuf::from(root).join(rel))
}

fn search(dir: &Path) {
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}

fn dylib(name: &str) {
    println!("cargo:rustc-link-lib=dylib={name}");
}
