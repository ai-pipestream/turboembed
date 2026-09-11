//! Compile-time cfg so device = "sycl" can refuse a CPU-only llama.cpp link.
//!
//! llama-cpp-sys-2 forwards `GGML_*` environment variables into CMake. A
//! SYCL build must set `GGML_SYCL=ON` (and typically `CMAKE_C_COMPILER=icx`
//! / `CMAKE_CXX_COMPILER=icpx`) *and* inject the ggml-sycl sources that
//! crates.io omits — `scripts/setup-llamacpp-sycl.sh` does both.

fn main() {
    println!("cargo:rerun-if-env-changed=GGML_SYCL");
    println!("cargo:rerun-if-env-changed=CMAKE_C_COMPILER");
    println!("cargo:rerun-if-env-changed=CMAKE_CXX_COMPILER");
    let sycl = std::env::var("GGML_SYCL")
        .map(|v| matches!(v.as_str(), "ON" | "On" | "on" | "1" | "true"))
        .unwrap_or(false);
    println!("cargo:rustc-check-cfg=cfg(llama_sycl)");
    if sycl {
        println!("cargo:rustc-cfg=llama_sycl");
    }
}
