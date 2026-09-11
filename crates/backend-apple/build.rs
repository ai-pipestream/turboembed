use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=../../native/mlx-engine/Sources");
    println!("cargo:rerun-if-changed=../../native/mlx-engine/Package.swift");
    println!("cargo:rerun-if-changed=../../native/mlx-engine/include/mlx_engine.h");

    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }

    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let engine_dir = manifest_dir.join("../../native/mlx-engine");
    let status = Command::new("swift")
        .args(["build", "-c", "release", "--package-path"])
        .arg(&engine_dir)
        .status()
        .expect("failed to spawn `swift build` for native/mlx-engine");
    if !status.success() {
        panic!("swift build of native/mlx-engine failed: {status}");
    }

    let search = engine_dir.join(".build/release");
    println!("cargo:rustc-link-search=native={}", search.display());
    println!("cargo:rustc-link-lib=dylib=MlxEngine");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", search.display());
}
