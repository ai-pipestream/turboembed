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
    let engine_dir = manifest_dir
        .join("../../native/mlx-engine")
        .canonicalize()
        .expect("native/mlx-engine");
    println!(
        "cargo:rerun-if-changed={}",
        engine_dir.join("build-metallib.sh").display()
    );

    let mut swift = Command::new("swift");
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
        if xcode.is_dir() {
            swift.env("DEVELOPER_DIR", &xcode);
        }
    }
    let status = swift
        .args(["build", "-c", "release", "--package-path"])
        .arg(&engine_dir)
        .status()
        .expect("failed to spawn `swift build` for native/mlx-engine");
    if !status.success() {
        panic!("swift build of native/mlx-engine failed: {status}");
    }

    let search = engine_dir.join(".build/release");
    let metallib = Command::new("sh")
        .arg(engine_dir.join("build-metallib.sh"))
        .arg(&search)
        .status()
        .expect("failed to spawn native/mlx-engine/build-metallib.sh");
    if !metallib.success() {
        panic!("build-metallib.sh failed: {metallib}");
    }

    println!("cargo:rustc-link-search=native={}", search.display());
    println!("cargo:rustc-link-lib=dylib=MlxEngine");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", search.display());

    // This package's tests pick up rustc-link-arg. The inferstream-apple
    // binary does not, so also drop the dylib + metallib next to cargo
    // artifacts (`@loader_path`) and emit the profile dir as metadata.
    if let Some(profile) = profile_dir() {
        stage_runtime(&search, &profile);
        let deps = profile.join("deps");
        if deps.is_dir() {
            stage_runtime(&search, &deps);
        }
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile.display());
    }
}

fn profile_dir() -> Option<PathBuf> {
    let out = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    out.ancestors().nth(3).map(PathBuf::from)
}

fn stage_runtime(from: &std::path::Path, dest: &std::path::Path) {
    let _ = std::fs::create_dir_all(dest);
    let _ = std::fs::create_dir_all(dest.join("Resources"));
    for name in ["libMlxEngine.dylib", "mlx.metallib", "default.metallib"] {
        let src = from.join(name);
        if src.is_file() {
            let _ = std::fs::copy(&src, dest.join(name));
        }
    }
    for name in ["mlx.metallib", "default.metallib"] {
        let src = from.join(name);
        if src.is_file() {
            let _ = std::fs::copy(&src, dest.join("Resources").join(name));
        }
    }
}
