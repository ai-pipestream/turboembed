//! Link the TurboEmbed C ABI.
//!
//! * **macOS:** `swift build` `libTurboEmbed.dylib` (`@_cdecl` → mlx-swift
//!   on Metal). Never link the C++ mock stub here — that would hide a fake
//!   `turboembed_embed("minilm")`.
//! * **elsewhere:** compile `native/turboembed/src/stub.cpp` (mock +
//!   NOT_IMPLEMENTED). No Python, no GPU libs.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest
        .join("../..")
        .canonicalize()
        .expect("workspace root");
    let header = root.join("include/turboembed.h");
    let stub = root.join("native/turboembed/src/stub.cpp");
    let apple = root.join("swift/Sources/TurboEmbedC/include/turboembed.h");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", stub.display());
    println!("cargo:rerun-if-changed={}", apple.display());
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rustc-env=INFERSTREAM_ROOT={}", root.display());

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        link_swift_mlx(&root);
    } else {
        compile_stub(&root, &stub);
    }
}

fn link_swift_mlx(root: &Path) {
    let swift_dir = root.join("swift");
    println!(
        "cargo:rerun-if-changed={}",
        swift_dir.join("Sources/TurboEmbed").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        swift_dir.join("Sources/MlxEngine").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        swift_dir.join("Package.swift").display()
    );

    let mut swift = Command::new("swift");
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
        if xcode.is_dir() {
            swift.env("DEVELOPER_DIR", &xcode);
        }
    }
    let status = swift
        .args([
            "build",
            "-c",
            "release",
            "--package-path",
        ])
        .arg(&swift_dir)
        .args(["--product", "TurboEmbed"])
        .status()
        .expect("failed to spawn `swift build --product TurboEmbed`");
    if !status.success() {
        panic!("swift build of TurboEmbed (MLX Metal) failed: {status}");
    }

    let search = swift_dir.join(".build/release");
    let dylib = search.join("libTurboEmbed.dylib");
    if !dylib.is_file() {
        panic!(
            "swift build did not emit {} — Metal MiniLM cannot be a stub",
            dylib.display()
        );
    }

    let metallib = Command::new("sh")
        .arg(root.join("scripts/build-apple-metallib.sh"))
        .arg(&search)
        .status()
        .expect("failed to spawn scripts/build-apple-metallib.sh");
    if !metallib.success() {
        panic!("build-apple-metallib.sh failed: {metallib}");
    }

    println!("cargo:rustc-link-search=native={}", search.display());
    println!("cargo:rustc-link-lib=dylib=TurboEmbed");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", search.display());

    if let Some(profile) = profile_dir() {
        stage_runtime(&search, &profile);
        let deps = profile.join("deps");
        if deps.is_dir() {
            stage_runtime(&search, &deps);
        }
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", profile.display());
    }
}

fn compile_stub(root: &Path, stub: &Path) {
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(stub)
        .include(root.join("include"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter");
    if std::env::var_os("CXX").is_none() && cfg!(target_os = "linux") {
        build.compiler("g++");
    }
    if let Ok(out) = Command::new("g++")
        .args(["-print-file-name=libstdc++.so"])
        .output()
    {
        if let Ok(path) = String::from_utf8(out.stdout) {
            if let Some(dir) = Path::new(path.trim()).parent() {
                if dir.join("libstdc++.so").exists() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
            }
        }
    }
    build.compile("turboembed_stub");
}

fn profile_dir() -> Option<PathBuf> {
    let out = PathBuf::from(std::env::var("OUT_DIR").ok()?);
    out.ancestors().nth(3).map(PathBuf::from)
}

fn stage_runtime(from: &Path, dest: &Path) {
    let _ = std::fs::create_dir_all(dest);
    let _ = std::fs::create_dir_all(dest.join("Resources"));
    for name in ["libTurboEmbed.dylib", "mlx.metallib", "default.metallib"] {
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
