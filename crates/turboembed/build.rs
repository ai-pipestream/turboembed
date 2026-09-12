//! Compile the C++ ABI stub and link it into this crate.
//!
//! Header: `include/turboembed.h`
//! Source: `native/turboembed/src/stub.cpp`
//! Needs a C++17 compiler (`g++` / `clang++`). No Python, no GPU libs.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let header = root.join("include/turboembed.h");
    let stub = root.join("native/turboembed/src/stub.cpp");
    let apple = root.join("swift/Sources/TurboEmbedC/include/turboembed.h");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", stub.display());
    println!("cargo:rerun-if-changed={}", apple.display());

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&stub)
        .include(root.join("include"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter");
    // This image's default `c++` is clang++ without a C++ stdlib. Prefer
    // g++ on Linux unless the caller set CXX.
    if std::env::var_os("CXX").is_none() && cfg!(target_os = "linux") {
        build.compiler("g++");
    }
    // rust-lld / clang `cc` look for -lstdc++ via libstdc++.so (the
    // unversioned symlink), which lives next to g++'s own libs — not
    // always on the default search path.
    if let Ok(out) = std::process::Command::new("g++")
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
    build.compile("turboembed_stub");
}
