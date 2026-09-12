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

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .file(&stub)
        .include(root.join("include"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
        .compile("turboembed_stub");
}
