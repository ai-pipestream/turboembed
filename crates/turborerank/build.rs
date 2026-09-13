//! Compile native/turborerank into the Rust crate.

use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let root = root.canonicalize().unwrap_or(root);

    let sources = [
        "native/turborerank/src/alloc.cpp",
        "native/turborerank/src/pack.cpp",
        "native/turborerank/src/wordpiece.cpp",
        "native/turborerank/src/safetensors.cpp",
        "native/turborerank/src/bert_cpu.cpp",
        "native/turborerank/src/engine.cpp",
    ];
    for rel in sources {
        println!("cargo:rerun-if-changed={}", root.join(rel).display());
    }
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/turborerank.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/reranker.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turborerank/src/internal.hpp").display()
    );

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
        .include(root.join("native/turborerank/src"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
        .define(
            "TURBORERANK_WORKSPACE_ROOT",
            format!("\"{}\"", escape_c_string(&root.to_string_lossy())).as_str(),
        );
    for rel in sources {
        build.file(root.join(rel));
    }
    if std::env::var_os("CXX").is_none() && cfg!(target_os = "linux") {
        build.compiler("g++");
    }
    link_libstdcxx();
    build.compile("turborerank_native");
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
