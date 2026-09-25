//! Builds the CUDA backend (core/cuda/) when the `cuda` feature is on;
//! without it, nothing. docs/cuda.md says what it needs and which
//! variables it reads:
//!
//! - TURBO_CUDA_ROOT: the toolkit's directory, with bin/nvcc, include/ and
//!   lib64/ (or lib/, targets/<arch>-linux/lib/, or lib/<arch>-linux-gnu/,
//!   where a distribution's package puts it, so /usr serves). Else
//!   CUDA_PATH, else CUDA_HOME, else /usr/local/cuda.
//! - TURBO_CUDA_ARCH: the SM architectures to compile for, comma
//!   separated, as nvcc numbers them (89 for sm_89). Default 89. Each gets
//!   its machine code, and the highest its PTX too, so a newer device can
//!   compile that when it loads the library.
//!
//! It also compiles in the benchmark records in benchmarks/records/, with
//! or without the feature (docs/benchmarks.md).
//!
//! The kernels and the host side are compiled by nvcc into a static
//! library linked into libturbo, against the toolkit's shared cudart and
//! cuBLAS, with the toolkit's library directory as a run path.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCES: [&str; 2] = ["cuda/kernels.cu", "cuda/backend.cpp"];
const DEPENDS: [&str; 3] = ["cuda/kernels.h", "../include/turbo/turbo.h", "../include/turbo/turbo_backend.h"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    records();
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    for f in SOURCES.iter().chain(&DEPENDS) {
        println!("cargo:rerun-if-changed={f}");
    }
    for v in ["TURBO_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "TURBO_CUDA_ARCH", "NVCC_CCBIN"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let root = toolkit_root();
    let nvcc = root.join("bin/nvcc");
    if !nvcc.is_file() {
        fail(&format!(
            "the cuda feature needs nvcc, and {} is not there: set TURBO_CUDA_ROOT to the CUDA toolkit's directory",
            nvcc.display()
        ));
    }
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned());
    let lib = ["lib64", "lib", &format!("targets/{arch}-linux/lib"), &format!("lib/{arch}-linux-gnu")]
        .iter()
        .map(|d| root.join(d))
        .find(|d| d.join("libcudart.so").exists())
        .unwrap_or_else(|| {
            fail(&format!(
                "no libcudart.so under {}/lib64, lib, targets/{arch}-linux/lib or lib/{arch}-linux-gnu",
                root.display()
            ))
        });
    let archs = archs();

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let list = archs.join(",");
    std::fs::write(
        out.join("turbo_cuda_build.h"),
        format!("/* Written by build.rs. */\n#define TURBO_CUDA_ARCHS \"{list}\"\n"),
    )
    .unwrap();
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let include = manifest.join("../include");
    let mut objects = Vec::new();
    for src in SOURCES {
        let obj = out.join(Path::new(src).file_name().unwrap()).with_extension("o");
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-c", "-O3", "-std=c++17", "-Xcompiler", "-fPIC,-Wall,-Wextra"])
            .arg("-I")
            .arg(&include)
            .arg("-I")
            .arg(&out)
            .arg("-o")
            .arg(&obj)
            .arg(manifest.join(src));
        if src.ends_with(".cu") {
            for (i, a) in archs.iter().enumerate() {
                let code = if i + 1 == archs.len() { format!("[sm_{a},compute_{a}]") } else { format!("sm_{a}") };
                cmd.arg("-gencode").arg(format!("arch=compute_{a},code={code}"));
            }
        }
        run(&mut cmd);
        objects.push(obj);
    }
    let archive = out.join("libturbo_cuda.a");
    let _ = std::fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_owned());
    run(Command::new(ar).arg("crs").arg(&archive).args(&objects));

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=turbo_cuda");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
}

/// Every `*.json` file in benchmarks/records/ as `EMBEDDED`, its file name
/// and its text, sorted by name, in OUT_DIR/records.rs. The core parses
/// them when a capability is first asked for; the library opens no file.
fn records() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let dir = manifest.join("../benchmarks/records");
    // A directory: cargo looks at every file under it.
    println!("cargo:rerun-if-changed={}", dir.display());
    let mut files: Vec<PathBuf> =
        std::fs::read_dir(&dir).map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect()).unwrap_or_default();
    files.retain(|p| p.is_file() && p.extension().is_some_and(|e| e == "json"));
    files.sort();
    let mut code = String::from("/* Written by build.rs. */\npub static EMBEDDED: &[(&str, &str)] = &[\n");
    for p in files {
        let name = p.file_name().unwrap().to_str().unwrap_or_else(|| fail(&format!("{}: not UTF-8", p.display())));
        let path = p.canonicalize().unwrap_or_else(|e| fail(&format!("{}: {e}", p.display())));
        let path = path.to_str().unwrap_or_else(|| fail(&format!("{}: not UTF-8", p.display())));
        // A record that is not even JSON fails the build here, by name; the
        // core's parser judges the rest when a capability is asked for.
        let bytes = std::fs::read(&p).unwrap_or_else(|e| fail(&format!("benchmarks/records/{name}: {e}")));
        let text =
            std::str::from_utf8(&bytes).unwrap_or_else(|e| fail(&format!("benchmarks/records/{name}: not UTF-8: {e}")));
        if let Err(e) = serde_json::from_str::<serde_json::Value>(text) {
            fail(&format!("benchmarks/records/{name}: not JSON: {e}"));
        }
        code.push_str(&format!("    ({name:?}, include_str!({path:?})),\n"));
    }
    code.push_str("];\n");
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    std::fs::write(out.join("records.rs"), code).unwrap();
}

fn toolkit_root() -> PathBuf {
    ["TURBO_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME"]
        .iter()
        .find_map(|v| env::var_os(v).filter(|s| !s.is_empty()))
        .map_or_else(|| PathBuf::from("/usr/local/cuda"), PathBuf::from)
}

/// TURBO_CUDA_ARCH as numbers nvcc takes after sm_, lowest first.
fn archs() -> Vec<String> {
    let given = env::var("TURBO_CUDA_ARCH").unwrap_or_else(|_| "89".to_owned());
    let mut archs: Vec<String> = given.split(',').map(|a| a.trim().trim_start_matches("sm_").to_owned()).collect();
    let digits = |a: &str| a.len() >= 2 && a.chars().take_while(char::is_ascii_digit).count() >= 2;
    if let Some(bad) = archs.iter().find(|a| !digits(a) || !a.chars().all(|c| c.is_ascii_alphanumeric())) {
        fail(&format!("TURBO_CUDA_ARCH={given}: {bad:?} is not an SM architecture such as 89"));
    }
    let number = |a: &String| a.chars().take_while(char::is_ascii_digit).collect::<String>().parse::<u32>().unwrap();
    archs.sort_by_key(number);
    archs.dedup();
    archs
}

/// Run a compiler step; its warnings are shown, and a failure stops the build.
fn run(cmd: &mut Command) {
    let out = cmd.output().unwrap_or_else(|e| fail(&format!("{cmd:?}: {e}")));
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        fail(&format!("{cmd:?} failed:\n{}{stderr}", String::from_utf8_lossy(&out.stdout)));
    }
    for line in stderr.lines().filter(|l| !l.trim().is_empty()) {
        println!("cargo:warning={line}");
    }
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    std::process::exit(1);
}
