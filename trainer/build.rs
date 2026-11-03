//! Builds the CUDA training kernels (trainer/cuda/train.cu) for the
//! features that ask for them:
//!
//! - `cuda`: nvcc compiles them for the device, into a static library
//!   linked with the toolkit's static cudart, as core/build.rs builds the
//!   CUDA backend. The toolkit is found the same way: TURBO_CUDA_ROOT,
//!   else CUDA_PATH, else CUDA_HOME, else /usr/local/cuda; TURBO_CUDA_ARCH
//!   names the SM architectures (default 89). docs/cuda.md says more.
//! - `cuda-shim`: the host C++ compiler (CXX, else c++) compiles the same
//!   source as plain C++ through trainer/cuda/shim.h, where a launch is
//!   loops over its threads and device memory is host memory, so the
//!   kernels and the host path run, and are tested, on any machine.
//!
//! Both at once is refused: one library serves the C API.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

const SOURCE: &str = "cuda/train.cu";
const DEPENDS: [&str; 2] = ["cuda/train.h", "cuda/shim.h"];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let cuda = env::var_os("CARGO_FEATURE_CUDA").is_some();
    let shim = env::var_os("CARGO_FEATURE_CUDA_SHIM").is_some();
    if cuda && shim {
        fail("the cuda and cuda-shim features are one or the other, not both");
    }
    if cuda || shim {
        for f in DEPENDS.iter().chain([&SOURCE]) {
            println!("cargo:rerun-if-changed={f}");
        }
        for v in ["TURBO_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "TURBO_CUDA_ARCH", "CXX"] {
            println!("cargo:rerun-if-env-changed={v}");
        }
    }
    if cuda {
        device();
    } else if shim {
        host();
    }
}

/// The kernels for the device, through nvcc.
fn device() {
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
        .find(|d| d.join("libcudart_static.a").exists())
        .unwrap_or_else(|| {
            fail(&format!(
                "no libcudart_static.a under {}/lib64, lib, targets/{arch}-linux/lib or lib/{arch}-linux-gnu",
                root.display()
            ))
        });
    let archs = archs();
    let (out, manifest) = dirs();
    let obj = out.join("train.o");
    let mut cmd = Command::new(&nvcc);
    cmd.args(["-c", "-O3", "-std=c++17", "-Xcompiler", "-fPIC,-Wall,-Wextra"]).arg("-o").arg(&obj);
    cmd.arg(manifest.join(SOURCE));
    for (i, a) in archs.iter().enumerate() {
        let code = if i + 1 == archs.len() { format!("[sm_{a},compute_{a}]") } else { format!("sm_{a}") };
        cmd.arg("-gencode").arg(format!("arch=compute_{a},code={code}"));
    }
    run(&mut cmd);
    archive(&out, &obj);
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=cudart_static");
    for l in ["dl", "pthread", "rt"] {
        println!("cargo:rustc-link-lib=dylib={l}");
    }
}

/// The kernels as plain C++ on the host, through the shim.
fn host() {
    let (out, manifest) = dirs();
    let cxx = env::var("CXX").ok().filter(|s| !s.is_empty()).unwrap_or_else(|| "c++".to_owned());
    let obj = out.join("train.o");
    let mut cmd = Command::new(cxx);
    cmd.args(["-c", "-O2", "-std=c++17", "-fPIC", "-Wall", "-Wextra", "-x", "c++", "-include"])
        .arg(manifest.join("cuda/shim.h"))
        .arg("-o")
        .arg(&obj)
        .arg(manifest.join(SOURCE));
    run(&mut cmd);
    archive(&out, &obj);
}

/// OUT_DIR and the crate's directory.
fn dirs() -> (PathBuf, PathBuf) {
    (PathBuf::from(env::var_os("OUT_DIR").unwrap()), PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()))
}

/// The object into a static library the crate links.
fn archive(out: &Path, obj: &Path) {
    let archive = out.join("libturbo_train_cuda.a");
    let _ = std::fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_owned());
    run(Command::new(ar).arg("crs").arg(&archive).arg(obj));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=turbo_train_cuda");
    if env::var_os("CARGO_FEATURE_CUDA_SHIM").is_some() {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
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

/// Runs a compiler step; its warnings are shown, and a failure stops the build.
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
