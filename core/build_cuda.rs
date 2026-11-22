//! The CUDA backend's build, for `cuda` (build.rs says what it reads).
//! nvcc drives the platform's host compiler: gcc or clang on Linux, MSVC's
//! cl.exe on Windows (#2453), whose flags, library names and archiver differ.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::{fail, object, run};

const SOURCES: [&str; 6] =
    ["cuda/kernels.cu", "cuda/deberta.cu", "cuda/backend.cpp", "cuda/autotune.cpp", "cuda/lstm.cu", "cuda/rotary.cpp"];
const DEPENDS: [&str; 6] =
    ["cuda/kernels.h", "cuda/deberta.h", "cuda/autotune.h", "cuda/rotary.h", "cuda/rotary.cuh", "cuda/portable.h"];
const API: [&str; 2] = ["../include/turbo/turbo.h", "../include/turbo/turbo_backend.h"];

/// What differs between the Linux (gcc or clang) and Windows (MSVC) builds.
struct Host {
    /// nvcc's -Xcompiler flags for the host compiler.
    host_flags: &'static str,
    /// The toolkit's library directories to look in, under its root.
    lib_dirs: Vec<String>,
    /// The static runtime and the shared cuBLAS's link files.
    cudart: &'static str,
    cublas: &'static str,
}

fn host() -> Host {
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_else(|_| env::consts::ARCH.to_owned());
    if msvc() {
        // /MD or /MT: the C runtime rustc links, dynamic unless the
        // target has crt-static; /EHsc: C++ exceptions; /Zc:__cplusplus:
        // the real language level, which CUTLASS reads; /bigobj: the
        // kernels' template instances.
        return Host {
            host_flags: if crt_static() {
                "/MT,/EHsc,/Zc:__cplusplus,/bigobj,/W3"
            } else {
                "/MD,/EHsc,/Zc:__cplusplus,/bigobj,/W3"
            },
            lib_dirs: vec!["lib/x64".to_owned()],
            cudart: "cudart_static.lib",
            cublas: "cublas.lib",
        };
    }
    Host {
        host_flags: "-fPIC,-Wall,-Wextra",
        lib_dirs: vec![
            "lib64".to_owned(),
            "lib".to_owned(),
            format!("targets/{arch}-linux/lib"),
            format!("lib/{arch}-linux-gnu"),
        ],
        cudart: "libcudart_static.a",
        cublas: "libcublas.so",
    }
}

fn msvc() -> bool {
    env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc")
}

fn crt_static() -> bool {
    env::var("CARGO_CFG_TARGET_FEATURE").is_ok_and(|f| f.split(',').any(|x| x == "crt-static"))
}

pub(crate) fn cuda() {
    for f in SOURCES.iter().chain(&DEPENDS).chain(&API) {
        println!("cargo:rerun-if-changed={f}");
    }
    for v in ["TURBO_CUDA_ROOT", "CUDA_PATH", "CUDA_HOME", "TURBO_CUDA_ARCH", "NVCC_CCBIN"] {
        println!("cargo:rerun-if-env-changed={v}");
    }

    let root = toolkit_root();
    let nvcc = root.join(if msvc() { "bin/nvcc.exe" } else { "bin/nvcc" });
    if !nvcc.is_file() {
        fail(&format!(
            "the cuda feature needs nvcc, and {} is not there: set TURBO_CUDA_ROOT to the CUDA toolkit's directory",
            nvcc.display()
        ));
    }
    let host = host();
    let cublas = env::var_os("CARGO_FEATURE_CUDA_CUBLAS").is_some();
    let needed = if cublas { format!("{} and {}", host.cudart, host.cublas) } else { host.cudart.to_owned() };
    let lib = host
        .lib_dirs
        .iter()
        .map(|d| root.join(d))
        .find(|d| d.join(host.cudart).exists() && (!cublas || d.join(host.cublas).exists()))
        .unwrap_or_else(|| fail(&format!("no {needed} under {}/{}", root.display(), host.lib_dirs.join(", "))));
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
    let cutlass = manifest.join("cuda/cutlass/include");
    let mut objects = Vec::new();
    for src in SOURCES {
        let obj = object(&out, src);
        let mut cmd = Command::new(&nvcc);
        cmd.args(["-c", "-O3", "-std=c++17", "-Xcompiler", host.host_flags]);
        if cublas {
            cmd.arg("-DTURBO_CUDA_CUBLAS=1");
        }
        if msvc() {
            // MSVC deprecates the standard fopen and getenv (C4996) in
            // favour of its own _s functions; the standard ones stay.
            cmd.arg("-D_CRT_SECURE_NO_WARNINGS");
        }
        cmd.arg("-I")
            .arg(&include)
            .arg("-isystem")
            .arg(&cutlass)
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
    archive(&out, &objects);

    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=turbo_cuda");
    println!("cargo:rustc-link-search=native={}", lib.display());
    // The runtime linked in, so the library needs the driver alone; the
    // runtime's own needs after it. MSVC's C++ and C runtimes come with
    // the objects' default libraries.
    println!("cargo:rustc-link-lib=static=cudart_static");
    // NVIDIA builds cudart_static.lib with /MT, so it asks for LIBCMT,
    // the static C runtime; with the dynamic one that is a second C
    // runtime (LNK4098). This crate's own links drop it here, and
    // backend.cpp carries the same directive for a program that links
    // the crate (#2453).
    if msvc() && !crt_static() {
        println!("cargo:rustc-link-arg=/NODEFAULTLIB:libcmt");
    }
    if !msvc() {
        for l in ["dl", "pthread", "rt"] {
            println!("cargo:rustc-link-lib=dylib={l}");
        }
    }
    if cublas {
        // cuBLAS stays the toolkit's shared library: found through its
        // directory as a run path on Linux, through PATH on Windows.
        println!("cargo:rustc-link-lib=dylib=cublas");
        if !msvc() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
        }
    }
    if !msvc() {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
}

/// The objects as one static library rustc links as `turbo_cuda`:
/// libturbo_cuda.a by ar, or turbo_cuda.lib by MSVC's lib.exe.
fn archive(out: &Path, objects: &[PathBuf]) {
    if msvc() {
        let archive = out.join("turbo_cuda.lib");
        let _ = std::fs::remove_file(&archive);
        let mut out_arg = std::ffi::OsString::from("/OUT:");
        out_arg.push(&archive);
        run(Command::new("lib.exe").arg("/NOLOGO").arg(out_arg).args(objects));
        return;
    }
    let archive = out.join("libturbo_cuda.a");
    let _ = std::fs::remove_file(&archive);
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_owned());
    run(Command::new(ar).arg("crs").arg(&archive).args(objects));
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
