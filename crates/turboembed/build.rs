//! Link the TurboEmbed C ABI.
//!
//! * **macOS:** `swift build` `libTurboEmbed.dylib` (`@_cdecl` → mlx-swift
//!   on Metal). Never link the C++ mock stub here — that would hide a fake
//!   `turboembed_embed("minilm")`.
//! * **elsewhere:** compile `native/turboembed/src/stub.cpp` (mock + dispatch).
//!   Feature `genai`: also `native/turboembed/src/genai.cpp`.
//!   Feature `ort-cuda`: define `TURBOEMBED_ORT_CUDA` for the ORT hooks.
//!   Fails the GenAI build if OpenVINO is missing — never a silent stub.

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let root = root.canonicalize().unwrap_or(root);
    let header = root.join("include/turboembed.h");
    let buffer_header = root.join("include/turbo_buffer.h");
    let stub = root.join("native/turboembed/src/stub.cpp");
    let genai_cpp = root.join("native/turboembed/src/genai.cpp");
    let genai_hpp = root.join("native/turboembed/src/genai.hpp");
    let apple = root.join("swift/Sources/TurboEmbedC/include/turboembed.h");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", buffer_header.display());
    println!("cargo:rerun-if-changed={}", stub.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/arena.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/cuda.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/ze.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/metal.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turbo_buffer/src/metal.mm").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("scripts/build-turbo-buffer-apple.sh").display()
    );
    println!("cargo:rerun-if-changed={}", genai_cpp.display());
    println!("cargo:rerun-if-changed={}", genai_hpp.display());
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turboembed/src/hailo.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turboembed/src/hailo.hpp").display()
    );
    println!("cargo:rerun-if-env-changed=HAILORT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=HAILORT_INCLUDE_DIR");
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/wordpiece/vocab_load.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/wordpiece/encode.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("third_party/utf8proc/utf8proc.c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("third_party/utf8proc/utf8proc.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("third_party/utf8proc/utf8proc_data.c").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("third_party/nlohmann/json.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/wordpiece/bert_unicode_categories.hpp")
            .display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/wordpiece/vocab.hpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("include/wordpiece.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turboembed/src/pool_cuda.cu").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        root.join("native/turboembed/src/pool_cuda.h").display()
    );
    println!("cargo:rerun-if-changed={}", apple.display());
    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OPENVINO_GENAI_DIR");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OpenVINO_DIR");
    println!("cargo:rerun-if-env-changed=DEVELOPER_DIR");
    println!("cargo:rerun-if-env-changed=TURBOEMBED_DISABLE_CUDA");
    println!("cargo:rerun-if-env-changed=TURBOEMBED_DISABLE_ZE");
    println!("cargo:rustc-check-cfg=cfg(turboembed_cuda)");

    println!(
        "cargo:rustc-env=TURBOEMBED_WORKSPACE_ROOT={}",
        root.display()
    );
    println!("cargo:rustc-env=INFERSTREAM_ROOT={}", root.display());

    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rerun-if-env-changed=TURBOEMBED_PREPARED_SDK");
    if std::env::var_os("CARGO_FEATURE_PREPARED").is_some() {
        assert_eq!(
            target_os, "linux",
            "the prepared SDK currently targets Linux"
        );
        let prefix = PathBuf::from(
            std::env::var_os("TURBOEMBED_PREPARED_SDK")
                .expect("set TURBOEMBED_PREPARED_SDK to the installed native SDK prefix"),
        );
        let lib = prefix.join("lib");
        assert!(
            lib.join("libturboembed_prepared.so").is_file(),
            "prepared SDK library is missing"
        );
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=dylib=turboembed_prepared");
    }
    if target_os == "macos" {
        link_swift_mlx(&root);
    } else {
        compile_stub(&root, &stub, &genai_cpp);
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

    let archive = Command::new("sh")
        .arg(root.join("scripts/build-turbo-buffer-apple.sh"))
        .status()
        .expect("failed to spawn scripts/build-turbo-buffer-apple.sh");
    if !archive.success() {
        panic!("libturbo_buffer_apple.a (Metal SHARED arena) failed: {archive}");
    }

    let mut swift = Command::new("swift");
    if std::env::var_os("DEVELOPER_DIR").is_none() {
        let xcode = PathBuf::from("/Applications/Xcode.app/Contents/Developer");
        if xcode.is_dir() {
            swift.env("DEVELOPER_DIR", &xcode);
        }
    }
    let status = swift
        .args(["build", "-c", "release", "--package-path"])
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

fn compile_stub(root: &Path, stub: &Path, genai_cpp: &Path) {
    let genai = std::env::var("CARGO_FEATURE_GENAI").is_ok();
    let ort_cuda = std::env::var("CARGO_FEATURE_ORT_CUDA").is_ok();
    let hailo = std::env::var("CARGO_FEATURE_HAILO").is_ok();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(stub)
        .file(root.join("third_party/utf8proc/utf8proc.c"))
        .file(root.join("native/wordpiece/vocab_load.cpp"))
        .file(root.join("native/wordpiece/encode.cpp"))
        .file(root.join("native/turbo_buffer/src/arena.cpp"))
        .file(root.join("native/turbo_buffer/src/cuda.cpp"))
        .file(root.join("native/turbo_buffer/src/ze.cpp"))
        .file(root.join("native/turbo_buffer/src/metal.cpp"))
        .include(root.join("include"))
        .include(root.join("third_party"))
        .include(root.join("native/wordpiece"))
        .include(root.join("native/turboembed/src"))
        .include(root.join("native/turbo_buffer/src"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
        .define("UTF8PROC_STATIC", None)
        .define(
            "TURBOEMBED_WORKSPACE_ROOT",
            format!("\"{}\"", escape_c_string(&root.to_string_lossy())).as_str(),
        );

    if std::env::var_os("CXX").is_none() && cfg!(target_os = "linux") {
        build.compiler("g++");
    }
    link_libstdcxx();

    if ort_cuda {
        build.define("TURBOEMBED_ORT_CUDA", None);
        if cuda_enabled() {
            build.define("TURBO_BUFFER_CUDA", "1");
            build.include("/usr/include");
            build.include("/usr/local/cuda/include");
            println!("cargo:rustc-cfg=turboembed_cuda");
            println!("cargo:rustc-link-lib=cudart");
            if let Some(dir) = cuda_lib_dir() {
                println!("cargo:rustc-link-search=native={dir}");
            }
        }
    }

    if genai {
        let ov = match find_openvino() {
            Ok(v) => v,
            Err(msg) => {
                eprintln!(
                    "error: feature `genai` is enabled but OpenVINO GenAI was not found.\n\
                     {msg}\n\
                     Install OpenVINO + OpenVINO GenAI, source setupvars.sh, \
                     and rebuild. See docs/intel-genai-embed.md."
                );
                std::process::exit(1);
            }
        };
        build
            .file(genai_cpp)
            .define("TURBOEMBED_GENAI", None)
            .flag_if_supported("-Wno-missing-field-initializers");
        for dir in &ov.include_dirs {
            build.include(dir);
        }
        for dir in &ov.lib_dirs {
            println!("cargo:rustc-link-search=native={dir}");
            println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        }
        println!("cargo:rustc-link-lib=dylib=openvino");
        println!("cargo:rustc-link-lib=dylib=openvino_genai");
        if ov.has_tokenizers {
            println!("cargo:rustc-link-lib=dylib=openvino_tokenizers");
        }
        if level_zero_present() {
            build.define("TURBO_BUFFER_ZE", "1");
            build.include("/usr/include");
            println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
            println!("cargo:rustc-link-lib=dylib=ze_loader");
        }
    }

    if hailo {
        // Raspberry Pi AI HAT+ provider. HailoRT ships headers at
        // /usr/include/hailo/ and a *versioned* soname only
        // (libhailort.so.4.x for Hailo-8/8L, libhailort.so.5.x for
        // Hailo-10H; no .so symlink, no pkg-config), so locate both
        // explicitly. Overrides: HAILORT_INCLUDE_DIR / HAILORT_LIB_DIR.
        let include_dirs: Vec<PathBuf> = match std::env::var_os("HAILORT_INCLUDE_DIR") {
            Some(dir) => vec![PathBuf::from(dir)],
            None => ["/usr/include", "/usr/local/include"]
                .iter()
                .map(PathBuf::from)
                .filter(|d| d.join("hailo/hailort.h").is_file())
                .collect(),
        };
        if include_dirs.is_empty() {
            eprintln!(
                "error: feature `hailo` is enabled but hailo/hailort.h was not found.\n\
                 Install the HailoRT dev files on the Pi: `sudo apt install dkms hailo-all` \\\n                 (Hailo-8/8L) or `hailo-h10-all` (Hailo-10H), or set HAILORT_INCLUDE_DIR.\n\
                 See docs/hailo-embed.md."
            );
            std::process::exit(1);
        }
        for dir in &include_dirs {
            build.include(dir);
        }
        let mut linked = false;
        let mut search_dirs: Vec<PathBuf> = Vec::new();
        if let Some(dir) = std::env::var_os("HAILORT_LIB_DIR") {
            search_dirs.push(PathBuf::from(dir));
        }
        for cand in ["/usr/lib", "/usr/local/lib", "/usr/lib/aarch64-linux-gnu"] {
            search_dirs.push(PathBuf::from(cand));
        }
        for dir in &search_dirs {
            if let Some(soname) = newest_hailort_soname(dir) {
                println!("cargo:rustc-link-search=native={}", dir.display());
                // The Pi debs ship only a versioned soname (no .so symlink),
                // and cargo's link-lib parser rejects the `-l:` exact-name
                // form — pass the shared object to the linker by full path.
                println!("cargo:rustc-link-arg={}", dir.join(&soname).display());
                linked = true;
                break;
            }
        }
        if !linked {
            eprintln!(
                "error: feature `hailo` is enabled but libhailort.so.* was not found in \
                 /usr/lib, /usr/local/lib, /usr/lib/aarch64-linux-gnu.\n\
                 Install HailoRT on the Pi (`sudo apt install hailo-all` / \
                 `hailo-h10-all`) or set HAILORT_LIB_DIR.\n\
                 See docs/hailo-embed.md."
            );
            std::process::exit(1);
        }
        build
            .file(root.join("native/turboembed/src/hailo.cpp"))
            .define("TURBOEMBED_HAILO", None);
    }

    build.compile(if genai {
        "turboembed_genai"
    } else if hailo {
        "turboembed_hailo"
    } else {
        "turboembed_stub"
    });

    if ort_cuda && cuda_enabled() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true)
            .cpp(true)
            .std("c++17")
            .include(root.join("include"))
            .include(root.join("native/turboembed/src"))
            .file(root.join("native/turboembed/src/pool_cuda.cu"))
            .flag("-O2")
            .flag("-arch=native")
            .flag("-allow-unsupported-compiler")
            .flag("-ccbin=g++-13")
            .flag_if_supported("--expt-relaxed-constexpr");
        nvcc.compile("turboembed_pool_cuda");
    }
}

fn cuda_enabled() -> bool {
    if std::env::var_os("TURBOEMBED_DISABLE_CUDA").is_some() {
        return false;
    }
    if Command::new("nvcc").arg("--version").output().is_err() {
        return false;
    }
    Path::new("/usr/include/cuda_runtime.h").exists()
        || Path::new("/usr/local/cuda/include/cuda_runtime.h").exists()
}

fn cuda_lib_dir() -> Option<String> {
    for dir in [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
    ] {
        let p = Path::new(dir);
        if p.join("libcudart.so").exists() || p.join("libcudart.so.12").exists() {
            return Some(dir.to_string());
        }
    }
    None
}

fn newest_hailort_soname(dir: &Path) -> Option<String> {
    let mut best: Option<String> = None;
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        // libhailort.so.<major>.<minor>.<patch> — no bare .so symlink in the
        // Pi debs; pick the lexicographic max (4.24.0 > 4.9.0 is wrong
        // lexicographically, so compare parsed versions).
        let Some(ver) = name.strip_prefix("libhailort.so.") else {
            continue;
        };
        let parsed: Option<(u64, u64, u64)> = {
            let mut it = ver.split('.');
            match (it.next(), it.next(), it.next(), it.next()) {
                (Some(a), Some(b), Some(c), None) => match (a.parse(), b.parse(), c.parse()) {
                    (Ok(a), Ok(b), Ok(c)) => Some((a, b, c)),
                    _ => None,
                },
                _ => None,
            }
        };
        let Some(ver_triple) = parsed else { continue };
        let better = match &best {
            None => true,
            Some(cur) => {
                let cur_ver = cur
                    .strip_prefix("libhailort.so.")
                    .and_then(|v| {
                        let mut it = v.split('.');
                        match (it.next(), it.next(), it.next()) {
                            (Some(a), Some(b), Some(c)) => {
                                match (a.parse(), b.parse(), c.parse()) {
                                    (Ok(a), Ok(b), Ok(c)) => Some((a, b, c)),
                                    _ => None,
                                }
                            }
                            _ => None,
                        }
                    })
                    .unwrap_or((0, 0, 0));
                ver_triple > cur_ver
            }
        };
        if better {
            best = Some(name.to_string());
        }
    }
    best
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
            if let Some(dir) = Path::new(path.trim()).parent() {
                if dir.join("libstdc++.so").exists() {
                    println!("cargo:rustc-link-search=native={}", dir.display());
                }
            }
        }
    }
}

struct OpenVinoPaths {
    include_dirs: Vec<String>,
    lib_dirs: Vec<String>,
    has_tokenizers: bool,
}

fn find_openvino() -> Result<OpenVinoPaths, String> {
    if let Ok(paths) = from_pkg_config() {
        return Ok(paths);
    }

    let mut roots = Vec::new();
    for key in [
        "OPENVINO_GENAI_DIR",
        "OPENVINO_DIR",
        "OpenVINO_DIR",
        "INTEL_OPENVINO_DIR",
    ] {
        if let Ok(val) = std::env::var(key) {
            if !val.is_empty() {
                roots.push(PathBuf::from(val));
            }
        }
    }
    for candidate in [
        "/work/opt/openvino_genai",
        "/work/opt/openvino_genai_ubuntu26_2026.3.1.0_x86_64",
        "/opt/intel/openvino",
        "/opt/intel/openvino_2026",
        "/opt/intel/openvino_2025",
        "/usr",
    ] {
        roots.push(PathBuf::from(candidate));
    }

    for root in roots {
        if let Some(paths) = from_root(&root) {
            return Ok(paths);
        }
        if let Some(paths) = from_root(&root.join("runtime")) {
            return Ok(paths);
        }
    }

    Err("searched pkg-config (openvino / openvino_genai) and \
         OPENVINO_DIR / INTEL_OPENVINO_DIR / /work/opt/openvino_genai / \
         /opt/intel/openvino*"
        .into())
}

fn level_zero_present() -> bool {
    if std::env::var_os("TURBOEMBED_DISABLE_ZE").is_some() {
        return false;
    }
    Path::new("/usr/include/level_zero/ze_api.h").exists()
        && (Path::new("/usr/lib/x86_64-linux-gnu/libze_loader.so").exists()
            || Path::new("/usr/lib/x86_64-linux-gnu/libze_loader.so.1").exists())
}

fn from_pkg_config() -> Result<OpenVinoPaths, String> {
    let ov = pkg_config_libs("openvino")?;
    let genai = pkg_config_libs("openvino_genai").or_else(|_| pkg_config_libs("openvino-genai"))?;
    let mut include_dirs = ov.0;
    for d in genai.0 {
        if !include_dirs.contains(&d) {
            include_dirs.push(d);
        }
    }
    let mut lib_dirs = ov.1;
    for d in genai.1 {
        if !lib_dirs.contains(&d) {
            lib_dirs.push(d);
        }
    }
    let has_tokenizers = lib_dirs.iter().any(|d| {
        let p = Path::new(d);
        p.join("libopenvino_tokenizers.so").exists()
            || p.join("libopenvino_tokenizers.so.1").exists()
    });
    Ok(OpenVinoPaths {
        include_dirs,
        lib_dirs,
        has_tokenizers,
    })
}

fn pkg_config_libs(name: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let out = Command::new("pkg-config")
        .args(["--cflags", "--libs", name])
        .output()
        .map_err(|e| format!("pkg-config: {e}"))?;
    if !out.status.success() {
        return Err(format!("pkg-config {name} failed"));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut includes = Vec::new();
    let mut libs = Vec::new();
    for tok in text.split_whitespace() {
        if let Some(dir) = tok.strip_prefix("-I") {
            includes.push(dir.to_string());
        } else if let Some(dir) = tok.strip_prefix("-L") {
            libs.push(dir.to_string());
        }
    }
    if includes.is_empty() && libs.is_empty() {
        return Err(format!("pkg-config {name} returned no -I/-L"));
    }
    Ok((includes, libs))
}

fn from_root(root: &Path) -> Option<OpenVinoPaths> {
    let include = [
        root.join("include"),
        root.join("include/openvino"),
        root.join("../include"),
    ]
    .into_iter()
    .find(|p| {
        p.join("openvino/openvino.hpp").exists()
            || p.join("openvino/genai/rag/text_embedding_pipeline.hpp")
                .exists()
    });

    let genai_include = [
        root.join("include"),
        root.parent().unwrap_or(root).join("include"),
        PathBuf::from("/usr/include"),
    ]
    .into_iter()
    .find(|p| {
        p.join("openvino/genai/rag/text_embedding_pipeline.hpp")
            .exists()
    });

    let lib = [
        root.join("lib/intel64"),
        root.join("lib64"),
        root.join("lib"),
        root.join("../lib/intel64"),
        root.join("../lib"),
    ]
    .into_iter()
    .find(|p| p.join("libopenvino.so").exists() || p.join("libopenvino.so.2500").exists());

    let include_dir = include.or(genai_include)?;
    let lib_dir = lib?;
    let mut include_dirs = vec![include_dir.display().to_string()];
    if let Some(extra) = [
        lib_dir.parent().unwrap_or(&lib_dir).join("include"),
        PathBuf::from("/usr/include"),
    ]
    .into_iter()
    .find(|p| {
        p.join("openvino/genai/rag/text_embedding_pipeline.hpp")
            .exists()
    }) {
        let s = extra.display().to_string();
        if !include_dirs.contains(&s) {
            include_dirs.push(s);
        }
    }
    if !include_dirs.iter().any(|d| {
        Path::new(d)
            .join("openvino/genai/rag/text_embedding_pipeline.hpp")
            .exists()
    }) {
        return None;
    }
    let has_tokenizers = lib_dir.join("libopenvino_tokenizers.so").exists()
        || lib_dir.join("libopenvino_tokenizers.so.1").exists();
    Some(OpenVinoPaths {
        include_dirs,
        lib_dirs: vec![lib_dir.display().to_string()],
        has_tokenizers,
    })
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
