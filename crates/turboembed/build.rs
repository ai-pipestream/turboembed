//! Compile the TurboEmbed C ABI.
//!
//! Header: `include/turboembed.h`
//! Always: `native/turboembed/src/stub.cpp` (mock + dispatch)
//! Feature `genai`: also `native/turboembed/src/genai.cpp`
//! (`ov::genai::TextEmbeddingPipeline` on GPU). Fails the build if
//! OpenVINO GenAI is missing — never silently compiles a stub.

use std::path::{Path, PathBuf};

fn main() {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.join("../..");
    let root = root.canonicalize().unwrap_or(root);
    let header = root.join("include/turboembed.h");
    let stub = root.join("native/turboembed/src/stub.cpp");
    let genai_cpp = root.join("native/turboembed/src/genai.cpp");
    let genai_hpp = root.join("native/turboembed/src/genai.hpp");
    let apple = root.join("swift/Sources/TurboEmbedC/include/turboembed.h");

    println!("cargo:rerun-if-changed={}", header.display());
    println!("cargo:rerun-if-changed={}", stub.display());
    println!("cargo:rerun-if-changed={}", genai_cpp.display());
    println!("cargo:rerun-if-changed={}", genai_hpp.display());
    println!("cargo:rerun-if-changed={}", apple.display());
    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OPENVINO_GENAI_DIR");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OpenVINO_DIR");

    println!(
        "cargo:rustc-env=TURBOEMBED_WORKSPACE_ROOT={}",
        root.display()
    );

    let genai = std::env::var("CARGO_FEATURE_GENAI").is_ok();
    let ort_cuda = std::env::var("CARGO_FEATURE_ORT_CUDA").is_ok();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file(&stub)
        .include(root.join("include"))
        .include(root.join("native/turboembed/src"))
        .warnings(true)
        .flag_if_supported("-Wno-unused-parameter")
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
            .file(&genai_cpp)
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
    }

    build.compile(if genai {
        "turboembed_genai"
    } else {
        "turboembed_stub"
    });
}

fn escape_c_string(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn link_libstdcxx() {
    if let Ok(out) = std::process::Command::new("g++")
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

    Err(
        "searched pkg-config (openvino / openvino_genai) and \
         OPENVINO_DIR / INTEL_OPENVINO_DIR / /work/opt/openvino_genai / \
         /opt/intel/openvino*"
            .into(),
    )
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
    let out = std::process::Command::new("pkg-config")
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
