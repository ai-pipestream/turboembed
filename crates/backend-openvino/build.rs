//! Build the cxx bridge only when the `genai` feature is on.
//!
//! Looks for OpenVINO / OpenVINO GenAI via (in order):
//!   * `pkg-config` (`openvino`, `openvino_genai`)
//!   * `OPENVINO_GENAI_DIR`, `OPENVINO_DIR`, `INTEL_OPENVINO_DIR`
//!   * `/opt/intel/openvino`, `/opt/intel/openvino_2025`, `/opt/intel/openvino_2026`
//!
//! Fails the build with an actionable message if the feature is on and
//! the runtime cannot be found — never silently compiles a stub.

fn main() {
    println!("cargo:rerun-if-changed=cxx/text_embedding.cpp");
    println!("cargo:rerun-if-changed=cxx/text_embedding.hpp");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-env-changed=OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OPENVINO_GENAI_DIR");
    println!("cargo:rerun-if-env-changed=INTEL_OPENVINO_DIR");
    println!("cargo:rerun-if-env-changed=OpenVINO_DIR");

    if std::env::var("CARGO_FEATURE_GENAI").is_err() {
        return;
    }

    let ov = match find_openvino() {
        Ok(v) => v,
        Err(msg) => {
            eprintln!(
                "error: feature `genai` is enabled but OpenVINO GenAI was not found.\n\
                 {msg}\n\
                 Install OpenVINO + OpenVINO GenAI, source setupvars.sh / \
                 /opt/intel/oneapi/setvars.sh, and rebuild.\n\
                 See docs/intel-genai-embed.md."
            );
            std::process::exit(1);
        }
    };

    let mut build = cxx_build::bridge("src/ffi.rs");
    build
        .file("cxx/text_embedding.cpp")
        .include("cxx")
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-missing-field-initializers");
    for dir in &ov.include_dirs {
        build.include(dir);
    }
    build.compile("inferstream_ov_genai");

    for dir in &ov.lib_dirs {
        println!("cargo:rustc-link-search=native={dir}");
    }
    println!("cargo:rustc-link-lib=dylib=openvino");
    println!("cargo:rustc-link-lib=dylib=openvino_genai");
    // Tokenizers are usually a plugin (libopenvino_tokenizers.so) loaded at
    // runtime next to the other OpenVINO libs; still link if present so the
    // loader records the dependency.
    if ov.has_tokenizers {
        println!("cargo:rustc-link-lib=dylib=openvino_tokenizers");
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
                roots.push(std::path::PathBuf::from(val));
            }
        }
    }
    for candidate in [
        "/opt/intel/openvino",
        "/opt/intel/openvino_2026",
        "/opt/intel/openvino_2025",
        "/usr",
    ] {
        roots.push(std::path::PathBuf::from(candidate));
    }

    for root in roots {
        if let Some(paths) = from_root(&root) {
            return Ok(paths);
        }
        // CMake install prefix often nests runtime/ under the toolkit root.
        if let Some(paths) = from_root(&root.join("runtime")) {
            return Ok(paths);
        }
    }

    Err("searched pkg-config (openvino / openvino_genai) and \
         OPENVINO_DIR / INTEL_OPENVINO_DIR / /opt/intel/openvino*"
        .into())
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
        let p = std::path::Path::new(d);
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

fn from_root(root: &std::path::Path) -> Option<OpenVinoPaths> {
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
        std::path::PathBuf::from("/usr/include"),
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
    // GenAI headers may live next to OpenVINO headers or in a sibling prefix.
    let mut include_dirs = vec![include_dir.display().to_string()];
    if let Some(extra) = [
        lib_dir.parent().unwrap_or(&lib_dir).join("include"),
        std::path::PathBuf::from("/usr/include"),
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
        std::path::Path::new(d)
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
