//! The no-JSON-front-end rule (AGENTS.md, docs/demo.md), checked where it
//! can be exact: the demo does not depend on serde or serde_json, the
//! server does not outside its tests, no build script that compiles the
//! protos adds serde to the generated types, and the demo routes nothing
//! under `/api/`.

use std::path::Path;

/// What the JSON gateway from #1271 left on main, which #1360 removes.
/// The list only shrinks: an entry that no longer holds fails the test
/// below, so whoever removes a violation also removes its entry here and
/// it cannot come back unnoticed.
const PENDING_1360: [&str; 4] = [
    "demo/build.rs: serde attributes on the generated proto types",
    "demo/src/lib.rs: an \"/api/\" route",
    "turbo-kserve-demo depends on serde",
    "turbo-kserve-demo depends on serde_json",
];

/// The packages whose JSON dependencies are checked, and whether their
/// dev-dependencies count: the server's tests read JSON manifests.
const PACKAGES: [(&str, bool); 2] = [("turbo-kserve-demo", true), ("turbo-kserve", false)];

fn json_dependencies(root: &Path, found: &mut Vec<String>) {
    let out = std::process::Command::new(env!("CARGO"))
        .current_dir(root)
        .args(["metadata", "--offline", "--no-deps", "--format-version", "1"])
        .output()
        .expect("Cargo reads workspace manifests");
    assert!(out.status.success(), "Cargo metadata: {}", String::from_utf8_lossy(&out.stderr));
    let metadata: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    for package in metadata["packages"].as_array().unwrap() {
        let Some(&(name, dev)) = PACKAGES.iter().find(|(n, _)| package["name"] == *n) else { continue };
        for dep in package["dependencies"].as_array().unwrap() {
            let dep_name = dep["name"].as_str().unwrap();
            if matches!(dep_name, "serde" | "serde_json") && (dev || dep["kind"] != "dev") {
                found.push(format!("{name} depends on {dep_name}"));
            }
        }
    }
}

/// Every build.rs under `root` that runs a protobuf code generator.
fn proto_build_scripts(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<_> = rd.map(|e| e.unwrap().path()).collect();
    entries.sort();
    for p in entries {
        let name = p.file_name().unwrap().to_str().unwrap_or("");
        if p.is_dir() {
            if !name.starts_with('.') && name != "node_modules" && name != "target" && !name.starts_with("target-") {
                proto_build_scripts(root, &p, out);
            }
        } else if name == "build.rs" {
            let text = std::fs::read_to_string(&p).unwrap();
            if text.contains("prost_build") {
                out.push(p.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/"));
            }
        }
    }
}

fn code_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|l| !l.trim_start().starts_with("//"))
}

fn rust_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for p in rd.map(|e| e.unwrap().path()) {
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// Each violation of the rule under `root`, sorted.
pub fn violations(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    json_dependencies(root, &mut found);
    let mut scripts = Vec::new();
    proto_build_scripts(root, root, &mut scripts);
    for s in scripts {
        let text = std::fs::read_to_string(root.join(&s)).unwrap();
        if code_lines(&text).any(|l| l.contains("serde")) {
            found.push(format!("{s}: serde attributes on the generated proto types"));
        }
    }
    let mut sources = Vec::new();
    rust_files(&root.join("demo/src"), &mut sources);
    for f in sources {
        let text = std::fs::read_to_string(&f).unwrap();
        if code_lines(&text).any(|l| l.contains("\"/api/")) {
            let rel = f.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/");
            found.push(format!("{rel}: an \"/api/\" route"));
        }
    }
    found.sort();
    found.dedup();
    found
}

#[test]
fn the_front_end_speaks_only_protobuf() {
    let root = crate::common::workspace();
    let found = violations(&root);
    let new: Vec<_> = found.iter().filter(|v| !PENDING_1360.contains(&v.as_str())).collect();
    assert!(new.is_empty(), "the no-JSON-front-end rule (AGENTS.md) is broken by:\n  {}", fmt(&new));
    let gone: Vec<_> = PENDING_1360.iter().filter(|p| !found.iter().any(|v| v == *p)).collect();
    assert!(gone.is_empty(), "fixed, so remove from PENDING_1360 in bench/tests/tree/front_end.rs:\n  {}", fmt(&gone));
    let mut scripts = Vec::new();
    proto_build_scripts(&root, &root, &mut scripts);
    assert!(scripts.len() >= 2, "only {scripts:?} found compiling protos; the scan is too narrow");
}

fn fmt<T: std::fmt::Display>(v: &[T]) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("\n  ")
}

/// Each check, shown failing on a tree that breaks it and passing on one
/// that does not.
#[test]
fn each_front_end_check_fails_on_a_planted_violation() {
    let dir = crate::common::scratch("front-end-rule");
    let write = |path: &str, text: &str| {
        let p = dir.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    };
    write("Cargo.toml", "[workspace]\nmembers=[\"demo\",\"server\"]\nresolver=\"2\"\n");
    let manifest = |name: &str, deps: &str| {
        format!(
            "[package]\nname=\"{name}\"\nversion=\"0.1.0\"\nbuild=\"build.rs\"\n[lib]\npath=\"src/lib.rs\"\n{deps}\n"
        )
    };
    write("server/src/lib.rs", "");
    write(
        "server/build.rs",
        "fn main() {\n    // serde is named in a comment only.\n    tonic_prost_build::configure();\n}\n",
    );
    let clean_demo = || {
        write("demo/Cargo.toml", &manifest("turbo-kserve-demo", "[dependencies]\nprost=\"0.14\""));
        write("demo/build.rs", "fn main() {\n    tonic_prost_build::configure();\n}\n");
        write(
            "demo/src/lib.rs",
            "// GET /api/status was a JSON route.\nconst ROUTE: &str = \"/turbo.result.v1.AnalysisService/Capabilities\";\n",
        );
        write("server/Cargo.toml", &manifest("turbo-kserve", "[dev-dependencies]\nserde_json=\"1\""));
    };
    clean_demo();
    assert_eq!(violations(&dir), Vec::<String>::new());

    write(
        "demo/Cargo.toml",
        &manifest("turbo-kserve-demo", "[dev-dependencies]\njs={package=\"serde_json\",version=\"1\"}"),
    );
    assert_eq!(violations(&dir), ["turbo-kserve-demo depends on serde_json"]);
    clean_demo();
    write(
        "server/Cargo.toml",
        &manifest("turbo-kserve", "[dependencies]\nserde={version=\"1\",features=[\"derive\"]}"),
    );
    assert_eq!(violations(&dir), ["turbo-kserve depends on serde"]);
    clean_demo();
    write(
        "demo/build.rs",
        "fn main() {\n    tonic_prost_build::configure().message_attribute(\".\", \"#[derive(serde::Serialize)]\");\n}\n",
    );
    assert_eq!(violations(&dir), ["demo/build.rs: serde attributes on the generated proto types"]);
    clean_demo();
    write("demo/src/routes/mod.rs", "fn r() -> &'static str {\n    \"/api/infer\"\n}\n");
    assert_eq!(violations(&dir), ["demo/src/routes/mod.rs: an \"/api/\" route"]);
}
