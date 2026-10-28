//! The RPC-path part of the no-JSON-front-end rule (AGENTS.md,
//! docs/demo.md), checked where it can be exact. JSON elsewhere, such as
//! a form at the edge, is not this test's concern. On the RPC path: the
//! demo does not depend on serde or serde_json, the server does not
//! outside its tests, no build script that compiles the protos adds serde
//! to the generated types, and the demo routes nothing under `/api/`.
//! In the page (demo/web, issue #1361): the hand-written TypeScript
//! names no JSON transport, the Connect transport is binary and POST
//! only, and the one `fetch` is the upload PUT.

use std::path::Path;

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

/// What the page's own TypeScript may not say outside a comment: each
/// is a way JSON gets onto the RPC path or a call leaves Connect-ES.
const WEB_FORBIDDEN: [&str; 9] = [
    "JSON.",
    "toJson",
    "fromJson",
    "application/json",
    "connect+json",
    "useHttpGet: true",
    "useBinaryFormat: false",
    "XMLHttpRequest",
    "/api/",
];

/// The page's transport settings, each required in its client module.
const WEB_TRANSPORT: [&str; 2] = ["useBinaryFormat: true", "useHttpGet: false"];

/// The one module that may call `fetch`: the upload PUT.
const WEB_UPLOAD: &str = "demo/web/src/upload.ts";

/// The hand-written TypeScript under `dir`, without the generated stubs.
fn typescript_files(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for p in rd.map(|e| e.unwrap().path()) {
        let name = p.file_name().unwrap().to_str().unwrap_or("");
        if p.is_dir() {
            if name != "gen" && name != "node_modules" {
                typescript_files(&p, out);
            }
        } else if name.ends_with(".ts") {
            out.push(p);
        }
    }
}

fn web_code_lines(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|l| {
        let t = l.trim_start();
        !t.starts_with("//") && !t.starts_with('*') && !t.starts_with("/*")
    })
}

/// Each violation of the rule in the page's sources under `root`, sorted.
pub fn web_violations(root: &Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut files = Vec::new();
    typescript_files(&root.join("demo/web/src"), &mut files);
    typescript_files(&root.join("demo/web/test"), &mut files);
    for f in files {
        let rel = f.strip_prefix(root).unwrap().to_str().unwrap().replace('\\', "/");
        let text = std::fs::read_to_string(&f).unwrap();
        for line in web_code_lines(&text) {
            if let Some(w) = WEB_FORBIDDEN.iter().find(|w| line.contains(*w)) {
                found.push(format!("{rel}: {w}"));
            }
            if line.contains("fetch(") && rel != WEB_UPLOAD {
                found.push(format!("{rel}: fetch outside the upload module"));
            }
        }
    }
    let rpc = std::fs::read_to_string(root.join("demo/web/src/rpc.ts")).unwrap_or_default();
    for t in WEB_TRANSPORT {
        if !web_code_lines(&rpc).any(|l| l.contains(t)) {
            found.push(format!("demo/web/src/rpc.ts: no {t}"));
        }
    }
    found.sort();
    found.dedup();
    found
}

#[test]
fn the_front_end_speaks_only_protobuf() {
    let root = crate::common::workspace();
    let mut found = violations(&root);
    found.extend(web_violations(&root));
    assert!(found.is_empty(), "the no-JSON-front-end rule (AGENTS.md) is broken by:\n  {}", fmt(&found));
    // The server compiles the protos; the demo no longer does, as it
    // decodes no message.
    let mut scripts = Vec::new();
    proto_build_scripts(&root, &root, &mut scripts);
    assert!(
        scripts.iter().any(|s| s == "server/build.rs"),
        "only {scripts:?} found compiling protos; the scan is too narrow"
    );
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

/// The page's checks, each shown failing on planted sources.
#[test]
fn each_web_check_fails_on_a_planted_violation() {
    let dir = crate::common::scratch("front-end-web-rule");
    let write = |path: &str, text: &str| {
        let p = dir.join(path);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    };
    let clean = || {
        write(
            "demo/web/src/rpc.ts",
            "const t = createConnectTransport({ useBinaryFormat: true, useHttpGet: false });\n",
        );
        write(
            "demo/web/src/upload.ts",
            "// Not JSON.parse: binary.\nconst r = await fetch(url, { method: \"PUT\" });\n",
        );
        write("demo/web/src/gen/x_pb.ts", "const j = JSON.stringify(x);\n");
        write("demo/web/src/tabs/a.ts", " * Never toJson.\nconst a = 1;\n");
        write("demo/web/test/b.test.ts", "const m = fromBinary(S, b);\n");
    };
    clean();
    assert_eq!(web_violations(&dir), Vec::<String>::new());
    write("demo/web/src/tabs/a.ts", "const a = JSON.stringify(m);\n");
    assert_eq!(web_violations(&dir), ["demo/web/src/tabs/a.ts: JSON."]);
    clean();
    write("demo/web/test/b.test.ts", "const m = fromJson(S, j);\n");
    assert_eq!(web_violations(&dir), ["demo/web/test/b.test.ts: fromJson"]);
    clean();
    write(
        "demo/web/src/tabs/a.ts",
        "const r = await fetch(\"/x\", { headers: { \"content-type\": \"application/json\" } });\n",
    );
    assert_eq!(
        web_violations(&dir),
        ["demo/web/src/tabs/a.ts: application/json", "demo/web/src/tabs/a.ts: fetch outside the upload module"]
    );
    clean();
    write("demo/web/src/rpc.ts", "const t = createConnectTransport({ useBinaryFormat: false, useHttpGet: false });\n");
    assert_eq!(
        web_violations(&dir),
        ["demo/web/src/rpc.ts: no useBinaryFormat: true", "demo/web/src/rpc.ts: useBinaryFormat: false"]
    );
}
