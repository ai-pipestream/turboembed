use std::path::Path;

pub fn regex_dependencies(root: &Path) -> (usize, Vec<String>) {
    // Cargo resolves renames, inherited dependencies, target tables and TOML syntax.
    // --no-deps inspects declarations without fetching or building dependencies.
    let out = std::process::Command::new(env!("CARGO"))
        .current_dir(root)
        .args(["metadata", "--offline", "--no-deps", "--format-version", "1"])
        .output()
        .expect("Cargo reads workspace manifests");
    assert!(out.status.success(), "Cargo metadata: {}", String::from_utf8_lossy(&out.stderr));
    let metadata: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let members = metadata["workspace_members"].as_array().unwrap();
    let packages = metadata["packages"].as_array().unwrap();
    let mut found = Vec::new();
    let mut seen = 0;
    for package in packages.iter().filter(|p| members.contains(&p["id"])) {
        seen += 1;
        if package["dependencies"].as_array().unwrap().iter().any(|dep| dep["name"] == "regex") {
            found.push(package["name"].as_str().unwrap().to_owned());
        }
    }
    found.sort();
    (seen, found)
}

#[test]
fn cargo_dependency_forms_are_checked() {
    let dir = crate::common::scratch("dependency-forms");
    let forms = [
        ("compact", "[dependencies]\nregex=\"1\""),
        ("quoted", "[dependencies]\n\"regex\" = \"1\""),
        ("table", "[dev-dependencies.regex]\nversion=\"1\""),
        ("alias", "[build-dependencies]\nrx={package=\"regex\",version=\"1\"}"),
        ("targeted", "[target.'cfg(unix)'.dependencies]\nregex={version=\"1\",optional=true}"),
        ("inherited", "[dependencies]\nrx.workspace=true"),
        ("ordinary", "[dependencies]\nregex = \"1\""),
    ];
    std::fs::write(dir.join("Cargo.toml"),
        "[workspace]\nmembers=[\"*case\"]\nresolver=\"2\"\n[workspace.dependencies]\nrx={package=\"regex\",version=\"1\"}\n").unwrap();
    for (name, form) in forms {
        let package = dir.join(format!("{name}case"));
        std::fs::create_dir(&package).unwrap();
        std::fs::write(package.join("lib.rs"), "").unwrap();
        std::fs::write(
            package.join("Cargo.toml"),
            format!("[package]\nname=\"{name}case\"\nversion=\"0.1.0\"\n[lib]\npath=\"lib.rs\"\n{form}\n"),
        )
        .unwrap();
    }
    let clean = dir.join("cleancase");
    std::fs::create_dir(&clean).unwrap();
    std::fs::write(clean.join("lib.rs"), "").unwrap();
    std::fs::write(
        clean.join("Cargo.toml"),
        concat!(
            "[package]\nname=\"cleancase\"\nversion=\"0.1.0\"\n[lib]\npath=\"lib.rs\"\n",
            "[dependencies]\nregex={package=\"unicode-ident\",version=\"1\"}\n",
        ),
    )
    .unwrap();
    let (_, found) = regex_dependencies(&dir);
    let mut expected: Vec<_> = forms.iter().map(|(name, _)| format!("{name}case")).collect();
    expected.sort();
    assert_eq!(found, expected);
}
