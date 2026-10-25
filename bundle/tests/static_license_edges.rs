use serde_json::{Value, json};
use std::fs;
use std::path::{Path, PathBuf};
use turbo::bundle::{Bundle, sha256_hex};
use turbo_bundle::distill::from_base::{self, Options};

const TEXT: &[u8] = b"Test licence text, preserved byte for byte.\n";

fn copy(from: &Path, to: &Path) {
    fs::create_dir_all(to).unwrap();
    for entry in fs::read_dir(from).unwrap() {
        let entry = entry.unwrap();
        let dest = to.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy(&entry.path(), &dest);
        } else {
            fs::copy(entry.path(), dest).unwrap();
        }
    }
}

fn fixture(name: &str, license: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!("turbo-license-edge-{}-{name}", std::process::id()));
    fs::create_dir_all(&root).unwrap();
    let base = root.join("base");
    copy(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle"), &base);
    let path = base.join("manifest.json");
    let mut m: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    let file = base.join(license);
    if file.exists() {
        // Keep the original tokenizer/reference valid at another base path.
        let moved = format!("original/{license}");
        fs::create_dir_all(base.join(&moved).parent().unwrap()).unwrap();
        fs::rename(&file, base.join(&moved)).unwrap();
        relocate(&mut m, license, &moved);
    }
    fs::create_dir_all(file.parent().unwrap()).unwrap();
    fs::write(file, TEXT).unwrap();
    m["model"]["license_file"] = json!(license);
    m["files"].as_array_mut().unwrap().push(json!({
        "path": license, "size": TEXT.len(), "sha256": sha256_hex(TEXT)
    }));
    fs::write(path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    Bundle::open(&base).unwrap().read_verified(license).unwrap();
    root
}

fn relocate(value: &mut Value, from: &str, to: &str) {
    match value {
        Value::String(s) if s == from => *s = to.into(),
        Value::Array(a) => a.iter_mut().for_each(|v| relocate(v, from, to)),
        Value::Object(o) => o.values_mut().for_each(|v| relocate(v, from, to)),
        _ => {}
    }
}

fn options() -> Options {
    Options {
        id: "turbo-testdata/license-edge".into(),
        revision: "1".into(),
        pca_dims: 16,
        sif_coefficient: 1e-4,
        dtype: "F32".into(),
        max_length: 32,
    }
}

#[test]
fn recipe_checks_the_bases_pinned_license_bytes() {
    let root = fixture("hash", "LICENSE");
    let base = root.join("base");
    let mut altered = TEXT.to_vec();
    altered[0] = b'X';
    for bytes in [altered, TEXT[..TEXT.len() - 1].to_vec()] {
        fs::write(base.join("LICENSE"), bytes).unwrap();
        assert!(Bundle::open(&base).unwrap().read_verified("LICENSE").is_err());
        assert!(
            from_base::recipe(&base, &options(), &root.join("work")).is_err(),
            "recipe accepted licence bytes that the base's verified reader rejects"
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[cfg(unix)]
fn recipe_checks_license_symlink_containment() {
    let root = fixture("link", "LICENSE");
    let base = root.join("base");
    fs::write(root.join("outside-license"), TEXT).unwrap();
    fs::remove_file(base.join("LICENSE")).unwrap();
    std::os::unix::fs::symlink("../outside-license", base.join("LICENSE")).unwrap();
    assert!(Bundle::open(&base).unwrap().read_verified("LICENSE").is_err());
    assert!(
        from_base::recipe(&base, &options(), &root.join("work")).is_err(),
        "recipe accepted a licence link resolving outside the base"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn generated_weights_do_not_replace_license_text() {
    sealed_copy("weights/static.safetensors", "collision");
}

fn sealed_copy(source: &str, name: &str) {
    let root = fixture(name, source);
    let base = root.join("base");
    let mut recipe = from_base::recipe(&base, &options(), &root.join("work")).unwrap();
    let output = root.join("output");
    turbo_bundle::distill::stage_with(&recipe, &base, &output, &[], &mut |_| {}).unwrap();
    turbo_bundle::distill::seal(&mut recipe, &output).unwrap();
    let bundle = Bundle::open(&output).unwrap();
    let license = bundle.manifest.model.license_file.as_ref().unwrap();
    assert!(
        bundle.read_verified(license).unwrap() == TEXT,
        "sealed licence entry points at generated weights instead of the licence"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn licence_paths_do_not_collide_with_other_generated_outputs() {
    for (i, source) in [
        "quality/texts.jsonl",
        "reference/reference.safetensors",
        "weights/static.safetensors.report.json",
        "reference/cases.json",
        "tokenizer.json",
        "quality",
        "legal/nested/LICENSE",
    ]
    .iter()
    .enumerate()
    {
        sealed_copy(source, &format!("other-output-{i}"));
    }
}

#[test]
fn absent_or_null_license_file_needs_no_copy() {
    let root = fixture("optional", "LICENSE");
    let path = root.join("base/manifest.json");
    let mut m: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for null in [false, true] {
        if null {
            m["model"]["license_file"] = Value::Null;
        } else {
            m["model"].as_object_mut().unwrap().remove("license_file");
        }
        fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
        let r = from_base::recipe(&root.join("base"), &options(), &root.join("work")).unwrap();
        assert!(r.manifest["model"].get("license_file").is_none());
        assert_eq!(r.local.len(), 1);
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[cfg(unix)]
fn contained_license_symlink_keeps_verified_bytes() {
    let root = fixture("contained", "legal/LICENSE");
    let base = root.join("base");
    fs::rename(base.join("legal/LICENSE"), base.join("legal/text")).unwrap();
    std::os::unix::fs::symlink("text", base.join("legal/LICENSE")).unwrap();
    let r = from_base::recipe(&base, &options(), &root.join("work")).unwrap();
    let output = root.join("output");
    turbo_bundle::seal::stage(&r, &base, &output).unwrap();
    let destination = r.manifest["model"]["license_file"].as_str().unwrap();
    assert_eq!(fs::read(output.join(destination)).unwrap(), TEXT);
    fs::remove_dir_all(root).unwrap();
}
