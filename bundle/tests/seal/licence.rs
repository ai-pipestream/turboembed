use super::*;

/// The zoo's licence classes through the bundle tool (#944): every recipe
/// maps to one, a seal records it, and a class that needs acceptance is
/// fetched only once the caller accepts the licence.

#[test]
fn every_recipe_maps_to_a_licence_class() {
    let mut n = 0;
    for entry in fs::read_dir(root().join("bundle/recipes")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "json") {
            continue;
        }
        let r = Recipe::load(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        turbo_bundle::licence::manifest_class(&r.manifest).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        n += 1;
    }
    assert!(n > 10, "{n} recipes");
}

#[test]
fn a_seal_records_the_class_and_verify_checks_it() {
    let (bundle, out) = sealed("licence-class", |_| {});
    out.unwrap();
    let path = bundle.join("manifest.json");
    let mut m: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(m["model"]["license_class"], json!("CLASS_PERMISSIVE"));

    m["model"]["license_class"] = json!("CLASS_GPL");
    fs::write(&path, serde_json::to_vec_pretty(&m).unwrap()).unwrap();
    let e = seal::verify(&bundle).unwrap_err();
    assert!(e.contains("license_class"), "{e}");
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

#[test]
fn a_seal_under_a_refused_licence_is_refused() {
    let (bundle, out) = sealed("licence-refused", |r| r.manifest["model"]["license"] = json!("CC-BY-ND-4.0"));
    let e = out.unwrap_err();
    assert!(e.contains("CC-BY-ND-4.0") && e.contains("refused"), "{e}");
    assert!(!bundle.join("manifest.json").exists());
    fs::remove_dir_all(bundle.parent().unwrap()).unwrap();
}

/// The upstream files are already there with pinned hashes, so the fetch
/// reads no network.
#[test]
fn a_non_commercial_model_is_fetched_only_once_accepted() {
    let d = scratch("licence-nc");
    let up = upstream(&d);
    let p = tiny_recipe(&d);
    let mut r: Value = serde_json::from_slice(&fs::read(&p).unwrap()).unwrap();
    for u in r["upstream"].as_array_mut().unwrap() {
        let bytes = fs::read(up.join(u["path"].as_str().unwrap())).unwrap();
        u["sha256"] = json!(turbo::bundle::sha256_hex(&bytes));
    }
    r["manifest"]["model"]["license"] = json!("CC-BY-NC-4.0");
    fs::write(&p, serde_json::to_vec_pretty(&r).unwrap()).unwrap();

    let e = turbo_bundle::fetch::fetch(&Recipe::load(&p).unwrap(), &up, false).unwrap_err();
    assert!(e.contains("CC-BY-NC-4.0") && e.contains("--accept-terms"), "{e}");
    assert!(!up.join(turbo_bundle::fetch::PROVENANCE).exists(), "refused before anything is written");

    turbo_bundle::fetch::fetch(&Recipe::load(&p).unwrap(), &up, true).unwrap();
    let got: Value = serde_json::from_slice(&fs::read(up.join(turbo_bundle::fetch::PROVENANCE)).unwrap()).unwrap();
    assert_eq!(got["license_class"], json!("CLASS_LOCAL_ONLY"));
    assert_eq!(got["terms_accepted"], json!(true));
    fs::remove_dir_all(&d).unwrap();
}
