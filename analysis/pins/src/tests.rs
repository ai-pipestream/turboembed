use super::*;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-pins-{name}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn every_line_of_models_txt_parses_and_no_digest_is_pinned_twice() {
    let own = parse_models(MODELS).unwrap();
    assert!(own.len() >= 40, "{} pins", own.len());
    let mut seen = std::collections::HashSet::new();
    for p in &own {
        assert!(seen.insert(p.sha256), "{} is pinned twice", p.sha256);
    }
}

#[test]
fn the_zoo_and_the_references_are_pinned_too() {
    assert!(pins().iter().filter(|p| p.source == "analysis/zoo/index/zoo.json").count() >= 90);
    let text = "# a comment\n\
                1111111111111111111111111111111111111111111111111111111111111111 https://example.org/a/kept.bin\n\
                2222222222222222222222222222222222222222222222222222222222222222 https://example.org/models-1.5/gone.bin\n";
    let refs = reference_pins(text);
    assert_eq!(refs.len(), 1);
    assert_eq!((refs[0].name, refs[0].kind), ("kept.bin", Kind::File));
}

/// No OpenNLP 1.5 model is pinned anywhere (#1509), whatever a references
/// file still lists.
#[test]
fn no_retired_model_is_pinned() {
    for name in
        ["en-pos-maxent.bin", "en-pos-perceptron.bin", "en-chunker.bin", "en-ner-person.bin", "en-ner-location.bin"]
    {
        assert!(!pins().iter().any(|p| p.name == name), "{name} is pinned");
    }
    for hex in [
        "645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d",
        "7861a0c2f134d9c12a022a1ba501e88bc7039f6db72b4140e1bafd1fb5ef76cc",
    ] {
        assert!(lookup(hex).is_none(), "{hex} is pinned");
    }
}

#[test]
fn an_ill_formed_line_is_refused() {
    for bad in [
        "sha256/0000  file  x\n",
        "blobs/sha256/abc  file  x\n",
        "blobs/sha256/0000000000000000000000000000000000000000000000000000000000000000  blob  x\n",
        "blobs/sha256/0000000000000000000000000000000000000000000000000000000000000000 file x\n",
        "blobs/sha256/0000000000000000000000000000000000000000000000000000000000000000  file   x\n",
    ] {
        assert!(parse_models(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_pinned_bundle_passes_and_a_changed_one_fails() {
    let tiny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    assert_eq!(digest(&tiny).unwrap().0, Kind::Bundle);
    check(&tiny).unwrap();
    assert_eq!(require(&tiny), tiny);
    let dir = scratch("bundle");
    let mut manifest = fs::read(tiny.join("manifest.json")).unwrap();
    manifest.push(b'\n');
    fs::write(dir.join("manifest.json"), manifest).unwrap();
    let e = check(&dir).unwrap_err();
    assert!(e.contains("is not pinned") && e.contains("blobs/sha256/"), "{e}");
    assert!(std::panic::catch_unwind(|| require(&dir)).is_err());
}

#[test]
fn a_bundle_changed_after_a_passing_require_is_refused() {
    let tiny = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let dir = scratch("changed");
    let manifest = fs::read(tiny.join("manifest.json")).unwrap();
    fs::write(dir.join("manifest.json"), &manifest).unwrap();
    assert_eq!(require(&dir), dir);
    let mut changed = manifest;
    changed.push(b'\n');
    fs::write(dir.join("manifest.json"), changed).unwrap();
    assert!(std::panic::catch_unwind(|| require(&dir)).is_err());
}

#[test]
fn a_file_and_a_tree_are_hashed_as_models_txt_says() {
    let dir = scratch("tree");
    fs::create_dir_all(dir.join("sub")).unwrap();
    fs::write(dir.join("b.bin"), b"b").unwrap();
    fs::write(dir.join("sub/a.bin"), b"a").unwrap();
    let (kind, hex) = digest(&dir).unwrap();
    assert_eq!(kind, Kind::Tree);
    let listing = format!("{}  b.bin\n{}  sub/a.bin\n", sha256_hex(b"b"), sha256_hex(b"a"));
    assert_eq!(hex, sha256_hex(listing.as_bytes()));
    assert_eq!(digest(&dir.join("b.bin")).unwrap(), (Kind::File, sha256_hex(b"b")));
    // A file pinned by the digest of a directory's listing is not that file.
    assert!(check(&dir.join("b.bin")).is_err());
}

#[test]
fn a_package_is_pinned_by_its_own_manifest() {
    let dir = scratch("package");
    fs::write(dir.join("gliner.json"), b"{}").unwrap();
    fs::write(dir.join("other.bin"), b"x").unwrap();
    assert_eq!(digest(&dir).unwrap(), (Kind::Package, sha256_hex(b"{}")));
}

#[test]
fn a_directory_of_pinned_files_passes_and_one_unpinned_file_fails_it() {
    let dir = scratch("checkpoint");
    let zoo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle/manifest.json");
    fs::copy(&zoo, dir.join("a.json")).unwrap();
    // The tiny bundle's manifest is pinned as a bundle, not as a file.
    assert!(check(&dir).is_err());
    let dir = scratch("checkpoint-files");
    fs::write(dir.join("x.bin"), b"not a model").unwrap();
    let e = check(&dir).unwrap_err();
    assert!(e.contains("files not pinned: x.bin"), "{e}");
}

/// A GLiREL-shaped package: glirel.json beside encoder/, whose deberta.json
/// glirel.json does not name.
fn glirel_package(name: &str, encoder: &[u8]) -> PathBuf {
    let dir = scratch(name);
    fs::write(dir.join("glirel.json"), br#"{"version": 1}"#).unwrap();
    fs::write(dir.join("heads.safetensors"), b"heads").unwrap();
    fs::create_dir_all(dir.join("encoder")).unwrap();
    fs::write(dir.join("encoder/deberta.json"), encoder).unwrap();
    fs::write(dir.join("encoder/weights.safetensors"), encoder).unwrap();
    dir
}

#[test]
fn a_package_whose_inner_encoder_is_swapped_is_refused() {
    let package = sha256_hex(br#"{"version": 1}"#);
    let encoder = sha256_hex(b"encoder");
    let pinned = |hex: &str| (hex == package || hex == encoder).then_some((Kind::Package, "pin"));
    let dir = glirel_package("glirel", b"encoder");
    check_with(&dir, &pinned).unwrap();
    // glirel.json is unchanged and still pinned; encoder/ is replaced,
    // manifest and weights together.
    let swapped = glirel_package("glirel-swapped", b"another encoder");
    let e = check_with(&swapped, &pinned).unwrap_err();
    assert!(e.contains("inner model is not bound") && e.contains("encoder"), "{e}");
    // Without a pin of its own, an encoder the outer manifest names by
    // digest is bound by it, as gliner.json names encoder/deberta.json.
    let dir = scratch("gliner");
    let named = format!(r#"{{"encoder_sha256": "{encoder}"}}"#);
    fs::write(dir.join("gliner.json"), &named).unwrap();
    fs::create_dir_all(dir.join("encoder")).unwrap();
    fs::write(dir.join("encoder/deberta.json"), b"encoder").unwrap();
    let outer = sha256_hex(named.as_bytes());
    check_with(&dir, &|hex: &str| (hex == outer).then_some((Kind::Package, "pin"))).unwrap();
    fs::write(dir.join("encoder/deberta.json"), b"another encoder").unwrap();
    assert!(check_with(&dir, &|hex: &str| (hex == outer).then_some((Kind::Package, "pin"))).is_err());
}
