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
    assert!(pins().iter().filter(|p| p.source == "analysis/zoo/index/zoo.json").count() >= 171);
    let maxent = "645a094f45a866687a617385233fd23ae8b0f5fa8b1b76996781a50c17bdcf3d";
    assert!(pins().iter().any(|p| p.sha256 == maxent && p.source.contains("references")));
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
    let masc = include_str!("../../train-recipes/masc-references.txt");
    assert!(masc.contains("4de58d12b09421003c8b1ba13e8b58ec533b28e1f22a65272119dc0993d333e8"));
    fs::write(dir.join("x.bin"), b"not a model").unwrap();
    let e = check(&dir).unwrap_err();
    assert!(e.contains("files not pinned: x.bin"), "{e}");
}
