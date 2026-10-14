//! The `turbo-zoo` command: what it shows before the user accepts terms,
//! and what it records.

#![cfg(feature = "http")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use turbo_zoo::{Acceptances, Licence, LocalOnlyReason, Requirement, Sha256, Sha512};

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-cli-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Runs `turbo-zoo` as the user alice with its home in `home`.
fn zoo(home: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_turbo-zoo"))
        .args(args)
        .env("TURBO_ZOO_HOME", home)
        .env("USER", "alice")
        .output()
        .unwrap()
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8(bytes.to_vec()).unwrap()
}

#[test]
fn accepting_a_non_commercial_licence_shows_that_it_stays_local() {
    let home = TempDir::new("accept");
    let out = zoo(&home.0, &["accept", "CC-BY-NC-SA-4.0"]);
    assert!(!out.status.success(), "{out:?}");
    let shown = text(&out.stdout);
    assert!(
        shown.contains("CC-BY-NC-SA-4.0: read the text at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode")
    );
    assert!(shown.contains(LocalOnlyReason::NonCommercial.notice()), "{shown}");
    assert!(text(&out.stderr).contains("nothing recorded"));
    assert!(!home.0.join("acceptances.json").exists());

    let out = zoo(&home.0, &["accept", "CC-BY-NC-SA-4.0", "--yes"]);
    assert!(out.status.success(), "{out:?}");
    assert!(text(&out.stdout).contains(LocalOnlyReason::NonCommercial.notice()));
    let acc = Acceptances::load(&home.0.join("acceptances.json")).unwrap();
    assert!(acc.has(&"alice".parse().unwrap(), &Requirement::of(Licence::CcByNcSa40).unwrap()));
    assert!(!acc.has(&"alice".parse().unwrap(), &Requirement::of(Licence::CcByNc40).unwrap()));
}

#[test]
fn restricted_terms_show_the_hosts_notice() {
    let home = TempDir::new("restricted");
    let out = zoo(&home.0, &["accept", "LicenseRef-RestrictedPublic", "https://example.org/terms"]);
    assert!(!out.status.success(), "{out:?}");
    let shown = text(&out.stdout);
    assert!(shown.contains(LocalOnlyReason::HostTerms.notice()), "{shown}");
    assert!(!shown.contains(LocalOnlyReason::NonCommercial.notice()), "{shown}");
}

#[test]
fn refused_and_permissive_licences_are_not_offered_for_acceptance() {
    let home = TempDir::new("refused");
    for (licence, says) in [
        ("CC-BY-NC-ND-4.0", "refused (no-derivatives terms)"),
        ("CC-BY-ND-4.0", "refused (no-derivatives terms)"),
        ("Apache-2.0", "nothing to accept"),
    ] {
        let out = zoo(&home.0, &["accept", licence, "--yes"]);
        assert!(!out.status.success(), "{licence}: {out:?}");
        assert!(out.stdout.is_empty(), "{licence}: {out:?}");
        let err = text(&out.stderr);
        assert!(err.contains(licence) && err.contains(says), "{err}");
    }
    assert!(!home.0.join("acceptances.json").exists());
}

#[test]
fn show_says_a_non_commercial_artifact_stays_local() {
    let home = TempDir::new("show");
    let bytes = b"model bytes";
    let index = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{
  "id": "m",
  "kind": "KIND_MODEL",
  "format": "FORMAT_OPENNLP_MODEL",
  "version": "1",
  "license": "Apache-2.0",
  "license_terms": "https://example.org/m/LICENSE",
  "trained_on": [{{"name": "UD_English-GUM r2.16", "license": "CC-BY-NC-SA-4.0", "license_terms": "https://example.org/gum/README.md"}}],
  "class": "CLASS_LOCAL_ONLY",
  "origin": "https://example.org/",
  "files": [{{"path": "m.bin", "url": "https://example.org/m.bin", "size": {}, "sha256": "{}", "sha512": "{}"}}]
}}]}}"#,
        bytes.len(),
        Sha256::of(bytes),
        Sha512::of(bytes),
    );
    let path = home.0.join("zoo.json");
    fs::write(&path, index).unwrap();
    let out = zoo(&home.0, &["--index", path.to_str().unwrap(), "show", "m"]);
    assert!(out.status.success(), "{out:?}");
    let shown = text(&out.stdout);
    assert!(shown.contains("class: CLASS_LOCAL_ONLY"), "{shown}");
    assert!(
        shown.contains("accept first: CC-BY-NC-SA-4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode"),
        "{shown}"
    );
    assert!(shown.contains("origin: https://example.org/ (fetched from there only)"), "{shown}");
    assert!(shown.contains(&format!("local only: {}", LocalOnlyReason::NonCommercial.notice())), "{shown}");
}
