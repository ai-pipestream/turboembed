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

/// What the user reads before accepting non-commercial terms: the address
/// of the licence, what the licence itself allows, and, apart from it,
/// this project's policy, which accepting agrees to.
const NON_COMMERCIAL_SHOWN: &str = "\
CC-BY-NC-SA-4.0: read the text at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
The licence grants its rights for non-commercial purposes only.
This project's policy: the zoo requests data under this licence only at addresses under its original source, and \
the project never ships, bundles or redistributes the data or any model trained on it. By accepting, you agree to \
treat any model you train on it as non-commercial and to keep that model on your machine. Model weights under \
this licence are held to the same: the project never ships or redistributes them or anything converted from them, \
and you agree to use them, and what you convert from them, for non-commercial purposes only and on your machine.
";

#[test]
fn accepting_a_non_commercial_licence_shows_its_terms_apart_from_the_policy() {
    let home = TempDir::new("accept");
    let out = zoo(&home.0, &["accept", "CC-BY-NC-SA-4.0"]);
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(text(&out.stdout), NON_COMMERCIAL_SHOWN);
    assert_eq!(
        text(&out.stdout),
        format!(
            "CC-BY-NC-SA-4.0: read the text at https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\n{}\n{}\n",
            LocalOnlyReason::NonCommercial.terms(),
            LocalOnlyReason::NonCommercial.policy()
        )
    );
    assert!(text(&out.stderr).contains("nothing recorded"));
    assert!(!home.0.join("acceptances.json").exists());

    let out = zoo(&home.0, &["accept", "CC-BY-NC-SA-4.0", "--yes"]);
    assert!(out.status.success(), "{out:?}");
    assert_eq!(text(&out.stdout), format!("{NON_COMMERCIAL_SHOWN}alice accepts CC-BY-NC-SA-4.0\n"));
    let acc = Acceptances::load(&home.0.join("acceptances.json")).unwrap();
    assert!(acc.has(&"alice".parse().unwrap(), &Requirement::of(Licence::CcByNcSa40).unwrap()));
    assert!(!acc.has(&"alice".parse().unwrap(), &Requirement::of(Licence::CcByNc40).unwrap()));
}

#[test]
fn restricted_terms_show_the_hosts_terms_apart_from_the_policy() {
    let home = TempDir::new("restricted");
    let out = zoo(&home.0, &["accept", "LicenseRef-RestrictedPublic", "https://example.org/terms"]);
    assert!(!out.status.success(), "{out:?}");
    assert_eq!(
        text(&out.stdout),
        format!(
            "LicenseRef-RestrictedPublic: read the text at https://example.org/terms\n{}\n{}\n",
            LocalOnlyReason::HostTerms.terms(),
            LocalOnlyReason::HostTerms.policy()
        )
    );
}

#[test]
fn refused_and_permissive_licences_are_not_offered_for_acceptance() {
    let home = TempDir::new("refused");
    // `revoke` names them the same way, so a refused `LicenseRef-` licence
    // is not asked for a terms address there either.
    for args in [
        &["accept", "CC-BY-NC-ND-4.0", "--yes"][..],
        &["accept", "CC-BY-ND-4.0", "--yes"],
        &["accept", "Apache-2.0", "--yes"],
        // A refused `LicenseRef-` licence has no text of its own: it is
        // named as refused, not asked for a terms address.
        &["accept", "LicenseRef-ResearchOnly"],
        &["accept", "LicenseRef-Unconfirmed", "--yes"],
        &["accept", "LicenseRef-PlatformTerms", "https://example.org/terms", "--yes"],
        &["revoke", "LicenseRef-ResearchOnly"],
        &["revoke", "LicenseRef-Unconfirmed"],
        &["revoke", "LicenseRef-PlatformTerms", "https://example.org/terms"],
        &["revoke", "CC-BY-NC-ND-4.0"],
        &["revoke", "Apache-2.0"],
    ] {
        let licence = args[1];
        let says = match licence {
            "Apache-2.0" => format!("Apache-2.0 is CLASS_PERMISSIVE: there is nothing to {}", args[0]),
            _ => {
                let reason = licence.parse::<Licence>().unwrap().refusal().unwrap();
                format!("{licence}: refused ({reason}); the zoo fetches nothing under it")
            }
        };
        let out = zoo(&home.0, args);
        assert!(!out.status.success(), "{args:?}: {out:?}");
        assert!(out.stdout.is_empty(), "{args:?}: {out:?}");
        assert_eq!(text(&out.stderr), format!("turbo-zoo: {says}\n"), "{args:?}");
    }
    assert!(!home.0.join("acceptances.json").exists());
}

#[test]
fn show_gives_a_non_commercial_artifacts_terms_and_policy() {
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
  "attribution": "Trained on UD_English-GUM r2.16 (CC-BY-NC-SA-4.0).",
  "origin": "https://example.org/models/m/",
  "files": [{{"path": "m.bin", "url": "https://example.org/models/m/1/m.bin", "size": {}, "sha256": "{}", "sha512": "{}"}}]
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
    assert!(
        shown.contains(
            "origin: https://example.org/models/m/ (every file's address lies under it; redirects are followed over \
             HTTPS only)"
        ),
        "{shown}"
    );
    let terms = format!("local only: {}\n", LocalOnlyReason::NonCommercial.terms());
    let policy = format!("local only: {}\n", LocalOnlyReason::NonCommercial.policy());
    assert!(shown.contains("attribution: Trained on UD_English-GUM r2.16 (CC-BY-NC-SA-4.0).\n"), "{shown}");
    assert!(shown.contains(&format!("{terms}{policy}")), "{shown}");
}

#[test]
fn show_prints_attribution_for_builtin_entries() {
    let home = TempDir::new("show-builtin");
    for (id, expected) in [
        (
            "opennlp-ca-ud-ancora-sentence-1.3-2.5.4",
            "attribution: Apache OpenNLP UD models 1.3 (Apache-2.0, \
             https://archive.apache.org/dist/opennlp/models/ud-models-1.3/README), trained on UD_Catalan-AnCora \
             r2.16 (CC-BY-4.0, https://github.com/UniversalDependencies/UD_Catalan-AnCora/blob/r2.16/README.md).",
        ),
        (
            "turbo-en-masc-3.0.0-sentence-1",
            "attribution: Trained on MASC 3.0.0, the Manually Annotated Sub-Corpus of the American National Corpus",
        ),
        ("fastino-gliner2-base-v1", "attribution: fastino/gliner2-base-v1 at f9634218e53580c56edf0de97ca1a7d3f1c2354e"),
    ] {
        let out = zoo(&home.0, &["show", id]);
        assert!(out.status.success(), "{id}: {out:?}");
        let shown = text(&out.stdout);
        assert!(shown.contains(expected), "{id}: {shown}");
    }
}
