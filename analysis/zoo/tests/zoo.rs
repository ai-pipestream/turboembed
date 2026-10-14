//! The zoo's licence rules and integrity checks, on a small index of one
//! artifact per class served from memory.

use std::cell::Cell;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use turbo_zoo::{
    Acceptances, Check, GplAcceptance, Index, Licence, LicenceClass, Policy, Sha256, Sha512, Transport, Url, UserId,
    Zoo, ZooError,
};

/// Serves fixed bytes per address and counts the requests.
struct Memory {
    files: HashMap<String, Vec<u8>>,
    gets: Cell<usize>,
}

impl Memory {
    fn new(files: &[(&str, &[u8])]) -> Memory {
        Memory { files: files.iter().map(|(u, b)| (u.to_string(), b.to_vec())).collect(), gets: Cell::new(0) }
    }
}

impl Transport for Memory {
    fn get(&self, url: &Url) -> Result<Vec<u8>, String> {
        self.gets.set(self.gets.get() + 1);
        self.files.get(url.as_str()).cloned().ok_or_else(|| "404".to_string())
    }
}

/// One index entry for `bytes`, with its own licence and training data.
fn entry(id: &str, license: Licence, trained_on: &[Licence], class: LicenceClass, bytes: &[u8]) -> String {
    let trained: Vec<String> = trained_on
        .iter()
        .map(|l| {
            format!(
                r#"{{"name": "data under {l}", "license": "{l}", "license_terms": "https://example.org/data/{l}"}}"#
            )
        })
        .collect();
    format!(
        r#"{{
      "id": "{id}",
      "kind": "KIND_MODEL",
      "component": "COMPONENT_TOKENIZER",
      "language": "en",
      "format": "FORMAT_OPENNLP_MODEL",
      "version": "1",
      "license": "{license}",
      "license_terms": "https://example.org/{id}/LICENSE",
      "trained_on": [{trained}],
      "class": "{class}",
      "files": [{{
        "path": "{id}.bin",
        "url": "https://example.org/{id}.bin",
        "size": {size},
        "sha256": "{sha256}",
        "sha512": "{sha512}"
      }}]
    }}"#,
        trained = trained.join(", "),
        size = bytes.len(),
        sha256 = Sha256::of(bytes),
        sha512 = Sha512::of(bytes),
    )
}

fn index_of(entries: &[String]) -> String {
    format!("{{\"zoo_version\": 1, \"artifacts\": [{}]}}", entries.join(", "))
}

const BYTES: &[u8] = b"model bytes";

fn url(id: &str) -> String {
    format!("https://example.org/{id}.bin")
}

/// One artifact per class, and the cases of a class coming from the
/// training data rather than the artifact's own licence.
fn sample() -> String {
    use Licence::*;
    use LicenceClass::*;
    index_of(&[
        entry("permissive", Apache2, &[CcBy40], Permissive, BYTES),
        entry("share-alike", CcBySa40, &[], ShareAlike, BYTES),
        entry("trained-on-share-alike", Apache2, &[CcBySa40], ShareAlike, BYTES),
        entry("lgpl", Lgpl30Only, &[], Lgpl, BYTES),
        entry("gpl", Gpl30Only, &[], Gpl, BYTES),
        entry("non-commercial", CcByNcSa40, &[], Refused, BYTES),
        entry("trained-on-non-commercial", Apache2, &[CcByNcSa40], Refused, BYTES),
        entry("unconfirmed", Apache2, &[Unconfirmed], Refused, BYTES),
        entry("lgpl-and-share-alike", Lgpl21Only, &[CcBySa30], ShareAlike, BYTES),
    ])
}

fn transport() -> Memory {
    let ids = [
        "permissive",
        "share-alike",
        "trained-on-share-alike",
        "lgpl",
        "gpl",
        "non-commercial",
        "trained-on-non-commercial",
        "unconfirmed",
        "lgpl-and-share-alike",
    ];
    let urls: Vec<String> = ids.iter().map(|i| url(i)).collect();
    Memory::new(&urls.iter().map(|u| (u.as_str(), BYTES)).collect::<Vec<_>>())
}

struct TempDir(PathBuf);

impl TempDir {
    fn new(name: &str) -> TempDir {
        let p = std::env::temp_dir().join(format!("turbo-zoo-{name}-{}", std::process::id()));
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

fn alice() -> UserId {
    "alice".parse().unwrap()
}

#[test]
fn permissive_is_fetched_without_acceptance_and_then_served_from_the_cache() {
    let dir = TempDir::new("permissive");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let none = Acceptances::default();
    let got = zoo.fetch("permissive", &alice(), &none).unwrap();
    let path = got.single().unwrap().to_path_buf();
    assert_eq!(path, dir.0.join("permissive").join("permissive.bin"));
    assert_eq!(fs::read(&path).unwrap(), BYTES);
    assert_eq!(t.gets.get(), 1);
    zoo.fetch("permissive", &alice(), &none).unwrap();
    assert_eq!(t.gets.get(), 1, "a verified cached file is not fetched again");
    assert!(!dir.0.join("permissive").join("permissive.bin.partial").exists());
}

#[test]
fn refused_artifacts_are_not_listed_and_never_fetched() {
    let dir = TempDir::new("refused");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let listed: Vec<&str> = zoo.index().list().iter().map(|a| a.id.as_str()).collect();
    assert_eq!(listed, ["permissive", "share-alike", "trained-on-share-alike", "lgpl", "gpl", "lgpl-and-share-alike"]);
    for (id, licence) in [
        ("non-commercial", Licence::CcByNcSa40),
        ("trained-on-non-commercial", Licence::CcByNcSa40),
        ("unconfirmed", Licence::Unconfirmed),
    ] {
        match zoo.fetch(id, &alice(), &Acceptances::default()) {
            Err(ZooError::Refused { id: got, licences }) => {
                assert_eq!(got.as_str(), id);
                assert_eq!(licences, [licence]);
            }
            other => panic!("{id}: {other:?}"),
        }
    }
    assert_eq!(t.gets.get(), 0);
    let msg = zoo.fetch("non-commercial", &alice(), &Acceptances::default()).unwrap_err().to_string();
    assert!(msg.contains("refused") && msg.contains("non-commercial"), "{msg}");
}

#[test]
fn share_alike_lgpl_and_gpl_need_the_licence_accepted() {
    let dir = TempDir::new("accept");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let none = Acceptances::default();
    for (id, class, missing) in [
        ("share-alike", LicenceClass::ShareAlike, vec![Licence::CcBySa40]),
        ("trained-on-share-alike", LicenceClass::ShareAlike, vec![Licence::CcBySa40]),
        ("lgpl", LicenceClass::Lgpl, vec![Licence::Lgpl30Only]),
        ("gpl", LicenceClass::Gpl, vec![Licence::Gpl30Only]),
        ("lgpl-and-share-alike", LicenceClass::ShareAlike, vec![Licence::CcBySa30, Licence::Lgpl21Only]),
    ] {
        match zoo.fetch(id, &alice(), &none) {
            Err(ZooError::NotAccepted { id: got, class: c, user, missing: m }) => {
                assert_eq!(got.as_str(), id);
                assert_eq!(c, class);
                assert_eq!(user, "alice");
                assert_eq!(m, missing, "{id}");
            }
            other => panic!("{id}: {other:?}"),
        }
    }
    assert_eq!(t.gets.get(), 0, "nothing is fetched before the licence is accepted");
    assert!(!dir.0.join("share-alike").exists());

    let mut acc = Acceptances::default();
    for l in [Licence::CcBySa40, Licence::Lgpl30Only, Licence::Gpl30Only, Licence::CcBySa30] {
        acc.accept(&alice(), l, 1_792_000_000).unwrap();
    }
    for id in ["share-alike", "trained-on-share-alike", "lgpl", "gpl"] {
        zoo.fetch(id, &alice(), &acc).unwrap();
    }
    // Both licences are needed, not one of them.
    assert!(matches!(
        zoo.fetch("lgpl-and-share-alike", &alice(), &acc),
        Err(ZooError::NotAccepted { ref missing, .. }) if missing == &[Licence::Lgpl21Only]
    ));
    acc.accept(&alice(), Licence::Lgpl21Only, 1_792_000_000).unwrap();
    zoo.fetch("lgpl-and-share-alike", &alice(), &acc).unwrap();
}

#[test]
fn acceptance_is_per_user_and_checked_on_every_call() {
    let dir = TempDir::new("per-user");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let mut acc = Acceptances::default();
    acc.accept(&alice(), Licence::CcBySa40, 0).unwrap();
    zoo.fetch("share-alike", &alice(), &acc).unwrap();
    let bob: UserId = "bob".parse().unwrap();
    assert!(matches!(zoo.fetch("share-alike", &bob, &acc), Err(ZooError::NotAccepted { .. })));
    // A revoked acceptance stops the zoo handing out the cached file.
    acc.revoke(&alice(), Licence::CcBySa40);
    assert!(matches!(zoo.fetch("share-alike", &alice(), &acc), Err(ZooError::NotAccepted { .. })));
}

#[test]
fn gpl_acceptance_can_be_turned_off() {
    let dir = TempDir::new("gpl");
    let t = transport();
    let lax = Policy { gpl: GplAcceptance::NotRequired };
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), lax, &t);
    zoo.fetch("gpl", &alice(), &Acceptances::default()).unwrap();
    // LGPL and CC BY-SA still need it.
    assert!(matches!(zoo.fetch("lgpl", &alice(), &Acceptances::default()), Err(ZooError::NotAccepted { .. })));
    assert!(matches!(zoo.fetch("share-alike", &alice(), &Acceptances::default()), Err(ZooError::NotAccepted { .. })));
}

fn tampered(check: Check, served: &[u8], sha512_of: &[u8]) {
    let dir = TempDir::new(&format!("tamper-{check}"));
    // The entry's size and SHA-256 are BYTES'; its SHA-512 may be of
    // other bytes, to reach the last check.
    let mut e = entry("m", Licence::Apache2, &[], LicenceClass::Permissive, BYTES);
    e = e.replace(&Sha512::of(BYTES).to_string(), &Sha512::of(sha512_of).to_string());
    let t = Memory::new(&[(&url("m"), served)]);
    let zoo = Zoo::new(Index::parse(index_of(&[e]).as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    match zoo.fetch("m", &alice(), &Acceptances::default()) {
        Err(ZooError::Integrity { id, path, check: c, expected, actual }) => {
            assert_eq!((id.as_str(), path.as_str(), c), ("m", "m.bin", check));
            assert_ne!(expected, actual);
        }
        other => panic!("{check}: {other:?}"),
    }
    assert!(!dir.0.join("m").join("m.bin").exists(), "a file that fails its check is not written");
    assert!(!dir.0.join("m").join("m.bin.partial").exists());
}

#[test]
fn a_size_mismatch_fails() {
    tampered(Check::Size, b"model bytes!", BYTES);
}

#[test]
fn a_sha256_mismatch_fails() {
    tampered(Check::Sha256, b"model bytez", BYTES);
}

#[test]
fn a_sha512_mismatch_fails() {
    tampered(Check::Sha512, BYTES, b"other");
}

#[test]
fn a_corrupt_cached_file_is_fetched_again() {
    let dir = TempDir::new("corrupt");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let none = Acceptances::default();
    let path = zoo.fetch("permissive", &alice(), &none).unwrap().files[0].clone();
    fs::write(&path, b"model bytez").unwrap();
    zoo.fetch("permissive", &alice(), &none).unwrap();
    assert_eq!(t.gets.get(), 2);
    assert_eq!(fs::read(&path).unwrap(), BYTES);
}

#[test]
fn offline_serves_the_cache_only() {
    let dir = TempDir::new("offline");
    let index = Index::parse(sample().as_bytes()).unwrap();
    let none = Acceptances::default();
    let offline = Zoo::new(index.clone(), dir.0.clone(), Policy::default(), &turbo_zoo::Offline);
    assert!(matches!(offline.fetch("permissive", &alice(), &none), Err(ZooError::Transport { .. })));
    let t = transport();
    Zoo::new(index, dir.0.clone(), Policy::default(), &t).fetch("permissive", &alice(), &none).unwrap();
    offline.fetch("permissive", &alice(), &none).unwrap();
}

#[cfg(unix)]
#[test]
fn a_symlink_in_the_cache_is_refused() {
    let dir = TempDir::new("symlink");
    let outside = TempDir::new("symlink-outside");
    let cache = dir.0.join("cache");
    fs::create_dir_all(&cache).unwrap();
    std::os::unix::fs::symlink(&outside.0, cache.join("permissive")).unwrap();
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), cache, Policy::default(), &t);
    assert!(matches!(zoo.fetch("permissive", &alice(), &Acceptances::default()), Err(ZooError::Io { .. })));
    assert!(!outside.0.join("permissive.bin").exists());
}

#[test]
fn unknown_ids_are_not_found() {
    let index = Index::parse(sample().as_bytes()).unwrap();
    assert!(matches!(index.get("nothing"), Err(ZooError::NotFound(_))));
}

#[test]
fn the_index_is_parsed_strictly() {
    let good = index_of(&[entry("m", Licence::Apache2, &[], LicenceClass::Permissive, BYTES)]);
    assert!(Index::parse(good.as_bytes()).is_ok());
    let sha256 = Sha256::of(BYTES).to_string();
    let bad = [
        // The class must be the one the licences give.
        index_of(&[entry("m", Licence::Apache2, &[Licence::CcBySa40], LicenceClass::Permissive, BYTES)]),
        index_of(&[entry("m", Licence::Apache2, &[], LicenceClass::ShareAlike, BYTES)]),
        good.replace("\"zoo_version\": 1", "\"zoo_version\": 2"),
        good.replace("\"version\": \"1\",", "\"version\": \"1\", \"note\": \"x\","),
        good.replace("\"version\": \"1\",", ""),
        good.replace("Apache-2.0", "Apache 2.0"),
        good.replace("KIND_MODEL", "MODEL"),
        good.replace("COMPONENT_TOKENIZER", "COMPONENT_SPELLER"),
        good.replace("\"m.bin\"", "\"../m.bin\""),
        good.replace("\"m.bin\"", "\"/m.bin\""),
        good.replace("\"m.bin\"", "\"a//m.bin\""),
        good.replace("https://example.org/m.bin", "http://example.org/m.bin"),
        good.replace(&sha256, &sha256.to_uppercase()),
        good.replace("\"id\": \"m\"", "\"id\": \"M\""),
        good.replace("\"id\": \"m\"", "\"id\": \"..\""),
        good.replace("\"language\": \"en\"", "\"language\": \"english\""),
        good.replace(&format!("\"size\": {}", BYTES.len()), &format!("\"size\": \"{}\"", BYTES.len())),
        good.replace("\"class\": \"CLASS_PERMISSIVE\"", "\"class\": \"PERMISSIVE\""),
        index_of(&[
            entry("m", Licence::Apache2, &[], LicenceClass::Permissive, BYTES),
            entry("m", Licence::Mit, &[], LicenceClass::Permissive, BYTES),
        ]),
    ];
    for b in &bad {
        assert!(matches!(Index::parse(b.as_bytes()), Err(ZooError::Index(_))), "{b}");
    }
    let no_files = good.split("\"files\"").next().unwrap().to_string() + "\"files\": []}]}";
    assert!(matches!(Index::parse(no_files.as_bytes()), Err(ZooError::Index(_))), "{no_files}");
    // Optional fields may be left out.
    let bare = good.replace("\"component\": \"COMPONENT_TOKENIZER\",", "").replace("\"language\": \"en\",", "");
    assert!(Index::parse(bare.as_bytes()).is_ok(), "{bare}");
}
