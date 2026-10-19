//! The zoo's licence rules and integrity checks, on a small index of one
//! artifact per class served from memory.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use turbo_zoo::{
    Acceptances, Check, GplAcceptance, Index, Licence, LicenceClass, Policy, Requirement, Sha256, Sha512, Transport,
    Url, UserId, Zoo, ZooError,
};

/// Serves fixed bytes per address and counts the requests.
struct Memory {
    files: HashMap<String, Vec<u8>>,
    gets: Cell<usize>,
    /// The size the zoo expected, per request.
    sizes: RefCell<Vec<u64>>,
}

impl Memory {
    fn new(files: &[(&str, &[u8])]) -> Memory {
        Memory {
            files: files.iter().map(|(u, b)| (u.to_string(), b.to_vec())).collect(),
            gets: Cell::new(0),
            sizes: RefCell::new(Vec::new()),
        }
    }
}

impl Transport for Memory {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        self.sizes.borrow_mut().push(size);
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
        "url": "https://example.org/models/{id}.bin",
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
    format!("https://example.org/models/{id}.bin")
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
        with_origin(entry("non-commercial", CcByNcSa40, &[], LocalOnly, BYTES)),
        with_origin(entry("trained-on-non-commercial", Apache2, &[CcByNcSa40], LocalOnly, BYTES)),
        entry("unconfirmed", Apache2, &[Unconfirmed], Refused, BYTES),
        entry("lgpl-and-share-alike", Lgpl21Only, &[CcBySa30], ShareAlike, BYTES),
        with_origin(entry("restricted", RestrictedPublic, &[], LocalOnly, BYTES)),
        with_origin(entry("trained-on-restricted", Apache2, &[RestrictedPublic], LocalOnly, BYTES)),
        with_origin(entry("restricted-and-share-alike", Apache2, &[CcBySa40, RestrictedPublic], LocalOnly, BYTES)),
        with_origin(entry("restricted-and-non-commercial", RestrictedPublic, &[CcByNcSa40], LocalOnly, BYTES)),
        entry("no-derivatives", CcByNd40, &[], Refused, BYTES),
        entry("trained-on-non-commercial-no-derivatives", Apache2, &[CcByNcNd40], Refused, BYTES),
        entry("non-commercial-and-no-derivatives", Apache2, &[CcByNcSa40, CcByNcNd40], Refused, BYTES),
    ])
}

/// The entry with an origin its files lie under, `https://example.org/models/`.
fn with_origin(entry: String) -> String {
    entry.replace("\"files\": [", "\"origin\": \"https://example.org/models/\",\n      \"files\": [")
}

/// The terms of licences with a single text.
fn reqs(licences: &[Licence]) -> Vec<Requirement> {
    licences.iter().map(|l| Requirement::of(*l).unwrap()).collect()
}

/// The terms `entry` gives a `LicenseRef-RestrictedPublic` licence of the
/// artifact `id` itself, or of its training data when `id` is `None`.
fn restricted_terms(id: Option<&str>) -> Requirement {
    let text = match id {
        Some(id) => format!("https://example.org/{id}/LICENSE"),
        None => "https://example.org/data/LicenseRef-RestrictedPublic".to_owned(),
    };
    Requirement { licence: Licence::RestrictedPublic, text }
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
        "restricted",
        "trained-on-restricted",
        "restricted-and-share-alike",
        "restricted-and-non-commercial",
        "no-derivatives",
        "trained-on-non-commercial-no-derivatives",
        "non-commercial-and-no-derivatives",
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
    assert_eq!(
        listed,
        [
            "permissive",
            "share-alike",
            "trained-on-share-alike",
            "lgpl",
            "gpl",
            "non-commercial",
            "trained-on-non-commercial",
            "lgpl-and-share-alike",
            "restricted",
            "trained-on-restricted",
            "restricted-and-share-alike",
            "restricted-and-non-commercial",
        ]
    );
    for (id, licence) in [
        ("no-derivatives", Licence::CcByNd40),
        ("trained-on-non-commercial-no-derivatives", Licence::CcByNcNd40),
        ("non-commercial-and-no-derivatives", Licence::CcByNcNd40),
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
    let msg = zoo
        .fetch("trained-on-non-commercial-no-derivatives", &alice(), &Acceptances::default())
        .unwrap_err()
        .to_string();
    assert!(msg.contains("refused") && msg.contains("CC-BY-NC-ND-4.0 (no-derivatives terms)"), "{msg}");
}

#[test]
fn share_alike_lgpl_and_gpl_need_the_licence_accepted() {
    let dir = TempDir::new("accept");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let none = Acceptances::default();
    for (id, class, missing) in [
        ("share-alike", LicenceClass::ShareAlike, reqs(&[Licence::CcBySa40])),
        ("trained-on-share-alike", LicenceClass::ShareAlike, reqs(&[Licence::CcBySa40])),
        ("lgpl", LicenceClass::Lgpl, reqs(&[Licence::Lgpl30Only])),
        ("gpl", LicenceClass::Gpl, reqs(&[Licence::Gpl30Only])),
        ("lgpl-and-share-alike", LicenceClass::ShareAlike, reqs(&[Licence::CcBySa30, Licence::Lgpl21Only])),
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
        Err(ZooError::NotAccepted { ref missing, .. }) if missing == &reqs(&[Licence::Lgpl21Only])
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
    acc.revoke(&alice(), &Requirement::of(Licence::CcBySa40).unwrap());
    assert!(matches!(zoo.fetch("share-alike", &alice(), &acc), Err(ZooError::NotAccepted { .. })));
}

#[test]
fn restricted_data_needs_its_hosts_terms_accepted() {
    let dir = TempDir::new("restricted");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let none = Acceptances::default();
    let own = restricted_terms(Some("restricted"));
    let data = restricted_terms(None);
    for (id, missing) in [
        ("restricted", vec![own.clone()]),
        ("trained-on-restricted", vec![data.clone()]),
        ("restricted-and-share-alike", vec![Requirement::of(Licence::CcBySa40).unwrap(), data.clone()]),
    ] {
        match zoo.fetch(id, &alice(), &none) {
            Err(ZooError::NotAccepted { class, missing: m, .. }) => {
                assert_eq!(class, LicenceClass::LocalOnly, "{id}");
                assert_eq!(m, missing, "{id}");
            }
            other => panic!("{id}: {other:?}"),
        }
    }
    assert_eq!(t.gets.get(), 0);

    // One host's terms never cover another's.
    let mut acc = Acceptances::default();
    acc.accept_terms(&alice(), &data, 0).unwrap();
    assert!(matches!(zoo.fetch("restricted", &alice(), &acc), Err(ZooError::NotAccepted { .. })));
    zoo.fetch("trained-on-restricted", &alice(), &acc).unwrap();
    acc.accept_terms(&alice(), &own, 0).unwrap();
    zoo.fetch("restricted", &alice(), &acc).unwrap();
    // Every licence that needs it is still needed.
    assert!(matches!(
        zoo.fetch("restricted-and-share-alike", &alice(), &acc),
        Err(ZooError::NotAccepted { ref missing, .. }) if missing == &reqs(&[Licence::CcBySa40])
    ));
    acc.accept(&alice(), Licence::CcBySa40, 0).unwrap();
    zoo.fetch("restricted-and-share-alike", &alice(), &acc).unwrap();
}

/// Non-commercial data, and every model trained on it, is local-only: the
/// zoo fetches it from its origin after the user accepts the licence's
/// legal code, under either GPL policy, and per user.
#[test]
fn non_commercial_data_needs_its_licence_accepted() {
    let nc = Requirement::of(Licence::CcByNcSa40).unwrap();
    let host = restricted_terms(Some("restricted-and-non-commercial"));
    for policy in [Policy::default(), Policy { gpl: GplAcceptance::NotRequired }] {
        let dir = TempDir::new(&format!("non-commercial-{:?}", policy.gpl));
        let t = transport();
        let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), policy, &t);
        let none = Acceptances::default();
        for (id, missing) in [
            ("non-commercial", vec![nc.clone()]),
            ("trained-on-non-commercial", vec![nc.clone()]),
            ("restricted-and-non-commercial", vec![host.clone(), nc.clone()]),
        ] {
            match zoo.fetch(id, &alice(), &none) {
                Err(ZooError::NotAccepted { class, missing: m, .. }) => {
                    assert_eq!(class, LicenceClass::LocalOnly, "{id}");
                    assert_eq!(m, missing, "{id}");
                }
                other => panic!("{id}: {other:?}"),
            }
        }
        assert_eq!(t.gets.get(), 0, "nothing is fetched before the licence is accepted");
        assert!(!dir.0.join("non-commercial").exists());

        // Another non-commercial licence does not cover it.
        let mut acc = Acceptances::default();
        acc.accept(&alice(), Licence::CcByNc40, 0).unwrap();
        acc.accept(&alice(), Licence::CcByNcSa30, 0).unwrap();
        assert!(matches!(
            zoo.fetch("non-commercial", &alice(), &acc),
            Err(ZooError::NotAccepted { ref missing, .. }) if missing == std::slice::from_ref(&nc)
        ));
        assert_eq!(t.gets.get(), 0);

        acc.accept(&alice(), Licence::CcByNcSa40, 1_792_000_000).unwrap();
        let got = zoo.fetch("non-commercial", &alice(), &acc).unwrap();
        assert_eq!(got.single().unwrap(), dir.0.join("non-commercial").join("non-commercial.bin"));
        assert_eq!(fs::read(got.single().unwrap()).unwrap(), BYTES);
        zoo.fetch("trained-on-non-commercial", &alice(), &acc).unwrap();
        assert_eq!(t.gets.get(), 2);
        // Restricted data beside it still needs its host's terms.
        assert!(matches!(
            zoo.fetch("restricted-and-non-commercial", &alice(), &acc),
            Err(ZooError::NotAccepted { ref missing, .. }) if missing == std::slice::from_ref(&host)
        ));
        acc.accept_terms(&alice(), &host, 0).unwrap();
        zoo.fetch("restricted-and-non-commercial", &alice(), &acc).unwrap();
        assert_eq!(t.gets.get(), 3);

        // Acceptance is per user and checked on every call, cached file
        // or not.
        let bob: UserId = "bob".parse().unwrap();
        assert!(matches!(zoo.fetch("non-commercial", &bob, &acc), Err(ZooError::NotAccepted { .. })));
        acc.revoke(&alice(), &nc);
        assert!(matches!(zoo.fetch("non-commercial", &alice(), &acc), Err(ZooError::NotAccepted { .. })));
    }
}

#[test]
fn local_only_data_and_models_trained_on_it_are_never_shippable() {
    let index = Index::parse(sample().as_bytes()).unwrap();
    let shippable: Vec<&str> = index.shippable().map(|a| a.id.as_str()).collect();
    assert_eq!(shippable, ["permissive"]);
    for id in [
        "restricted",
        "trained-on-restricted",
        "restricted-and-share-alike",
        "non-commercial",
        "trained-on-non-commercial",
        "restricted-and-non-commercial",
    ] {
        let a = index.get(id).unwrap();
        assert_eq!(a.class, LicenceClass::LocalOnly, "{id}");
        assert!(!a.class.may_bundle() && !a.class.may_redistribute(), "{id}");
    }
    // A model trained locally on restricted or non-commercial data is
    // local-only too.
    assert_eq!(turbo_zoo::class_of([Licence::Apache2, Licence::RestrictedPublic]), LicenceClass::LocalOnly);
    assert_eq!(turbo_zoo::class_of([Licence::Apache2, Licence::CcByNcSa40]), LicenceClass::LocalOnly);
    assert_eq!(turbo_zoo::class_of([Licence::Apache2, Licence::CcByNc40]), LicenceClass::LocalOnly);
    assert_eq!(turbo_zoo::class_of([Licence::Apache2, Licence::CcByNcNd40]), LicenceClass::Refused);
}

#[test]
fn local_only_artifacts_name_their_files_under_their_origin() {
    use Licence::*;
    use LicenceClass::*;
    let restricted = entry("r", RestrictedPublic, &[], LocalOnly, BYTES);
    let non_commercial = entry("r", CcByNcSa40, &[], LocalOnly, BYTES);
    let trained_on_non_commercial = entry("r", Apache2, &[CcByNc40], LocalOnly, BYTES);
    for e in [&restricted, &non_commercial, &trained_on_non_commercial] {
        assert!(Index::parse(index_of(&[with_origin(e.clone())]).as_bytes()).is_ok(), "{e}");
    }
    let bad = [
        // No origin.
        restricted.clone(),
        non_commercial.clone(),
        trained_on_non_commercial.clone(),
        // A file on another host than the origin's.
        with_origin(restricted.replace("https://example.org/models/r.bin", "https://mirror.example.net/models/r.bin")),
        with_origin(
            non_commercial.replace("https://example.org/models/r.bin", "https://mirror.example.net/models/r.bin"),
        ),
        // An origin on an artifact that is not restricted.
        with_origin(entry("p", Apache2, &[], Permissive, BYTES)),
        with_origin(entry("s", Apache2, &[CcBySa40], ShareAlike, BYTES)),
        // The class must still be the most restrictive one.
        with_origin(entry("r", Apache2, &[RestrictedPublic], Gpl, BYTES)),
        with_origin(entry("r", RestrictedPublic, &[Gpl30Only], Gpl, BYTES)),
        with_origin(entry("r", Apache2, &[CcByNcSa40], ShareAlike, BYTES)),
        with_origin(entry("r", RestrictedPublic, &[CcByNcNd40], LocalOnly, BYTES)),
        with_origin(entry("r", CcByNcSa40, &[CcByNd40], LocalOnly, BYTES)),
    ];
    for b in bad {
        let text = index_of(&[b]);
        assert!(matches!(Index::parse(text.as_bytes()), Err(ZooError::Index(_))), "{text}");
    }
    // The host is compared without regard to case, but the path is not
    // ignored: a file must lie under the origin's path.
    let upper =
        with_origin(restricted.clone()).replace("\"https://example.org/models/\"", "\"https://EXAMPLE.org/models/\"");
    assert!(Index::parse(index_of(&[upper]).as_bytes()).is_ok());
    let elsewhere =
        with_origin(restricted).replace("\"https://example.org/models/\"", "\"https://example.org/terms/\"");
    assert!(matches!(Index::parse(index_of(&[elsewhere]).as_bytes()), Err(ZooError::Index(_))));
}

/// A non-commercial entry for UD English GUM whose origin is `origin` and
/// whose only file is fetched from `url`.
fn gum(origin: &str, url: &str) -> String {
    entry("gum", Licence::CcByNcSa40, &[], LicenceClass::LocalOnly, BYTES)
        .replace("https://example.org/models/gum.bin", url)
        .replace("\"files\": [", &format!("\"origin\": \"{origin}\",\n      \"files\": ["))
}

/// The origin is the publisher's own location, not its whole host: a copy
/// on the same host, or at a path that only shares a string prefix with
/// the origin's, is not the original.
#[test]
fn an_origin_is_a_location_not_a_host() {
    const REPO: &str = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM";
    let file = format!("{REPO}/r2.16/en_gum-ud-train.conllu");
    let parses = |origin: &str, url: &str| Index::parse(index_of(&[gum(origin, url)]).as_bytes());
    let with_slash = format!("{REPO}/");
    for origin in [REPO, with_slash.as_str()] {
        // Under the origin's path, with or without its trailing `/`.
        for ok in [file.clone(), format!("{REPO}/en_gum-ud-train.conllu"), format!("{file}?raw=true"), REPO.to_owned()]
        {
            assert!(parses(origin, &ok).is_ok(), "{origin} {ok}");
        }
        for bad in [
            // A mirror on the same host.
            "https://raw.githubusercontent.com/someone/UD_English-GUM/r2.16/en_gum-ud-train.conllu".to_owned(),
            // A sibling path that shares a string prefix with the origin's.
            format!("{REPO}-copy/r2.16/en_gum-ud-train.conllu"),
            format!("{REPO}2/en_gum-ud-train.conllu"),
            // Paths compare case for case.
            "https://raw.githubusercontent.com/universaldependencies/UD_English-GUM/r2.16/x.conllu".to_owned(),
            // Another host, and one written to look like the origin's.
            "https://github.com/UniversalDependencies/UD_English-GUM/raw/r2.16/en_gum-ud-train.conllu".to_owned(),
            "https://raw.githubusercontent.com@mirror.example.net/UniversalDependencies/UD_English-GUM/x".to_owned(),
            "https://raw.githubusercontent.com:8443/UniversalDependencies/UD_English-GUM/x".to_owned(),
            // Steps out of the origin's path, plain or encoded.
            format!("{REPO}/../UD_English-GUM-copy/en_gum-ud-train.conllu"),
            format!("{REPO}/%2e%2e/UD_English-GUM-copy/en_gum-ud-train.conllu"),
            format!("{REPO}/%2E./UD_English-GUM-copy/en_gum-ud-train.conllu"),
            format!("{REPO}/./en_gum-ud-train.conllu"),
            format!("{REPO}/..%2fUD_English-GUM-copy/en_gum-ud-train.conllu"),
            format!("{REPO}/..%5CUD_English-GUM-copy/en_gum-ud-train.conllu"),
            // A query on the origin's path is not a path under it.
            format!("{REPO}?file=en_gum-ud-train.conllu"),
        ] {
            match parses(origin, &bad) {
                Err(ZooError::Index(m)) => assert!(m.contains("not under its origin"), "{origin} {bad}: {m}"),
                other => panic!("{origin} {bad}: {other:?}"),
            }
        }
        // A backslash, escaped once for Rust and once for JSON, is not in
        // an address at all.
        let backslash = format!("{REPO}/..\\\\UD_English-GUM-copy/en_gum-ud-train.conllu");
        match parses(origin, &backslash) {
            Err(ZooError::Index(m)) => assert!(m.contains("without a backslash"), "{origin} {backslash}: {m}"),
            other => panic!("{origin} {backslash}: {other:?}"),
        }
    }
    // The host of the origin is compared without regard to case.
    let upper = "https://RAW.GitHubUserContent.com/UniversalDependencies/UD_English-GUM/";
    assert!(parses(upper, &file).is_ok());
    // An origin is a location: no user information, query, fragment or
    // step out of its path.
    for origin in [
        format!("{REPO}?ref=r2.16"),
        format!("{REPO}#readme"),
        format!("{REPO}/r2.16/.."),
        format!("{REPO}/%2E%2E/"),
        // On `mirror.example.net`, though it reads as GitHub's host.
        "https://raw.githubusercontent.com@mirror.example.net/UniversalDependencies/UD_English-GUM/".to_owned(),
    ] {
        match parses(&origin, &file) {
            Err(ZooError::Index(m)) => assert!(m.contains("is not a location"), "{origin}: {m}"),
            other => panic!("{origin}: {other:?}"),
        }
    }
    // Every file must lie under it, not one of them.
    let two = gum(&with_slash, &file).replace(
        "\"files\": [{",
        &format!(
            "\"files\": [{{\"path\": \"copy.conllu\", \"url\": \"{REPO}-copy/x.conllu\", \"size\": {}, \"sha256\": \"{}\", \"sha512\": \"{}\"}}, {{",
            BYTES.len(),
            Sha256::of(BYTES),
            Sha512::of(BYTES)
        ),
    );
    match Index::parse(index_of(&[two]).as_bytes()) {
        Err(ZooError::Index(m)) => assert!(m.contains("copy.conllu") && m.contains("not under its origin"), "{m}"),
        other => panic!("{other:?}"),
    }
}

/// An origin names a path on its host, so a whole host is not one; a file
/// at the root of a host is named by its own address.
#[test]
fn a_whole_host_is_not_an_origin() {
    const FILE: &str = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM/r2.16/x.conllu";
    let parses = |origin: &str, url: &str| Index::parse(index_of(&[gum(origin, url)]).as_bytes());
    for origin in [
        "https://raw.githubusercontent.com",
        "https://raw.githubusercontent.com/",
        "https://raw.githubusercontent.com//",
        "https://RAW.githubusercontent.com/",
    ] {
        match parses(origin, FILE) {
            Err(ZooError::Index(m)) => assert!(m.contains("is not a location"), "{origin}: {m}"),
            other => panic!("{origin}: {other:?}"),
        }
    }
    let root = "https://example.org/gum.bin";
    assert!(parses(root, root).is_ok());
    for origin in [root.to_owned(), format!("{root}/")] {
        match parses(&origin, "https://example.org/other.bin") {
            Err(ZooError::Index(m)) => assert!(m.contains("not under its origin"), "{origin}: {m}"),
            other => panic!("{origin}: {other:?}"),
        }
    }
}

/// Forms a server may decode or strip into a step out of a directory: a
/// `%` encoded again, a malformed escape a lenient decoder may repair, a
/// `;` that starts segment parameters, plain or encoded, and encoded bytes
/// that are not UTF-8, overlong forms of `.`, `/` and `\` among them. None
/// is accepted in a file's address or in an origin, in upper or lower
/// case.
#[test]
fn encoded_and_parameter_steps_are_refused() {
    const REPO: &str = "https://raw.githubusercontent.com/UniversalDependencies/UD_English-GUM";
    let file = format!("{REPO}/r2.16/en_gum-ud-train.conllu");
    let parses = |origin: &str, url: &str| Index::parse(index_of(&[gum(origin, url)]).as_bytes());
    let forms = [
        // An encoded `%`.
        "%252e%252e",
        "%252E%252E",
        "..%252f",
        // Malformed escapes, and IIS's `%u` form.
        "%%32%65%%32%65",
        "%2%65%2%65",
        "..%%35%63",
        "%u002e%u002e",
        "..%",
        "..%2",
        "%zz",
        // A `;`, plain or encoded.
        "..;",
        ".;x",
        "r2.16;jsessionid=1",
        "..%3b",
        "..%3B",
        // An encoded NUL.
        "..%00",
        // Overlong forms.
        "%c0%ae%c0%ae",
        "%C0%AE%C0%AE",
        "..%c0%af",
        "..%c1%9c",
        "%e0%80%ae%e0%80%ae",
        "%E0%9F%BF",
        "%f0%80%80%ae%f0%80%80%ae",
        "%f0%8f%bf%bf",
        // Five- and six-byte forms, and bytes no UTF-8 holds.
        "%f8%80%80%80%ae",
        "%fc%80%80%80%80%ae",
        "%f5%80%80%80",
        "%ff",
        // A surrogate, and a value past U+10FFFF.
        "%ed%a0%80",
        "%f4%90%80%80",
        // A bad, lone or missing continuation byte.
        "%e0%00%ae",
        "%c3a",
        "%80",
        "%c3",
    ];
    for form in forms {
        let bad = format!("{REPO}/{form}/UD_English-GUM-copy/en_gum-ud-train.conllu");
        match parses(REPO, &bad) {
            Err(ZooError::Index(m)) => assert!(m.contains("not under its origin"), "{bad}: {m}"),
            other => panic!("{bad}: {other:?}"),
        }
        let origin = format!("{REPO}/{form}/");
        match parses(&origin, &file) {
            Err(ZooError::Index(m)) => assert!(m.contains("is not a location"), "{origin}: {m}"),
            other => panic!("{origin}: {other:?}"),
        }
    }
    // Percent-encoding that cannot make a step is kept: a space, and
    // well-formed UTF-8 at the edges of each length, U+0800, U+0FFF,
    // U+10000, U+1F600 and U+10FFFF among them.
    for encoded in [
        "a%20b",
        "%c3%a9",
        "%c2%80",
        "%DF%BF",
        "%e0%a0%80",
        "%E0%BF%BF",
        "%ed%9f%bf",
        "%ee%80%80",
        "%f0%90%80%80",
        "%F0%9F%98%80",
        "%f4%8f%bf%bf",
    ] {
        let ok = format!("{REPO}/r2.16/{encoded}.conllu");
        assert!(parses(REPO, &ok).is_ok(), "{ok}");
        let origin = format!("{REPO}/{encoded}/");
        assert!(parses(&origin, &format!("{origin}x.conllu")).is_ok(), "{origin}");
    }
    // A character cut off by the end of the path.
    for cut in ["%c3", "%f0%9f%98"] {
        let bad = format!("{REPO}/r2.16/x{cut}");
        match parses(REPO, &bad) {
            Err(ZooError::Index(m)) => assert!(m.contains("not under its origin"), "{bad}: {m}"),
            other => panic!("{bad}: {other:?}"),
        }
        let origin = format!("{REPO}/x{cut}");
        match parses(&origin, &origin) {
            Err(ZooError::Index(m)) => assert!(m.contains("is not a location"), "{origin}: {m}"),
            other => panic!("{origin}: {other:?}"),
        }
    }
    // A `;` in the query is not in the path.
    assert!(parses(REPO, &format!("{REPO}/r2.16/x.conllu?a;b")).is_ok());
    // An origin with empty segments after a non-empty one is a location.
    assert!(parses("https://example.org/a//", "https://example.org/a//x.conllu").is_ok());
}

/// An address names a host and holds no backslash.
#[test]
fn an_address_has_a_host_and_no_backslash() {
    for ok in ["https://example.org", "https://example.org/x", "https://a@example.org:8443/x?y#z"] {
        assert!(ok.parse::<Url>().is_ok(), "{ok}");
    }
    for (bad, says) in [
        ("https://", "names no host"),
        ("https:///example.org/x", "names no host"),
        ("https://@/x", "names no host"),
        ("https://user@/x", "names no host"),
        ("https://:443/x", "names no host"),
        ("https://user@:443/x", "names no host"),
        ("https://?x", "names no host"),
        ("https://#x", "names no host"),
        ("https://example.org\\x", "without a backslash"),
        ("https://example.org/a\\b", "without a backslash"),
        ("https://example.org/a b", "printable ASCII"),
        ("http://example.org/x", "is not an https:// address"),
    ] {
        match bad.parse::<Url>() {
            Err(m) => assert!(m.contains(says), "{bad}: {m}"),
            Ok(u) => panic!("{bad}: {u}"),
        }
    }
    // In an index too, for a file's address and for an origin.
    let file_without_host = entry("m", Licence::Apache2, &[], LicenceClass::Permissive, BYTES)
        .replace("https://example.org/models/m.bin", "https:///example.org/models/m.bin");
    let origin_without_host = gum("https:///raw.githubusercontent.com/x/", "https://raw.githubusercontent.com/x/y");
    for e in [file_without_host, origin_without_host] {
        match Index::parse(index_of(&[e]).as_bytes()) {
            Err(ZooError::Index(m)) => assert!(m.contains("names no host"), "{m}"),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn gpl_acceptance_can_be_turned_off() {
    let dir = TempDir::new("gpl");
    let t = transport();
    let lax = Policy { gpl: GplAcceptance::NotRequired };
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), lax, &t);
    zoo.fetch("gpl", &alice(), &Acceptances::default()).unwrap();
    // LGPL, CC BY-SA, restricted and non-commercial data still need it.
    for id in
        ["lgpl", "share-alike", "restricted", "trained-on-restricted", "non-commercial", "trained-on-non-commercial"]
    {
        assert!(matches!(zoo.fetch(id, &alice(), &Acceptances::default()), Err(ZooError::NotAccepted { .. })), "{id}");
    }
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

/// A symlink above a nested file is found before any directory is made
/// below it, so nothing is created outside the cache.
#[cfg(unix)]
#[test]
fn a_symlink_above_a_nested_path_creates_nothing_outside() {
    let dir = TempDir::new("nested-symlink");
    let outside = TempDir::new("nested-symlink-outside");
    let cache = dir.0.join("cache");
    fs::create_dir_all(&cache).unwrap();
    std::os::unix::fs::symlink(&outside.0, cache.join("m")).unwrap();
    let e = entry("m", Licence::Apache2, &[], LicenceClass::Permissive, BYTES)
        .replace("\"path\": \"m.bin\"", "\"path\": \"sub/deeper/m.bin\"");
    let t = Memory::new(&[(&url("m"), BYTES)]);
    let zoo = Zoo::new(Index::parse(index_of(&[e]).as_bytes()).unwrap(), cache.clone(), Policy::default(), &t);
    assert!(matches!(zoo.fetch("m", &alice(), &Acceptances::default()), Err(ZooError::Io { .. })));
    assert!(!outside.0.join("sub").exists());
    assert_eq!(fs::read_dir(&outside.0).unwrap().count(), 0);
    assert_eq!(t.gets.get(), 0);

    // The same with the symlink one level further down.
    fs::remove_file(cache.join("m")).unwrap();
    fs::create_dir(cache.join("m")).unwrap();
    std::os::unix::fs::symlink(&outside.0, cache.join("m").join("sub")).unwrap();
    assert!(matches!(zoo.fetch("m", &alice(), &Acceptances::default()), Err(ZooError::Io { .. })));
    assert!(!outside.0.join("deeper").exists());

    // With plain directories it is fetched into place.
    fs::remove_file(cache.join("m").join("sub")).unwrap();
    let got = zoo.fetch("m", &alice(), &Acceptances::default()).unwrap();
    assert_eq!(got.single().unwrap(), cache.join("m").join("sub").join("deeper").join("m.bin"));
}

/// A symlink left at the temporary name is removed, never written
/// through, and never renamed onto the file.
#[cfg(unix)]
#[test]
fn a_symlinked_temporary_file_is_never_written_through() {
    let dir = TempDir::new("partial-symlink");
    let outside = TempDir::new("partial-symlink-outside");
    let victim = outside.0.join("victim");
    fs::write(&victim, b"untouched").unwrap();
    let cache = dir.0.join("cache");
    fs::create_dir_all(cache.join("permissive")).unwrap();
    std::os::unix::fs::symlink(&victim, cache.join("permissive").join("permissive.bin.partial")).unwrap();
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), cache.clone(), Policy::default(), &t);
    let got = zoo.fetch("permissive", &alice(), &Acceptances::default()).unwrap();
    let path = got.single().unwrap();
    assert_eq!(fs::read(&victim).unwrap(), b"untouched");
    assert!(!fs::symlink_metadata(path).unwrap().file_type().is_symlink());
    assert_eq!(fs::read(path).unwrap(), BYTES);
    assert!(fs::symlink_metadata(cache.join("permissive").join("permissive.bin.partial")).is_err());
}

#[test]
fn the_transport_is_told_the_expected_size() {
    let dir = TempDir::new("size");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    zoo.fetch("permissive", &alice(), &Acceptances::default()).unwrap();
    assert_eq!(*t.sizes.borrow(), [BYTES.len() as u64]);
}

/// A broken acceptance file blocks only artifacts that need acceptance,
/// and the error says what is wrong.
#[test]
fn acceptances_are_read_only_when_needed() {
    let dir = TempDir::new("lazy");
    let t = transport();
    let zoo = Zoo::new(Index::parse(sample().as_bytes()).unwrap(), dir.0.clone(), Policy::default(), &t);
    let broken = || Err(ZooError::Acceptances("accepted[0]: broken".into()));
    zoo.fetch_with("permissive", &alice(), broken).unwrap();
    match zoo.fetch_with("share-alike", &alice(), broken) {
        Err(ZooError::Acceptances(m)) => assert!(m.contains("accepted[0]"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(t.gets.get(), 1);
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
        good.replace("https://example.org/models/m.bin", "http://example.org/models/m.bin"),
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
