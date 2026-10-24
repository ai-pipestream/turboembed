//! The zoo's licence rules and integrity checks, on a small index of one
//! artifact per class served from memory.

#[path = "zoo/origin.rs"]
mod origin;

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
      "attribution": "Example attribution for {id}.",
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
