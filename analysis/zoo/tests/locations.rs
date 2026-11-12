//! More than one location per file (#2192): `also` lists further
//! addresses of the same bytes, tried in order after `url`, each answer
//! checked as the file before it is handed over. Only a permissive
//! artifact may list any.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Mutex;

use turbo_zoo::{Acceptances, Index, LicenceClass, Mirror, Policy, Sha256, Sha512, Transport, Url, UserId, Zoo};

const TEXT: &[u8] = b"hello hello hello\n";
const OTHER: &[u8] = b"hello hello HELLO\n";

/// TEXT as Python's `gzip.compress(..., mtime=0)` writes it.
const TEXT_GZ: [u8; 29] = [
    0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0xff, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x90,
    0x5c, 0x00, 0x3b, 0x7c, 0x8a, 0xdf, 0x12, 0x00, 0x00, 0x00,
];

const FIRST: &str = "https://first.example.org/hello.txt";
const SECOND: &str = "https://second.example.org/copies/hello.txt";
const THIRD: &str = "https://third.example.org/hello.txt";

/// One artifact `hello` of one file at `url`, with `rest` spliced in
/// after the address (an `also` list, a `download` entry).
fn index_json(licence: &str, class: &str, extra: &str, rest: &str) -> String {
    format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT",
          "version": "1", "license": "{licence}", "license_terms": "https://first.example.org/LICENSE",
          "trained_on": [], "class": "{class}", "attribution": "Example attribution."{extra},
          "files": [{{"path": "hello.txt", "url": "{FIRST}"{rest}, "size": {}, "sha256": "{}", "sha512": "{}"}}]}}]}}"#,
        TEXT.len(),
        Sha256::of(TEXT),
        Sha512::of(TEXT),
    )
}

fn permissive(rest: &str) -> Index {
    Index::parse(index_json("Apache-2.0", "CLASS_PERMISSIVE", "", rest).as_bytes()).unwrap()
}

fn also(urls: &[&str]) -> String {
    let list: Vec<String> = urls.iter().map(|u| format!("{u:?}")).collect();
    format!(r#", "also": [{}]"#, list.join(", "))
}

/// Serves each address what the test set for it, refuses every other
/// address as a dead host does, and records the addresses asked.
struct Hosts {
    serve: BTreeMap<&'static str, &'static [u8]>,
    asked: Mutex<Vec<String>>,
}

impl Hosts {
    fn new(serve: &[(&'static str, &'static [u8])]) -> Hosts {
        Hosts { serve: serve.iter().copied().collect(), asked: Mutex::new(Vec::new()) }
    }

    fn asked(&self) -> Vec<String> {
        self.asked.lock().unwrap().clone()
    }
}

impl Transport for Hosts {
    fn get(&self, url: &Url, _size: u64) -> Result<Vec<u8>, String> {
        self.asked.lock().unwrap().push(url.to_string());
        self.serve.get(url.as_str()).map(|b| b.to_vec()).ok_or_else(|| "connection refused".to_owned())
    }
}

fn cache(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-locations-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// Fetches `hello` through `transport` and returns the file's bytes, or
/// the error's text, and whether the cache holds the file afterwards.
fn fetch(name: &str, index: Index, transport: &dyn Transport) -> (Result<Vec<u8>, String>, bool) {
    let dir = cache(name);
    let zoo = Zoo::new(index, dir.clone(), Policy::default(), transport);
    let user: UserId = "alice".parse().unwrap();
    let got = zoo
        .fetch("hello", &user, &Acceptances::default())
        .map(|f| std::fs::read(&f.files[0]).unwrap())
        .map_err(|e| e.to_string());
    let kept = dir.join("hello").join("hello.txt").is_file();
    let _ = std::fs::remove_dir_all(&dir);
    (got, kept)
}

#[test]
fn a_dead_first_location_and_a_live_second_one_install_the_artifact() {
    let hosts = Hosts::new(&[(SECOND, TEXT)]);
    let (got, kept) = fetch("dead-first", permissive(&also(&[SECOND])), &hosts);
    assert_eq!(got.unwrap(), TEXT);
    assert!(kept);
    assert_eq!(hosts.asked(), [FIRST, SECOND]);
}

#[test]
fn the_first_location_that_serves_the_file_is_the_only_one_asked() {
    let hosts = Hosts::new(&[(FIRST, TEXT), (SECOND, OTHER)]);
    let (got, _) = fetch("first-wins", permissive(&also(&[SECOND])), &hosts);
    assert_eq!(got.unwrap(), TEXT);
    assert_eq!(hosts.asked(), [FIRST]);
}

#[test]
fn a_first_location_serving_other_bytes_passes_the_request_to_the_next() {
    let hosts = Hosts::new(&[(FIRST, OTHER), (THIRD, TEXT)]);
    let (got, kept) = fetch("other-first", permissive(&also(&[SECOND, THIRD])), &hosts);
    assert_eq!(got.unwrap(), TEXT);
    assert!(kept);
    assert_eq!(hosts.asked(), [FIRST, SECOND, THIRD]);
}

#[test]
fn a_second_location_serving_other_bytes_is_refused_and_nothing_is_kept() {
    let hosts = Hosts::new(&[(SECOND, OTHER)]);
    let (got, kept) = fetch("other-second", permissive(&also(&[SECOND])), &hosts);
    let e = got.unwrap_err();
    assert!(e.contains(&format!("{FIRST}: connection refused")), "{e}");
    assert!(e.contains(SECOND) && e.contains("SHA-256"), "{e}");
    assert!(!kept);
}

#[test]
fn a_file_served_compressed_is_taken_packed_or_unpacked_from_any_location() {
    let download = format!(
        r#", "download": {{"compression": "gzip", "download_bytes": {}, "download_sha256": "{}"}}"#,
        TEXT_GZ.len(),
        Sha256::of(&TEXT_GZ)
    );
    // `download` follows the digests in the canonical form.
    let json = index_json("Apache-2.0", "CLASS_PERMISSIVE", "", &also(&[SECOND, THIRD]))
        .replace(r#""}]}]}"#, &format!(r#""{download}}}]}}]}}"#));
    for (name, serve) in [("gz", &TEXT_GZ[..]), ("plain", TEXT)] {
        let hosts = Hosts { serve: [(THIRD, serve)].into_iter().collect(), asked: Mutex::new(Vec::new()) };
        let (got, _) = fetch(name, Index::parse(json.as_bytes()).unwrap(), &hosts);
        assert_eq!(got.unwrap(), TEXT, "{name}");
    }
}

#[test]
fn a_mirror_rewrites_every_location() {
    let moved = "https://mirror.example.org/zoo/second.example.org/copies/hello.txt";
    let hosts = Hosts::new(&[(moved, TEXT)]);
    let mirror = Mirror::new("https://mirror.example.org/zoo", hosts).unwrap();
    let (got, _) = fetch("mirror", permissive(&also(&[SECOND])), &mirror);
    assert_eq!(got.unwrap(), TEXT);
}

#[test]
fn an_index_without_the_field_reads_and_writes_as_before() {
    let index = permissive("");
    let file = &index.get("hello").unwrap().files[0];
    assert!(file.also.is_empty());
    assert_eq!(file.locations().map(Url::as_str).collect::<Vec<_>>(), [FIRST]);
    assert!(!Index::to_canonical_json(index.list()).contains("also"));

    let index = permissive(&also(&[SECOND, THIRD]));
    let file = &index.get("hello").unwrap().files[0];
    assert_eq!(file.locations().map(Url::as_str).collect::<Vec<_>>(), [FIRST, SECOND, THIRD]);
    let written = Index::to_canonical_json(index.list());
    let (url, list) = (written.find("\"url\"").unwrap(), written.find("\"also\"").unwrap());
    assert!(url < list && list < written.find("\"size\"").unwrap(), "{written}");
    assert_eq!(Index::parse(written.as_bytes()).unwrap(), index);
}

#[test]
fn a_further_location_on_an_artifact_that_is_not_permissive_fails_to_load() {
    let origin = r#", "origin": "https://first.example.org/hello.txt""#;
    for (licence, class, extra) in [
        ("LGPL-3.0-only", LicenceClass::Lgpl, ""),
        ("CC-BY-SA-4.0", LicenceClass::ShareAlike, ""),
        ("GPL-3.0-only", LicenceClass::Gpl, ""),
        ("CC-BY-NC-4.0", LicenceClass::LocalOnly, origin),
        // Under the origin too: a local-only file has one address.
        ("LicenseRef-RestrictedPublic", LicenceClass::LocalOnly, origin),
    ] {
        let one = index_json(licence, class.name(), extra, "");
        assert!(Index::parse(one.as_bytes()).is_ok(), "{licence}");
        let under = "https://first.example.org/hello.txt/copy";
        for second in [SECOND, under] {
            let e = Index::parse(index_json(licence, class.name(), extra, &also(&[second])).as_bytes()).unwrap_err();
            let e = e.to_string();
            assert!(e.contains("hello.txt lists a further location") && e.contains(class.name()), "{licence}: {e}");
        }
    }
}

#[test]
fn a_location_listed_twice_or_malformed_fails_to_load() {
    for (list, why) in [
        (also(&[FIRST]), "twice"),
        (also(&[SECOND, SECOND]), "twice"),
        (also(&["http://second.example.org/hello.txt"]), "https://"),
        (also(&["https:///hello.txt"]), "names no host"),
        (r#", "also": "https://second.example.org/hello.txt""#.to_owned(), "sequence"),
    ] {
        let e = Index::parse(index_json("Apache-2.0", "CLASS_PERMISSIVE", "", &list).as_bytes()).unwrap_err();
        assert!(e.to_string().contains(why), "{list}: {e}");
    }
}
