//! Importing an OCI artifact pinned by digest (#2213), against a small
//! registry stand-in started in-process that answers the distribution
//! API as a registry does: a token challenge, manifests and blobs by
//! digest, and a redirect to storage that refuses a token. The files are
//! the repository's own small sealed bundle under testdata. The ignored
//! test reads a real registry.
#![cfg(feature = "http")]

use std::collections::BTreeMap;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use turbo_scratch::Scratch;
use turbo_zoo::import::{BundleImport, Described, OciImport, bundle, oci};
use turbo_zoo::{
    Acceptances, Check, Format, Index, Kind, Licence, ModelType, OciRegistry, Offline, Policy, Sha256, Sha512, UserId,
    Zoo, ZooError,
};

const HOST: &str = "registry.example.org";
const NAME: &str = "models/tiny-bert";
const PATHS: [&str; 4] =
    ["manifest.json", "reference/reference.safetensors", "tokenizer.json", "weights/model.safetensors"];

fn sealed() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}

fn bytes_of(path: &str) -> Vec<u8> {
    fs::read(sealed().join(path)).unwrap()
}

/// One layer as ORAS writes it for a file.
fn layer(path: &str, bytes: &[u8]) -> String {
    format!(
        r#"{{"mediaType": "application/vnd.oci.image.layer.v1.tar", "digest": "sha256:{}", "size": {},
            "annotations": {{"org.opencontainers.image.title": "{path}"}}}}"#,
        Sha256::of(bytes),
        bytes.len()
    )
}

/// An artifact manifest of `layers`.
fn manifest_of(layers: &[String]) -> Vec<u8> {
    format!(
        r#"{{"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
            "artifactType": "application/vnd.turboembed.bundle",
            "config": {{"mediaType": "application/vnd.oci.empty.v1+json",
              "digest": "sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a", "size": 2}},
            "layers": [{}]}}"#,
        layers.join(", ")
    )
    .into_bytes()
}

/// What the stand-in holds, by digest, and each request it saw: the
/// target and whether it carried a token.
#[derive(Default)]
struct Registry {
    manifests: BTreeMap<String, Vec<u8>>,
    blobs: BTreeMap<String, Vec<u8>>,
    /// Where a manifest request is sent on, as a web host that is not a
    /// registry answers one.
    manifests_moved_to: Option<String>,
    /// A path that is no part of the registry's API and where it sends a
    /// request for it on.
    hops: BTreeMap<String, String>,
    /// The repository, when it is not [`NAME`].
    name: Option<&'static str>,
    seen: Mutex<Vec<(String, bool)>>,
}

impl Registry {
    /// The sealed bundle pushed as one artifact; its manifest's digest.
    fn with_bundle() -> (Registry, Sha256) {
        let mut r = Registry::default();
        let layers: Vec<String> = PATHS.iter().map(|p| layer(p, &bytes_of(p))).collect();
        for p in PATHS {
            r.push_blob(bytes_of(p));
        }
        let pin = r.push_manifest(manifest_of(&layers));
        (r, pin)
    }

    fn push_blob(&mut self, bytes: Vec<u8>) {
        self.blobs.insert(Sha256::of(&bytes).to_string(), bytes);
    }

    fn push_manifest(&mut self, bytes: Vec<u8>) -> Sha256 {
        let pin = Sha256::of(&bytes);
        self.manifests.insert(pin.to_string(), bytes);
        pin
    }

    /// The status, extra header lines and body for one request. Every
    /// call under `/v2/` needs the token `/token` gives; the weights are
    /// redirected to `/storage/<digest>`, which refuses a request that
    /// carries a token, as a registry's object store does.
    fn answer(&self, base: &str, target: &str, token: Option<&str>) -> (&'static str, String, Vec<u8>) {
        let name = self.name.unwrap_or(NAME);
        let ok = |body: Vec<u8>| ("200 OK", String::new(), body);
        let missing = ("404 Not Found", String::new(), b"{}".to_vec());
        if target.starts_with("/token?") {
            let asked =
                target.contains("service=stand-in") && target.contains(&format!("scope=repository:{name}:pull"));
            return if asked { ok(br#"{"token": "anonymous-pull"}"#.to_vec()) } else { missing };
        }
        if let Some(hex) = target.strip_prefix("/storage/") {
            return match (token, self.blobs.get(hex)) {
                (Some(_), _) => ("400 Bad Request", String::new(), b"a token reached storage".to_vec()),
                (None, Some(bytes)) => ok(bytes.clone()),
                (None, None) => missing,
            };
        }
        if let Some(to) = self.hops.get(target) {
            return ("302 Found", format!("Location: {to}\r\n"), Vec::new());
        }
        if token != Some("anonymous-pull") {
            let challenge = format!(
                "WWW-Authenticate: Bearer realm=\"{base}/token\",service=\"stand-in\",scope=\"repository:{name}:pull\"\r\n"
            );
            return ("401 Unauthorized", challenge, b"{}".to_vec());
        }
        let under = format!("/v2/{name}/");
        let Some(rest) = target.strip_prefix(under.as_str()) else { return missing };
        if let (Some(_), Some(to)) = (rest.strip_prefix("manifests/"), &self.manifests_moved_to) {
            return ("302 Found", format!("Location: {to}\r\n"), Vec::new());
        }
        if let Some(hex) = rest.strip_prefix("manifests/sha256:") {
            return self.manifests.get(hex).map_or(missing, |m| ok(m.clone()));
        }
        match rest.strip_prefix("blobs/sha256:") {
            Some(hex) if self.blobs.get(hex).is_some_and(|b| b.len() > 1_000_000) => {
                ("307 Temporary Redirect", format!("Location: /storage/{hex}\r\n"), Vec::new())
            }
            Some(hex) => self.blobs.get(hex).map_or(missing, |b| ok(b.clone())),
            None => missing,
        }
    }
}

/// Serves `registry` until the test ends, one request per connection;
/// a client for it, with `registry.example.org` standing for it.
fn serve(registry: Registry) -> (OciRegistry, Arc<Registry>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let registry = Arc::new(registry);
    let (held, at) = (registry.clone(), base.clone());
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let target = line.split(' ').nth(1).unwrap_or_default().to_owned();
            let mut token = None;
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim_end().is_empty() {
                    break;
                }
                if let Some(t) = h.trim_end().to_ascii_lowercase().strip_prefix("authorization: bearer ") {
                    token = Some(t.to_owned());
                }
            }
            held.seen.lock().unwrap().push((target.clone(), token.is_some()));
            let (status, extra, body) = held.answer(&at, &target, token.as_deref());
            let head =
                format!("HTTP/1.1 {status}\r\n{extra}Content-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&body).unwrap();
        }
    });
    (OciRegistry::with(HOST, &base).unwrap(), registry)
}

fn described() -> Described {
    Described {
        license_terms: "https://models.example.org/tiny-bert/LICENSE".parse().unwrap(),
        trained_on: Vec::new(),
        attribution: Some("Example attribution for tiny.".into()),
        origin: None,
        component: None,
        language: None,
    }
}

fn spec(pin: &Sha256) -> OciImport {
    OciImport {
        id: "tiny".parse().unwrap(),
        reference: format!("{HOST}/{NAME}@sha256:{pin}"),
        bundle: false,
        kind: Kind::Model,
        format: Format::Safetensors,
        version: None,
        license: Licence::Apache2,
        described: described(),
        keep: None,
    }
}

#[test]
fn the_imported_entry_is_the_one_written_by_hand_for_the_same_layers() {
    let (registry, pin) = Registry::with_bundle();
    let (client, registry) = serve(registry);
    let files: Vec<String> = PATHS
        .iter()
        .map(|p| {
            let b = bytes_of(p);
            let (sha256, sha512) = (Sha256::of(&b), Sha512::of(&b));
            format!(
                r#"{{"path": "{p}", "url": "https://{HOST}/v2/{NAME}/blobs/sha256:{sha256}", "size": {},
                    "sha256": "{sha256}", "sha512": "{sha512}"}}"#,
                b.len()
            )
        })
        .collect();
    let by_hand = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "tiny", "kind": "KIND_MODEL", "format": "FORMAT_SAFETENSORS",
          "version": "sha256:{pin}", "license": "Apache-2.0",
          "license_terms": "https://models.example.org/tiny-bert/LICENSE", "trained_on": [],
          "class": "CLASS_PERMISSIVE", "attribution": "Example attribution for tiny.", "files": [{}]}}]}}"#,
        files.join(", ")
    );
    let by_hand = Index::parse(by_hand.as_bytes()).unwrap();
    let imported = oci(&client, spec(&pin)).unwrap();
    assert_eq!(&imported, by_hand.get("tiny").unwrap());

    // One challenge and one token for the repository; the token went to
    // the registry on every later call and never to its storage.
    let seen = registry.seen.lock().unwrap().clone();
    assert_eq!(seen[0], (format!("/v2/{NAME}/manifests/sha256:{pin}"), false));
    assert!(seen[1].0.starts_with("/token?") && !seen[1].1, "{seen:?}");
    assert_eq!(seen.iter().filter(|(t, _)| t.starts_with("/token?")).count(), 1, "{seen:?}");
    assert!(seen[2..].iter().all(|(t, token)| *token == t.starts_with("/v2/")), "{seen:?}");
    assert_eq!(seen.iter().filter(|(t, _)| t.starts_with("/storage/")).count(), 1, "{seen:?}");

    // The zoo then fetches the entry through the same client.
    let cache = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-import-oci-{}", std::process::id())));
    let zoo = Zoo::new(by_hand, cache.to_path_buf(), Policy::default(), &client);
    let user: UserId = "alice".parse().unwrap();
    let fetched = zoo.fetch("tiny", &user, &Acceptances::default()).unwrap();
    for (local, path) in fetched.files.iter().zip(PATHS) {
        assert_eq!(fs::read(local).unwrap(), bytes_of(path), "{path}");
    }
}

#[test]
fn layers_that_are_a_sealed_bundle_import_as_an_encoder_bundle() {
    let (registry, pin) = Registry::with_bundle();
    let (client, _) = serve(registry);
    let mut as_bundle = spec(&pin);
    as_bundle.bundle = true;
    let imported = oci(&client, as_bundle).unwrap();
    assert_eq!((imported.model_type, imported.format), (ModelType::EncoderBundle, Format::Bundle));
    // The sealed directory's own entry, but for each file's address.
    let from_dir = BundleImport {
        id: "tiny".parse().unwrap(),
        base: "https://models.example.org/tiny-bert".parse().unwrap(),
        described: described(),
    };
    let mut expected = bundle(&sealed(), from_dir).unwrap();
    for f in &mut expected.files {
        f.url = format!("https://{HOST}/v2/{NAME}/blobs/sha256:{}", f.sha256).parse().unwrap();
    }
    assert_eq!(imported, expected);
}

#[test]
fn an_import_that_keeps_its_files_downloads_each_once() {
    let (registry, pin) = Registry::with_bundle();
    let (client, registry) = serve(registry);
    let blobs = || registry.seen.lock().unwrap().iter().filter(|(t, _)| t.contains("/blobs/")).count();
    let cache = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-import-oci-keep-{}", std::process::id())));
    let mut keeping = spec(&pin);
    keeping.keep = Some(cache.to_path_buf());
    // An import whose entry is refused, every layer read by then, keeps none.
    let mut refused = keeping.clone();
    (refused.bundle, refused.license) = (true, Licence::Mit);
    assert!(oci(&client, refused).unwrap_err().to_string().contains("the index says MIT"));
    assert!(!cache.join("tiny").exists());
    let entry = oci(&client, keeping.clone()).unwrap();
    assert_eq!(entry, oci(&client, spec(&pin)).unwrap(), "keeping changes nothing of the entry");
    for p in PATHS {
        assert_eq!(fs::read(cache.join("tiny").join(p)).unwrap(), bytes_of(p), "{p}");
    }

    // The same import again asks for the manifest and no blob.
    let asked = blobs();
    assert_eq!(oci(&client, keeping.clone()).unwrap(), entry);
    assert_eq!(blobs(), asked);

    // A fetch of the entry finds every file, with no source to ask.
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&entry)).as_bytes()).unwrap();
    let zoo = Zoo::new(index, cache.to_path_buf(), Policy::default(), &Offline);
    let user: UserId = "alice".parse().unwrap();
    assert_eq!(zoo.fetch("tiny", &user, &Acceptances::default()).unwrap().files.len(), PATHS.len());

    // A kept file that holds other bytes is read again and put right.
    let tokenizer = cache.join("tiny").join("tokenizer.json");
    let mut other = bytes_of("tokenizer.json");
    other[0] ^= 1;
    fs::write(&tokenizer, &other).unwrap();
    assert_eq!(oci(&client, keeping).unwrap(), entry);
    assert_eq!(blobs(), asked + 1);
    assert_eq!(fs::read(&tokenizer).unwrap(), bytes_of("tokenizer.json"));
}

#[test]
fn a_reference_that_may_move_is_refused_before_anything_is_asked() {
    let (registry, pin) = Registry::with_bundle();
    let (client, registry) = serve(registry);
    for reference in [format!("{HOST}/{NAME}:latest"), format!("{HOST}/{NAME}"), format!("{NAME}@sha256:{pin}")] {
        let mut moving = spec(&pin);
        moving.reference = reference.clone();
        assert!(matches!(oci(&client, moving), Err(ZooError::Import { .. })), "{reference}");
    }
    assert!(registry.seen.lock().unwrap().is_empty());
    // A digest the registry does not hold.
    let absent = Sha256::of(b"no such manifest");
    assert!(matches!(oci(&client, spec(&absent)), Err(ZooError::Transport { .. })));
}

#[test]
fn a_manifest_request_sent_on_to_another_host_is_reported_as_a_redirect() {
    // The page the request is sent on to is served, and is not read.
    let (mut registry, pin) = Registry::with_bundle();
    registry.manifests_moved_to = Some("https://www.example.org/".to_owned());
    let (client, registry) = serve(registry);
    let e = oci(&client, spec(&pin)).unwrap_err();
    assert!(matches!(e, ZooError::Transport { .. }), "{e:?}");
    let e = e.to_string();
    assert!(e.contains("redirected to https://www.example.org/, which is not the registry"), "{e}");
    assert!(!e.contains("SHA-256"), "{e}");
    assert!(registry.seen.lock().unwrap().iter().all(|(t, _)| !t.contains("/blobs/")));

    // A hop that stays on the registry and a second one that leaves it.
    let (mut registry, pin) = Registry::with_bundle();
    registry.manifests_moved_to = Some("/moved/manifest".to_owned());
    registry.hops.insert("/moved/manifest".to_owned(), "https://www.example.org/page".to_owned());
    let (client, registry) = serve(registry);
    let e = oci(&client, spec(&pin)).unwrap_err().to_string();
    assert!(e.contains("redirected to https://www.example.org/page, which is not the registry"), "{e}");
    let seen: Vec<String> = registry.seen.lock().unwrap().iter().map(|(t, _)| t.clone()).collect();
    assert_eq!(seen.last().map(String::as_str), Some("/moved/manifest"), "{seen:?}");
}

#[test]
fn a_repository_named_like_an_api_resource_is_read_as_any_other() {
    // `manifests` and `blobs` as segments of the name: the weights' blob
    // is still redirected to storage and followed there.
    let name = "models/manifests/blobs/tiny";
    let (mut registry, pin) = Registry::with_bundle();
    registry.name = Some(name);
    let (client, registry) = serve(registry);
    let mut named = spec(&pin);
    named.reference = format!("{HOST}/{name}@sha256:{pin}");
    let entry = oci(&client, named).unwrap();
    assert_eq!(entry.files.len(), PATHS.len());
    assert!(entry.files.iter().all(|f| f.url.as_str().starts_with(&format!("https://{HOST}/v2/{name}/blobs/sha256:"))));
    let seen = registry.seen.lock().unwrap();
    assert_eq!(seen.iter().filter(|(t, _)| t.starts_with("/storage/")).count(), 1, "{seen:?}");
    assert_eq!(seen.iter().filter(|(t, _)| t.starts_with("/token?")).count(), 1, "{seen:?}");
}

#[test]
fn bytes_that_are_not_the_ones_a_digest_names_are_refused() {
    // A manifest served under another digest.
    let (mut registry, pin) = Registry::with_bundle();
    let other = Sha256::of(b"another manifest");
    let served = registry.manifests[&pin.to_string()].clone();
    registry.manifests.insert(other.to_string(), served);
    let (client, _) = serve(registry);
    let e = oci(&client, spec(&other)).unwrap_err();
    assert!(matches!(e, ZooError::Transport { .. }) && e.to_string().contains("whose SHA-256 is"), "{e}");

    // A blob with a changed byte, and one cut short, under the digest
    // the manifest gives.
    for (path, check, cut) in [("tokenizer.json", Check::Sha256, false), ("manifest.json", Check::Size, true)] {
        let (mut registry, pin) = Registry::with_bundle();
        let mut bytes = bytes_of(path);
        let hex = Sha256::of(&bytes).to_string();
        bytes[0] ^= 1;
        if cut {
            bytes.pop();
        }
        registry.blobs.insert(hex, bytes);
        let (client, _) = serve(registry);
        match oci(&client, spec(&pin)) {
            Err(ZooError::Integrity { path: p, check: c, .. }) => assert_eq!((p.as_str(), c), (path, check)),
            other => panic!("{path}: {other:?}"),
        }
    }

    // A blob longer than its layer says is not read past that size.
    let (mut registry, pin) = Registry::with_bundle();
    let mut bytes = bytes_of("tokenizer.json");
    let hex = Sha256::of(&bytes).to_string();
    bytes.extend_from_slice(b"more");
    registry.blobs.insert(hex, bytes);
    let (client, _) = serve(registry);
    assert!(matches!(oci(&client, spec(&pin)), Err(ZooError::Transport { .. })));
}

#[test]
fn a_manifest_the_importer_cannot_take_is_refused() {
    let file = layer("tokenizer.json", &bytes_of("tokenizer.json"));
    let untitled = file.replace("org.opencontainers.image.title", "org.example.other");
    let sha512 = file.replace("sha256:", "sha512:");
    let packed = file.replace(r#""annotations": {"#, r#""annotations": {"io.deis.oras.content.unpack": "true", "#);
    let escaping = file.replace(r#""tokenizer.json""#, r#""../tokenizer.json""#);
    for (manifest, why) in [
        (manifest_of(&[untitled]), "layers[0] names no file"),
        (manifest_of(&[packed]), "is an archive of a directory, not a file"),
        (manifest_of(&[sha512]), "not a SHA-256 one"),
        (manifest_of(&[escaping]), "names a file the zoo does not take"),
        (manifest_of(&[]), "lists no layers"),
        (br#"{"schemaVersion": 2, "manifests": []}"#.to_vec(), "names an image index"),
        (b"not json".to_vec(), "is not JSON"),
        (manifest_of(&[file.clone(), file.clone()]), "tokenizer.json listed twice"),
    ] {
        let mut registry = Registry::default();
        registry.push_blob(bytes_of("tokenizer.json"));
        let pin = registry.push_manifest(manifest);
        let (client, _) = serve(registry);
        let e = oci(&client, spec(&pin)).unwrap_err().to_string();
        assert!(e.contains(why), "{why}: {e}");
    }
    // Layers that are not a bundle, asked for as one.
    let mut registry = Registry::default();
    registry.push_blob(bytes_of("tokenizer.json"));
    let pin = registry.push_manifest(manifest_of(&[file]));
    let (client, _) = serve(registry);
    let mut as_bundle = spec(&pin);
    as_bundle.bundle = true;
    let e = oci(&client, as_bundle).unwrap_err().to_string();
    assert!(e.contains("a bundle needs manifest.json among its layers"), "{e}");
}

/// The cache a network test keeps its downloads in: the directory
/// `TURBO_TEST_ZOO_CACHE` names, which is left in place so the next run
/// downloads nothing it holds, or else a directory of the test's own.
fn network_cache(name: &str) -> (PathBuf, Option<Scratch>) {
    match std::env::var_os("TURBO_TEST_ZOO_CACHE") {
        Some(dir) => (PathBuf::from(dir), None),
        None => {
            let own = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-{name}-{}", std::process::id())));
            (own.to_path_buf(), Some(own))
        }
    }
}

/// Imports the artifact `TURBO_TEST_OCI_REFERENCE` names
/// (`<registry>/<name>@sha256:<digest>`, each layer a file with its
/// title) from its real registry, with an anonymous token where the
/// registry asks for one, and fetches the entry back through the zoo:
/// `TURBO_TEST_OCI_REFERENCE=... cargo test -p turbo-zoo --test import_oci -- --ignored`.
/// Needs the network. With `TURBO_TEST_ZOO_CACHE=<dir>` the files are
/// kept there and a later run downloads none of them again.
#[test]
#[ignore]
fn a_real_artifact_imports_and_fetches_from_its_registry() {
    let reference = std::env::var("TURBO_TEST_OCI_REFERENCE")
        .expect("set TURBO_TEST_OCI_REFERENCE to <registry>/<name>@sha256:<digest>");
    let client = OciRegistry::new();
    let mut real = spec(&Sha256::of(b""));
    real.reference = reference;
    // The import keeps what it reads, so the fetch below downloads nothing.
    let (cache, _own) = network_cache("import-oci-real");
    real.keep = Some(cache.clone());
    let entry = oci(&client, real).unwrap();
    assert!(!entry.files.is_empty());
    let index = Index::parse(Index::to_canonical_json(std::slice::from_ref(&entry)).as_bytes()).unwrap();
    let zoo = Zoo::new(index, cache, Policy::default(), &Offline);
    let user: UserId = "turbo-zoo-test".parse().unwrap();
    let fetched = zoo.fetch("tiny", &user, &Acceptances::default()).unwrap();
    for (local, file) in fetched.files.iter().zip(&entry.files) {
        let bytes = fs::read(local).unwrap();
        assert_eq!((bytes.len() as u64, Sha256::of(&bytes)), (file.size, file.sha256), "{}", file.path);
        println!("{}\t{}\t{}", file.path, file.size, file.url);
    }
}
