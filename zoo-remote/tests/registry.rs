//! The kind registry (#1744): a user's holders open into a source list
//! through it, a kind the caller adds opens like a built-in one, and an
//! unknown scheme or a sign-in a kind does not take is refused.

use std::path::PathBuf;

use turbo_zoo::{Acceptances, Blobs, Holder, Holders, Index, Offline, Policy, Sha256, Sha512, SignIn, UserId, Zoo};
use turbo_zoo_remote::{Kind, Opened, Registry};

const TEXT: &[u8] = b"hello hello hello\n";

fn index() -> Index {
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{{"id": "hello", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT",
          "version": "1", "license": "Apache-2.0", "license_terms": "https://example.org/LICENSE", "trained_on": [],
          "class": "CLASS_PERMISSIVE", "attribution": "Example.",
          "files": [{{"path": "hello.txt", "url": "https://example.org/hello.txt", "size": {}, "sha256": "{}",
            "sha512": "{}"}}]}}]}}"#,
        TEXT.len(),
        Sha256::of(TEXT),
        Sha512::of(TEXT),
    );
    Index::parse(json.as_bytes()).unwrap()
}

fn temp(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-registry-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn alice() -> UserId {
    "alice".parse().unwrap()
}

/// A kind of the caller's own: `team://<name>` is the store a team keeps
/// under one directory, by name.
struct Team(PathBuf);

impl Kind for Team {
    fn schemes(&self) -> &[&str] {
        &["team"]
    }

    fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
        let name = holder.address.strip_prefix("team://").unwrap_or_default();
        if name.is_empty() || name.contains('/') {
            return Err(format!("{}: a team is one name", holder.address));
        }
        Ok(Opened::Source(Box::new(Blobs::new(self.0.join(name)))))
    }
}

#[test]
fn a_users_holders_open_in_order_and_the_first_store_keeps() {
    let root = temp("order");
    Blobs::new(root.join("teams/nlp")).put(TEXT).unwrap();
    let mut registry = Registry::builtin();
    registry.register(Team(root.join("teams")));
    let mut holders = Holders::default();
    let local = format!("file://{}", root.join("local").display());
    holders.register(&alice(), Holder::new("mine", &local, SignIn::Environment).unwrap()).unwrap();
    holders.register(&alice(), Holder::new("team", "team://nlp", SignIn::Environment).unwrap()).unwrap();
    let path = root.join("holders.json");
    holders.save(&path).unwrap();

    let list = Holders::load(&path).unwrap();
    let sources = registry.sources(list.list(&alice()), &alice(), Some(Offline)).unwrap();
    assert_eq!(sources.len(), 3, "both holders, then origin");
    let zoo = Zoo::new(index(), root.join("cache"), Policy::default(), &sources);
    let got = zoo.fetch("hello", &alice(), &Acceptances::default()).map(|f| std::fs::read(&f.files[0]).unwrap());
    assert_eq!(got.unwrap(), TEXT);
    // The team's file was kept in the user's own store as it was read.
    assert_eq!(std::fs::read(Blobs::new(root.join("local")).path(&Sha256::of(TEXT))).unwrap(), TEXT);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn an_unknown_scheme_names_the_ones_the_registry_knows() {
    let mut registry = Registry::builtin();
    let holder = Holder::new("t", "team://nlp", SignIn::Environment).unwrap();
    let e = registry.open(&holder, &alice()).err().unwrap();
    assert!(e.contains("a source is file://, grpc://, grpcs://, s3://, gdrive:// or origin"), "{e}");
    registry.register(Team(temp("unknown")));
    assert_eq!(registry.schemes().collect::<Vec<_>>(), ["file", "grpc", "grpcs", "s3", "gdrive", "team"]);
    assert!(registry.open(&holder, &alice()).is_ok());
    let e = registry.spec("team://a/b", &alice(), Offline).err().unwrap();
    assert!(e.contains("a team is one name"), "{e}");
    assert_eq!(registry.spec("team://a, origin", &alice(), Offline).unwrap().len(), 2);
    let e = Registry::empty().open(&holder, &alice()).err().unwrap();
    assert!(e.contains("no kind of source is registered"), "{e}");
}

#[test]
fn a_kind_registered_later_takes_its_scheme_over() {
    let root = temp("over");
    Blobs::new(root.join("file")).put(TEXT).unwrap();
    struct Elsewhere;
    impl Kind for Elsewhere {
        fn schemes(&self) -> &[&str] {
            &["file"]
        }
        fn open(&self, holder: &Holder, _user: &UserId) -> Result<Opened, String> {
            Err(format!("{}: not here", holder.address))
        }
    }
    let mut registry = Registry::builtin();
    registry.register(Elsewhere);
    let e = registry.spec("file:///zoo", &alice(), Offline).err().unwrap();
    assert!(e.contains("not here"), "{e}");
    assert_eq!(registry.schemes().filter(|s| *s == "file").count(), 1);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_sign_in_a_kind_does_not_take_is_refused() {
    let registry = Registry::builtin();
    let key = || SignIn::ApiKey { key: "k".into() };
    for address in ["file:///zoo", "grpc://zoo.example:8081", "s3://models/zoo"] {
        let e = registry.open(&Holder::new("h", address, key()).unwrap(), &alice()).err().unwrap();
        assert!(e.contains("environment only") || e.contains("needs the s3 feature"), "{address}: {e}");
    }
    let grant = SignIn::GoogleOauth { client_id: "c".into(), client_secret: None, refresh_token: "r".into() };
    let bad = Holder::new("d", "gdrive://a.b", grant.clone()).unwrap();
    let e = registry.open(&bad, &alice()).err().unwrap();
    if cfg!(feature = "gdrive") {
        assert!(e.contains("a folder id is"), "{e}");
        // Opening asks Google for nothing: the token is fetched on first use.
        let drive = Holder::new("d", "gdrive://root/models", grant).unwrap();
        assert!(registry.open(&drive, &alice()).is_ok());
        let keyed = Holder::new("d", "gdrive://root", key()).unwrap();
        assert!(registry.open(&keyed, &alice()).is_ok());
    } else {
        assert!(e.contains("needs the gdrive feature"), "{e}");
    }
    let named = Holders::default();
    let opened = registry.sources(named.list(&alice()), &alice(), None::<Offline>).unwrap();
    assert!(opened.is_empty());
}
