//! The Hugging Face copies of the permissive `turbo-*` models (#2192):
//! each file's second location, at a pinned commit of the model's own
//! repository.

use turbo_zoo::{Index, LicenceClass, Transport, Url};

/// Each artifact published on Hugging Face and the commit of its
/// repository, `pipestream-ai/<id>`, that holds the indexed files.
const COPIES: [(&str, &str); 8] = [
    ("turbo-en-masc-3.0.0-modernbert-base-polarity-head-1", "c1be62ddcbf4d10ff5c6882d0a699d09f3d8f537"),
    ("turbo-en-goemotions-modernbert-base-head-1", "d5c044091ac2a655404f34a6143f35c6f0873ee3"),
    ("turbo-en-civil-comments-modernbert-base-head-1", "e5442f56eb13771d5e466469d91efaec5dd1f460"),
    ("turbo-en-civil-comments-modernbert-base-context-head-1", "d889545c3117fab305801f819c973acc30413fbd"),
    ("turbo-en-wnut17-modernbert-base-l12-open-span-1", "12bf5f641b2af47d4dcee403962292ef77a94e3a"),
    ("turbo-en-litbank-modernbert-base-coref-2", "d8def155bd40e5b14c138d4ed504e6914c78cc16"),
    ("turbo-en-masc-pos-modernbert-base-2", "2a47799b816e1a22d58b210a6040b78e7006d684"),
    ("turbo-en-masc-pos-modernbert-base-bilstm256x2-1", "68c03ef3ca2731c282a8e51c64de0e21172afa8e"),
];

#[test]
fn each_published_model_lists_its_hugging_face_copy_at_a_pinned_commit() {
    let index = Index::builtin();
    for (id, commit) in COPIES {
        assert!(commit.len() == 40 && commit.bytes().all(|c| c.is_ascii_hexdigit()), "{id}");
        let a = index.get(id).unwrap();
        assert_eq!(a.class, LicenceClass::Permissive, "{id}");
        for f in &a.files {
            let copy = format!("https://huggingface.co/pipestream-ai/{id}/resolve/{commit}/{}", f.path.as_str());
            assert_eq!(f.also.iter().map(Url::as_str).collect::<Vec<_>>(), [copy.as_str()], "{id} {}", f.path);
            assert!(f.url.as_str().starts_with("https://github.com/ai-pipestream/turboembed/releases/download/"));
        }
    }
}

#[test]
fn no_other_artifact_lists_a_further_location() {
    for a in Index::builtin().list() {
        let listed = a.files.iter().any(|f| !f.also.is_empty());
        assert_eq!(listed, COPIES.iter().any(|(id, _)| *id == a.id.as_str()), "{}", a.id);
    }
}

/// HTTPS to Hugging Face only: every other address is refused before a
/// connection is made, as a blocked host would be.
#[cfg(feature = "http")]
struct HuggingFaceOnly(turbo_zoo::Http);

#[cfg(feature = "http")]
impl Transport for HuggingFaceOnly {
    fn get(&self, url: &Url, size: u64) -> Result<Vec<u8>, String> {
        if !url.as_str().starts_with("https://huggingface.co/") {
            return Err("blocked: only huggingface.co is reachable in this test".into());
        }
        self.0.get(url, size)
    }
}

/// Installs each published model from Hugging Face alone, the release
/// assets' host blocked: `cargo test -p turbo-zoo -- --ignored`. Needs
/// the network.
#[test]
#[ignore]
#[cfg(feature = "http")]
fn each_published_model_installs_from_hugging_face_alone() {
    let cache = std::env::temp_dir().join(format!("turbo-zoo-hugging-face-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&cache);
    let user: turbo_zoo::UserId = "turbo-zoo-test".parse().unwrap();
    let transport = HuggingFaceOnly(turbo_zoo::Http::new());
    let zoo = turbo_zoo::Zoo::new(Index::builtin(), cache.clone(), turbo_zoo::Policy::default(), &transport);
    for (id, _) in COPIES {
        let f = zoo.fetch(id, &user, &turbo_zoo::Acceptances::default()).unwrap();
        for (path, file) in f.files.iter().zip(&f.artifact.files) {
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(turbo_zoo::Sha256::of(&bytes), file.sha256, "{id} {}", file.path);
            assert_eq!(turbo_zoo::Sha512::of(&bytes), file.sha512, "{id} {}", file.path);
        }
    }
    std::fs::remove_dir_all(&cache).unwrap();
}
