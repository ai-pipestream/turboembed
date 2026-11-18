//! SuggestPrefix and LookupStemSurfaces on a warm cache (#2341): checking
//! a lexicon image reads every table, so it is done once when the image
//! is mapped, and a lookup after that walks the trie alone. The check
//! count is the calling thread's, so other tests do not move it.

use turbo_suggest::SurfaceLexiconBuilder;
use turbo_suggest::lexicon::image_checks_on_this_thread as checks;

use super::SuggestCache;
use crate::proto::vocabulary::SuggestLexiconArtifactDescriptor;
use crate::vocabulary::Store;

/// Stores a lexicon of `words`, each `(surface, stem, count)`, under `id`.
fn put(store: &Store, id: &str, words: &[(&str, &str, u64)]) {
    let mut builder = SurfaceLexiconBuilder::new();
    for &(surface, stem, count) in words {
        builder.observe(surface, stem, count);
    }
    let bytes = builder.build_bytes().unwrap();
    let desc = SuggestLexiconArtifactDescriptor {
        artifact_id: id.to_owned(),
        language: "en".into(),
        entry_count: words.len() as u64,
        stem_count: 0,
        byte_size: bytes.len() as u64,
        sha256: turbo_maxent::sha256(&bytes).to_hex(),
    };
    store.put_suggest(desc, &bytes).unwrap();
}

fn store(name: &str) -> (turbo_scratch::Scratch, Store) {
    let dir = turbo_scratch::Scratch::at(std::env::temp_dir().join(format!("{name}-{}", std::process::id())));
    std::fs::create_dir_all(&*dir).unwrap();
    let store = Store::open(&dir).unwrap();
    (dir, store)
}

#[test]
fn a_warm_lookup_does_not_check_the_image_again() {
    let (_dir, store) = store("suggest-warm-lookup");
    put(&store, "lex", &[("running", "run", 9), ("runner", "run", 4), ("rust", "rust", 2)]);
    let cache = SuggestCache::default();

    let before = checks();
    let cold = cache.complete(&store, "lex", "run", 10).unwrap();
    assert_eq!(checks() - before, 1, "mapping the image checks it once");
    let surfaces: Vec<&str> = cold.iter().map(|c| c.surface.as_str()).collect();
    assert_eq!(surfaces, ["running", "runner"]);

    let warm = checks();
    for _ in 0..3 {
        assert_eq!(cache.complete(&store, "lex", "run", 10).unwrap(), cold);
        let stems = cache.surfaces(&store, "lex", "run", 10).unwrap();
        assert_eq!(
            stems.iter().map(|s| (s.surface.as_str(), s.count)).collect::<Vec<_>>(),
            [("running", 9), ("runner", 4)]
        );
    }
    assert_eq!(checks() - warm, 0, "a warm lookup checked the whole image again");
}

#[test]
fn a_replaced_lexicon_is_checked_and_served_anew() {
    let (_dir, store) = store("suggest-warm-replace");
    put(&store, "lex", &[("running", "run", 9)]);
    let cache = SuggestCache::default();
    assert_eq!(cache.complete(&store, "lex", "ru", 10).unwrap().len(), 1);

    assert!(store.delete("lex").unwrap());
    put(&store, "lex", &[("rubber", "rubber", 3), ("ruby", "rubi", 7)]);
    let before = checks();
    let got = cache.complete(&store, "lex", "ru", 10).unwrap();
    assert_eq!(got.iter().map(|c| c.surface.as_str()).collect::<Vec<_>>(), ["ruby", "rubber"]);
    assert_eq!(checks() - before, 1, "the new image is checked once");
    assert!(cache.complete(&store, "lex", "zz", 10).unwrap().is_empty());
    assert_eq!(checks() - before, 1);
}
