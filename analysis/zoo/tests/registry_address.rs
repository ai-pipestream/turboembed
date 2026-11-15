//! An entry whose file is at a registry's blob address, as `import-oci`
//! writes one, fetched through what `turbo-zoo fetch` and a zoo server
//! read an index's addresses with ([`Sources::https`]). The test reads a
//! real registry, so it is ignored.
#![cfg(feature = "http")]

use std::fs;

use turbo_scratch::Scratch;
use turbo_zoo::{Acceptances, Http, Index, Policy, Sha256, Sources, UserId, Zoo};

/// The published POS model, kept on Docker Hub as an artifact whose
/// layers are its files.
const ID: &str = "turbo-en-masc-pos-modernbert-base-2";
const REPOSITORY: &str = "https://registry-1.docker.io/v2/pipestreamai/turbo-en-masc-pos-modernbert-base-2";

/// `cargo test -p turbo-zoo --test registry_address -- --ignored`. Needs
/// the network. Reads one small file of the model, 430,352 bytes.
#[test]
#[ignore]
fn a_file_at_a_registry_blob_address_is_fetched_with_the_registrys_token() {
    // The built-in entry with one file, its address the registry's blob.
    let mut entry = Index::builtin().get(ID).unwrap().clone();
    entry.files.retain(|f| f.path.as_str() == "reference/reference.safetensors");
    let file = &mut entry.files[0];
    file.url = format!("{REPOSITORY}/blobs/sha256:{}", file.sha256).parse().unwrap();
    file.also.clear();
    let (size, sha256) = (file.size, file.sha256);
    let index = || Index::parse(Index::to_canonical_json(std::slice::from_ref(&entry)).as_bytes()).unwrap();
    let user: UserId = "turbo-zoo-test".parse().unwrap();

    // Asked plainly, the registry wants a token and the fetch fails.
    let cache = Scratch::at(std::env::temp_dir().join(format!("turbo-zoo-registry-address-{}", std::process::id())));
    let plain = Http::new();
    let zoo = Zoo::new(index(), cache.to_path_buf(), Policy::default(), &plain);
    match zoo.fetch(ID, &user, &Acceptances::default()) {
        Err(e) => assert!(e.to_string().contains("401"), "{e}"),
        Ok(_) => panic!("a registry served a blob without its token"),
    }

    let sources = Sources::https();
    let zoo = Zoo::new(index(), cache.to_path_buf(), Policy::default(), &sources);
    let fetched = zoo.fetch(ID, &user, &Acceptances::default()).unwrap();
    let bytes = fs::read(&fetched.files[0]).unwrap();
    assert_eq!((bytes.len() as u64, Sha256::of(&bytes)), (size, sha256));
}
