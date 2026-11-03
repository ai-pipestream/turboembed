//! The checked corpus fetch-and-pin recipes of `analysis/formats/recipes`
//! (#1533): the set parses and is deterministically ordered, every pin is
//! structurally valid, and, when the environment names a cache directory,
//! the cached artifacts hash to the pinned SHA-256s. Nothing here needs
//! the network; the hash check reads only files a cache directory already
//! holds.

#[path = "../src/recipes.rs"]
mod recipes;

use std::path::{Path, PathBuf};

use recipes::{Kind, Recipe, Tier, parse_dir};

/// The checked-in recipes, next to the crate manifest.
const RECIPES_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/recipes");

/// The corpora the issue names.
const CORPORA: [&str; 13] = [
    "fineweb-2-sample",
    "flores-plus",
    "germeval-2014",
    "litbank",
    "masc-3.0.0",
    "multiconer-v2",
    "tatoeba-2026-10-10",
    "ud-catalan-ancora",
    "ud-english-ewt",
    "ud-finnish-ftb",
    "ud-italian-markit",
    "ud-spanish-ancora",
    "wnut-17",
];

fn checked_in() -> Vec<Recipe> {
    parse_dir(Path::new(RECIPES_DIR)).unwrap_or_else(|e| panic!("{RECIPES_DIR}: {e}"))
}

#[test]
fn the_checked_in_recipes_parse() {
    let recipes = checked_in();
    let names: Vec<_> = recipes.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, CORPORA, "the set is exactly the issue's corpora, sorted");
}

#[test]
fn every_pin_is_structurally_valid() {
    for recipe in checked_in() {
        assert_eq!(recipe.sha256.len(), 64, "{}: 64 hex digits", recipe.name);
        assert!(recipe.sha256.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        for (field, url) in [("url", &recipe.url), ("license-url", &recipe.license_url)] {
            assert!(url.starts_with("https://"), "{}: {field} is https", recipe.name);
            assert!(recipe.license.is_some(), "{}: names the licence it read", recipe.name);
        }
    }
}

#[test]
fn tiers_follow_the_licence_classes() {
    for recipe in checked_in() {
        let expected = match recipe.name.as_str() {
            "ud-english-ewt" => Tier::AcceptFirst,
            "wnut-17" | "flores-plus" => Tier::EvalOnly,
            _ => Tier::Default,
        };
        assert_eq!(recipe.tier, expected, "{} tier", recipe.name);
    }
}

#[test]
fn wnut_17_stays_out_of_the_default_tier() {
    let recipes = checked_in();
    let wnut = recipes.iter().find(|r| r.name == "wnut-17").expect("the wnut-17 recipe");
    assert_eq!(wnut.tier, Tier::EvalOnly);
    assert!(
        wnut.notes.iter().any(|n| n.contains("Blocker") && n.contains("platform terms")),
        "the platform-terms blocker is recorded in the recipe"
    );
}

#[test]
fn ordering_is_deterministic_across_reads() {
    let a = checked_in();
    let b = checked_in();
    assert_eq!(a, b);
    let mut sorted = a.clone();
    sorted.sort_by(|x, y| x.name.cmp(&y.name));
    assert_eq!(a, sorted, "sorted by name");
    let mut names: Vec<_> = a.iter().map(|r| r.name.clone()).collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), a.len(), "no two recipes share a name");
}

#[test]
fn rejects_two_recipes_with_one_name() {
    let dir = std::env::temp_dir().join(format!("recipes-dup-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    let one = checked_in().pop().expect("one recipe").to_string();
    std::fs::write(dir.join("a.recipe"), &one).unwrap();
    std::fs::write(dir.join("b.recipe"), one).unwrap();
    assert!(matches!(parse_dir(&dir).unwrap_err().kind, Kind::DuplicateName(_)));
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn ignores_files_that_are_not_recipes() {
    let dir = std::env::temp_dir().join(format!("recipes-mix-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir(&dir).unwrap();
    std::fs::write(dir.join("README.txt"), "not a recipe").unwrap();
    std::fs::write(dir.join("one.recipe"), checked_in().pop().expect("one recipe").to_string()).unwrap();
    assert_eq!(parse_dir(&dir).unwrap().len(), 1);
    std::fs::remove_dir_all(&dir).unwrap();
}

// The SHA-256 the hash check below compares cached files against, as a
// self-contained implementation: FIPS 180-4, eight rounds unrolled per
// 64-byte block.

const K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98,
    0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8,
    0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819,
    0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
    0xc67178f2,
];

fn sha256_hex(bytes: &[u8]) -> String {
    let mut h: [u32; 8] =
        [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    let mut msg = bytes.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());
    let (blocks, rest) = msg.as_chunks::<64>();
    debug_assert!(rest.is_empty());
    for block in blocks {
        let mut w = [0u32; 64];
        for (i, word) in w[..16].iter_mut().enumerate() {
            *word = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

#[test]
fn sha256_matches_the_fips_vectors() {
    assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
    let million_a = vec![b'a'; 1_000_000];
    assert_eq!(sha256_hex(&million_a), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

/// When `TURBO_RECIPES_CACHE` names a directory, every recipe whose
/// artifact the directory holds under the recipe's name is hashed and
/// compared with the pin. A directory with no matching files, or an
/// environment without the variable, checks nothing; nothing is fetched.
#[test]
fn cached_artifacts_hash_to_the_pins() {
    let Some(cache) = std::env::var_os("TURBO_RECIPES_CACHE").map(PathBuf::from) else { return };
    if !cache.is_dir() {
        return;
    }
    let mut checked = 0;
    for recipe in checked_in() {
        let file = cache.join(&recipe.name);
        if !file.is_file() {
            continue;
        }
        let bytes = std::fs::read(&file).unwrap();
        assert_eq!(sha256_hex(&bytes), recipe.sha256, "{} at {}", recipe.name, file.display());
        checked += 1;
    }
    eprintln!("TURBO_RECIPES_CACHE: compared {checked} cached artifact(s) against the pins");
}
