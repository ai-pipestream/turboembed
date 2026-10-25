//! Origin, URL, cache integrity, symlink and strict index checks for `tests/zoo.rs`.

use super::*;

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

/// A symlink left at a temporary name is never written through and never
/// renamed onto the file: each write makes a name of its own.
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
    assert!(fs::symlink_metadata(cache.join("permissive").join("permissive.bin.partial")).unwrap().is_symlink());
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

#[test]
fn attribution_is_required_when_a_licence_asks_for_it() {
    use Licence::*;
    use LicenceClass::*;
    let without_attribution = |e: String| e.replace("      \"attribution\": \"Example attribution for m.\",\n", "");
    // Licences that ask for attribution (on the artifact or on its training data) reject an absent attribution.
    for (lic, trained, class) in [
        (Apache2, &[][..], Permissive),
        (Mit, &[][..], Permissive),
        (CcBy40, &[][..], Permissive),
        (Cc0, &[CcBy40][..], Permissive),
        (CcBySa40, &[][..], ShareAlike),
        (Lgpl30Only, &[][..], Lgpl),
        (Gpl30Only, &[][..], Gpl),
    ] {
        let missing = without_attribution(entry("m", lic, trained, class, BYTES));
        match Index::parse(index_of(&[missing]).as_bytes()) {
            Err(ZooError::Index(msg)) => assert!(msg.contains("requires attribution"), "{lic}: {msg}"),
            other => panic!("{lic}: {other:?}"),
        }
    }
    // Empty, whitespace-only, or control-character attribution is rejected even when optional.
    for bad_attr in ["", "   ", "Example\\nattribution."] {
        let e = entry("m", Cc0, &[], Permissive, BYTES).replace("Example attribution for m.", bad_attr);
        assert!(matches!(Index::parse(index_of(&[e]).as_bytes()), Err(ZooError::Index(_))), "{bad_attr:?}");
    }
    // Public-domain / dedication / restricted-public entries may omit attribution.
    for (lic, trained, class, add_origin) in [
        (Cc0, &[][..], Permissive, false),
        (Pddl10, &[Cc0][..], Permissive, false),
        (PublicDomain, &[][..], Permissive, false),
        (RestrictedPublic, &[Cc0][..], LocalOnly, true),
    ] {
        let mut e = without_attribution(entry("m", lic, trained, class, BYTES));
        if add_origin {
            e = with_origin(e);
        }
        let parsed = Index::parse(index_of(&[e]).as_bytes()).unwrap();
        assert_eq!(parsed.get("m").unwrap().attribution, None);
    }
}
