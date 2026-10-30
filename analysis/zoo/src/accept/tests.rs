//! Tests of the acceptance file.

use super::*;

fn user(s: &str) -> UserId {
    s.parse().unwrap()
}

fn req(l: Licence) -> Requirement {
    Requirement::of(l).unwrap()
}

fn restricted(text: &str) -> Requirement {
    Requirement { licence: Licence::RestrictedPublic, text: text.to_owned() }
}

#[test]
fn restricted_terms_are_accepted_per_host() {
    let imdb = restricted("https://ai.stanford.edu/~amaas/data/sentiment/");
    let other = restricted("https://example.org/terms");
    let mut a = Acceptances::default();
    // There is no one text to accept for the licence as a whole.
    assert!(a.accept(&user("alice"), Licence::RestrictedPublic, 0).is_err());
    a.accept_terms(&user("alice"), &imdb, 7).unwrap();
    assert!(a.has(&user("alice"), &imdb));
    assert!(!a.has(&user("alice"), &other));
    assert!(!a.has(&user("bob"), &imdb));
    let back = Acceptances::parse(a.to_canonical_json().as_bytes()).unwrap();
    assert_eq!(back, a);
    assert!(a.revoke(&user("alice"), &imdb));
    assert!(!a.has(&user("alice"), &imdb));
    // The text must be an https address.
    assert!(a.accept_terms(&user("alice"), &restricted("not a url"), 0).is_err());
    assert!(a.accept_terms(&user("alice"), &restricted("http://example.org/terms"), 0).is_err());
    // A licence with its own text accepts that text only.
    let wrong = Requirement { licence: Licence::CcBySa40, text: "https://example.org/terms".into() };
    assert!(a.accept_terms(&user("alice"), &wrong, 0).is_err());
}

#[test]
fn accept_round_trips_through_the_file() {
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcBySa40, 1_792_000_000).unwrap();
    a.accept(&user("alice"), Licence::Lgpl30Only, 1_792_000_001).unwrap();
    a.accept(&user("bob"), Licence::CcBySa40, 1_792_000_002).unwrap();
    // A second acceptance keeps the first time.
    a.accept(&user("alice"), Licence::CcBySa40, 1_800_000_000).unwrap();
    let text = a.to_canonical_json();
    let back = Acceptances::parse(text.as_bytes()).unwrap();
    assert_eq!(back, a);
    assert_eq!(back.to_canonical_json(), text);
    assert_eq!(back.iter().next().unwrap().accepted_at, 1_792_000_000);
    assert!(back.has(&user("alice"), &req(Licence::CcBySa40)));
    assert!(!back.has(&user("carol"), &req(Licence::CcBySa40)));
    assert!(!back.has(&user("bob"), &req(Licence::Lgpl30Only)));
}

#[test]
fn the_canonical_form_is_documented() {
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcBySa40, 1_792_000_000).unwrap();
    assert_eq!(
        a.to_canonical_json(),
        r#"{
  "acceptances_version": 1,
  "accepted": [
    {
      "user": "alice",
      "license": "CC-BY-SA-4.0",
      "license_text": "https://creativecommons.org/licenses/by-sa/4.0/legalcode",
      "accepted_at": 1792000000
    }
  ]
}
"#
    );
}

#[test]
fn permissive_and_refused_licences_cannot_be_accepted() {
    let mut a = Acceptances::default();
    assert!(a.accept(&user("alice"), Licence::Apache2, 0).is_err());
    assert!(a.accept(&user("alice"), Licence::CcByNcNd40, 0).is_err());
    assert!(a.accept(&user("alice"), Licence::CcByNd40, 0).is_err());
    assert!(a.accept(&user("alice"), Licence::Unconfirmed, 0).is_err());
    assert_eq!(a, Acceptances::default());
    let nd = r#"{"acceptances_version":1,"accepted":[{"user":"a","license":"CC-BY-NC-ND-4.0","license_text":"https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode","accepted_at":1}]}"#;
    assert!(Acceptances::parse(nd.as_bytes()).is_err());
}

/// A non-commercial licence is accepted as one text, its legal code,
/// for every host, and each licence on its own.
#[test]
fn non_commercial_licences_are_accepted_by_their_legal_code() {
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcByNcSa40, 9).unwrap();
    assert!(a.has(&user("alice"), &req(Licence::CcByNcSa40)));
    assert!(!a.has(&user("alice"), &req(Licence::CcByNcSa30)));
    assert!(!a.has(&user("alice"), &req(Licence::CcByNc40)));
    assert!(!a.has(&user("bob"), &req(Licence::CcByNcSa40)));
    let back = Acceptances::parse(a.to_canonical_json().as_bytes()).unwrap();
    assert_eq!(back, a);
    assert!(
        a.to_canonical_json()
            .contains("\"license_text\": \"https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode\"")
    );
    // A host's page in place of the legal code is not the licence's text.
    let host = Requirement {
        licence: Licence::CcByNcSa40,
        text: "https://github.com/UniversalDependencies/UD_English-GUM/blob/r2.16/README.md".into(),
    };
    assert!(a.accept_terms(&user("alice"), &host, 0).is_err());
    assert!(a.revoke(&user("alice"), &req(Licence::CcByNcSa40)));
    assert_eq!(a, Acceptances::default());
}

#[test]
fn revoke_removes_one_users_acceptance() {
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::Gpl30Only, 0).unwrap();
    a.accept(&user("bob"), Licence::Gpl30Only, 0).unwrap();
    assert!(a.revoke(&user("alice"), &req(Licence::Gpl30Only)));
    assert!(!a.revoke(&user("alice"), &req(Licence::Gpl30Only)));
    assert!(!a.has(&user("alice"), &req(Licence::Gpl30Only)));
    assert!(a.has(&user("bob"), &req(Licence::Gpl30Only)));
}

#[test]
fn parse_is_strict() {
    let ok = r#"{"acceptances_version":1,"accepted":[{"user":"a","license":"CC-BY-SA-4.0","license_text":"https://creativecommons.org/licenses/by-sa/4.0/legalcode","accepted_at":1}]}"#;
    assert!(Acceptances::parse(ok.as_bytes()).is_ok());
    for bad in [
        ok.replace("\"acceptances_version\":1", "\"acceptances_version\":3"),
        ok.replace("\"acceptances_version\":1", "\"acceptances_version\":0"),
        ok.replace("\"accepted_at\":1", "\"accepted_at\":1,\"note\":\"x\""),
        ok.replace("\"accepted_at\":1", "\"accepted_at\":\"1\""),
        ok.replace(",\"accepted_at\":1", ""),
        ok.replace("CC-BY-SA-4.0", "cc-by-sa-4.0"),
        ok.replace("CC-BY-SA-4.0\",\"license_text\":\"https://creativecommons.org/licenses/by-sa/4.0/legalcode", "MIT\",\"license_text\":\"https://opensource.org/license/mit"),
        ok.replace("by-sa/4.0/legalcode", "by-sa/3.0/legalcode"),
        ok.replace("\"user\":\"a\"", "\"user\":\"\""),
        ok.replace("\"user\":\"a\"", "\"user\":\" a\""),
        ok.replace("}]}", "},{\"user\":\"a\",\"license\":\"CC-BY-SA-4.0\",\"license_text\":\"https://creativecommons.org/licenses/by-sa/4.0/legalcode\",\"accepted_at\":2}]}"),
    ] {
        assert!(Acceptances::parse(bad.as_bytes()).is_err(), "{bad}");
    }
}

#[test]
fn a_missing_file_is_no_acceptances() {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-accept-{}", std::process::id()));
    let path = dir.join("absent").join("acceptances.json");
    assert_eq!(Acceptances::load(&path).unwrap(), Acceptances::default());
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcBySa30, 5).unwrap();
    a.save(&path).unwrap();
    assert_eq!(Acceptances::load(&path).unwrap(), a);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn an_unusable_entry_is_named_with_the_file() {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-accept-bad-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("acceptances.json");
    let text = r#"{"acceptances_version":1,"accepted":[{"user":"a","license":"CC-BY-SA-4.0","license_text":"https://creativecommons.org/licenses/by-sa/4.0/legalcode","accepted_at":1},{"user":"a","license":"MIT","license_text":"https://opensource.org/license/mit","accepted_at":2}]}"#;
    fs::write(&path, text).unwrap();
    let msg = Acceptances::load(&path).unwrap_err().to_string();
    assert!(msg.contains(&path.display().to_string()) && msg.contains("accepted[1]") && msg.contains("MIT"), "{msg}");
    fs::remove_dir_all(&dir).unwrap();
}

/// A symlink left at a temporary name is never written through: each
/// write makes a name of its own, and leaves no temporary file.
#[cfg(unix)]
#[test]
fn save_never_writes_through_a_symlinked_temporary_file() {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-accept-link-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(dir.join("home")).unwrap();
    let victim = dir.join("victim");
    fs::write(&victim, b"untouched").unwrap();
    let path = dir.join("home").join("acceptances.json");
    std::os::unix::fs::symlink(&victim, dir.join("home").join("acceptances.json.partial")).unwrap();
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcBySa40, 5).unwrap();
    a.save(&path).unwrap();
    assert_eq!(fs::read(&victim).unwrap(), b"untouched");
    assert!(!fs::symlink_metadata(&path).unwrap().file_type().is_symlink());
    assert_eq!(Acceptances::load(&path).unwrap(), a);
    let names: Vec<_> = fs::read_dir(dir.join("home")).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert_eq!(names.len(), 2, "acceptances.json and the symlink left before: {names:?}");
    fs::remove_dir_all(&dir).unwrap();
}

fn shown(artifact: &str, text: &str) -> Shown {
    Shown { artifact: Some(artifact.parse().unwrap()), text_sha256: Some(Sha256::of(text.as_bytes())) }
}

/// An acceptance from a page names the artifact and the digest of the
/// text it displayed, and the file moves to version 2 to hold them.
#[test]
fn a_page_acceptance_is_version_2_and_round_trips() {
    let r = req(Licence::CcByNcSa40);
    let mut a = Acceptances::default();
    a.accept_shown(&user("alice"), &r, &shown("jackboyla-glirel-large-v0", &r.text), 1_792_000_000).unwrap();
    let text = a.to_canonical_json();
    assert_eq!(
        text,
        format!(
            r#"{{
  "acceptances_version": 2,
  "accepted": [
    {{
      "user": "alice",
      "license": "CC-BY-NC-SA-4.0",
      "license_text": "https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode",
      "accepted_at": 1792000000,
      "artifact": "jackboyla-glirel-large-v0",
      "license_text_sha256": "{}"
    }}
  ]
}}
"#,
            Sha256::of(r.text.as_bytes())
        )
    );
    let back = Acceptances::parse(text.as_bytes()).unwrap();
    assert_eq!(back, a);
    assert_eq!(back.to_canonical_json(), text);
    // Accepting the same terms again, from another page, keeps the first.
    a.accept_shown(&user("alice"), &r, &shown("other-model", &r.text), 1_800_000_000).unwrap();
    assert_eq!(a.to_canonical_json(), text);
    // Revoking the last version 2 acceptance writes version 1 again.
    a.accept(&user("bob"), Licence::CcBySa40, 3).unwrap();
    assert!(a.revoke(&user("alice"), &r));
    assert!(a.to_canonical_json().contains("\"acceptances_version\": 1,"));
}

#[test]
fn a_digest_of_other_text_is_refused() {
    let r = req(Licence::CcByNcSa40);
    let mut a = Acceptances::default();
    let wrong = shown("jackboyla-glirel-large-v0", "https://example.org/terms");
    assert!(a.accept_shown(&user("alice"), &r, &wrong, 1).is_err());
    assert_eq!(a, Acceptances::default());
    let ok = shown("jackboyla-glirel-large-v0", &r.text);
    a.accept_shown(&user("alice"), &r, &ok, 1).unwrap();
    let text = a.to_canonical_json();
    let other = Sha256::of(b"https://example.org/terms").to_string();
    let bad = text.replace(&Sha256::of(r.text.as_bytes()).to_string(), &other);
    assert!(Acceptances::parse(bad.as_bytes()).is_err());
}

/// A version 1 file still reads, and holds no version 2 field.
#[test]
fn version_1_reads_and_refuses_version_2_fields() {
    let v1 = r#"{"acceptances_version":1,"accepted":[{"user":"a","license":"CC-BY-SA-4.0","license_text":"https://creativecommons.org/licenses/by-sa/4.0/legalcode","accepted_at":1}]}"#;
    let a = Acceptances::parse(v1.as_bytes()).unwrap();
    assert_eq!(a.iter().next().unwrap().artifact, None);
    let with_artifact = v1.replace("\"accepted_at\":1", "\"accepted_at\":1,\"artifact\":\"m\"");
    assert!(Acceptances::parse(with_artifact.as_bytes()).is_err());
    let v2 = with_artifact.replace("\"acceptances_version\":1", "\"acceptances_version\":2");
    let b = Acceptances::parse(v2.as_bytes()).unwrap();
    assert_eq!(b.iter().next().unwrap().artifact.as_ref().map(|x| x.as_str()), Some("m"));
    // A version 2 file without the new fields is written back as version 1.
    let plain = v1.replace("\"acceptances_version\":1", "\"acceptances_version\":2");
    let c = Acceptances::parse(plain.as_bytes()).unwrap();
    assert!(c.to_canonical_json().contains("\"acceptances_version\": 1,"));
}

#[test]
fn of_user_lists_one_users_acceptances_in_order() {
    let mut a = Acceptances::default();
    a.accept(&user("alice"), Licence::CcBySa40, 1).unwrap();
    a.accept(&user("bob"), Licence::CcBySa40, 2).unwrap();
    a.accept(&user("alice"), Licence::Lgpl30Only, 3).unwrap();
    let alice = user("alice");
    let times: Vec<u64> = a.of_user(&alice).map(|x| x.accepted_at).collect();
    assert_eq!(times, [1, 3]);
    assert_eq!(a.find(&alice, &req(Licence::Lgpl30Only)).unwrap().accepted_at, 3);
}
