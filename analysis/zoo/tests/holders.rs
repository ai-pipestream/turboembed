//! The holders file (#1744): each user's registered sources, in order,
//! kept in one canonical form, refused when malformed, and written
//! readable by its owner alone.

use turbo_zoo::{Holder, Holders, SignIn, UserId, ZooError};

fn user(name: &str) -> UserId {
    name.parse().unwrap()
}

fn drive() -> Holder {
    let sign_in =
        SignIn::GoogleOauth { client_id: "client-1".into(), client_secret: None, refresh_token: "refresh-1".into() };
    Holder::new("My Drive", "gdrive://root/models", sign_in).unwrap()
}

fn disk() -> Holder {
    Holder::new("disk", "file:///zoo", SignIn::Environment).unwrap()
}

fn temp(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-holders-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn holders_round_trip_in_one_canonical_form_with_the_file_kept_private() {
    let mut h = Holders::default();
    h.register(&user("bob"), disk()).unwrap();
    h.register(&user("alice"), drive()).unwrap();
    h.register(&user("alice"), disk()).unwrap();
    let dir = temp("round");
    let path = dir.join("holders.json");
    h.save(&path).unwrap();
    let back = Holders::load(&path).unwrap();
    assert_eq!(back, h);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text, h.to_canonical_json());
    assert!(text.find("\"alice\"").unwrap() < text.find("\"bob\"").unwrap(), "users in order");
    assert!(!text.contains("\"sign_in\": {\n        \"kind\": \"environment\""), "the default is left out");
    assert!(text.contains("\"kind\": \"google_oauth\""), "{text}");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o077, 0);
    }
    assert_eq!(Holders::load(&dir.join("none.json")).unwrap(), Holders::default());
    let _ = std::fs::remove_dir_all(&dir);
    let names: Vec<_> = back.list(&user("alice")).iter().map(|h| h.name.as_str()).collect();
    assert_eq!(names, ["My Drive", "disk"]);
    assert_eq!(back.users().count(), 2);
    assert!(back.list(&user("carol")).is_empty());
}

#[test]
fn register_replaces_by_name_in_place_and_holders_move_and_go() {
    let alice = user("alice");
    let mut h = Holders::default();
    h.register(&alice, drive()).unwrap();
    h.register(&alice, disk()).unwrap();
    let moved = Holder::new("My Drive", "gdrive://root/other", SignIn::ApiKey { key: "k".into() }).unwrap();
    h.register(&alice, moved.clone()).unwrap();
    assert_eq!(h.list(&alice), [moved.clone(), disk()]);
    h.move_to(&alice, "disk", 0).unwrap();
    assert_eq!(h.list(&alice), [disk(), moved.clone()]);
    h.move_to(&alice, "disk", 99).unwrap();
    assert_eq!(h.list(&alice), [moved, disk()]);
    assert!(matches!(h.move_to(&alice, "none", 0), Err(ZooError::Holders(_))));
    assert_eq!(h.remove(&alice, "disk"), Some(disk()));
    assert_eq!(h.remove(&alice, "disk"), None);
    h.remove(&alice, "My Drive").unwrap();
    assert_eq!(h.users().count(), 0);
    assert!(!h.to_canonical_json().contains("alice"), "a user with no holder is left out");
}

#[test]
fn a_bad_holder_or_file_is_refused() {
    for (name, address) in [
        ("", "file:///zoo"),
        (" disk", "file:///zoo"),
        ("a,b", "file:///zoo"),
        ("tab\there", "file:///zoo"),
        (&"x".repeat(65) as &str, "file:///zoo"),
        ("disk", "/zoo"),
        ("disk", "file://"),
        ("disk", "File:///zoo"),
        ("disk", "file:///a,b"),
    ] {
        assert!(Holder::new(name, address, SignIn::Environment).is_err(), "{name:?} {address:?}");
    }
    assert!(Holder::new("é 模型", "file:///zoo", SignIn::Environment).is_ok(), "names are not ASCII only");
    assert!(Holder::new("k", "gdrive://root", SignIn::ApiKey { key: "a key".into() }).is_err());
    let e = format!("{:?}", drive());
    assert!(!e.contains("refresh-1") && !e.contains("client-1"), "Debug shows no secret: {e}");

    let ok = r#"{"holders_version": 1, "users": [{"user": "alice", "holders": [
        {"name": "disk", "address": "file:///zoo"}]}]}"#;
    assert_eq!(Holders::parse(ok.as_bytes()).unwrap().list(&user("alice")), [disk()]);
    for (bad, says) in [
        (r#"{"holders_version": 2, "users": []}"#, "holders_version 2"),
        (r#"{"holders_version": 1, "users": [], "extra": 1}"#, "unknown field"),
        (
            r#"{"holders_version": 1, "users": [{"user": "a", "holders": [
               {"name": "d", "address": "file:///z", "colour": "red"}]}]}"#,
            "unknown field",
        ),
        (
            r#"{"holders_version": 1, "users": [{"user": "a", "holders": [{"name": "d", "address": "file:///z",
               "sign_in": {"kind": "api_key", "key": "k", "more": 1}}]}]}"#,
            "unknown field",
        ),
        (
            r#"{"holders_version": 1, "users": [{"user": "a", "holders": [
               {"name": "d", "address": "file:///z"}, {"name": "d", "address": "file:///y"}]}]}"#,
            "named twice",
        ),
        (
            r#"{"holders_version": 1, "users": [{"user": "a", "holders": []}, {"user": "a", "holders": []}]}"#,
            "listed twice",
        ),
        (r#"{"holders_version": 1, "users": [{"user": " a", "holders": []}]}"#, "not a user name"),
    ] {
        let e = Holders::parse(bad.as_bytes()).unwrap_err().to_string();
        assert!(e.contains(says), "{bad}: {e}");
    }
}
