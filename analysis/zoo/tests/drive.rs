//! A Google Drive folder as a source (#1736), against a small Drive API
//! stand-in started in-process: `files.list` queries answered from a
//! tree of folders and files, `files.get?alt=media` from their bytes,
//! and the zoo's own checks on whatever the folder serves.
#![cfg(feature = "http")]

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

use turbo_zoo::{
    Acceptances, Blobs, DriveAuth, GoogleDrive, Index, Policy, Sha256, Sha512, Sources, Transport, UserId, Zoo,
    ZooError,
};

const TEXT: &[u8] = b"hello hello hello\n";
const SA: &[u8] = b"a share-alike word list\n";
const FOLDER: &str = "application/vnd.google-apps.folder";

/// A permissive artifact `hello` and a share-alike one `sa`.
fn index() -> Index {
    let artifact = |id: &str, licence: &str, class: &str, bytes: &[u8]| {
        format!(
            r#"{{"id": "{id}", "kind": "KIND_LEXICON", "format": "FORMAT_TEXT", "version": "1",
              "license": "{licence}", "license_terms": "https://example.org/{id}/LICENSE", "trained_on": [],
              "class": "{class}", "attribution": "Example attribution for {id}.",
              "files": [{{"path": "lists/{id}.txt", "url": "https://example.org/{id}.txt", "size": {}, "sha256": "{}",
                "sha512": "{}"}}]}}"#,
            bytes.len(),
            Sha256::of(bytes),
            Sha512::of(bytes),
        )
    };
    let json = format!(
        r#"{{"zoo_version": 1, "artifacts": [{}, {}]}}"#,
        artifact("hello", "Apache-2.0", "CLASS_PERMISSIVE", TEXT),
        artifact("sa", "CC-BY-SA-4.0", "CLASS_SHARE_ALIKE", SA),
    );
    Index::parse(json.as_bytes()).unwrap()
}

/// One entry in the stand-in's tree: id, parent id, name, and the bytes
/// of a file (`None` for a folder).
type Entry = (&'static str, &'static str, String, Option<Vec<u8>>);

/// The request lines and headers (names lower-cased) the stand-in saw.
type Seen = Arc<Mutex<Vec<(String, Vec<String>)>>>;

/// Serves `tree` until the test ends, one request per connection.
fn stand_in(tree: Vec<Entry>) -> (String, Seen) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let seen: Seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for conn in listener.incoming() {
            let mut conn = conn.unwrap();
            let mut reader = BufReader::new(conn.try_clone().unwrap());
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            let target = line.split(' ').nth(1).unwrap_or_default().to_owned();
            let mut headers = Vec::new();
            loop {
                let mut h = String::new();
                reader.read_line(&mut h).unwrap();
                if h.trim_end().is_empty() {
                    break;
                }
                headers.push(h.trim_end().to_ascii_lowercase());
            }
            log.lock().unwrap().push((target.clone(), headers));
            let (status, body) = answer(&tree, &target);
            let head = format!("HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            conn.write_all(head.as_bytes()).unwrap();
            conn.write_all(&body).unwrap();
        }
    });
    (endpoint, seen)
}

fn answer(tree: &[Entry], target: &str) -> (&'static str, Vec<u8>) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let params: Vec<(&str, String)> =
        query.split('&').filter_map(|kv| kv.split_once('=')).map(|(k, v)| (k, decode(v))).collect();
    let param = |k: &str| params.iter().find(|(n, _)| *n == k).map(|(_, v)| v.clone());
    if param("supportsAllDrives").as_deref() != Some("true") {
        return ("400 Bad Request", b"supportsAllDrives".to_vec());
    }
    if path == "/drive/v3/files" {
        let q = param("q").unwrap_or_default();
        let (name, rest) = quoted(q.strip_prefix("name = '").unwrap());
        let (parent, rest) = quoted(rest.strip_prefix(" and '").unwrap());
        let folder = rest.contains("mimeType = '");
        let files: Vec<String> = tree
            .iter()
            .filter(|(_, p, n, bytes)| *p == parent && *n == name && bytes.is_none() == folder)
            .map(|(id, _, _, bytes)| match bytes {
                Some(b) => format!(r#"{{"id": "{id}", "size": "{}"}}"#, b.len()),
                None => format!(r#"{{"id": "{id}", "mimeType": "{FOLDER}"}}"#),
            })
            .collect();
        return ("200 OK", format!(r#"{{"files": [{}]}}"#, files.join(", ")).into_bytes());
    }
    let id = path.strip_prefix("/drive/v3/files/").unwrap_or_default();
    match tree.iter().find(|(i, ..)| *i == id) {
        Some((_, _, _, Some(bytes))) if param("alt").as_deref() == Some("media") => ("200 OK", bytes.clone()),
        _ => ("404 Not Found", Vec::new()),
    }
}

/// The query string literal at the start of `s`, unescaped, and what
/// follows its closing quote.
fn quoted(s: &str) -> (String, &str) {
    let mut out = String::new();
    let mut chars = s.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => out.push(chars.next().unwrap().1),
            '\'' => return (out, &s[i + 1..]),
            c => out.push(c),
        }
    }
    panic!("unterminated literal in {s}")
}

fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            out.push(u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap(), 16).unwrap());
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

/// A store root `top` with `blobs/sha256/<hex of TEXT>` below it.
fn blob_tree(bytes: &[u8]) -> Vec<Entry> {
    vec![
        ("blobsid", "top", "blobs".into(), None),
        ("shaid", "blobsid", "sha256".into(), None),
        ("fileid", "shaid", Sha256::of(TEXT).to_string(), Some(bytes.to_vec())),
    ]
}

fn cache(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("turbo-zoo-drive-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn alice() -> UserId {
    "alice".parse().unwrap()
}

#[test]
fn a_file_is_read_from_the_blob_layout_with_the_bearer_token() {
    let (endpoint, seen) = stand_in(blob_tree(TEXT));
    let drive = GoogleDrive::with("gdrive://top", Some(&endpoint), DriveAuth::Token("tok-1".into())).unwrap();
    let dir = cache("blob");
    let zoo = Zoo::new(index(), dir.clone(), Policy::default(), &drive);
    let got = zoo.fetch("hello", &alice(), &Acceptances::default()).map(|f| std::fs::read(&f.files[0]).unwrap());
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(got.unwrap(), TEXT);
    let seen = seen.lock().unwrap();
    // blobs, sha256, the file, then its bytes.
    assert_eq!(seen.len(), 4, "{seen:?}");
    for (target, headers) in seen.iter() {
        assert!(headers.contains(&"authorization: bearer tok-1".to_string()), "{target}: {headers:?}");
        assert!(!target.contains("key="), "{target}");
    }
    assert!(seen[3].0.starts_with("/drive/v3/files/fileid?alt=media"), "{}", seen[3].0);
}

#[test]
fn folder_ids_are_remembered_between_files() {
    let (endpoint, seen) = stand_in(blob_tree(TEXT));
    let drive = GoogleDrive::with("gdrive://top", Some(&endpoint), DriveAuth::Token("t".into())).unwrap();
    let index = index();
    let a = index.get("hello").unwrap();
    assert_eq!(drive.get_file(a, &a.files[0]).unwrap(), TEXT);
    assert_eq!(drive.get_file(a, &a.files[0]).unwrap(), TEXT);
    // The second read asks only for the file and its bytes.
    assert_eq!(seen.lock().unwrap().len(), 6);
}

#[test]
fn a_file_not_in_the_blob_layout_is_read_at_its_artifact_path_below_named_folders() {
    let tree = vec![
        ("data", "root", "it's data".into(), None),
        ("models", "data", "models".into(), None),
        ("hello", "models", "hello".into(), None),
        ("lists", "hello", "lists".into(), None),
        ("other", "lists", "hello.txt".into(), Some(b"other bytes, same name\n".to_vec())),
        ("textid", "lists", "hello.txt".into(), Some(TEXT.to_vec())),
    ];
    let (endpoint, seen) = stand_in(tree);
    let address = "gdrive://root/it's data/models";
    let drive = GoogleDrive::with(address, Some(&endpoint), DriveAuth::ApiKey("key/1".into())).unwrap();
    let index = index();
    let a = index.get("hello").unwrap();
    assert_eq!(drive.get_file(a, &a.files[0]).unwrap(), TEXT);
    for (target, headers) in seen.lock().unwrap().iter() {
        assert!(target.ends_with("&key=key%2F1"), "{target}");
        assert!(!headers.iter().any(|h| h.starts_with("authorization")), "{headers:?}");
    }
}

#[test]
fn what_the_folder_lacks_or_serves_wrong_falls_through_to_the_next_source() {
    let store = cache("store");
    Blobs::new(&store).put(TEXT).unwrap();
    let cases = [(vec![], "gdrive://top: no blobs/sha256/"), (blob_tree(b"hello hello hellO\n"), "SHA-256 is")];
    for (tree, why) in cases {
        let (endpoint, _) = stand_in(tree);
        let drive = || GoogleDrive::with("gdrive://top", Some(&endpoint), DriveAuth::Token("t".into())).unwrap();
        let empty = cache("empty");
        let lacking = Sources::new().then("gdrive://top", drive()).then("empty", Blobs::new(&empty));
        let alone = Zoo::new(index(), cache("alone"), Policy::default(), &lacking);
        let err = alone.fetch("hello", &alice(), &Acceptances::default()).unwrap_err().to_string();
        assert!(err.contains(why) && err.contains("empty: "), "{err}");
        let sources = Sources::new().then("gdrive://top", drive()).then("store", Blobs::new(&store));
        let dir = cache("through");
        let zoo = Zoo::new(index(), dir.clone(), Policy::default(), &sources);
        let got = zoo.fetch("hello", &alice(), &Acceptances::default()).map(|f| std::fs::read(&f.files[0]).unwrap());
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(got.unwrap(), TEXT);
    }
    let _ = std::fs::remove_dir_all(&store);
    let _ = std::fs::remove_dir_all(cache("alone"));
}

#[test]
fn terms_not_accepted_are_refused_before_drive_is_asked() {
    let (endpoint, seen) = stand_in(blob_tree(TEXT));
    let drive = GoogleDrive::with("gdrive://top", Some(&endpoint), DriveAuth::Token("t".into())).unwrap();
    let zoo = Zoo::new(index(), cache("refused"), Policy::default(), &drive);
    let err = zoo.fetch("sa", &alice(), &Acceptances::default()).unwrap_err();
    assert!(matches!(err, ZooError::NotAccepted { .. }), "{err}");
    assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn bad_addresses_endpoints_and_credentials_are_refused_before_any_request() {
    let token = || DriveAuth::Token("t".into());
    for bad in ["gdrive://", "gdrive://a b", "gdrive://id/../x", "gdrive://id//x", "drive://id"] {
        assert!(GoogleDrive::with(bad, None, token()).is_err(), "{bad}");
    }
    for bad in ["ftp://host", "http://", "http://host/path", "https://user@host"] {
        assert!(GoogleDrive::with("gdrive://id", Some(bad), token()).is_err(), "{bad}");
    }
    assert!(GoogleDrive::with("gdrive://id", None, DriveAuth::Token(String::new())).is_err());
    assert!(GoogleDrive::with("gdrive://id", None, DriveAuth::ApiKey("a key".into())).is_err());
    let none = GoogleDrive::from_vars("gdrive://id", |_| None).err().unwrap();
    assert!(none.contains("GOOGLE_OAUTH_ACCESS_TOKEN"), "{none}");
    let empty = GoogleDrive::from_vars("gdrive://id", |_| Some(String::new()));
    assert!(empty.is_err());
    let key = GoogleDrive::from_vars("gdrive://id", |k| (k == "GOOGLE_API_KEY").then(|| "k".into()));
    assert_eq!(key.unwrap().address(), "gdrive://id");
    let url = "https://example.org/x".parse().unwrap();
    let drive = GoogleDrive::with("gdrive://id", None, token()).unwrap();
    assert!(drive.get(&url, 8).unwrap_err().contains("not its address"));
}
