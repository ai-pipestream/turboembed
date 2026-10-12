//! Reads archives java.util.zip.ZipOutputStream wrote (fixtures/zip).

mod common;

use std::borrow::Cow;

use turbo_maxent::{Method, Zip, crc32};

#[test]
fn entries_match_what_java_wrote() {
    let index = std::fs::read_to_string(common::fixtures().join("zip/index.txt")).unwrap();
    let mut checked = 0;
    for line in index.lines().filter(|l| !l.starts_with('#')) {
        let f: Vec<&str> = line.split(' ').collect();
        let (archive, name, method, size, crc) = (f[0], f[1], f[2], f[3].parse::<usize>().unwrap(), f[4]);
        let zip = Zip::new(std::fs::read(common::fixtures().join("zip").join(archive)).unwrap()).unwrap();
        let expected_method = if method == "stored" { Method::Stored } else { Method::Deflated };
        assert_eq!(zip.method(name), Some(expected_method), "{archive} {name}");
        assert_eq!(zip.size(name), Some(size), "{archive} {name}");
        let data = zip.entry(name).unwrap_or_else(|e| panic!("{archive} {name}: {e}"));
        assert_eq!(data.len(), size);
        assert_eq!(format!("{:08x}", crc32(&data)), crc);
        assert_eq!(matches!(data, Cow::Borrowed(_)), expected_method == Method::Stored, "{archive} {name}");
        checked += 1;
    }
    assert_eq!(checked, 7);
}

#[test]
fn names_comments_and_contents() {
    let zip = Zip::new(std::fs::read(common::fixtures().join("zip/deflated.zip")).unwrap()).unwrap();
    let names: Vec<&[u8]> = zip.names().collect();
    assert_eq!(names, [&b"text.txt"[..], b"dir/random.bin", b"empty.txt", "näme.txt".as_bytes()]);
    assert_eq!(&*zip.entry("näme.txt").unwrap(), b"hello zip\n");
    assert!(zip.entry("empty.txt").unwrap().is_empty());
    assert!(!zip.contains("missing"));

    let stored = Zip::new(std::fs::read(common::fixtures().join("zip/stored.zip")).unwrap()).unwrap();
    assert_eq!(&*stored.entry("small.txt").unwrap(), b"hello zip\n");
    assert_eq!(stored.entry("text.txt").unwrap(), zip.entry("text.txt").unwrap());
    assert_eq!(stored.entry("random.bin").unwrap(), zip.entry("dir/random.bin").unwrap());
}

#[test]
fn a_damaged_deflated_entry_is_refused() {
    let mut bytes = std::fs::read(common::fixtures().join("zip/deflated.zip")).unwrap();
    // The first entry's data starts after the 30-byte local header and the
    // 8-byte name; flipping a byte in the middle breaks the stream or the CRC.
    bytes[30 + 8 + 100] ^= 0x55;
    let zip = Zip::new(bytes).unwrap();
    assert!(zip.entry("text.txt").is_err());
    assert_eq!(&*zip.entry("näme.txt").unwrap(), b"hello zip\n");
}
