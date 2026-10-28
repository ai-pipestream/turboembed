use super::*;

fn utf8_len(s: &JavaString) -> usize {
    s.to_str().unwrap().len()
}

#[test]
fn long_strings_are_chunked_by_utf8_bytes() {
    // 70000 ASCII bytes: two chunks of 65535 and 4465 bytes.
    let long = JavaString::new(&"x".repeat(70000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 4465]);
    // Three byte characters do not straddle a chunk end.
    let long = JavaString::new(&"\u{20AC}".repeat(30000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 24465]);
    assert_eq!(utf8_len(&long), 90000);
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&long).unwrap();
    let signature = b"CHUNKED-MODEL-PARAMS:2";
    assert_eq!(&out.bytes[..2 + signature.len()], [&[0, signature.len() as u8][..], signature].concat());
    assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 24465);
}

#[test]
fn a_chunk_of_supplementary_characters_overflows_write_utf() {
    // 16383 emoji fit a 65532 byte UTF-8 chunk but are 98298 bytes of
    // modified UTF-8, which writeUTF refuses, as the JVM's does.
    let long = JavaString::new(&"\u{1F600}".repeat(17000));
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks[0].len(), 16383 * 6);
    let mut out = ModelOutput { bytes: Vec::new() };
    assert_eq!(
        out.write_utf(&long),
        Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 16383 * 6 })
    );
}

#[test]
fn a_nul_is_one_utf8_byte_but_two_written() {
    assert_eq!(next_char(b"\xC0\x80a", 0), Some((2, 1)));
    assert_eq!(next_char(b"\xC3\xA9", 0), Some((2, 2)));
    assert_eq!(next_char(b"\xC0", 0), None);
    assert_eq!(next_char(b"\xE2\x82", 0), None);
    // 40000 NULs: 80000 bytes of modified UTF-8, so the string is
    // chunked, but 40000 bytes of UTF-8, so it is one chunk, whose
    // writeUTF throws in the JVM.
    let nuls = JavaString::new(&"\0".repeat(40000));
    assert_eq!(nuls.len(), 80000);
    let chunks = split_by_utf8_length(&nuls);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [80000]);
    let mut out = ModelOutput { bytes: Vec::new() };
    assert_eq!(out.write_utf(&nuls), Err(Error::StringTooLong { what: "chunk of a model parameter", bytes: 80000 }));
    // NULs after the first chunk make a second chunk of twice their count.
    let late = JavaString::new(&format!("{}{}", "x".repeat(65535), "\0".repeat(1000)));
    let chunks = split_by_utf8_length(&late);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 2000]);
    let mut out = ModelOutput { bytes: Vec::new() };
    out.write_utf(&late).unwrap();
    let signature = b"CHUNKED-MODEL-PARAMS:2";
    assert_eq!(out.bytes.len(), 2 + signature.len() + 2 + 65535 + 2 + 2000);
}

#[test]
fn a_lone_surrogate_cuts_a_chunked_string() {
    let mut units: Vec<u16> = vec![b'y' as u16; 70000];
    units[66000] = 0xD83D;
    let long = JavaString::from_utf16(&units);
    let chunks = split_by_utf8_length(&long);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [65535, 465]);

    units[0] = 0xD83D;
    let at_start = JavaString::from_utf16(&units);
    let chunks = split_by_utf8_length(&at_start);
    assert_eq!(chunks.iter().map(|c| c.len()).collect::<Vec<_>>(), [0]);
}
