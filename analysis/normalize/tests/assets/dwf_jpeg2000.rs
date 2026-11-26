//! The DWF header and the JPEG 2000 types.

use super::media_types::be32_at;
use super::*;

// ---------------------------------------------------------------------------
// AssetDwfHeaderTest

/// `AssetDwfHeaderTest.header`: `(DWF V` the version `)`, cut or
/// zero-filled to `length`, with two non-ASCII bytes at the end of a
/// longer one.
fn dwf(version: &str, length: usize) -> Vec<u8> {
    let mut b = format!("(DWF V{version})").into_bytes();
    b.resize(length, 0);
    if length > 14 {
        b[length - 2] = 0xFB;
        b[length - 1] = 0xFF;
    }
    b
}

fn assert_dwf(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "dwf", "model/vnd.dwf", transport);
}

#[test]
fn dwf_version_and_transport() {
    for version in ["00.00", "00.55", "00.60", "06.00", "06.01", "07.00", "99.99"] {
        for transport in TRANSPORTS {
            assert_dwf(&dwf(version, 80), transport);
        }
    }
}

#[test]
fn dwf_encoded_note() {
    let bytes = b"(DWF Version notes) This text describes a drawing file header.";
    for transport in TRANSPORTS {
        assert!(detect(&encode(bytes, transport)).is_empty());
    }
}

#[test]
fn dwf_version_byte() {
    for offset in [6, 7, 9, 10] {
        let values = [0u8, b'\t', b'\n', b'\r', b' ', b'/', b'0', b'1', b'8', b'9', b':', b'A', b'a', 0x7F, 0x80, 0xFF];
        for value in values {
            let mut bytes = dwf("00.55", 80);
            bytes[offset] = value;
            if value.is_ascii_digit() {
                assert_dwf(&bytes, "standard");
                assert_dwf(&bytes, "uri");
            } else {
                assert_unrecognized_default(&bytes);
            }
        }
    }
}

#[test]
fn dwf_punctuation_byte() {
    for offset in [8, 11] {
        for value in [0u8, b'\t', b'\n', b'\r', b' ', b'.', b')', b'0', b'9', b',', b'/', 0x7F, 0x80, 0xFF] {
            let mut bytes = dwf("00.55", 80);
            bytes[offset] = value;
            if value == if offset == 8 { b'.' } else { b')' } {
                assert_dwf(&bytes, "standard");
            } else {
                assert_unrecognized_default(&bytes);
            }
        }
    }
}

#[test]
fn dwf_truncated_header() {
    for length in 0..12 {
        assert_unrecognized_default(&dwf("00.55", length));
    }
}

#[test]
fn dwf_header_without_drawing() {
    for length in [12, 13, 16, 23, 24] {
        assert_dwf(&dwf("00.55", length), "uri");
    }
}

#[test]
fn dwf_changed_prefix() {
    for offset in 0..6 {
        for value in [0u8, b'x', 0x80, 0xFF] {
            let mut bytes = dwf("00.55", 80);
            bytes[offset] = value;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn dwf_misplaced_version() {
    for version in ["0.55", "000.55", "00.5", "00.555", "00.55 ", " 00.55", "00,55"] {
        assert_unrecognized_default(&dwf(version, 80));
    }
}

#[test]
fn dwf_declared_media_type() {
    for version in ["00.55", "00,55", "aa.bb"] {
        let bytes = dwf(version, 80);
        let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), if version == "00.55" { "dwf" } else { "octet-stream" });
        assert_eq!(a.media_type(&text), "application/octet-stream");
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn dwf_w2d_prefix() {
    let mut bytes = dwf("06.00", 80);
    bytes[1..4].copy_from_slice(b"W2D");
    for transport in TRANSPORTS {
        assert!(detect(&encode(&bytes, transport)).is_empty());
    }
}

/// CRC-32 (ISO 3309), for the zip fixture.
fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 == 1 { crc >> 1 ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    !crc
}

/// A zip archive holding `content` stored as `name`: a local file header,
/// the data, a central directory entry and the end record.
fn stored_zip(name: &str, content: &[u8]) -> Vec<u8> {
    let (crc, size, name_len) = (crc32(content), content.len() as u32, name.len() as u16);
    let mut z = Vec::new();
    let put16 = |z: &mut Vec<u8>, v: u16| z.extend_from_slice(&v.to_le_bytes());
    z.extend_from_slice(b"PK\x03\x04");
    for v in [20u16, 0, 0, 0, 0x21] {
        put16(&mut z, v);
    }
    for v in [crc, size, size] {
        z.extend_from_slice(&v.to_le_bytes());
    }
    put16(&mut z, name_len);
    put16(&mut z, 0);
    z.extend_from_slice(name.as_bytes());
    z.extend_from_slice(content);
    let directory = z.len() as u32;
    z.extend_from_slice(b"PK\x01\x02");
    for v in [20u16, 20, 0, 0, 0, 0x21] {
        put16(&mut z, v);
    }
    for v in [crc, size, size] {
        z.extend_from_slice(&v.to_le_bytes());
    }
    for v in [name_len, 0, 0, 0, 0] {
        put16(&mut z, v);
    }
    z.extend_from_slice(&[0; 8]);
    z.extend_from_slice(name.as_bytes());
    let directory_size = z.len() as u32 - directory;
    z.extend_from_slice(b"PK\x05\x06");
    for v in [0u16, 0, 1, 1] {
        put16(&mut z, v);
    }
    z.extend_from_slice(&directory_size.to_le_bytes());
    z.extend_from_slice(&directory.to_le_bytes());
    put16(&mut z, 0);
    z
}

#[test]
fn dwf_zip_container() {
    let bytes = stored_zip("drawing.dwf", &dwf("00.55", 80));
    for transport in TRANSPORTS {
        assert_identified(&AssetDetector::new(), &bytes, "zip", "application/zip", transport);
    }
}

// ---------------------------------------------------------------------------
// AssetJpeg2000TypeTest

const JP2_SIGNATURE: &[u8] = b"\x00\x00\x00\x0cjP  \r\n\x87\n";

/// `AssetJpeg2000TypeTest.header`: the signature box, then an ftyp box
/// with a standard (24) or extended (1, then 32 as eight bytes) length,
/// the brand, minor version 0 and a compatibility list of the brand and
/// `jp2 `.
fn jp2(brand: &str, extended: bool) -> Vec<u8> {
    let mut b = JP2_SIGNATURE.to_vec();
    b.extend_from_slice(&(if extended { 1u32 } else { 24 }).to_be_bytes());
    b.extend_from_slice(b"ftyp");
    if extended {
        b.extend_from_slice(&32u64.to_be_bytes());
    }
    b.extend_from_slice(brand.as_bytes());
    b.extend_from_slice(&[0; 4]);
    b.extend_from_slice(brand.as_bytes());
    b.extend_from_slice(b"jp2 ");
    b.resize(80, 0);
    b
}

#[test]
fn jp2_file_brands() {
    for (brand, format, media_type) in [
        ("jp2 ", "jp2", "image/jp2"),
        ("jpx ", "jpx", "image/jpx"),
        ("jpm ", "jpm", "image/jpm"),
        ("mjp2", "mj2", "video/mj2"),
    ] {
        for extended in [false, true] {
            assert_identified_everywhere(&jp2(brand, extended), format, media_type);
        }
    }
}

#[test]
fn jp2_incomplete_file_type() {
    for extended in [false, true] {
        for length in 0..if extended { 36 } else { 28 } {
            assert_unrecognized_default(&jp2("jp2 ", extended)[..length]);
        }
    }
}

fn assert_jp2(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "jp2", "image/jp2", transport);
}

#[test]
fn jp2_fixed_fields_without_image_data() {
    for extended in [false, true] {
        let bytes = &jp2("jp2 ", extended)[..if extended { 36 } else { 28 }];
        assert_jp2(bytes, "standard");
        assert_jp2(bytes, "uri");
    }
}

#[test]
fn jp2_minor_version() {
    for extended in [false, true] {
        let mut bytes = jp2("jp2 ", extended);
        be32_at(&mut bytes, if extended { 32 } else { 24 }, u32::MAX);
        assert_jp2(&bytes, "uri");
    }
}

#[test]
fn jp2_insufficient_standard_box_length() {
    for length in [2, 7, 8, 11, 12, 15] {
        let mut bytes = jp2("jp2 ", false);
        be32_at(&mut bytes, 12, length);
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn jp2_insufficient_extended_box_length() {
    for length in [0u64, 1, 8, 16, 20, 23] {
        let mut bytes = jp2("jp2 ", true);
        bytes[20..28].copy_from_slice(&length.to_be_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn jp2_standard_length_values() {
    for length in [0u32, 16, 20, 24, 32, 0x8000_0000, 0xFFFF_FFFC] {
        let mut bytes = jp2("jp2 ", false);
        be32_at(&mut bytes, 12, length);
        assert_jp2(&bytes, "uri");
    }
}

#[test]
fn jp2_extended_length_values() {
    for length in [24u64, 28, 32, 0x6A70_3220, 0xFFFF_FFFF, 0x1_0000_0000, i64::MAX as u64, i64::MIN as u64, u64::MAX] {
        let mut bytes = jp2("jpx ", true);
        bytes[20..28].copy_from_slice(&length.to_be_bytes());
        assert_identified(&AssetDetector::new(), &bytes, "jpx", "image/jpx", "uri");
    }
}

#[test]
fn jp2_unknown_brand() {
    for brand in ["????", "jp2x", "JP2 ", "jpxb", "mj2 ", "jpip", "    "] {
        assert_unrecognized_default(&jp2(brand, false));
        assert_unrecognized_default(&jp2(brand, true));
    }
}

#[test]
fn jp2_incorrect_box_type() {
    for kind in ["free", "jp2h", "FTYP", "xxxx"] {
        let mut bytes = jp2("jp2 ", false);
        bytes[16..20].copy_from_slice(kind.as_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn jp2_displaced_file_type_box() {
    for offset in [11, 13, 16, 24, 32] {
        let source = jp2("jp2 ", false);
        let mut bytes = padded(JP2_SIGNATURE, 80);
        bytes[offset..offset + 24].copy_from_slice(&source[12..36]);
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn jp2_incorrect_signature() {
    for offset in 0..12 {
        let mut bytes = jp2("jp2 ", false);
        bytes[offset] ^= 1;
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn jp2_supplied_media_type() {
    for brand in ["jpx ", "????"] {
        let bytes = jp2(brand, true);
        let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), if brand == "jpx " { "jpx" } else { "octet-stream" });
        assert_eq!(a.media_type(&text), "application/octet-stream");
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}
