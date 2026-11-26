//! The eight-byte PNG-family signatures, AIFF forms, WebVTT, and the
//! Android and GRIB headers.

use super::*;

// ---------------------------------------------------------------------------
// AssetFullSignatureTest: the eight-byte PNG-family signatures.

const FULL_SIGNATURES: [([u8; 8], &str, &str); 4] = [
    (*b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (*b"\x8aMNG\r\n\x1a\n", "mng", "video/x-mng"),
    (*b"\x8bJNG\r\n\x1a\n", "jng", "image/x-jng"),
    (*b"\x89HDF\r\n\x1a\n", "hdf", "application/x-hdf"),
];

fn signature_header(signature: &[u8; 8], length: usize) -> Vec<u8> {
    let mut b = signature.to_vec();
    b.resize(length, 0);
    b
}

#[test]
fn full_signatures_complete() {
    for (signature, format, media_type) in FULL_SIGNATURES {
        let bytes = signature_header(&signature, 64);
        let known = magic_format(&bytes).unwrap();
        assert_eq!((known.name(), known.media_type()), (format, media_type));
        assert_identified_everywhere(&bytes, format, media_type);
    }
}

#[test]
fn full_signature_without_file_contents() {
    for (signature, format, media_type) in FULL_SIGNATURES {
        assert_identified(&AssetDetector::new(), &signature_header(&signature, 8), format, media_type, "uri");
    }
}

#[test]
fn full_signatures_truncated_or_changed() {
    for (signature, _, _) in FULL_SIGNATURES {
        for position in 0..8 {
            let truncated = signature_header(&signature, position);
            assert_eq!(magic_format(&truncated), None);
            assert_unrecognized_default(&truncated);
            let mut changed = signature_header(&signature, 64);
            changed[position] ^= 1;
            assert_eq!(magic_format(&changed), None);
            assert_unrecognized_default(&changed);
        }
    }
}

#[test]
fn full_signature_byte_after() {
    for (signature, format, media_type) in FULL_SIGNATURES {
        for value in [0u8, 63, 128, 255] {
            let mut bytes = signature_header(&signature, 64);
            bytes[8] = value;
            assert_identified(&AssetDetector::new(), &bytes, format, media_type, "standard");
        }
    }
}

#[test]
fn full_signature_supplied_media_type() {
    for (signature, _, _) in FULL_SIGNATURES {
        let bytes = signature_header(&signature, 7);
        let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn full_signature_supplied_jng_video_media_type() {
    let bytes = signature_header(&FULL_SIGNATURES[2].0, 8);
    let text = format!("data:video/x-jng;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("jng", "video/x-jng"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetAiffFormTest

/// `FORM`, a big-endian size and a four-byte form type, in 64 bytes.
fn aiff(kind: &str, size: u32) -> Vec<u8> {
    let mut b = b"FORM".to_vec();
    b.extend_from_slice(&size.to_be_bytes());
    b.extend_from_slice(kind.as_bytes());
    b.resize(64, 0);
    b
}

fn assert_aiff(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "aiff", "audio/x-aiff", transport);
}

#[test]
fn aiff_audio_forms() {
    for kind in ["AIFF", "AIFC"] {
        for transport in TRANSPORTS {
            assert_aiff(&aiff(kind, 56), transport);
        }
    }
}

#[test]
fn aiff_form_size_is_not_signature() {
    for kind in ["AIFF", "AIFC"] {
        for size in [4, 56, 65536, 0x00FF_FFFE, 0x0100_0000, 0x1000_0000, 0x7FFF_FFFE] {
            assert_aiff(&aiff(kind, size), "standard");
            assert_aiff(&aiff(kind, size), "uri");
        }
    }
}

#[test]
fn aiff_incomplete_form_type() {
    for kind in ["AIFF", "AIFC"] {
        for length in 0..12 {
            assert_unrecognized_default(&aiff(kind, 56)[..length]);
        }
    }
}

#[test]
fn aiff_form_type_without_local_chunks() {
    for kind in ["AIFF", "AIFC"] {
        assert_aiff(&aiff(kind, 4)[..12], "uri");
    }
}

#[test]
fn aiff_other_form_types() {
    for kind in ["ILBM", "8SVX", "ANIM", "PBM ", "MAUD", "????", "aiff", "aifc"] {
        assert_unrecognized_default(&aiff(kind, 56));
    }
}

#[test]
fn aiff_byte_after_form_type() {
    for kind in ["AIFFX", "AIFCX"] {
        assert_aiff(&aiff(kind, 56), "standard");
    }
}

#[test]
fn aiff_incorrect_form_type_offset() {
    for kind in ["AIFF", "AIFC"] {
        for offset in [4, 7, 9, 12, 20] {
            let mut bytes = aiff("????", 56);
            bytes[offset..offset + 4].copy_from_slice(kind.as_bytes());
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn aiff_incorrect_container() {
    for container in ["RIFF", "RIFX", "LIST", "CAT ", "F0RM", "form"] {
        let mut bytes = aiff("AIFF", 56);
        bytes[..4].copy_from_slice(container.as_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn aiff_supplied_type_for_other_form() {
    let bytes = aiff("ILBM", 56);
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetWebVttTest

const CUE: &str = "00:00.000 --> 00:03.000\nThe trail passes a café. Follow the blue signs. \u{1F9ED}\n";

fn assert_vtt(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "vtt", "text/vtt", transport);
}

fn vtt_headers(tails: &[&str]) -> Vec<String> {
    ["", "\u{FEFF}"].iter().flat_map(|bom| tails.iter().map(move |tail| format!("{bom}WEBVTT{tail}"))).collect()
}

/// Text as UTF-16 bytes in either byte order.
pub(super) fn utf16(text: &str, big_endian: bool) -> Vec<u8> {
    text.encode_utf16().flat_map(|u| if big_endian { u.to_be_bytes() } else { u.to_le_bytes() }).collect()
}

#[test]
fn vtt_caption_file() {
    for header in vtt_headers(&["\n\n", "\r\r", "\r\n\r\n", " OpenNLP captions\n\n", "\tOpenNLP captions\n\n"]) {
        for transport in TRANSPORTS {
            assert_vtt(format!("{header}{CUE}").as_bytes(), transport);
        }
    }
}

#[test]
fn vtt_boundary_byte() {
    for bom in ["", "\u{FEFF}"] {
        for value in 0..=255u8 {
            let signature = format!("{bom}WEBVTT");
            let mut bytes = signature.clone().into_bytes();
            bytes.resize(80, b'x');
            bytes[signature.len()] = value;
            if matches!(value, b'\t' | b'\n' | b'\r' | b' ') {
                assert_vtt(&bytes, "standard");
                assert_vtt(&bytes, "uri");
            } else {
                assert_unrecognized_default(&bytes);
            }
        }
    }
}

#[test]
fn vtt_header_without_cue() {
    for header in vtt_headers(&["", "\t", "\n", "\r", "\r\n", " "]) {
        assert_vtt(header.as_bytes(), "uri");
        // Bare, the header is shorter than a bare run may be.
        assert!(detect(&std64(header.as_bytes())).is_empty());
    }
}

#[test]
fn vtt_truncated_header() {
    for bom in ["", "\u{FEFF}"] {
        let bytes = format!("{bom}WEBVTT").into_bytes();
        for length in 0..bytes.len() {
            assert_unrecognized_default(&bytes[..length]);
        }
    }
}

#[test]
fn vtt_invalid_prefix() {
    for prefix in [
        "webvtt",
        "WebVTT",
        "WEBVT",
        "WEBVTTX",
        " WEBVTT",
        "\nWEBVTT",
        "\u{FEFF}\u{FEFF}WEBVTT",
        "\u{FFFE}WEBVTT",
        "WEB\u{FEFF}VTT",
        "\u{FEFF}webvtt",
    ] {
        assert_unrecognized_default(format!("{prefix}\n\n{CUE}").as_bytes());
    }
}

#[test]
fn vtt_utf16_header() {
    for big_endian in [false, true] {
        assert_unrecognized_default(&utf16(&format!("\u{FEFF}WEBVTT\n\n{CUE}"), big_endian));
    }
}

#[test]
fn vtt_declared_type() {
    for media_type in ["text/vtt", "text/plain"] {
        let bytes = format!("WEBVTTX\n\n{CUE}").into_bytes();
        let text = format!("data:{media_type};base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), &media_type[5..]);
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn vtt_manual_caption_example() {
    let captions = "\u{FEFF}WEBVTT\tTrail captions\n\n00:00.000 --> 00:03.000\nTurn left at the bridge.\n";
    let text = format!("{DATA_URI}{}", std64(captions.as_bytes()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("vtt", "text/vtt"));
    assert_eq!(a.decode(&text).unwrap(), captions.as_bytes());
}

// ---------------------------------------------------------------------------
// AssetAndroidGribTest

/// Little-endian chunk header: type, header length, total length.
fn chunk(out: &mut Vec<u8>, kind: u16, header: u16, total: u32) {
    out.extend_from_slice(&kind.to_le_bytes());
    out.extend_from_slice(&header.to_le_bytes());
    out.extend_from_slice(&total.to_le_bytes());
}

fn le32(out: &mut Vec<u8>, values: &[i32]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

/// `AssetAndroidGribTest.binaryXml`: an Android binary XML document with
/// `chunks` before its string pool.
fn binary_xml(chunks: &[u16]) -> Vec<u8> {
    let total = 112 + chunks.len() * 8;
    let mut b = Vec::with_capacity(total);
    chunk(&mut b, 3, 8, total as u32);
    for &kind in chunks {
        chunk(&mut b, kind, 8, 8);
    }
    chunk(&mut b, 1, 28, 44);
    le32(&mut b, &[1, 0, 0x0100, 32, 0, 0]);
    b.extend_from_slice(b"\x08\x08manifest\0\0");
    chunk(&mut b, 0x0102, 16, 36);
    le32(&mut b, &[1, -1, -1, 0]);
    b.extend_from_slice(&[20, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    chunk(&mut b, 0x0103, 16, 24);
    le32(&mut b, &[1, -1, -1, 0]);
    assert_eq!(b.len(), total);
    b
}

/// `AssetAndroidGribTest.gribHeader`.
fn grib(edition: u8) -> Vec<u8> {
    let mut b = b"GRIB\0\0\0".to_vec();
    b.push(edition);
    b.extend_from_slice(&80u64.to_be_bytes());
    b.resize(80, 0);
    b[19] = 0xFB;
    b[20] = 0xFF;
    b
}

fn android_or_grib(format: &str) -> (Vec<u8>, &'static str) {
    if format == "axml" { (binary_xml(&[]), "application/vnd.android.axml") } else { (grib(2), "application/x-grib") }
}

#[test]
fn android_xml_layout() {
    for chunks in [&[][..], &[0x0180], &[0x7FFE], &[0x7FFE, 0x0180]] {
        for transport in TRANSPORTS {
            let bytes = binary_xml(chunks);
            assert_identified(&AssetDetector::new(), &bytes, "axml", "application/vnd.android.axml", transport);
        }
    }
}

#[test]
fn grib_edition() {
    for edition in [0, 1, 2, 3, 127, 128, 255] {
        for transport in TRANSPORTS {
            assert_identified(&AssetDetector::new(), &grib(edition), "grb", "application/x-grib", transport);
        }
    }
}

#[test]
fn android_grib_changed_or_truncated_signature() {
    for format in ["axml", "grb"] {
        for offset in 0..4 {
            let (mut bytes, _) = android_or_grib(format);
            assert_unrecognized_default(&bytes[..offset]);
            bytes[offset] ^= 0xFF;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn android_grib_signature_without_complete_file() {
    for format in ["axml", "grb"] {
        for length in [4, 5, 7, 8, 12, 16] {
            let (bytes, media_type) = android_or_grib(format);
            assert_identified(&AssetDetector::new(), &bytes[..length], format, media_type, "uri");
        }
    }
}
