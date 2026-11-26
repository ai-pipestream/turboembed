//! Declared media types, timestamped data and format variants.

use super::*;

// ---------------------------------------------------------------------------
// AssetMediaTypeTest

/// `AssetMediaTypeTest.assertAsset`.
fn assert_asset(text: &str, bytes: &[u8], format: &str, media_type: &str) -> Asset {
    let a = one(text);
    assert_eq!((a.format_name(text), a.media_type(text)), (format, media_type), "{bytes:02x?}");
    assert_eq!(a.decoded_len, bytes.len() as u64);
    assert_eq!(a.decode(text).unwrap(), bytes);
    a
}

/// `AssetMediaTypeTest.mediaTypes`.
fn media_type_cases() -> [(Vec<u8>, &'static str, &'static str); 5] {
    [
        (padded(b"\x09\x00\x04\x00\x00\x00\x10\x00", 32), "xls", "application/vnd.ms-excel.sheet.2"),
        (padded(b"\x09\x02\x06\x00\x00\x00\x10\x00", 32), "xls", "application/vnd.ms-excel.sheet.3"),
        (padded(b"\x09\x04\x06\x00\x00\x00\x10\x00", 32), "xls", "application/vnd.ms-excel.sheet.4"),
        (pem("DSA PARAMETERS"), "pem-parameters", "application/x-x509-dsa-parameters"),
        (pem("EC PARAMETERS"), "pem-parameters", "application/x-x509-ec-parameters"),
    ]
}

#[test]
fn every_signature_retains_its_media_type() {
    for &(magic, format) in MAGICS {
        let name = format.name();
        let mut bytes = if name == "svg" { svg() } else { padded(magic, if name == "emf" { 44 } else { 32 }) };
        match name {
            // EMR_HEADER requires the additional signature at byte 40.
            "emf" => bytes[40..44].copy_from_slice(b" EMF"),
            // Excel signatures also require a supported document type: 0x0010 is a worksheet.
            "xls" => bytes[6] = 0x10,
            // The block type also requires byte-order magic at byte 8.
            "pcapng" => bytes[8..12].copy_from_slice(&0x1A2B_3C4Du32.to_be_bytes()),
            // FORM requires an AIFF or AIFC type at byte 8.
            "aiff" => bytes[8..12].copy_from_slice(b"AIFF"),
            // The shared signature requires an ftyp box with the JP2 brand.
            "jp2" => {
                bytes[12..16].copy_from_slice(&20u32.to_be_bytes());
                bytes[16..24].copy_from_slice(b"ftypjp2 ");
                bytes[28..32].copy_from_slice(b"jp2 ");
            }
            "dex" => bytes[..8].copy_from_slice(&dex("035", 8)),
            "vtt" => bytes[magic.len()] = b'\n',
            "dwf" => bytes[6..12].copy_from_slice(b"00.55)"),
            "ics" => bytes[magic.len()..magic.len() + 2].copy_from_slice(b"\r\n"),
            _ => {}
        }
        let encoded = std64(&bytes);
        assert_asset(&encoded, &bytes, name, format.media_type());
        assert_asset(&format!("{DATA_URI}{encoded}"), &bytes, name, format.media_type());
    }
}

#[test]
fn bare_payload_media_type() {
    for (bytes, format, media_type) in media_type_cases() {
        let prefix = "Payload: [";
        let encoded = std64(&bytes);
        let a = assert_asset(&format!("{prefix}{encoded}]"), &bytes, format, media_type);
        let expected = Span::new(prefix.len() as u32, (prefix.len() + encoded.len()) as u32);
        assert_eq!((a.span, a.payload), (expected, expected));
    }
}

#[test]
fn data_uri_media_type_inference() {
    for (bytes, format, media_type) in media_type_cases() {
        let text = format!("{DATA_URI}{}", std64(&bytes));
        let a = assert_asset(&text, &bytes, format, media_type);
        assert_eq!(a.span, Span::new(0, text.len() as u32));
        assert_eq!(a.payload, Span::new(DATA_URI.len() as u32, text.len() as u32));
    }
}

#[test]
fn data_uri_supplied_type_is_preserved() {
    let supplied = "application/octet-stream;charset=binary";
    for (bytes, format, media_type) in media_type_cases() {
        assert_ne!(supplied, media_type);
        assert_asset(&format!("data:{supplied};base64,{}", std64(&bytes)), &bytes, format, supplied);
    }
}

/// `AssetMediaTypeTest.riffHeader`: size 24 and a form of any length.
fn riff_header(form: &str) -> Vec<u8> {
    let mut b = b"RIFF".to_vec();
    b.extend_from_slice(&24u32.to_le_bytes());
    b.extend_from_slice(form.as_bytes());
    b.resize(32, 0);
    b
}

#[test]
fn riff_media_types() {
    for (form, format, media_type) in
        [("WEBP", "webp", "image/webp"), ("WAVE", "wav", "audio/wav"), ("AVI ", "avi", "video/x-msvideo")]
    {
        let header = riff_header(form);
        let encoded = std64(&header);
        assert_asset(&encoded, &header, format, media_type);
        assert_asset(&format!("{DATA_URI}{encoded}"), &header, format, media_type);
    }
}

#[test]
fn unknown_riff_type_is_rejected() {
    for form in ["NONE", "WEB", "WAV", "AVI", ""] {
        assert_unrecognized_default(&riff_header(form));
    }
}

// ---------------------------------------------------------------------------
// AssetTimestampedDataTest

const TSD_OID: &str = "2a864886f70d010910011f";

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len()).step_by(2).map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap()).collect()
}

/// `AssetTimestampedDataTest.header`: a SEQUENCE of the given length
/// holding the OID, in 80 bytes with two non-ASCII bytes at 70.
fn tsd(sequence_length: &str, oid_length: &str) -> Vec<u8> {
    let mut b = padded(&from_hex(&format!("30{sequence_length}06{oid_length}{TSD_OID}")), 80);
    b[70] = 0xFB;
    b[71] = 0xFF;
    b
}

fn assert_tsd(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "tsd", "application/timestamped-data", transport);
}

#[test]
fn tsd_length_and_transport() {
    let cases = [
        ("80", "0b"),
        ("0d", "0b"),
        ("7f", "0b"),
        ("8180", "0b"),
        ("81ff", "0b"),
        ("820100", "0b"),
        ("83010000", "0b"),
        ("84ffffffff", "0b"),
        ("850100000000", "0b"),
        ("8900000000000000000d", "0b"),
        ("80", "810b"),
        ("80", "82000b"),
        ("80", "8900000000000000000b"),
        ("82000f", "82000b"),
    ];
    for (sequence, oid) in cases {
        for transport in TRANSPORTS {
            assert_tsd(&tsd(sequence, oid), transport);
        }
    }
}

#[test]
fn tsd_different_identifier() {
    for offset in 4..15 {
        for bit in 0..8 {
            let mut bytes = tsd("80", "0b");
            bytes[offset] ^= 1 << bit;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn tsd_short_envelope() {
    for length in [0u8, 1, 2, 4, 8, 11, 12] {
        let value = format!("{length:02x}");
        assert_unrecognized_default(&tsd(&value, "0b"));
        assert_unrecognized_default(&tsd(&format!("81{value}"), "0b"));
        assert_unrecognized_default(&tsd(&format!("890000000000000000{value}"), "0b"));
    }
}

#[test]
fn tsd_envelope_excludes_identifier_length() {
    for length in ["0d", "0e", "810d", "82000e"] {
        assert_unrecognized_default(&tsd(length, "82000b"));
    }
}

#[test]
fn tsd_invalid_identifier_header() {
    for prefix in [
        "3180060b",
        "1080060b",
        "30ff060b",
        "30fe060b",
        "3080260b",
        "3080040b",
        "30800680",
        "308006ff",
        "308006fe",
        "3080060a",
        "3080060c",
        "308006810a",
        "3080068180",
        "30800685010000000b",
        "30800500060b",
        "003080060b",
    ] {
        assert_unrecognized_default(&padded(&from_hex(&format!("{prefix}{TSD_OID}")), 80));
    }
}

#[test]
fn tsd_truncated_identifier() {
    for length in 0..15 {
        assert_unrecognized_default(&tsd("80", "0b")[..length]);
    }
}

#[test]
fn tsd_truncated_long_header() {
    for prefix in ["308900000000000000000d060b", "3080068900000000000000000b", "3082000d060b"] {
        let bytes = from_hex(&format!("{prefix}{TSD_OID}"));
        for length in 0..bytes.len() {
            assert_unrecognized_default(&bytes[..length]);
        }
    }
}

#[test]
fn tsd_identifier_without_complete_file() {
    for length in [15, 16, 20, 23, 24] {
        assert_tsd(&tsd("80", "0b")[..length], "uri");
    }
}

#[test]
fn tsd_identifier_beyond_header_limit() {
    assert_unrecognized_default(&tsd("8a0000000000000000000d", "0b"));
    assert_unrecognized_default(&tsd("80", "8a0000000000000000000b"));
}

#[test]
fn tsd_declared_type() {
    for media_type in ["application/octet-stream", "application/pkcs7-mime", "application/timestamped-data"] {
        let mut bytes = tsd("80", "0b");
        bytes[14] = 2;
        let text = format!("data:{media_type};base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), &media_type[12..]);
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

// ---------------------------------------------------------------------------
// AssetFormatVariantsTest

/// `magic` zero-filled to 80 bytes.
fn variant(magic: &str) -> Vec<u8> {
    padded(&from_hex(magic), 80)
}

pub(super) fn be32_at(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_be_bytes());
}

fn le32_at(bytes: &mut [u8], at: usize, value: u32) {
    bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
}

const AXX_MAGIC: &str = "c0b9072e4f93f146a015792ca1d9e821";

#[test]
fn au_encoding() {
    for encoding in [1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 27] {
        let mut bytes = vec![0u8; 80];
        for (i, v) in [0x2E73_6E64, 28, 80 - 28, encoding, 8000, 1].into_iter().enumerate() {
            be32_at(&mut bytes, i * 4, v);
        }
        for transport in TRANSPORTS {
            assert_identified(&AssetDetector::new(), &bytes, "au", "audio/basic", transport);
        }
    }
}

/// The AU file Java Sound writes for 64 A-law samples at 8 kHz, mono:
/// the 24-byte header (magic, data offset, data size, encoding 27,
/// sample rate, channels) and the samples.
#[test]
fn java_sound_alaw() {
    let mut bytes = vec![0u8; 24];
    for (i, v) in [0x2E73_6E64, 24, 64, 27, 8000, 1].into_iter().enumerate() {
        be32_at(&mut bytes, i * 4, v);
    }
    bytes.extend_from_slice(&[0xD5; 64]);
    for transport in TRANSPORTS {
        assert_identified(&AssetDetector::new(), &bytes, "au", "audio/basic", transport);
    }
}

#[test]
fn las_version() {
    for minor in 0..=5 {
        let mut bytes = variant("4c415346");
        bytes[24] = 1;
        bytes[25] = minor;
        for transport in TRANSPORTS {
            assert_identified(&AssetDetector::new(), &bytes, "las", "application/x-asprs", transport);
        }
    }
}

#[test]
fn architecture_count() {
    for magic in ["cafebabf", "bebafeca", "bfbafeca"] {
        for count in [1u32, 2, 19, 20, 255, i32::MAX as u32, i32::MIN as u32, u32::MAX] {
            let mut bytes = variant(magic);
            if magic == "cafebabf" {
                be32_at(&mut bytes, 4, count);
            } else {
                le32_at(&mut bytes, 4, count);
            }
            for transport in TRANSPORTS {
                assert_identified(
                    &AssetDetector::new(),
                    &bytes,
                    "macho-fat",
                    "application/x-mach-o-universal",
                    transport,
                );
            }
        }
    }
}

#[test]
fn shared_java_magic() {
    for value in [1, 2, 20, 61, 255] {
        let mut bytes = variant("cafebabe");
        be32_at(&mut bytes, 4, value);
        for transport in TRANSPORTS {
            assert_identified(&AssetDetector::new(), &bytes, "class", "application/java-vm", transport);
        }
    }
}

#[test]
fn axcrypt_version() {
    for version in [1, 2, 3, 4] {
        let mut bytes = variant(AXX_MAGIC);
        le32_at(&mut bytes, 16, 21);
        bytes[20] = 2;
        le32_at(&mut bytes, 37, 10);
        bytes[41] = 3;
        bytes[42] = version;
        for transport in TRANSPORTS {
            assert_identified(&AssetDetector::new(), &bytes, "axx", "application/x-axcrypt", transport);
        }
    }
}

#[test]
fn variant_changed_or_truncated_signature() {
    for magic in ["2e736e64", "4c415346", AXX_MAGIC, "cafebabf", "bebafeca", "bfbafeca"] {
        for offset in 0..magic.len() / 2 {
            let mut bytes = variant(magic);
            bytes[offset] = b'?';
            assert_unrecognized_default(&bytes);
            assert_unrecognized_default(&from_hex(magic)[..offset]);
        }
    }
}
