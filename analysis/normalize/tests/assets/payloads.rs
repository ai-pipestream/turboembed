//! Image metadata, transport validation, and the EMF, PEM and MPX
//! signatures.

use super::*;

// ---------------------------------------------------------------------------
// AssetImageMetadataTest

#[test]
fn invalid_metadata_does_not_discard_asset() {
    let mut wrong_type = png(5, 7);
    wrong_type[12] = b'X';
    let mut wrong_length = png(5, 7);
    wrong_length[11] = 12;
    let mut wrong_png_signature = png(5, 7);
    wrong_png_signature[0] = 0;
    let mut wrong_gif_signature = gif(5, 7);
    wrong_gif_signature[0] = b'X';
    let cases: [(&str, &str, Vec<u8>); 13] = [
        ("PNG zero width", "png", png(0, 7)),
        ("PNG zero height", "png", png(5, 0)),
        ("PNG unsigned width overflow", "png", png(i32::MIN, 7)),
        ("PNG unsigned height overflow", "png", png(5, i32::MIN)),
        ("PNG largest unsigned width", "png", png(-1, 7)),
        ("PNG wrong first chunk", "png", wrong_type),
        ("PNG wrong IHDR length", "png", wrong_length),
        ("PNG declared without its signature", "png", wrong_png_signature),
        ("PNG missing height bytes", "png", png(5, 7)[..23].to_vec()),
        ("GIF zero width", "gif", gif(0, 7)),
        ("GIF zero height", "gif", gif(5, 0)),
        ("GIF declared without its signature", "gif", wrong_gif_signature),
        ("GIF missing height bytes", "gif", gif(5, 7)[..9].to_vec()),
    ];
    for (label, format, bytes) in cases {
        let text = format!("data:image/{format};base64,{}", std64(&bytes));
        let found = detect(&text);
        assert_eq!(found.len(), 1, "{label}");
        let a = found[0];
        assert_eq!(a.format_name(&text), format, "{label}");
        assert_eq!((a.width, a.height), (None, None), "{label}");
        assert_eq!(a.decoded_len, bytes.len() as u64);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn png_dimension_limits() {
    for d in [1, i32::MAX] {
        let a = one(&std64(&png(d, d)));
        assert_eq!((a.width, a.height), (Some(d as u32), Some(d as u32)));
    }
}

#[test]
fn gif_dimension_limits() {
    for d in [1, 65535] {
        let a = one(&std64(&gif(d, d)));
        assert_eq!((a.width, a.height), (Some(u32::from(d)), Some(u32::from(d))));
    }
}

// ---------------------------------------------------------------------------
// AssetTransportValidationTest

fn transport_pdf() -> String {
    std64(format!("%PDF-1.7{}", "a".repeat(40)).as_bytes())
}

fn jwt_parts() -> (String, String) {
    (url64(br#"{"alg":"HS256"}"#), url64(br#"{"sub":"123"}"#))
}

#[test]
fn malformed_bare_payload_is_not_reported() {
    let pdf = transport_pdf();
    for (name, suffix) in [
        ("padding after a complete group", "="),
        ("padding count inconsistent with length", "AA="),
        ("mixed alphabets", "+_AA"),
        ("wrapped URL alphabet", "\r\nAA-_"),
    ] {
        assert!(detect(&format!("{pdf}{suffix}")).is_empty(), "{name}");
    }
}

#[test]
fn malformed_data_payload_is_not_reported() {
    for suffix in ["=", "AA=", "+_AA"] {
        assert!(detect(&format!("data:application/pdf;base64,{}{suffix}", transport_pdf())).is_empty(), "{suffix}");
    }
}

#[test]
fn valid_payload_suffixes_decode_with_accurate_length() {
    for suffix in ["", "AA", "AAA", "AA==", "AAA=", "+/AA", "-_AA"] {
        let encoded = format!("{}{suffix}", transport_pdf());
        let expected = unbase64(&encoded);
        let a = one(&encoded);
        assert_eq!(a.payload.end as usize, encoded.len(), "{suffix}");
        assert_eq!(a.decoded_len, expected.len() as u64, "{suffix}");
        assert_eq!(a.decode(&encoded).unwrap(), expected);
    }
}

#[test]
fn short_data_payloads_are_detected() {
    for value in ["A", "AB", "ABC"] {
        let text = format!("data:text/plain;base64,{}", std64(value.as_bytes()));
        let a = one(&text);
        assert_eq!(a.format_name(&text), "plain");
        assert_eq!(a.decoded_len, value.len() as u64);
        assert_eq!(a.decode(&text).unwrap(), value.as_bytes());
    }
}

#[test]
fn data_media_type_parameters_are_not_part_of_the_format() {
    let text = "data:text/plain;charset=UTF-8;base64,SGVsbG8=";
    let a = one(text);
    assert_eq!(a.format_name(text), "plain");
    assert_eq!(a.media_type(text), "text/plain;charset=UTF-8");
}

#[test]
fn malformed_jwt_signature_encoding_is_rejected() {
    let (h, c) = jwt_parts();
    for signature in ["A", "AAAAA", "AAAA=", "AAAA=="] {
        let text = format!("{h}.{c}.{signature}");
        assert!(detect(&text).iter().all(|a| a.format != AssetFormat::Jwt), "{signature}");
    }
}

#[test]
fn unpadded_jwt_signature_lengths_are_accepted() {
    let (h, c) = jwt_parts();
    for signature in ["AA", "AAA", "AAAA"] {
        let text = format!("{h}.{c}.{signature}");
        let a = one(&text);
        assert_eq!(a.format, AssetFormat::Jwt);
        assert_eq!(a.span.end as usize, text.len());
    }
}

// ---------------------------------------------------------------------------
// AssetEmfSignatureTest

/// `AssetEmfSignatureTest.header`: record type 1 and size 88 (little
/// endian), 0xFB filler to byte 40, `signature` at `offset`.
fn emf(length: usize, offset: usize, signature: &[u8]) -> Vec<u8> {
    let mut b = vec![0u8; length];
    b[..4].copy_from_slice(&1u32.to_le_bytes());
    b[4..8].copy_from_slice(&88u32.to_le_bytes());
    b[8..40].fill(0xFB);
    b[offset..offset + signature.len()].copy_from_slice(signature);
    b
}

#[test]
fn emf_signature_beyond_initial_header() {
    let cases = [
        (44, "standard"),
        (45, "standard"),
        (46, "standard"),
        (88, "standard"),
        (44, "url"),
        (45, "url"),
        (46, "url"),
        (88, "url"),
        (88, "mime64"),
        (88, "mime76"),
        (44, "uri"),
        (45, "uri"),
        (46, "uri"),
        (88, "uri"),
    ];
    for (length, transport) in cases {
        let bytes = emf(length, 40, b" EMF");
        let text = encode(&bytes, transport);
        let a = one(&text);
        assert_eq!((a.format_name(&text), a.media_type(&text)), ("emf", "image/emf"), "{length} {transport}");
        assert_eq!(a.decoded_len, length as u64);
        assert_eq!(a.span, Span::new(0, text.len() as u32));
        let start = if transport == "uri" { DATA_URI.len() as u32 } else { 0 };
        assert_eq!(a.payload, Span::new(start, text.len() as u32));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn emf_truncated_signature_is_rejected() {
    for length in [0, 1, 3, 4, 23, 24, 39, 40, 41, 42, 43] {
        assert_unrecognized_default(&emf(88, 40, b" EMF")[..length]);
    }
}

#[test]
fn emf_incorrect_signature_is_rejected() {
    for signature in [&b""[..], b"EMF", b"FME ", b" EMX", b" emf", b"\0EMF", b" EM"] {
        assert_unrecognized_default(&emf(88, 40, signature));
    }
}

#[test]
fn emf_incorrect_signature_offset_is_rejected() {
    for offset in [8, 24, 36, 39, 41, 44, 80] {
        assert_unrecognized_default(&emf(88, offset, b" EMF"));
    }
}

#[test]
fn emf_incorrect_record_type_is_rejected() {
    for record in [0u32, 2, 0x0100_0001, 0x0100_0000, u32::MAX] {
        let mut bytes = emf(88, 40, b" EMF");
        bytes[..4].copy_from_slice(&record.to_le_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn emf_explicit_media_type_without_signature() {
    let bytes = emf(88, 40, b"");
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!(a.format_name(&text), "octet-stream");
    assert_eq!(a.media_type(&text), "application/octet-stream");
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetPemSignatureTest: a PEM envelope that is itself base64 encoded.

const PEM_HEADERS: [(&str, &str, &str); 11] = [
    ("CERTIFICATE", "pem-cert", "application/x-x509-cert"),
    ("DSA PARAMETERS", "pem-parameters", "application/x-x509-dsa-parameters"),
    ("EC PARAMETERS", "pem-parameters", "application/x-x509-ec-parameters"),
    ("DSA PRIVATE KEY", "pem-key", "application/x-x509-key"),
    ("RSA PRIVATE KEY", "pem-key", "application/x-x509-key"),
    ("PRIVATE KEY", "pem-key", "application/x-x509-key"),
    ("PUBLIC KEY", "pem-key", "application/x-x509-key"),
    ("DSA KEY", "pem-key", "application/x-x509-key"),
    ("RSA KEY", "pem-key", "application/x-x509-key"),
    ("KEY", "pem-key", "application/x-x509-key"),
    ("PKCS7", "p7s", "application/pkcs7-signature"),
];

fn opening(label: &str) -> Vec<u8> {
    format!("-----BEGIN {label}-----").into_bytes()
}

#[test]
fn pem_encoded_envelope() {
    for (label, format, media_type) in PEM_HEADERS {
        assert_identified_everywhere(&pem(label), format, media_type);
    }
}

#[test]
fn pem_truncated_marker() {
    for (label, _, _) in PEM_HEADERS {
        let open = opening(label);
        for length in 0..open.len() {
            assert_unrecognized_default(&open[..length]);
        }
    }
}

#[test]
fn pem_changed_marker() {
    for (label, _, _) in PEM_HEADERS {
        for offset in 0..opening(label).len() {
            let mut bytes = pem(label);
            bytes[offset] = b'?';
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn pem_opening_without_envelope() {
    for (label, format, media_type) in PEM_HEADERS {
        assert_identified(&AssetDetector::new(), &opening(label), format, media_type, "uri");
    }
}

#[test]
fn pem_similar_label() {
    for label in [
        "DSA PARAMETER",
        "DSA PARAMETERS NOTES",
        "EC PARAMETERS NOTES",
        "DSA PRIVATE KITCHEN",
        "RSA PRIVATE KITCHEN",
        "PKCS70",
        "PKCS7 NOTES",
        "CERTIFICATE--notes",
        "PRIVATE KEY--notes",
        "PUBLIC KEY---notes",
    ] {
        assert_unrecognized_default(&pem(label));
    }
}

#[test]
fn pem_declared_type() {
    for media_type in ["application/octet-stream", "application/x-x509-cert"] {
        let bytes = b"-----BEGIN CERTIFICATE--notes about a certificate";
        let text = format!("data:{media_type};base64,{}", std64(bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), &media_type[media_type.find('/').unwrap() + 1..]);
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

// ---------------------------------------------------------------------------
// AssetMpxSignatureTest

fn project(program: &str, separator: char) -> Vec<u8> {
    let s = separator;
    format!("MPX{s}{program}{s}4.0{s}ANSI\r\n0{s}Survey schedule\r\n0{s}??????~~~~~~\r\n").into_bytes()
}

#[test]
fn mpx_producer_and_transport() {
    for program in [
        "Microsoft Project for Windows",
        "Microsoft Project",
        "Primavera Systems",
        "Fieldwork Planner",
        "\tFieldwork Planner\t",
        "\"Survey, task planner\"",
        "\"Survey; task planner\"",
    ] {
        for separator in [',', ';'] {
            let bytes = project(program, separator);
            let standard = std64(&bytes);
            assert!(standard.contains('+') && standard.contains('/'));
            assert_identified_everywhere(&bytes, "mpx", "application/x-project");
        }
    }
}

#[test]
fn mpx_separator() {
    for separator in [0u8, b'\t', b'\n', b'\r', b' ', b'"', b',', b';', b':', b'|', b'0', b'A', 0x7F, 0x80, 0xFF] {
        let mut bytes = project("Fieldwork Planner", ',');
        bytes[3] = separator;
        if separator == b',' || separator == b';' {
            for transport in ["standard", "uri"] {
                assert_identified(&AssetDetector::new(), &bytes, "mpx", "application/x-project", transport);
            }
        } else {
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn mpx_changed_identifier() {
    for offset in 0..3 {
        for value in [0u8, b'x', 0x80, 0xFF] {
            let mut bytes = project("Fieldwork Planner", ',');
            bytes[offset] = value;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn mpx_truncated_identifier() {
    for length in 0..4 {
        assert_unrecognized_default(&project("Fieldwork Planner", ',')[..length]);
    }
}

#[test]
fn mpx_identifier_without_project() {
    for separator in [',', ';'] {
        let bytes = &project("Fieldwork Planner", separator)[..4];
        assert_identified(&AssetDetector::new(), bytes, "mpx", "application/x-project", "uri");
    }
}

#[test]
fn mpx_supplied_type() {
    for separator in ['|', ':'] {
        let bytes = project("Fieldwork Planner", separator);
        let text = format!("data:application/x-project;base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), "x-project");
        assert_eq!(a.media_type(&text), "application/x-project");
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn mpx_manual_example() {
    let project = "MPX;Fieldwork Planner;4.0;ANSI\r\n0;Survey schedule\r\n";
    let text = format!("{DATA_URI}{}", std64(project.as_bytes()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("mpx", "application/x-project"));
    assert_eq!(a.decode(&text).unwrap(), project.as_bytes());
}
