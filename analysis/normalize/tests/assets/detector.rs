//! OpenNLP's `CursorAssetDetectorTest`: what the detector finds in a text
//! and what it passes over.

use super::*;

// ---------------------------------------------------------------------------
// CursorAssetDetectorTest

#[test]
fn bare_png_run_with_dimensions() {
    let bytes = png(640, 480);
    let encoded = std64(&bytes);
    let text = format!("before {encoded} after");
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Png);
    assert_eq!(a.format_name(&text), "png");
    assert_eq!(a.media_type(&text), "image/png");
    assert_eq!((a.width, a.height), (Some(640), Some(480)));
    assert_eq!(a.decoded_len, bytes.len() as u64);
    assert_eq!(covered(&text, a.span), encoded);
    assert_eq!(a.decode(&text).unwrap(), bytes);
    assert_eq!(a.encoding, AssetEncoding::Base64);
}

#[test]
fn mime_wrapped_bare_payload() {
    let bytes = cat(&png(640, 480), &[0; 128]);
    let encoded = wrapped(&bytes, 76);
    let text = embed(&encoded);
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Png);
    assert_eq!(covered(&text, a.span), encoded);
    assert_eq!(covered(&text, a.payload), encoded);
    assert_eq!(a.decoded_len, bytes.len() as u64);
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

#[test]
fn bare_base64url_payload() {
    let bytes = cat(&png(8, 9), &[0xFB, 0xFF, 0xFF]);
    let encoded = url64(&bytes);
    assert!(encoded.contains(['-', '_']));
    let text = embed(&encoded);
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Png);
    assert_eq!(a.encoding, AssetEncoding::Base64Url);
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

#[test]
fn pem_certificate_envelope() {
    let mut certificate = vec![0u8; 160];
    certificate[..4].copy_from_slice(&[0x30, 0x82, 0x01, 0x20]);
    let body = wrapped(&certificate, 64);
    let pem = format!("-----BEGIN CERTIFICATE-----\r\n{body}\r\n-----END CERTIFICATE-----");
    let text = format!("certificate:\n{pem}\nend");
    let a = one(&text);
    assert_eq!(a.format_name(&text), "pem-cert");
    assert_eq!(a.media_type(&text), "application/x-x509-cert");
    assert_eq!(covered(&text, a.span), pem);
    assert_eq!(covered(&text, a.payload), body);
    assert_eq!(a.decoded_len, 160);
    assert_eq!(a.decode(&text).unwrap(), certificate);
}

const PEM_TYPES: [(&str, &str, &str); 10] = [
    ("CERTIFICATE", "pem-cert", "application/x-x509-cert"),
    ("X509 CRL", "pem-crl", "application/pkix-crl"),
    ("CERTIFICATE REQUEST", "pem-request", "application/pkcs10"),
    ("PKCS7", "pem-pkcs7", "application/pkcs7-signature"),
    ("CMS", "pem-cms", "application/pkcs7-mime"),
    ("PRIVATE KEY", "pem-key", "application/x-x509-key"),
    ("ENCRYPTED PRIVATE KEY", "pem-key", "application/x-x509-key"),
    ("PUBLIC KEY", "pem-key", "application/x-x509-key"),
    ("ATTRIBUTE CERTIFICATE", "pem-attribute-cert", "application/x-x509-attribute-cert"),
    ("EC PARAMETERS", "pem-parameters", "application/x-x509-ec-parameters"),
];

#[test]
fn pem_type_mappings() {
    for (label, format, media_type) in PEM_TYPES {
        let content = [0u8; 96];
        let body = wrapped(&content, 64).replace("\r\n", "\n");
        let pem = format!("-----BEGIN {label}-----\n{body}\n-----END {label}-----");
        let a = one(&pem);
        assert_eq!(a.format_name(&pem), format, "{label}");
        assert_eq!(a.media_type(&pem), media_type, "{label}");
        assert_eq!(a.decode(&pem).unwrap(), content);
    }
}

#[test]
fn json_web_token() {
    let claims = br#"{"sub":"1234567890","admin":true}"#;
    let (h, c, s) = (url64(br#"{"alg":"HS256","typ":"JWT"}"#), url64(claims), url64(&[0; 32]));
    let token = format!("{h}.{c}.{s}");
    let text = format!("token={token};");
    let a = one(&text);
    assert_eq!(a.format_name(&text), "jwt");
    assert_eq!(a.media_type(&text), "application/jwt");
    assert_eq!(covered(&text, a.span), token);
    assert_eq!(covered(&text, a.payload), c);
    assert_eq!(a.decoded_len, claims.len() as u64);
    assert_eq!(a.decode(&text).unwrap(), claims);
}

#[test]
fn unsecured_json_web_token() {
    let token = format!("{}.{}.", url64(br#"{"alg":"none"}"#), url64(br#"{"sub":"123"}"#));
    let a = one(&token);
    assert_eq!(covered(&token, a.span), token);
    assert_eq!(a.format, AssetFormat::Jwt);
}

#[test]
fn json_web_token_near_misses() {
    let header = url64(br#"{"typ":"JWT"}"#);
    let valid = url64(br#"{"alg":"HS256"}"#);
    let claims = url64(br#"{"sub":"123"}"#);
    let not_json = url64(b"plain text");
    let signature = url64(&[0; 32]);
    let none = url64(br#"{"alg":"none"}"#);
    let malformed = url64(br#"{"alg":"HS256",}"#);
    let empty_alg = url64(br#"{"alg":""}"#);
    for text in [
        format!("{header}.{claims}.{signature}"),
        format!("{valid}.{not_json}.{signature}"),
        format!("{valid}.{claims}"),
        format!("{valid}.{claims}."),
        format!("{none}.{claims}.{signature}"),
        format!("{malformed}.{claims}.{signature}"),
        format!("{empty_alg}.{claims}.{signature}"),
        format!("x{valid}.{claims}.{signature}"),
    ] {
        assert!(detect(&text).is_empty(), "{text}");
    }
}

#[test]
fn pem_envelope_near_misses() {
    let body = wrapped(&[0; 96], 64);
    assert!(detect(&format!("-----BEGIN CERTIFICATE-----\r\n{body}\r\n-----END PUBLIC KEY-----")).is_empty());
    assert!(detect(&format!("-----BEGIN CERTIFICATE-----\r\n{body}!\r\n-----END CERTIFICATE-----")).is_empty());
}

#[test]
fn data_uri_declares_the_media_type() {
    let bytes = png(2, 3);
    let uri = format!("data:image/png;base64,{}", std64(&bytes));
    let text = format!("<img src=\"{uri}\">");
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Png);
    assert_eq!(a.media_type(&text), "image/png");
    assert_eq!(covered(&text, a.span), uri);
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

#[test]
fn data_uri_with_unknown_header_keeps_the_declared_format() {
    let uri = format!("data:text/plain;base64,{}", std64(b"just some plain bytes padded out"));
    let text = embed(&uri);
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Declared);
    assert_eq!(a.format_name(&text), "plain");
    assert_eq!(a.media_type(&text), "text/plain");
}

#[test]
fn data_uri_with_a_malformed_media_type_declares_nothing() {
    // A declared type needs a type and subtype that are tokens, as the layer
    // asks: the sniffed format stands in for one that is not, and with none
    // the URI is not an asset.
    for media_type in ["text/", "text/;charset=utf-8", "/plain", "plain", "text//plain", "te(x)t/plain"] {
        let text = format!("data:{media_type};base64,{}", std64(&png(1, 1)));
        let layer = check_layer(&AssetDetector::new(), &text);
        assert_eq!((layer.len(), layer.format[0], layer.declared_type.get(0)), (1, AssetFormat::Png, ""));
        assert_eq!(one(&text).media_type(&text), "image/png");
        let plain = format!("data:{media_type};base64,{}", std64(b"just some plain bytes padded out"));
        assert!(detect(&plain).is_empty(), "{media_type}");
    }
}

#[test]
fn gif_jpeg_riff_pdf_zip_all_identify() {
    let runs = [
        std64(&padded(&gif(12, 34), 30)),
        std64(&padded(&[0xFF, 0xD8, 0xFF, 0xE0], 30)),
        std64(&riff(b"WEBP")),
        std64(&riff(b"WAVE")),
        std64(&riff(b"AVI ")),
        std64(&padded(b"%PDF-1.7 stub content", 30)),
        std64(&padded(b"PK\x03\x04", 30)),
    ];
    let text = runs.join("\n");
    let found = detect(&text);
    let formats: Vec<AssetFormat> = found.iter().map(|a| a.format).collect();
    use AssetFormat::*;
    assert_eq!(formats, [Gif, Jpeg, Webp, Wav, Avi, Pdf, Zip]);
    assert_eq!((found[0].width, found[0].height), (Some(12), Some(34)));
    assert_eq!(found[1].width, None);
    assert_eq!(found[4].media_type(&text), "video/x-msvideo");
}

/// `CursorAssetDetectorTest.knownFormatHeaders`.
const KNOWN_HEADERS: [(&str, &str, &[u8]); 28] = [
    ("tiff", "image/tiff", b"II\x2A\x00"),
    ("tiff", "image/tiff", b"MM\x00\x2A"),
    ("gzip", "application/gzip", b"\x1F\x8B\x08"),
    ("7z", "application/x-7z-compressed", b"7z\xBC\xAF\x27\x1C"),
    ("rar", "application/vnd.rar", b"Rar!\x1A\x07\x00"),
    ("flac", "audio/flac", b"fLaC"),
    ("ogg", "application/ogg", b"OggS"),
    ("midi", "audio/midi", b"MThd"),
    ("sqlite", "application/vnd.sqlite3", b"SQLite format 3\0"),
    ("elf", "application/x-elf", b"\x7FELF\x02\x01\x01"),
    ("pe", "application/vnd.microsoft.portable-executable", b"MZ\x90\x00"),
    ("class", "application/java-vm", b"\xCA\xFE\xBA\xBE\0\0\0\x37"),
    ("woff", "font/woff", b"wOFF"),
    ("woff2", "font/woff2", b"wOF2"),
    ("mp3", "audio/mpeg", b"ID3\x03\0\0"),
    ("ole2", "application/x-ole-storage", b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1"),
    ("zstd", "application/zstd", b"\x28\xB5\x2F\xFD"),
    ("wasm", "application/wasm", b"\0asm\x01\0\0\0"),
    ("rtf", "application/rtf", b"{\\rtf1\\ansi minimal"),
    ("xz", "application/x-xz", b"\xFD7zXZ\0"),
    ("matroska", "application/x-matroska", b"\x1A\x45\xDF\xA3"),
    ("psd", "image/vnd.adobe.photoshop", b"8BPS\0\x01"),
    ("ico", "image/vnd.microsoft.icon", b"\0\0\x01\0"),
    ("macho", "application/x-mach-o", b"\xFE\xED\xFA\xCE"),
    ("pem-cert", "application/x-x509-cert", b"-----BEGIN CERTIFICATE-----"),
    ("torrent", "application/x-bittorrent", b"d8:announce27:http"),
    ("php", "text/x-php", b"<?php echo 'shell';"),
    ("parquet", "application/x-parquet", b"PAR1"),
];

#[test]
fn expanded_magic_table_identifies() {
    for (format, media_type, header) in KNOWN_HEADERS {
        let encoded = std64(&padded(header, 30));
        let text = embed(&encoded);
        let a = one(&text);
        assert_eq!(a.format_name(&text), format, "{header:?}");
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(covered(&text, a.span), encoded);
    }
}

#[test]
fn tiff_prefix_with_broken_magic_byte_is_ignored() {
    let lookalike = std64(&padded(b"II\x2A\x01", 30));
    assert!(lookalike.starts_with("SUkqA"));
    assert!(detect(&embed(&lookalike)).is_empty());
}

#[test]
fn pe_prefix_with_broken_stub_byte_is_ignored() {
    let lookalike = std64(&padded(b"MZ\x90\x01", 30));
    assert!(lookalike.starts_with("TVqQA"));
    assert!(detect(&embed(&lookalike)).is_empty());
}

#[test]
fn short_runs_are_ignored() {
    assert!(detect("GET /9j/abc/thumbnail HTTP/1.1").is_empty());
}

#[test]
fn long_run_without_magic_is_ignored() {
    let encoded = std64(b"this is just text encoded as base64 for transport");
    assert!(detect(&format!("payload: {encoded}")).is_empty());
}

#[test]
fn magic_prefix_without_verifying_header_is_ignored() {
    let lookalike = std64(&riff(b"JUNK"));
    assert!(lookalike.starts_with("UklGR"));
    assert!(detect(&embed(&lookalike)).is_empty());
}

#[test]
fn plain_prose_yields_nothing() {
    assert!(
        detect(
            "Plain prose with ordinary words does not contain payloads, \
             not even AlphanumericRunsLikeThisOne0123456789 of some length."
        )
        .is_empty()
    );
}

#[test]
fn single_character_run_makes_progress() {
    assert!(detect("x").is_empty());
}

#[test]
fn truncated_payload_stays_decodable() {
    let encoded = base64(&padded(&png(1, 1), 30), STANDARD, false);
    // Cut so the run length is 1 mod 4, the length no base64 payload can end at.
    let cut = (encoded.len() - 4) / 4 * 4 + 1;
    let truncated = &encoded[..cut];
    let a = one(truncated);
    assert_eq!(a.payload.end as usize, cut - 1);
    assert!(!a.decode(truncated).unwrap().is_empty());
}

#[test]
fn declared_gif_with_png_payload_sniffs_png() {
    let text = embed(&format!("data:image/gif;base64,{}", std64(&png(9, 4))));
    let a = one(&text);
    assert_eq!(a.format, AssetFormat::Png);
    assert_eq!(a.media_type(&text), "image/gif");
    // The dimensions are read the sniffed format's way, from the PNG header.
    assert_eq!((a.width, a.height), (Some(9), Some(4)));
}

#[test]
fn payload_starting_at_index_zero() {
    let encoded = std64(&png(1, 1));
    let a = one(&format!("{encoded} trails"));
    assert_eq!(a.span, Span::new(0, encoded.len() as u32));
}

#[test]
fn data_uri_ending_at_the_end_of_the_text() {
    let text = format!("logo: data:image/png;base64,{}", std64(&png(1, 1)));
    assert_eq!(one(&text).span, Span::new(6, text.len() as u32));
}

#[test]
fn empty_text_yields_nothing() {
    assert!(detect("").is_empty());
}

#[test]
fn adjacent_data_uris_are_both_detected() {
    let bytes = cat(&png(1, 1), &[0]);
    let uri = format!("data:image/png;base64,{}", std64(&bytes));
    assert!(uri.ends_with("=="));
    let text = format!("{uri}{uri}");
    let found = detect(&text);
    assert_eq!(found.len(), 2);
    let n = uri.len() as u32;
    assert_eq!((found[0].span, found[1].span), (Span::new(0, n), Span::new(n, 2 * n)));
    assert_eq!(found[0].decode(&text).unwrap(), bytes);
    assert_eq!(found[1].decode(&text).unwrap(), bytes);
}

#[test]
fn data_uri_media_type_stops_at_unicode_whitespace() {
    for space in [" ", "\u{a0}", "\u{2003}", "\u{3000}", "\u{2028}"] {
        let text = format!("data:image/png{space}note;base64,{}", std64(&png(1, 1)));
        for a in detect(&text) {
            assert!(a.span.start > 0, "not a data URI: {a:?}");
            assert!(!a.media_type(&text).contains(space));
        }
    }
}

// testRejectsNull of the Java classes has no counterpart: a `&str` is never
// null, and the detector, fold and annotator take no nullable argument.
