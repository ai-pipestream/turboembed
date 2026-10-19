//! The embedded-asset detector and fold. The expectations of OpenNLP's 29
//! embedded-assets test classes come first, class by class, with OpenNLP's
//! UTF-16 offsets written as UTF-8 byte offsets (equal for the ASCII texts
//! they use). Then false positives with every addition on, UTF-8 offsets,
//! the three additions, the fold's alignment, and property loops over
//! generated text. Every detection goes through `Layer::check` and a
//! document (`assets_support::detect_with`), and every fold through the
//! alignment check of `assets_support::fold_checked`.

mod assets_support;

use assets_support::*;
use turbo_analysis::{Analyzer, Annotator, Bound, Document, Key, Layer, Span, View};
use turbo_normalize::assets::{DecodeError, MAGICS, magic_format};
use turbo_normalize::{Asset, AssetDetector, AssetFold, Chain, NormalizeStep, Normalizer, hash_config, normalize};
use turbo_normalize_layers::{AssetEncoding, AssetFormat, Assets, PlaceholderKind, Placeholders};

/// `CursorAssetDetectorTest.embed`.
fn embed(run: &str) -> String {
    format!("x {run} y")
}

/// A RIFF header of form `kind` with 20 zero bytes after it.
fn riff(kind: &[u8; 4]) -> Vec<u8> {
    let mut b = vec![b'R', b'I', b'F', b'F', 100, 0, 0, 0];
    b.extend_from_slice(kind);
    b.resize(32, 0);
    b
}

/// CRLF MIME wrapping at `width`, `CursorAssetDetectorTest.wrapped`.
fn wrapped(bytes: &[u8], width: usize) -> String {
    mime(bytes, width, "\r\n")
}

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

// ---------------------------------------------------------------------------
// EmbeddedAssetTest. The record checks of `EmbeddedAsset`'s constructor
// are `Assets::check` here; null and blank strings cannot occur (the
// format is an enum) and a length cannot be negative (`u64`).

/// `EmbeddedAssetTest.valid`: span 4..12, payload 8..12.
fn valid() -> Asset {
    Asset {
        span: Span::new(4, 12),
        payload: Span::new(8, 12),
        format: AssetFormat::Png,
        declared: None,
        encoding: AssetEncoding::Base64,
        decoded_len: 3,
        width: Some(5),
        height: Some(7),
    }
}

/// A one-row layer from `asset` and a declared type.
fn layer_of(asset: Asset, declared: &str) -> Assets {
    let mut layer = Assets::new(turbo_analysis::ViewId::SOURCE);
    layer.spans.push(asset.span);
    layer.payload.push(asset.payload);
    layer.format.push(asset.format);
    layer.declared_type.push(declared);
    layer.encoding.push(asset.encoding);
    layer.decoded_len.push(asset.decoded_len);
    layer.width.push(asset.width);
    layer.height.push(asset.height);
    layer
}

#[test]
fn layer_check_rejects_what_the_record_rejects() {
    assert!(layer_of(valid(), "").check().is_ok());
    for payload in [Span::new(3, 12), Span::new(4, 13), Span::new(8, 8)] {
        assert!(layer_of(Asset { payload, ..valid() }, "").check().is_err(), "{payload:?}");
    }
    for d in [Some(0), Some(i32::MAX as u32 + 1), Some(u32::MAX)] {
        assert!(layer_of(Asset { width: d, ..valid() }, "").check().is_err(), "{d:?}");
        assert!(layer_of(Asset { height: d, ..valid() }, "").check().is_err(), "{d:?}");
    }
    for d in [1, i32::MAX as u32] {
        assert!(layer_of(Asset { width: Some(d), height: Some(d), ..valid() }, "").check().is_ok());
    }
    assert!(layer_of(Asset { width: None, height: None, ..valid() }, "").check().is_ok());
    // OpenNLP's record allows one dimension without the other; its
    // detector never writes one, and the layer does not accept one.
    assert!(layer_of(Asset { height: None, ..valid() }, "").check().is_err());
    // A declared format needs a declared type with a slash after its first
    // character, and a declared type must have one.
    assert!(layer_of(Asset { format: AssetFormat::Declared, ..valid() }, "").check().is_err());
    assert!(layer_of(Asset { format: AssetFormat::Declared, ..valid() }, "text/plain").check().is_ok());
    assert!(layer_of(valid(), "/plain").check().is_err());
    assert!(layer_of(valid(), "plain").check().is_err());
    let mut short = layer_of(valid(), "");
    short.decoded_len.clear();
    assert!(short.check().is_err());
}

#[test]
fn layer_check_rejects_overlapping_rows() {
    let mut layer = layer_of(valid(), "");
    let second = layer_of(Asset { span: Span::new(10, 14), payload: Span::new(10, 14), ..valid() }, "");
    layer.spans.push(Span::new(10, 14));
    layer.payload.push(second.payload[0]);
    layer.format.push(AssetFormat::Png);
    layer.declared_type.push("");
    layer.encoding.push(AssetEncoding::Base64);
    layer.decoded_len.push(3);
    layer.width.push(None);
    layer.height.push(None);
    assert!(layer.check().is_err());
}

#[test]
fn accessors_round_trip() {
    let text = "padding QUJD";
    let a = valid();
    assert_eq!((a.span, a.payload), (Span::new(4, 12), Span::new(8, 12)));
    assert_eq!((a.format_name(text), a.media_type(text)), ("png", "image/png"));
    assert_eq!((a.decoded_len, a.width, a.height), (3, Some(5), Some(7)));
}

#[test]
fn decode_reads_the_payload_span_from_the_text() {
    // Payload characters at 8..12 are the base64 image of "ABC".
    assert_eq!(valid().decode("padding QUJD").unwrap(), b"ABC");
}

#[test]
fn decode_rejects_text_shorter_than_the_payload_span() {
    assert_eq!(valid().decode("short"), Err(DecodeError::TextTooShort));
    assert_eq!(DecodeError::TextTooShort.to_string(), "text is shorter than the payload span");
}

/// An asset whose span and payload are the whole of `encoded`.
fn whole(encoded: &str, encoding: AssetEncoding) -> Asset {
    let span = Span::new(0, encoded.len() as u32);
    Asset { span, payload: span, format: AssetFormat::Unknown, encoding, width: None, height: None, ..valid() }
}

#[test]
fn decode_accepts_wrapped_standard_base64() {
    let encoded = mime(b"ABCDEF", 4, "\r\n");
    assert_eq!(whole(&encoded, AssetEncoding::Base64).decode(&encoded).unwrap(), b"ABCDEF");
}

#[test]
fn decode_accepts_unpadded_base64url() {
    let encoded = url64(&[0xFB, 0xFF, 0xFF]);
    assert_eq!(whole(&encoded, AssetEncoding::Base64Url).decode(&encoded).unwrap(), [0xFB, 0xFF, 0xFF]);
}

#[test]
fn decode_rejects_mixed_base64_alphabets() {
    let err = whole("+_AA", AssetEncoding::Base64).decode("+_AA").unwrap_err();
    assert_eq!(err, DecodeError::MixedAlphabets);
    assert_eq!(err.to_string(), "payload mixes base64 and base64url alphabets");
}

// ---------------------------------------------------------------------------
// AssetFolderTest. OpenNLP folds a document that already holds the layer;
// `AssetFold` detects as it folds, so a missing layer or out-of-order rows
// cannot reach it, and its captions cannot be null.

fn annotated_text() -> String {
    format!("Before the image. {} After the image.", std64(&png(5, 7)))
}

fn two_asset_text() -> String {
    format!("One {} two {} three.", std64(&png(5, 7)), std64(&png(2, 3)))
}

/// The source span an output span maps back to.
fn back(text: &str, fold: &AssetFold, start: usize, end: usize) -> Span {
    let (_, a) = normalize(fold, text);
    a.span_to_parent(Span::new(start as u32, end as u32))
}

#[test]
fn caption_fold_keeps_surrounding_text_and_maps_offsets() {
    let text = annotated_text();
    let fold = AssetFold::new();
    let result = fold_checked(&fold, &text);
    assert_eq!(result, "Before the image. [png 5x7, 45 bytes] After the image.");
    let asset = one(&text);
    let (start, end) = (result.find('[').unwrap(), result.find(']').unwrap() + 1);
    assert_eq!(back(&text, &fold, start, end), asset.span);
    let after = result.find("After").unwrap();
    assert_eq!(&text[back(&text, &fold, after, after + 5).range()], "After");
}

#[test]
fn describer_composes_through_the_caption_function() {
    let text = annotated_text();
    let fold = AssetFold::new().with_caption("describer", |asset, text, out| {
        let content = asset.decode(text).unwrap();
        assert_eq!(asset.media_type(text), "image/png");
        out.push_str(&format!("a {} byte {} asset", content.len(), asset.media_type(text)));
    });
    assert_eq!(fold_checked(&fold, &text), "Before the image. a 45 byte image/png asset After the image.");
}

#[test]
fn empty_layer_folds_to_the_identical_text() {
    let fold = AssetFold::new();
    assert_eq!(fold_checked(&fold, "clean text"), "clean text");
    assert!(normalize(&fold, "clean text").1.edits().is_empty());
    assert!(check_layer(&AssetDetector::new(), "clean text").is_empty());
}

#[test]
fn two_asset_fold_keeps_intervening_text_and_maps_both_spans() {
    let text = two_asset_text();
    let fold = AssetFold::new();
    let result = fold_checked(&fold, &text);
    assert_eq!(result, "One [png 5x7, 45 bytes] two [png 2x3, 45 bytes] three.");
    let found = detect(&text);
    assert_eq!(found.len(), 2);
    let first = (result.find('[').unwrap(), result.find(']').unwrap() + 1);
    assert_eq!(back(&text, &fold, first.0, first.1), found[0].span);
    let second_start = first.1 + result[first.1..].find('[').unwrap();
    let second_end = second_start + result[second_start..].find(']').unwrap() + 1;
    assert_eq!(back(&text, &fold, second_start, second_end), found[1].span);
    let tail = result.find("three").unwrap();
    assert_eq!(&text[back(&text, &fold, tail, tail + 5).range()], "three");
}

// ---------------------------------------------------------------------------
// AssetAnnotatorTest

#[test]
fn provides_the_assets_layer() {
    assert_eq!(<AssetDetector as Annotator>::ID, "turbo:assets");
    assert_eq!(Assets::ID, "turbo:assets");
    assert_eq!(<AssetFold as Normalizer>::ID, "turbo:normalize:asset-fold");
    assert_eq!(<AssetFold as Annotator>::ID, "turbo:normalize:asset-fold");
}

#[test]
fn annotates_detected_assets() {
    let encoded = std64(&padded(b"%PDF-", 32));
    let layer = check_layer(&AssetDetector::new(), &format!("attached: {encoded}"));
    assert_eq!(layer.len(), 1);
    assert_eq!(layer.format[0], AssetFormat::Pdf);
    assert_eq!(layer.format_name(0), "pdf");
}

// testCleanDocumentGetsAnEmptyLayer, run through an analyzer.
#[test]
fn clean_document_gets_an_empty_layer() {
    let analyzer = turbo_analysis::Analyzer::builder().add(AssetDetector::new()).build().unwrap();
    let mut docs = vec![Document::new("no assets"), Document::new(annotated_text())];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert!(docs[0].get::<Assets>().unwrap().is_empty());
    assert_eq!(docs[1].get::<Assets>().unwrap().len(), 1);
}

#[test]
fn config_hashes_tell_the_configurations_apart() {
    let plain = AssetDetector::new();
    let hex = AssetDetector { hex_dumps: true, ..plain };
    let unknown = AssetDetector { report_unknown_base64: true, ..plain };
    let soft = AssetDetector { soft_line_breaks: true, ..plain };
    let hashes = [plain, hex, unknown, soft].map(|d| Annotator::config_hash(&d));
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(hashes[i], hashes[j]);
        }
    }
    let fold = AssetFold::new();
    let custom = AssetFold::new().with_caption("x", |_, _, out| out.push('x'));
    let other = AssetFold::new().with_caption("y", |_, _, out| out.push('x'));
    assert_ne!(Normalizer::config_hash(&fold), Normalizer::config_hash(&custom));
    assert_ne!(Normalizer::config_hash(&custom), Normalizer::config_hash(&other));
    assert_ne!(Normalizer::config_hash(&fold), Normalizer::config_hash(&AssetFold::with_detector(hex)));
    assert_eq!(Normalizer::config_hash(&fold), Annotator::config_hash(&fold));
}

// A malformed declared media type changed what is detected and folded, so
// the producers record version 1 where they recorded 0: output of the old
// and the new rule never carries one identity.
#[test]
fn producers_record_the_detector_version() {
    assert_eq!(<AssetDetector as Annotator>::VERSION, "1");
    assert_eq!(<AssetFold as Normalizer>::VERSION, "1");
    assert_eq!(<AssetFold as Annotator>::VERSION, "1");
    let text = "data:text//plain;base64,anVzdCBzb21lIHBsYWluIGJ5dGVzIHBhZGRlZCBvdXQ=";

    let fold = Bound::new(AssetFold::new()).write(0, "folded");
    let analyzer = Analyzer::builder().add(AssetDetector::new()).step(fold).build().unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let detected = docs[0].provenance(Key::<Assets>::DEFAULT.id()).unwrap();
    assert_eq!((detected.annotator, detected.version), ("turbo:assets", "1"));
    let folded = docs[0].provenance(Key::<Placeholders>::DEFAULT.id()).unwrap();
    assert_eq!((folded.annotator, folded.version), ("turbo:normalize:asset-fold", "1"));

    let step = Bound::new(NormalizeStep::new(AssetFold::new())).write(0, "folded");
    let analyzer = Analyzer::builder().step(step).build().unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let p = docs[0].provenance(Key::<View>::named("folded").id()).unwrap();
    assert_eq!((p.annotator, p.version), ("turbo:normalize:asset-fold", "1"));

    let fold = AssetFold::new();
    let parts = |version| vec![("turbo:normalize:asset-fold", version, Normalizer::config_hash(&fold))];
    let (old, new) = (hash_config(&parts("0")), hash_config(&parts("1")));
    let chain = Chain::new().then(AssetFold::new()).config_hash();
    assert_eq!(chain, new);
    assert_ne!(chain, old);
}

// ---------------------------------------------------------------------------
// AssetFalsePositiveTest

const NEAR_MISSES: [&str; 12] = [
    "release 3.0.0 is available",
    "request id 550e8400-e29b-41d4-a716-446655440000",
    "GET /9j/abc/thumbnail HTTP/1.1",
    "sha256 d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2d2",
    "AlphanumericRunsLikeThisOne0123456789 are prose",
    "eyJ0eXAiOiJKV1QifQ.eyJzdWIiOiIxMjMifQ.signature",
    "e25vdCBqc29ufQ.eyJzdWIiOiIxMjMifQ.c2ln",
    "-----BEGIN CERTIFICATE-----\nnot base64!\n-----END CERTIFICATE-----",
    "-----BEGIN PUBLIC KEY-----\nQUJD\n-----END PRIVATE KEY-----",
    "data:text/plain,SGVsbG8=",
    "mailto:data:image/png;base64,not-an-image",
    "the token a.b.c is punctuation",
];

/// Every addition on.
fn all_on() -> AssetDetector {
    AssetDetector { hex_dumps: true, report_unknown_base64: true, soft_line_breaks: true }
}

#[test]
fn near_miss_yields_nothing() {
    for fixture in NEAR_MISSES {
        assert!(detect(fixture).is_empty(), "{fixture}");
        assert!(detect_with(&all_on(), fixture).is_empty(), "{fixture}");
    }
}

#[test]
fn joined_corpus_yields_nothing() {
    let joined = NEAR_MISSES.join(" | ");
    assert!(detect(&joined).is_empty());
    assert!(detect_with(&all_on(), &joined).is_empty());
}

#[test]
fn control_text_still_finds_each_new_encoding_family() {
    let token = format!("{}.{}.{}", url64(br#"{"alg":"HS256"}"#), url64(br#"{"sub":"123"}"#), url64(&[0; 32]));
    let body = mime(&[0; 96], 64, "\n");
    let control = format!("{token}\n-----BEGIN CERTIFICATE-----\n{body}\n-----END CERTIFICATE-----");
    assert_eq!(detect(&control).len(), 2);
    assert_eq!(detect_with(&all_on(), &control).len(), 2);
}

// ---------------------------------------------------------------------------
// AssetsManualExampleTest

const MANUAL_TEXT: &str =
    "An image data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAHCAYAAAAAAAAAAAAAAAAAAAAAAAAA follows.";
const MANUAL_BARE: &str = "A report pastes iVBORw0KGgoAAAANSUhEUgAAAAUAAAAHCAYAAAAAAAAAAAAAAAAAAAAAAAAA into the body.";
const MANUAL_JWT: &str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjMifQ.AAAA";

#[test]
fn manual_detection_example() {
    let a = detect(MANUAL_TEXT)[0];
    assert_eq!((a.format_name(MANUAL_TEXT), a.media_type(MANUAL_TEXT)), ("png", "image/png"));
    assert_eq!((a.width, a.height, a.decoded_len), (Some(5), Some(7), 45));
    let bytes = a.decode(MANUAL_TEXT).unwrap();
    assert_eq!((bytes.len(), bytes[0]), (45, 0x89));
}

#[test]
fn manual_bare_run_example() {
    let a = detect(MANUAL_BARE)[0];
    assert_eq!((a.format_name(MANUAL_BARE), a.media_type(MANUAL_BARE)), ("png", "image/png"));
    assert_eq!((a.width, a.height, a.decoded_len), (Some(5), Some(7), 45));
    assert_eq!(a.payload, a.span);
}

#[test]
fn manual_jwt_example() {
    let a = detect(MANUAL_JWT)[0];
    assert_eq!(a.format, AssetFormat::Jwt);
    assert_eq!(a.media_type(MANUAL_JWT), "application/jwt");
    assert_eq!(a.decode(MANUAL_JWT).unwrap(), br#"{"sub":"123"}"#);
}

#[test]
fn manual_short_text_example() {
    let note = "data:text/plain;charset=UTF-8;base64,SGk=";
    let a = detect(note)[0];
    assert_eq!(a.decode(note).unwrap(), b"Hi");
    assert_eq!(a.format_name(note), "plain");
    assert_eq!(a.media_type(note), "text/plain;charset=UTF-8");
    assert_eq!(a.decoded_len, 2);
}

#[test]
fn manual_encoded_ec_parameters_example() {
    let parameters = "-----BEGIN EC PARAMETERS-----\nBggqhkjOPQMBBw==\n-----END EC PARAMETERS-----\n";
    let encoded = std64(parameters.as_bytes());
    let a = detect(&encoded)[0];
    assert_eq!(a.format_name(&encoded), "pem-parameters");
    assert_eq!(a.media_type(&encoded), "application/x-x509-ec-parameters");
    assert_eq!(a.decode(&encoded).unwrap(), parameters.as_bytes());
}

#[test]
fn manual_fold_example() {
    let fold = AssetFold::new();
    assert_eq!(fold_checked(&fold, MANUAL_TEXT), "An image [png 5x7, 45 bytes] follows.");
    assert_eq!(back(MANUAL_TEXT, &fold, 9, 28), detect(MANUAL_TEXT)[0].span);
}

#[test]
fn manual_describer_example() {
    let fold = AssetFold::new().with_caption("described", |asset, text, out| {
        out.push_str("described ");
        out.push_str(asset.media_type(text));
    });
    assert_eq!(fold_checked(&fold, MANUAL_TEXT), "An image described image/png follows.");
}

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

fn assert_unrecognized_default(bytes: &[u8]) {
    assert_unrecognized(&AssetDetector::new(), bytes);
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

// ---------------------------------------------------------------------------
// AssetExcelTypeTest

/// `AssetExcelTypeTest.header`: a BIFF2, BIFF3 or BIFF4 BOF record.
fn excel(version: u16, kind: u16, unused: u16) -> Vec<u8> {
    let record: u16 = match version {
        2 => 0x0009,
        3 => 0x0209,
        4 => 0x0409,
        _ => panic!("version must be 2, 3 or 4"),
    };
    let mut b = Vec::new();
    for field in [record, if version == 2 { 4 } else { 6 }, unused, kind, unused] {
        b.extend_from_slice(&field.to_le_bytes());
    }
    b.resize(32, 0);
    b
}

/// `AssetExcelTypeTest.assertFormat`: bare and as an untyped data URI.
fn assert_format(bytes: &[u8], format: &str, media_type: &str) {
    let encoded = std64(bytes);
    for text in [encoded.clone(), format!("{DATA_URI}{encoded}")] {
        let a = one(&text);
        assert_eq!((a.format_name(&text), a.media_type(&text)), (format, media_type));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn excel_worksheet_document_types() {
    for version in [2, 3, 4] {
        for kind in [16, 32, 64] {
            assert_format(&excel(version, kind, 0), "xls", &format!("application/vnd.ms-excel.sheet.{version}"));
        }
    }
}

#[test]
fn excel_workspace_document_types() {
    for version in [3, 4] {
        for unused in [0, 65535] {
            let media_type = format!("application/vnd.ms-excel.workspace.{version}");
            assert_format(&excel(version, 0x0100, unused), "xlw", &media_type);
        }
    }
}

#[test]
fn excel_unused_fields_are_ignored() {
    for version in [2, 3, 4] {
        assert_format(&excel(version, 0x0010, 0xBEEF), "xls", &format!("application/vnd.ms-excel.sheet.{version}"));
    }
}

#[test]
fn excel_unsupported_document_types_are_rejected() {
    let cases = [
        (2, 0),
        (2, 5),
        (2, 256),
        (2, 65535),
        (3, 0),
        (3, 5),
        (3, 6),
        (3, 272),
        (3, 65535),
        (4, 0),
        (4, 5),
        (4, 6),
        (4, 272),
        (4, 65535),
    ];
    for (version, kind) in cases {
        assert_unrecognized_default(&excel(version, kind, 0));
    }
}

#[test]
fn excel_truncated_bof_is_rejected() {
    let cases = [(2, 0), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (3, 4), (3, 6), (3, 7), (3, 8), (3, 9)];
    for (version, length) in cases.into_iter().chain([(4, 4), (4, 7), (4, 8), (4, 9)]) {
        let encoded = std64(&excel(version, 0x0010, 0)[..length]);
        assert!(detect(&format!("{DATA_URI}{encoded}")).is_empty(), "{version} {length}");
    }
}

#[test]
fn excel_complete_bof_without_file_body() {
    for version in [2u16, 3, 4] {
        let bytes = &excel(version, 0x0010, 0)[..if version == 2 { 8 } else { 10 }];
        let text = format!("{DATA_URI}{}", std64(bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), "xls");
        assert_eq!(a.media_type(&text), format!("application/vnd.ms-excel.sheet.{version}"));
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

#[test]
fn excel_supplied_type_remains_available_for_unknown_document_type() {
    let bytes = excel(3, 0xFFFF, 0);
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetPcapngSignatureTest

/// An empty pcapng section header block in either byte order.
fn section(little_endian: bool) -> Vec<u8> {
    let u32b = |v: u32| if little_endian { v.to_le_bytes() } else { v.to_be_bytes() };
    let u16b = |v: u16| if little_endian { v.to_le_bytes() } else { v.to_be_bytes() };
    let mut b = Vec::new();
    b.extend_from_slice(&u32b(0x0A0D_0D0A));
    b.extend_from_slice(&u32b(28));
    b.extend_from_slice(&u32b(0x1A2B_3C4D));
    b.extend_from_slice(&u16b(1));
    b.extend_from_slice(&u16b(0));
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&u32b(28));
    b
}

fn assert_pcapng(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "pcapng", "application/vnd.tcpdump.pcapng", transport);
}

#[test]
fn pcapng_empty_section_in_both_byte_orders() {
    for little in [false, true] {
        assert_pcapng(&section(little), "standard");
        assert_pcapng(&section(little), "uri");
    }
}

#[test]
fn pcapng_sections_with_different_byte_orders() {
    let bytes = cat(&section(true), &section(false));
    for transport in TRANSPORTS {
        assert_pcapng(&bytes, transport);
    }
}

#[test]
fn pcapng_incomplete_byte_order_magic_is_rejected() {
    for length in [0, 1, 3, 4, 5, 7, 8, 9, 10, 11] {
        assert_unrecognized_default(&section(false)[..length]);
        assert_unrecognized_default(&section(true)[..length]);
    }
}

#[test]
fn pcapng_signature_without_complete_block() {
    for little in [false, true] {
        assert_pcapng(&section(little)[..12], "uri");
    }
}

#[test]
fn pcapng_incorrect_byte_order_magic_is_rejected() {
    for magic in [
        0x0000_0000u32,
        0xFFFF_FFFF,
        0xA1B2_C3D4,
        0xD4C3_B2A1,
        0xA1B2_3C4D,
        0x4D3C_B2A1,
        0x1A2B_3C00,
        0x003C_2B1A,
        0x1A2B_3C4E,
        0x4E3C_2B1A,
    ] {
        let mut bytes = section(false);
        bytes[8..12].copy_from_slice(&magic.to_be_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn pcapng_incorrect_byte_order_offset_is_rejected() {
    for little in [false, true] {
        for offset in [4, 7, 9, 12, 24] {
            let mut bytes = section(little);
            bytes[8..12].fill(0);
            let magic = if little { 0x1A2B_3C4Du32.to_le_bytes() } else { 0x1A2B_3C4Du32.to_be_bytes() };
            bytes[offset..offset + 4].copy_from_slice(&magic);
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn pcapng_incorrect_section_type_is_rejected() {
    for kind in [0u32, 2, 0x0A0A_0A0A, 0x0D0A_0D0A, 0x0A0D_0D0B, 0x1A2B_3C4D] {
        let mut bytes = section(false);
        bytes[..4].copy_from_slice(&kind.to_be_bytes());
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn pcapng_explicit_media_type_with_incorrect_byte_order_magic() {
    let mut bytes = section(false);
    bytes[8..12].fill(0);
    let text = format!("data:application/octet-stream;base64,{}", std64(&bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("octet-stream", "application/octet-stream"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// ---------------------------------------------------------------------------
// AssetCalendarHeaderTest

const CALENDAR_BODY: &str = "PRODID:-//Fieldwork Planner//EN\r\nVERSION:2.0\r\nBEGIN:VEVENT\r\n\
    UID:survey@example.org\r\nDTSTAMP:20260907T090000Z\r\nDTSTART:20260908T100000Z\r\n\
    SUMMARY:Survey schedule\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n";
const VCALENDAR: &str = "BEGIN:VCALENDAR";

fn assert_calendar(calendar: &str, transport: &str) {
    assert_identified(&AssetDetector::new(), calendar.as_bytes(), "ics", "text/calendar", transport);
}

#[test]
fn calendar_opening() {
    let mut lines = vec![VCALENDAR.to_string(), "begin:vcalendar".to_string(), "BeGiN:vCaLeNdAr".to_string()];
    for at in 1..VCALENDAR.len() {
        for space in [" ", "\t"] {
            lines.push(format!("{}\r\n{space}{}", &VCALENDAR[..at], &VCALENDAR[at..]));
        }
    }
    for line in lines {
        for transport in TRANSPORTS {
            assert_calendar(&format!("{line}\r\n{CALENDAR_BODY}"), transport);
        }
    }
}

#[test]
fn calendar_invalid_opening() {
    for opening in [
        "BEGIN:VCALENDAR-NOTES\r\n",
        "BEGIN:VCALENDARS\r\n",
        "BEGIN:VCALENDAR notes\r\n",
        "BEGIN:VCALENDAR\0\r\n",
        "BEGIN:VCALENDAR\r\n -NOTES\r\n",
        "BEGIN:VCALENDAR\r\n\tS\r\n",
        "BEGIN:VCALENDAR\n",
        "BEGIN:VCALENDAR\r",
        "BEGIN:VCALENDAR\rX\n",
        "BEGIN:VCAL\r\n  ENDAR\r\n",
        "BEGIN:VCAL\n ENDAR\r\n",
        "BEGIN:VCALENDAR;\r\n",
        "BEGIN;X=1:VCALENDAR\r\n",
        "BEGIN:VCALENDAR:\r\n",
    ] {
        assert_unrecognized_default(format!("{opening}{CALENDAR_BODY}").as_bytes());
    }
}

#[test]
fn calendar_changed_byte() {
    for at in 0..VCALENDAR.len() {
        for value in [0u8, 0x1A, 0x3F, 0x7F, 0x80, 0xFE, 0xFF] {
            let mut bytes = format!("{VCALENDAR}\r\n{CALENDAR_BODY}").into_bytes();
            bytes[at] = value;
            assert_unrecognized_default(&bytes);
            assert!(detect(&url64(&bytes)).is_empty());
        }
    }
}

#[test]
fn calendar_truncation() {
    let opening = format!("{VCALENDAR}\r\n");
    for length in 0..=16 {
        assert_unrecognized_default(&opening.as_bytes()[..length]);
    }
}

#[test]
fn calendar_opening_only() {
    assert_calendar(&format!("{VCALENDAR}\r\n"), "uri");
}

#[test]
fn calendar_later_version() {
    assert_calendar(&format!("{VCALENDAR}\r\nX-NOTE:{}\r\n{CALENDAR_BODY}", "survey ".repeat(100)), "mime76");
}

#[test]
fn calendar_repeated_folding() {
    assert_calendar(&format!("B{}EGIN:VCALENDAR\r\n{CALENDAR_BODY}", "\r\n ".repeat(1000)), "mime64");
}

#[test]
fn calendar_folded_suffix() {
    let calendar = format!("{VCALENDAR}{}-NOTES\r\n{CALENDAR_BODY}", "\r\n\t".repeat(1000));
    assert_unrecognized_default(calendar.as_bytes());
}

#[test]
fn calendar_unterminated_fold() {
    assert_unrecognized_default(format!("{VCALENDAR}\r\n ").as_bytes());
}

#[test]
fn calendar_declared_type() {
    let bytes = b"BEGIN:VCALENDAR-NOTES\r\nSurvey schedule";
    let text = format!("data:text/calendar;base64,{}", std64(bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("calendar", "text/calendar"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

#[test]
fn calendar_manual_example() {
    let calendar = format!("begin:vcalendar\r\n{CALENDAR_BODY}");
    let text = format!("{DATA_URI}{}", std64(calendar.as_bytes()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("ics", "text/calendar"));
    assert_eq!(a.decode(&text).unwrap(), calendar.as_bytes());
}

// ---------------------------------------------------------------------------
// AssetDexMagicTest

fn assert_dex(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "dex", "application/x-dex", transport);
}

#[test]
fn dex_encoded_note() {
    let note = b"dex\nThis is a note about Android bytecode, not an executable file.";
    for transport in TRANSPORTS {
        assert!(detect(&encode(note, transport)).is_empty(), "{transport}");
    }
}

#[test]
fn dex_version_and_transport() {
    for version in ["009", "013", "035", "037", "038", "039", "040", "041", "000", "999"] {
        let mut bytes = dex(version, 80);
        bytes[13] = 0xFB;
        bytes[14] = 0xFF;
        for transport in TRANSPORTS {
            assert_dex(&bytes, transport);
        }
    }
}

#[test]
fn dex_version_digit() {
    for offset in 4..7 {
        for value in [0u8, b' ', b'/', b'0', b'1', b'8', b'9', b':', b'A', b'a', 0x7F, 0x80, 0xB2, 0xD9, 0xFF] {
            let mut bytes = dex("035", 80);
            bytes[offset] = value;
            if value.is_ascii_digit() {
                assert_dex(&bytes, "standard");
                assert_dex(&bytes, "uri");
            } else {
                assert_unrecognized_default(&bytes);
            }
        }
    }
}

#[test]
fn dex_nonzero_terminator() {
    for terminator in [1u8, 9, 10, 13, 32, 48, 127, 128, 255] {
        let mut bytes = dex("035", 80);
        bytes[7] = terminator;
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn dex_misplaced_terminator() {
    for offset in [8, 9, 12, 16, 24] {
        let mut bytes = dex("035", 80);
        bytes[7..].fill(b'x');
        bytes[offset] = 0;
        assert_unrecognized_default(&bytes);
    }
}

#[test]
fn dex_truncated_magic() {
    for length in 0..8 {
        assert_unrecognized_default(&dex("035", length));
    }
}

#[test]
fn dex_magic_without_complete_file() {
    for length in [8, 9, 12, 16, 23, 24] {
        assert_dex(&dex("035", length), "uri");
    }
}

#[test]
fn dex_changed_prefix() {
    for offset in 0..4 {
        for value in [0u8, b' ', b'D', b'X', b'\r', 0xFF] {
            let mut bytes = dex("035", 80);
            bytes[offset] = value;
            assert_unrecognized_default(&bytes);
        }
    }
}

#[test]
fn dex_declared_type_with_invalid_magic() {
    for media_type in ["application/octet-stream", "application/x-dex"] {
        let bytes = dex("abc", 80);
        let text = format!("data:{media_type};base64,{}", std64(&bytes));
        let a = one(&text);
        assert_eq!(a.format_name(&text), &media_type[media_type.find('/').unwrap() + 1..]);
        assert_eq!(a.media_type(&text), media_type);
        assert_eq!(a.decode(&text).unwrap(), bytes);
    }
}

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
fn utf16(text: &str, big_endian: bool) -> Vec<u8> {
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

fn be32_at(bytes: &mut [u8], at: usize, value: u32) {
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

// ---------------------------------------------------------------------------
// AssetSvgHeaderTest

const SVG_NAMESPACE: &str = "http://www.w3.org/2000/svg";

fn image() -> String {
    String::from_utf8(svg()).unwrap()
}

fn assert_svg(bytes: &[u8], transport: &str) {
    assert_identified(&AssetDetector::new(), bytes, "svg", "image/svg+xml", transport);
}

#[test]
fn svg_image() {
    let image = image();
    let ns = SVG_NAMESPACE;
    for xml in [
        image.clone(),
        format!("<?xml version='1.0'?>{image}"),
        format!("\u{FEFF}{image}"),
        format!(" \r\n\t<!-- Survey icon --><?creator fieldwork?>{image}"),
        format!("<drawing:svg xmlns:drawing='{ns}'/>"),
        format!("<svg1:svg xmlns:svg1='{ns}'/>"),
        format!("<svg:svg xmlns:svg='{ns}'/>"),
        format!("<\u{3b1}:svg xmlns:\u{3b1}='{ns}'/>"),
        format!("<svg data-note='&gt; &quot; &amp;' xmlns='{ns}'/>"),
        "<svg xmlns='http://www.w3.org/2000/sv&#x67;'/>".to_string(),
        format!("<!DOCTYPE svg SYSTEM 'urn:opennlp:unused-dtd'>{image}"),
    ] {
        for transport in TRANSPORTS {
            assert_svg(xml.as_bytes(), transport);
        }
    }
}

#[test]
fn svg_encoding() {
    for charset in ["UTF-8", "UTF-16", "UTF-16LE", "UTF-16BE"] {
        let xml = format!("<?xml version='1.0' encoding='{charset}'?>{}", image());
        let bytes = match charset {
            "UTF-8" => xml.into_bytes(),
            // Java's UTF-16 encoder writes a big-endian byte-order mark.
            "UTF-16" => utf16(&format!("\u{FEFF}{xml}"), true),
            "UTF-16LE" => utf16(&xml, false),
            _ => utf16(&xml, true),
        };
        assert_svg(&bytes, "standard");
        assert_svg(&bytes, "uri");
    }
}

#[test]
fn svg_not_svg() {
    for xml in [
        "<svg-notes>Survey</svg-notes>",
        "<svgNotes/>",
        "<svg/>",
        "<svg xmlns='urn:survey'/>",
        "<svg xmlns='http://www.w3.org/2000/svg-notes'/>",
        "<svg xmlns:other='http://www.w3.org/2000/svg'/>",
        "<svg xmlns='HTTP://www.w3.org/2000/svg'/>",
        "<svg xmlns='http://www.w3.org/2000/svg' xmlns='urn:survey'/>",
        "<svg xmlns='http://www.w3.org/2000/svg' a='1' a='2'/>",
        "<svg:notes xmlns:svg='http://www.w3.org/2000/svg'/>",
        "<svg:not-svg xmlns:svg='http://www.w3.org/2000/svg'/>",
        "<svg:svg/>",
        "<svg xmlns=http://www.w3.org/2000/svg/>",
        "<svg xmlns='http://www.w3.org/2000/svg'",
        "<svg",
        "<svg ",
        "<svg\0 xmlns='http://www.w3.org/2000/svg'/>",
        "<!-- <svg xmlns='http://www.w3.org/2000/svg'/> --><notes/>",
        "<notes><svg xmlns='http://www.w3.org/2000/svg'/></notes>",
        "<SVG xmlns='http://www.w3.org/2000/svg'/>",
    ] {
        assert_unrecognized_default(xml.as_bytes());
    }
}

#[test]
fn svg_later_namespace() {
    let xml = format!("<svg data-note='{}' xmlns='{SVG_NAMESPACE}'/>", "survey ".repeat(1000));
    assert_svg(xml.as_bytes(), "mime76");
}

#[test]
fn svg_header_limit() {
    let (prefix, suffix) = ("<svg data-note='", format!("' xmlns='{SVG_NAMESPACE}'/>"));
    let xml = format!("{prefix}{}{suffix}", "x".repeat(64 * 1024 - prefix.len() - suffix.len()));
    assert_svg(xml.as_bytes(), "uri");
    assert_unrecognized_default(format!("{prefix}x{}", &xml[prefix.len()..]).as_bytes());
}

// testBoundedSourceRead: a root tag that ends past the 64 KiB limit is
// not SVG. (That the scan reads no further than the limit is a property
// of the streaming decoder, which stops after `MAX_HEADER_BYTES`.)
#[test]
fn svg_bounded_source_read() {
    let xml = format!("<svg data-note='{}' xmlns='{SVG_NAMESPACE}'/>", "x".repeat(128 * 1024));
    assert_unrecognized_default(xml.as_bytes());
}

#[test]
fn svg_incomplete_image() {
    assert_svg(format!("<svg xmlns='{SVG_NAMESPACE}'><invalid").as_bytes(), "standard");
}

#[test]
fn svg_other_xml() {
    let bytes = b"<?xml version='1.0'?><notes/>";
    assert_identified(&AssetDetector::new(), bytes, "xml", "application/xml", "uri");
}

#[test]
fn svg_declared_type() {
    let bytes = b"<svg-notes>Survey</svg-notes>";
    let text = format!("data:image/svg+xml;base64,{}", std64(bytes));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("svg+xml", "image/svg+xml"));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

// testNoExternalRequests and testExternalNamespace: the scanner has no
// way to fetch anything, so a stylesheet or DTD URL changes nothing, and
// an attribute default an external DTD would supply is never applied.
#[test]
fn svg_external_references_are_never_read() {
    let url = "http://127.0.0.1:9/unused.dtd";
    let xml = format!("<?xml-stylesheet href='{url}'?><!DOCTYPE svg SYSTEM '{url}'>{}", image());
    assert_svg(xml.as_bytes(), "uri");
    assert_unrecognized_default(b"<!DOCTYPE svg SYSTEM 'file:///tmp/namespace.dtd'><svg/>");
}

#[test]
fn svg_internal_namespace_entity() {
    let xml = format!("<!DOCTYPE svg [<!ENTITY ns '{SVG_NAMESPACE}'>]><svg xmlns='&ns;'/>");
    assert_unrecognized_default(xml.as_bytes());
}

#[test]
fn svg_concurrent_detection() {
    std::thread::scope(|scope| {
        for thread in 0..4 {
            scope.spawn(move || {
                for index in (thread..32).step_by(4) {
                    assert_svg(format!("<!-- request {index} -->{}", image()).as_bytes(), "uri");
                    assert_unrecognized_default(b"<svg xmlns='urn:not-an-image'/>");
                }
            });
        }
    });
}

#[test]
fn svg_manual_example() {
    let text = format!("{DATA_URI}{}", std64(&svg()));
    let a = one(&text);
    assert_eq!((a.format_name(&text), a.media_type(&text)), ("svg", "image/svg+xml"));
    assert_eq!(a.decode(&text).unwrap(), svg());
}

// ---------------------------------------------------------------------------
// KnownMagicsTest. `magic_format` is `KnownMagics.formatOf`: the table
// alone, longest signature first. OpenNLP's precomputed base64 prefixes
// have no counterpart (the detector decodes and tests the first decoded
// byte instead), so their test becomes one that every signature's first
// byte passes that filter.

#[test]
fn every_signature_passes_the_first_byte_filter() {
    use turbo_normalize::assets::FIRST_BYTES;
    for &(magic, format) in MAGICS {
        assert!(FIRST_BYTES[usize::from(magic[0])], "{format:?}");
    }
    for b in [b'R', 0x30, b'B', b'b', b'<', b' ', b'\t', b'\r', b'\n', 0x00, 0xEF, 0xFE, 0xFF] {
        assert!(FIRST_BYTES[usize::from(b)], "{b:#x}");
    }
}

#[test]
fn entries_are_sorted_longest_first() {
    for pair in MAGICS.windows(2) {
        assert!(pair[0].0.len() >= pair[1].0.len(), "{:?} before {:?}", pair[0], pair[1]);
    }
}

#[test]
fn no_duplicate_magics() {
    let mut seen = std::collections::HashSet::new();
    for &(magic, format) in MAGICS {
        assert!(seen.insert(magic), "duplicate magic for {format:?}");
    }
}

#[test]
fn lookup_favors_the_longest_magic() {
    assert_eq!(magic_format(b"!<arch>\ndebian-binary junk").unwrap().name(), "deb");
    assert_eq!(magic_format(b"!<arch>\nsomething else").unwrap().name(), "ar");
}

#[test]
fn binary_dxf_sentinel() {
    let header = b"AutoCAD Binary DXF\r\n\x1a\0";
    assert_eq!(header.len(), 22);
    assert_eq!(magic_format(header).unwrap().name(), "dxf");
    let content = padded(header, 30);
    let encoded = std64(&content);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("dxf", "image/vnd.dxf"));
    assert_eq!(a.decode(&encoded).unwrap().len(), content.len());
}

const DXB_HEADER: &[u8] = b"AutoCAD DXB 1.0\r\n\x1a\0";

#[test]
fn dxb_point_drawing() {
    assert_eq!(DXB_HEADER.len(), 19);
    assert_eq!(magic_format(DXB_HEADER).unwrap().name(), "dxb");
    let drawing = cat(DXB_HEADER, &[2, 10, 0, 20, 0, 0]);
    let encoded = std64(&drawing);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("dxb", "image/vnd.dxb"));
    assert_eq!(a.decode(&encoded).unwrap(), drawing);
}

#[test]
fn truncated_dxb_sentinel_is_rejected() {
    for length in [0, 14, 15, 16, 17, 18] {
        assert_eq!(magic_format(&DXB_HEADER[..length]), None);
    }
}

#[test]
fn invalid_dxb_sentinel_is_rejected() {
    for sentinel in [
        &b"AutoCAD DXB 1.0\r\n0x1A00"[..],
        b"AutoCAD DXB 1.0\r\n^Z\0",
        b"AutoCAD DXB 1.0\n\r\x1a\0",
        b"AutoCAD DXB 1.0\r\n\x1a!",
    ] {
        let content = padded(sentinel, 30);
        assert_eq!(magic_format(&content), None);
        assert!(detect(&std64(&content)).is_empty());
    }
}

/// `KnownMagicsTest.openExrHeader`.
fn open_exr(flags: u32) -> Vec<u8> {
    let mut b = 20_000_630u32.to_le_bytes().to_vec();
    b.extend_from_slice(&(2 | flags).to_le_bytes());
    b.resize(32, 0);
    b
}

#[test]
fn open_exr_header_flags() {
    for flags in [0, 0x200, 0x400, 0x600, 0x800, 0xC00, 0x1000, 0x1400, 0x1800, 0x1C00] {
        let header = open_exr(flags);
        assert_eq!(magic_format(&header).unwrap().name(), "exr");
        let encoded = std64(&header);
        let a = one(&encoded);
        assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("exr", "image/x-exr"));
        assert_eq!(a.decode(&encoded).unwrap(), header);
    }
}

#[test]
fn declared_aces_type_is_preserved() {
    let header = open_exr(0);
    let uri = format!("data:image/aces;base64,{}", std64(&header));
    let a = one(&uri);
    assert_eq!((a.format_name(&uri), a.media_type(&uri)), ("exr", "image/aces"));
    assert_eq!(a.decode(&uri).unwrap(), header);
}

#[test]
fn truncated_open_exr_magic_is_rejected() {
    for length in 0..4 {
        assert_eq!(magic_format(&open_exr(0)[..length]), None);
    }
}

#[test]
fn big_endian_open_exr_magic_is_rejected() {
    let mut header = 20_000_630u32.to_be_bytes().to_vec();
    header.extend_from_slice(&2u32.to_be_bytes());
    header.resize(32, 0);
    assert_eq!(magic_format(&header), None);
    assert!(detect(&std64(&header)).is_empty());
}

#[test]
fn snappy_framed_stream() {
    let mut stream = 0x0000_06FFu32.to_le_bytes().to_vec();
    stream.extend_from_slice(b"sNaPpY");
    stream.extend_from_slice(&0x0000_12FEu32.to_le_bytes());
    stream.resize(32, 0);
    assert_eq!(magic_format(&stream).unwrap().name(), "sz");
    let encoded = std64(&stream);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("sz", "application/x-snappy-framed"));
    assert_eq!(a.decode(&encoded).unwrap(), stream);
}

#[test]
fn snappy_identifier_without_chunk_header_is_rejected() {
    let text = b"sNaPpY is the identifier, not the complete chunk";
    assert_eq!(magic_format(text), None);
    assert!(detect(&std64(text)).is_empty());
}

#[test]
fn tiff_byte_orders() {
    for little in [true, false] {
        for version in [42u16, 43] {
            let marker = if little { b'I' } else { b'M' };
            let mut header = vec![marker, marker];
            let mut put = |bytes: &[u8]| header.extend_from_slice(bytes);
            let order16 = |v: u16| if little { v.to_le_bytes() } else { v.to_be_bytes() };
            put(&order16(version));
            if version == 43 {
                put(&order16(8));
                put(&order16(0));
                put(&if little { 16u64.to_le_bytes() } else { 16u64.to_be_bytes() });
            } else {
                put(&if little { 8u32.to_le_bytes() } else { 8u32.to_be_bytes() });
            }
            header.resize(32, 0);
            assert_eq!(magic_format(&header), Some(AssetFormat::Tiff));
            let encoded = std64(&header);
            let a = one(&encoded);
            assert_eq!((a.format, a.media_type(&encoded)), (AssetFormat::Tiff, "image/tiff"));
            assert_eq!(a.decode(&encoded).unwrap(), header);
        }
    }
}

#[test]
fn xar_archive_media_type() {
    let mut header = [0u8; 28];
    header[..4].copy_from_slice(b"xar!");
    header[5] = 28;
    header[7] = 1;
    let encoded = std64(&header);
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("xar", "application/x-xar"));
}

#[test]
fn xpm_format_and_media_type() {
    let xpm = "/* XPM */\nstatic char *image[] = {\n\"1 1 1 1\",\n\". c #000000\",\n\".\"};\n";
    let encoded = std64(xpm.as_bytes());
    let a = one(&encoded);
    assert_eq!((a.format_name(&encoded), a.media_type(&encoded)), ("xpm", "image/x-xpixmap"));
}

#[test]
fn textual_hexadecimal_dxf_suffix_is_rejected() {
    assert_eq!(magic_format(b"AutoCAD Binary DXF\r\n0x1A00"), None);
}

#[test]
fn embedded_asset_constants_resolve() {
    use AssetFormat::*;
    let core = [
        Png, Jpeg, Gif, Pdf, Zip, Tiff, Gzip, SevenZip, Rar, Flac, Ogg, Midi, Sqlite, Elf, Pe, Class, Woff, Woff2, Mp3,
        Ole2, Zstd, Wasm,
    ];
    let names = [
        ("png", "image/png"),
        ("jpeg", "image/jpeg"),
        ("gif", "image/gif"),
        ("pdf", "application/pdf"),
        ("zip", "application/zip"),
        ("tiff", "image/tiff"),
        ("gzip", "application/gzip"),
        ("7z", "application/x-7z-compressed"),
        ("rar", "application/vnd.rar"),
        ("flac", "audio/flac"),
        ("ogg", "application/ogg"),
        ("midi", "audio/midi"),
        ("sqlite", "application/vnd.sqlite3"),
        ("elf", "application/x-elf"),
        ("pe", "application/vnd.microsoft.portable-executable"),
        ("class", "application/java-vm"),
        ("woff", "font/woff"),
        ("woff2", "font/woff2"),
        ("mp3", "audio/mpeg"),
        ("ole2", "application/x-ole-storage"),
        ("zstd", "application/zstd"),
        ("wasm", "application/wasm"),
    ];
    for (format, (name, media_type)) in core.into_iter().zip(names) {
        assert_eq!((format.name(), format.media_type()), (name, media_type));
        assert!(MAGICS.iter().any(|&(_, f)| f == format), "{name}");
    }
}

#[test]
fn riff_formats_are_not_table_entries() {
    for format in [AssetFormat::Webp, AssetFormat::Wav, AssetFormat::Avi] {
        assert!(MAGICS.iter().all(|&(_, f)| f != format));
    }
}

#[test]
fn matching_format_and_media_type_stay_together() {
    for &(magic, format) in MAGICS {
        assert_eq!(magic_format(magic), Some(format));
    }
}

#[test]
fn short_magics_are_limited_to_the_core_set() {
    for &(magic, format) in MAGICS {
        if magic.len() < 4 {
            assert!(matches!(format, AssetFormat::Jpeg | AssetFormat::Gzip | AssetFormat::Mp3), "{format:?}");
        }
    }
}

#[test]
fn format_names_round_trip() {
    for format in AssetFormat::SNIFFED {
        assert!(AssetFormat::from_names(format.name(), format.media_type()) == Some(format), "{format:?}");
        assert!(format.media_type().contains('/'));
    }
}

// ---------------------------------------------------------------------------
// False positives: text that looks like it might hold a payload, with the
// defaults and with every addition on.

const LOOKALIKES: [&str; 24] = [
    "Attachments",
    "Attachments: none. See AttachmentsAndEnclosuresOverview for details.",
    "deadbeef",
    "0xdeadbeef 0xcafebabe 0xfeedface",
    "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
    "commit 3f786850e387550fdab836ed7e6dc881de23001b",
    "sha256: e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "SHA-512 CF83E1357EEFB8BDF1542850D66D8007D620E4050B5715DC83F4A921D36CE9CE",
    "uuid 123e4567-e89b-12d3-a456-426614174000 and 00000000-0000-0000-0000-000000000000",
    "/usr/local/share/applications/ExampleViewer/ResourcesAndPreferences/Defaults",
    "https://example.org/a/very/long/path/WithCamelCaseSegments/And123/Numbers456",
    "AbstractSingletonProxyFactoryBeanImplementationHelperDelegate2",
    "Electroencephalographically, pneumonoultramicroscopicsilicovolcanoconiosis",
    "the MAC is 00:1a:2b:3c:4d:5e and the IPv6 is fe80::1ff:fe23:4567:890a",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
    "0123456789012345678901234567890123456789",
    "iVBORw0KGgo is how a PNG starts, but this run is far too short",
    "The ratio was 89 50 4e 47 in the survey, then 0d 0a 1a 0a.",
    "data:image/png;base64,",
    "data:,Hello%2C%20World",
    "-----BEGIN CERTIFICATE-----\n-----END CERTIFICATE-----",
    "eyJhbGciOiJIUzI1NiJ9 is a header without claims",
    "base64,base64,base64;base64,base64",
];

#[test]
fn lookalikes_yield_nothing() {
    for text in LOOKALIKES {
        assert!(detect(text).is_empty(), "{text}");
        assert!(detect_with(&all_on(), text).is_empty(), "{text}");
    }
    let joined = LOOKALIKES.join("\n");
    assert!(detect_with(&all_on(), &joined).is_empty());
}

// ---------------------------------------------------------------------------
// UTF-8 offsets: spans are byte offsets of the text, whatever precedes
// the asset.

#[test]
fn spans_are_utf8_byte_offsets() {
    let encoded = std64(&png(5, 7));
    for before in ["", "é ", "日本語の画像 ", "\u{1F5BC}\u{FE0F} ", "Ünïcödé: «"] {
        for after in ["", " é", " 日本語", "\u{1F600}", "»"] {
            let text = format!("{before}{encoded}{after}");
            let a = one(&text);
            assert_eq!(a.span, Span::new(before.len() as u32, (before.len() + encoded.len()) as u32), "{text}");
            assert_eq!(covered(&text, a.span), encoded);
        }
    }
}

#[test]
fn data_uri_after_multibyte_text() {
    let uri = format!("data:image/png;base64,{}", std64(&png(1, 1)));
    let text = format!("Ein Bild \u{1F4F7}: {uri} \u{2013}");
    let a = one(&text);
    let start = text.find("data:").unwrap();
    assert_eq!(a.span, Span::new(start as u32, (start + uri.len()) as u32));
    assert_eq!(a.declared, Some(Span::new((start + 5) as u32, (start + 14) as u32)));
}

/// The declared media type may run to offset 127 in UTF-16 code units (128
/// units), counted the way OpenNLP counts them: a supplementary character
/// is two units, whatever its UTF-8 length.
#[test]
fn media_type_length_is_counted_in_utf16_units() {
    let body = std64(&png(1, 1));
    let data_uri = |media_type: &str| {
        let text = format!("data:{media_type};base64,{body}");
        let found = detect(&text);
        assert_eq!(found.len(), 1, "{media_type}");
        found[0].span.start == 0
    };
    assert!(data_uri(&format!("x/{}", "a".repeat(126))));
    assert!(!data_uri(&format!("x/{}", "a".repeat(127))));
    assert!(data_uri(&format!("x/{}\u{1F600}", "a".repeat(124))));
    assert!(!data_uri(&format!("x/{}\u{1F600}", "a".repeat(125))));
    assert!(data_uri(&format!("x/{}", "é".repeat(126))));
}

#[test]
fn media_type_may_not_hold_white_space_or_quotes() {
    let body = std64(&png(1, 1));
    for bad in ["image/\"png", "image/'png", "image/\tpng", "image/\u{85}png", "image/\u{2029}png"] {
        let text = format!("data:{bad};base64,{body}");
        assert!(detect(&text).iter().all(|a| a.span.start > 0), "{bad:?}");
    }
    // A character outside a token does not end the media type, but it is
    // no media type: the URI is still found, with the sniffed one.
    let text = format!("data:image/p\u{e9}ng;base64,{body}");
    let a = one(&text);
    assert_eq!((a.span.start, a.media_type(&text)), (0, "image/png"));
}

// ---------------------------------------------------------------------------
// Addition: hex dumps.

fn hex_detector() -> AssetDetector {
    AssetDetector { hex_dumps: true, ..AssetDetector::new() }
}

/// `bytes` as lower-case hex pairs, each written with `prefix`, joined by
/// `separator`.
fn hex(bytes: &[u8], prefix: &str, separator: &str) -> String {
    bytes.iter().map(|b| format!("{prefix}{b:02x}")).collect::<Vec<_>>().join(separator)
}

#[test]
fn hex_dumps_in_every_form() {
    let bytes = png(640, 480);
    let forms = [
        hex(&bytes, "", ""),
        hex(&bytes, "", " "),
        hex(&bytes, "", ":"),
        hex(&bytes, "", ","),
        hex(&bytes, "", ", "),
        hex(&bytes, "0x", ", "),
        hex(&bytes, "0x", ","),
        hex(&bytes, "0x", " "),
        hex(&bytes, "0X", ":"),
        hex(&bytes, "", "").to_uppercase(),
        // xxd -p: 30 bytes a line.
        bytes.chunks(30).map(|l| hex(l, "", "")).collect::<Vec<_>>().join("\n"),
        // A C array: sixteen to a line, indented, comma at the line end.
        bytes.chunks(16).map(|l| format!("    {}", hex(l, "0x", ", "))).collect::<Vec<_>>().join(",\r\n"),
        // od-style lines of space-separated pairs.
        bytes.chunks(16).map(|l| hex(l, "", " ")).collect::<Vec<_>>().join(" \n\t"),
    ];
    for form in forms {
        let text = format!("dump: {form}\nend");
        let found = detect_with(&hex_detector(), &text);
        assert_eq!(found.len(), 1, "{form}");
        let a = found[0];
        assert_eq!((a.format, a.encoding), (AssetFormat::Png, AssetEncoding::Hex), "{form}");
        assert_eq!(covered(&text, a.span), form.trim_start());
        assert_eq!((a.width, a.height, a.decoded_len), (Some(640), Some(480), 45));
        assert_eq!(a.decode(&text).unwrap(), bytes);
        // Off by default.
        assert!(detect(&text).is_empty(), "{form}");
    }
}

#[test]
fn hex_dump_needs_32_bytes() {
    let bytes = padded(&png(1, 1)[..24], 32);
    let text = hex(&bytes, "", " ");
    assert_eq!(detect_with(&hex_detector(), &text).len(), 1);
    let short = hex(&bytes[..31], "", " ");
    assert!(detect_with(&hex_detector(), &short).is_empty());
}

#[test]
fn hex_dump_needs_a_recognized_signature() {
    let bytes: Vec<u8> = (0..64u8).map(|i| i.wrapping_mul(37).wrapping_add(11)).collect();
    assert_eq!(magic_format(&bytes), None);
    for separator in ["", " ", ":"] {
        assert!(detect_with(&hex_detector(), &hex(&bytes, "", separator)).is_empty());
    }
}

#[test]
fn hex_dump_stands_alone() {
    let pairs = hex(&png(1, 1), "", " ");
    for text in [format!("x{pairs}"), format!("_{pairs}"), format!("{pairs}g"), format!("9{pairs}")] {
        assert!(detect_with(&hex_detector(), &text).is_empty(), "{text}");
    }
    for text in [format!("({pairs})"), format!("[{pairs}]"), format!("\"{pairs}\".")] {
        assert_eq!(detect_with(&hex_detector(), &text).len(), 1, "{text}");
    }
}

#[test]
fn hex_dump_keeps_its_first_separator() {
    let bytes = png(1, 1);
    // The separator changes after the first 40 pairs: the dump ends there,
    // 40 bytes, still a PNG header.
    let text = format!("{} {}", hex(&bytes[..40], "", ":"), hex(&bytes[40..], "", ":"));
    let a = detect_with(&hex_detector(), &text)[0];
    assert_eq!(a.decoded_len, 40);
    // A prefix on some pairs only: the dump ends at the first pair without.
    let mixed = format!("{}, {}", hex(&bytes[..35], "0x", ", "), hex(&bytes[35..], "", ", "));
    assert_eq!(detect_with(&hex_detector(), &mixed)[0].decoded_len, 35);
    // An odd digit at the end is not part of a pair, and makes the run touch a word.
    assert!(detect_with(&hex_detector(), &format!("{}f", hex(&bytes, "", ""))).is_empty());
}

#[test]
fn hex_dump_layer_records_the_encoding() {
    let text = format!("a {} b", hex(&png(3, 4), "", " "));
    let layer = check_layer(&hex_detector(), &text);
    assert_eq!(layer.encoding, [AssetEncoding::Hex]);
    assert_eq!(layer.format_name(0), "png");
}

// ---------------------------------------------------------------------------
// Addition: base64 with no recognized signature.

fn unknown_detector() -> AssetDetector {
    AssetDetector { report_unknown_base64: true, ..AssetDetector::new() }
}

/// Bytes with no signature, from a fixed generator.
fn random_bytes(seed: u64, n: usize) -> Vec<u8> {
    let mut rng = Rng::new(seed);
    loop {
        let bytes = rng.bytes(n);
        if magic_format(&bytes).is_none() && !matches!(bytes[0], b'R' | 0x30 | b'<' | b'B' | b'b') {
            return bytes;
        }
    }
}

#[test]
fn unknown_base64_is_reported_when_on() {
    let bytes = random_bytes(7, 48);
    let encoded = std64(&bytes);
    let text = format!("blob {encoded} end");
    assert!(detect(&text).is_empty());
    let a = detect_with(&unknown_detector(), &text)[0];
    assert_eq!(
        (a.format, a.format_name(&text), a.media_type(&text)),
        (AssetFormat::Unknown, "unknown", "application/octet-stream")
    );
    assert_eq!(covered(&text, a.span), encoded);
    assert_eq!(a.decode(&text).unwrap(), bytes);
    assert_eq!((a.width, a.height), (None, None));
}

#[test]
fn unknown_base64_rule_edges() {
    let detector = unknown_detector();
    let reported = |run: &str| detect_with(&detector, &format!("x {run} y")).len() == 1;
    // 32 characters, every class, every pair changes class: reported.
    let alternating = "aB3dE5gH7jK9mN1pQ3sT5vW7yZ9bC2eF";
    assert_eq!(alternating.len(), 32);
    assert!(reported(alternating));
    // 31 characters: too short.
    assert!(!reported(&alternating[..31]));
    // Padding does not count toward the 32 characters.
    assert!(!reported(&format!("{}==", &alternating[..30])));
    assert!(reported(&format!("{}=", &format!("{alternating}aB3")[..35])));
    // A class missing: no digit, no upper case, no lower case.
    assert!(!reported("aBcDeFgHiJkLmNoPqRsTuVwXyZaBcDeF"));
    assert!(!reported("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6"));
    assert!(!reported("A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6"));
    // The change rate at the threshold: 31 pairs, 14 changes is 45.2%,
    // 13 changes is 41.9%.
    let at = |changes: usize| {
        let mut run = String::from("a");
        let mut class = 0;
        for i in 1..32 {
            if i <= changes {
                class = (class + 1) % 3;
            }
            run.push(['a', 'B', '7'][class]);
        }
        run
    };
    assert_eq!(14 * 100 / 31, 45);
    assert!(reported(&at(14)));
    assert!(!reported(&at(13)));
    // Words and hex never qualify.
    for word in
        ["Attachments", "deadbeef", "AttachmentsAttachmentsAttachments2024", "deadbeefdeadbeefdeadbeefdeadbeef0123"]
    {
        assert!(!reported(word), "{word}");
    }
}

#[test]
fn unknown_base64_does_not_hide_known_assets() {
    let known = std64(&png(5, 7));
    let unknown = std64(&random_bytes(11, 60));
    let text = format!("{unknown} {known}");
    let found = detect_with(&unknown_detector(), &text);
    let formats: Vec<AssetFormat> = found.iter().map(|a| a.format).collect();
    assert_eq!(formats, [AssetFormat::Unknown, AssetFormat::Png]);
}

/// A random run passes the rule almost always: of 200 runs of 48 bytes
/// (64 characters), none is missed.
#[test]
fn unknown_rule_accepts_random_data() {
    let mut rng = Rng::new(99);
    let mut missed = 0;
    for _ in 0..200 {
        let bytes = rng.bytes(48);
        if magic_format(&bytes).is_some() {
            continue;
        }
        let text = std64(&bytes);
        missed += usize::from(detect_with(&unknown_detector(), &text).is_empty());
    }
    assert_eq!(missed, 0);
}

// ---------------------------------------------------------------------------
// Addition: quoted-printable soft line breaks.

fn soft_detector() -> AssetDetector {
    AssetDetector { soft_line_breaks: true, ..AssetDetector::new() }
}

/// `encoded` broken with a soft line break every `width` characters.
fn soft_wrapped(encoded: &str, width: usize, line_break: &str) -> String {
    let lines: Vec<&str> = encoded.as_bytes().chunks(width).map(|l| std::str::from_utf8(l).unwrap()).collect();
    lines.join(&format!("={line_break}"))
}

#[test]
fn soft_line_breaks_join_a_bare_run() {
    let bytes = cat(&png(64, 32), &[0; 60]);
    for line_break in ["\r\n", "\n"] {
        for width in [10, 37, 75] {
            let wrapped = soft_wrapped(&std64(&bytes), width, line_break);
            let text = format!("Content: {wrapped}\n");
            let a = detect_with(&soft_detector(), &text)[0];
            assert_eq!(covered(&text, a.span), wrapped);
            assert_eq!((a.format, a.decoded_len), (AssetFormat::Png, bytes.len() as u64));
            assert_eq!(a.decode(&text).unwrap(), bytes);
            // Off, a run ends at the first soft break.
            let off = detect(&text);
            assert!(off.iter().all(|x| !covered(&text, x.span).contains('\n')), "{off:?}");
        }
    }
}

#[test]
fn soft_line_breaks_join_a_data_uri() {
    let bytes = png(9, 9);
    let wrapped = soft_wrapped(&std64(&bytes), 20, "\r\n");
    let text = format!("<img src=3D\"data:image/png;base64,{wrapped}\">");
    let a = detect_with(&soft_detector(), &text)[0];
    assert_eq!(a.media_type(&text), "image/png");
    assert_eq!(a.decode(&text).unwrap(), bytes);
    assert_eq!(a.decoded_len, bytes.len() as u64);
}

#[test]
fn soft_line_break_needs_a_payload_character_after_it() {
    // 44 bytes end in one `=`: before a line break that the payload does
    // not continue on, it is padding, with soft line breaks on or off.
    let encoded = std64(&png(1, 1)[..44]);
    assert!(encoded.ends_with("=") && !encoded.ends_with("=="));
    for tail in ["\r\n\r\n", "\n-", "\n ", "\r\n_"] {
        let text = format!("{encoded}{tail}tail");
        let on = detect_with(&soft_detector(), &text);
        assert_eq!(on, detect(&text), "{tail:?}");
        assert_eq!(covered(&text, on[0].span), encoded);
    }
    // A word on the next line is read as the rest of the payload: the cost
    // of the addition, and why it is off by default.
    let text = format!("{encoded}\r\ntail");
    assert_eq!(detect_with(&soft_detector(), &text)[0].span.end as usize, text.len());
    // A soft break with no payload after it ends the run before the `=`.
    let bare = std64(&png(1, 1));
    let text = format!("{bare}=\r\n\r\n");
    assert_eq!(detect_with(&soft_detector(), &text), detect(&text));
}

// ---------------------------------------------------------------------------
// The fold: captions, touching assets, the additions, the annotator.

#[test]
fn fold_captions_for_every_kind_of_asset() {
    let token = format!("{}.{}.", url64(br#"{"alg":"none"}"#), url64(br#"{"sub":"123"}"#));
    let pem = format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----", mime(&[0; 96], 64, "\n"));
    let cases = [
        (format!("see {}.", std64(&gif(12, 34))), "see [gif 12x34, 30 bytes]."),
        (format!("see {}.", std64(&padded(b"%PDF-1.7", 32))), "see [pdf, 32 bytes]."),
        ("note data:text/plain;charset=UTF-8;base64,SGk= end".to_string(), "note [plain, 2 bytes] end"),
        (format!("t={token};"), "t=[jwt, 13 bytes];"),
        (format!("cert:\n{pem}\nend"), "cert:\n[pem-cert, 96 bytes]\nend"),
    ];
    for (text, folded) in cases {
        assert_eq!(fold_checked(&AssetFold::new(), &text), folded);
    }
}

#[test]
fn write_caption_is_opennlp_caption() {
    let text = "padding QUJD";
    let mut out = String::from("keep ");
    turbo_normalize::assets::write_caption(&valid(), text, &mut out);
    assert_eq!(out, "keep [png 5x7, 3 bytes]");
    out.clear();
    turbo_normalize::assets::write_caption(&Asset { width: None, height: None, ..valid() }, text, &mut out);
    assert_eq!(out, "[png, 3 bytes]");
}

#[test]
fn touching_assets_keep_their_own_edits() {
    let uri = format!("data:image/png;base64,{}", std64(&cat(&png(1, 1), &[0])));
    let text = format!("a {uri}{uri} b");
    let fold = AssetFold::new();
    let caption = "[png 1x1, 46 bytes]";
    assert_eq!(fold_checked(&fold, &text), format!("a {caption}{caption} b"));
    let (_, alignment) = normalize(&fold, &text);
    let n = uri.len() as u32;
    let c = caption.len() as u32;
    let edits: Vec<(Span, Span)> = alignment.edits().iter().map(|e| (e.parent, e.child)).collect();
    assert_eq!(
        edits,
        [(Span::new(2, 2 + n), Span::new(2, 2 + c)), (Span::new(2 + n, 2 + 2 * n), Span::new(2 + c, 2 + 2 * c))]
    );
    // Each placeholder names its own asset, in the view read.
    let (_, spans) = placeholders(&fold, &text);
    assert_eq!(spans, [Span::new(2, 2 + n), Span::new(2 + n, 2 + 2 * n)]);
    // Each caption maps back to its own asset.
    assert_eq!(alignment.span_to_parent(Span::new(2, 3)), Span::new(2, 2 + n));
    assert_eq!(alignment.span_to_parent(Span::new(2 + c, 3 + c)), Span::new(2 + n, 2 + 2 * n));
}

#[test]
fn fold_with_the_additions() {
    let fold = AssetFold::with_detector(all_on());
    let unknown = std64(&random_bytes(5, 40));
    let text = format!("hex {} and {unknown}.", hex(&png(2, 2), "0x", ", "));
    assert_eq!(fold_checked(&fold, &text), "hex [png 2x2, 45 bytes] and [unknown, 40 bytes].");
    assert_eq!(fold_checked(&AssetFold::new(), &text), text);
}

#[test]
fn caption_function_can_be_empty() {
    let fold = AssetFold::new().with_caption("drop", |_, _, _| {});
    assert_eq!(fold_checked(&fold, &annotated_text()), "Before the image.  After the image.");
}

#[test]
fn fold_annotator_writes_asset_placeholders() {
    let text = two_asset_text();
    let doc = Document::new(text.as_str());
    let view = doc.get::<View>().unwrap();
    let (folded, rows) = AssetFold::new().annotate((view,)).unwrap();
    assert_eq!(folded.text(), "One [png 5x7, 45 bytes] two [png 2x3, 45 bytes] three.");
    let rows: &Placeholders = &rows;
    assert_eq!(rows.len(), 2);
    assert!(rows.kind.iter().all(|&k| k == PlaceholderKind::Asset));
}

// ---------------------------------------------------------------------------
// Property loops over generated text.

/// A piece of generated text: prose, an asset in some transport, or
/// noise that looks like one.
fn piece(rng: &mut Rng) -> String {
    let fixtures = [png(5, 7), gif(3, 4), padded(b"%PDF-1.4", 40), svg(), pem("EC PARAMETERS"), dex("035", 40)];
    match rng.below(14) {
        0 => rng
            .pick(&["the ", "Attachments ", "deadbeef ", "naïve ", "日本 ", "\u{1F600} ", "a.b.c ", "\n"])
            .to_string(),
        1 => rng.pick(&[" ", "\n", "\r\n", "", "\t", "\"", "(", ") ", "data:", "=", "-----BEGIN "]).to_string(),
        2 => encode(rng.pick(&fixtures), rng.pick(&TRANSPORTS)),
        3 => format!("data:image/png;base64,{}", std64(rng.pick(&fixtures))),
        4 => format!("-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n", mime(&rng.bytes(48), 64, "\n")),
        5 => format!("{}.{}.{}", url64(br#"{"alg":"HS256"}"#), url64(br#"{"sub":"1"}"#), url64(&rng.bytes(8))),
        6 => hex(rng.pick(&fixtures), rng.pick(&["", "0x"]), rng.pick(&[" ", ", ", ":"])),
        7 => {
            let n = rng.below(60);
            std64(&rng.bytes(n))
        }
        8 => {
            let n = rng.below(60);
            url64(&rng.bytes(n))
        }
        9 => soft_wrapped(&std64(rng.pick(&fixtures)), 1 + rng.below(70), "\r\n"),
        10 => {
            let mut s = std64(rng.pick(&fixtures));
            let cut = rng.below(s.len() + 1);
            s.truncate(cut);
            s
        }
        11 => mime(&cat(rng.pick(&fixtures), &rng.bytes(100)), *rng.pick(&[64, 76]), "\r\n"),
        12 => {
            let mut s = std64(rng.pick(&fixtures)).into_bytes();
            let at = rng.below(s.len());
            s[at] = *rng.pick(b"=-_+ .");
            String::from_utf8(s).unwrap()
        }
        _ => std64(&random_bytes(rng.next(), 24 + rng.below(40))),
    }
}

fn generated(rng: &mut Rng) -> String {
    (0..1 + rng.below(8)).map(|_| piece(rng)).collect()
}

#[test]
fn generated_texts_keep_every_invariant() {
    let configs = [AssetDetector::new(), hex_detector(), unknown_detector(), soft_detector(), all_on()];
    let mut rng = Rng::new(0x5EED);
    let mut found = 0;
    for _ in 0..600 {
        let text = generated(&mut rng);
        for config in &configs {
            // detect_with checks the layer, the document, and each decode.
            let assets = detect_with(config, &text);
            for pair in assets.windows(2) {
                assert!(pair[0].span.end <= pair[1].span.start, "{text:?}");
            }
            found += assets.len();
            let fold = AssetFold::with_detector(*config);
            let out = fold_checked(&fold, &text);
            if assets.is_empty() {
                assert_eq!(out, text);
            }
        }
        // The additions that only report more never hide what the
        // defaults report.
        let plain = detect(&text);
        let more = detect_with(&unknown_detector(), &text);
        assert!(plain.iter().all(|a| more.contains(a)), "{text:?}");
    }
    assert!(found > 1000, "{found}");
}

/// An asset of the default detector is found again, the same, in a text
/// that holds only it.
#[test]
fn generated_assets_stand_alone() {
    let mut rng = Rng::new(42);
    for _ in 0..600 {
        let text = generated(&mut rng);
        for a in detect(&text) {
            let alone = &text[a.span.range()];
            let again = detect(alone);
            assert_eq!(again.len(), 1, "{alone:?} from {text:?}");
            let b = again[0];
            assert_eq!((b.format, b.decoded_len, b.width, b.height), (a.format, a.decoded_len, a.width, a.height));
            assert_eq!(b.media_type(alone), a.media_type(&text));
            assert_eq!(b.decode(alone), a.decode(&text));
        }
    }
}

// ---------------------------------------------------------------------------
// The SVG scanner against the JDK's StAX parser, which OpenNLP's
// `SvgHeader` runs with DTD support off. Each expectation was read from
// that parser.

#[test]
fn svg_prolog_follows_the_jdk_parser() {
    let root = format!("<svg xmlns='{SVG_NAMESPACE}'/>");
    let cases: [(&str, bool); 31] = [
        // The internal subset is skipped unread up to its first `]`.
        ("<!DOCTYPE svg [<!A<TLIST>]>", true),
        ("<!DOCTYPE svg [ ' ]>", true),
        ("<!DOCTYPE svg [ %pe; garbage ]>", true),
        // A character XML does not allow: the JDK parser throws
        // MissingResourceException, which `SvgHeader` does not catch; here
        // the payload is not SVG.
        ("<!DOCTYPE svg [ \u{1} ]>", false),
        ("<!DOCTYPE svg [ ' ]>' ]>", false),
        ("<!DOCTYPE svg [ ]] ]>", false),
        ("<!DOCTYPE svg []  >", true),
        ("<!DOCTYPE svg [] x>", false),
        ("<!DOCTYPE svg [<!-- ]> -->]>", false),
        ("<!DOCTYPE svg[]>", true),
        ("<!DOCTYPE svg SYSTEM 'c'[]>", true),
        // Public identifiers hold PubidChar only; system literals anything.
        ("<!DOCTYPE svg PUBLIC 'a]b' 'c'>", false),
        ("<!DOCTYPE svg PUBLIC 'a\tb' 'c'>", false),
        ("<!DOCTYPE svg PUBLIC \"a'b\" 'c'>", true),
        ("<!DOCTYPE svg SYSTEM 'a]b<\t'>", true),
        // White space is required between the identifiers and after SYSTEM.
        ("<!DOCTYPE svg PUBLIC 'ab''c'>", false),
        ("<!DOCTYPE svg SYSTEM'c'>", false),
        ("<!DOCTYPE svg SYSTEM 'c'><!DOCTYPE svg SYSTEM 'c'>", false),
        // A processing instruction target may hold a colon, not be `xml`.
        ("<?x:l data?>", true),
        ("<?XmL data?>", false),
        // Encoding names: the JDK's IANA names and aliases, not Java's.
        ("<?xml version='1.0' encoding='utf-8'?>", true),
        ("<?xml version='1.0' encoding='UTF8'?>", false),
        ("<?xml version='1.0' encoding='ISO8859-1'?>", false),
        ("<?xml version='1.0' encoding='latin1'?>", true),
        ("<?xml version='1.0' encoding='L1'?>", true),
        ("<?xml version='1.0' encoding='CP819'?>", true),
        ("<?xml version='1.0' encoding='US'?>", true),
        ("<?xml version='1.0' encoding='IBM367'?>", true),
        ("<?xml version='1.0' encoding='ANSI_X3.4-1968'?>", true),
        ("<?xml version='1.0' encoding='UTF-16'?>", false),
        ("<?xml version='1.0' encoding='ISO_646.IRV:1991'?>", false),
    ];
    for (prolog, is_svg) in cases {
        let bytes = format!("{prolog}{root}").into_bytes();
        let text = format!("{DATA_URI}{}", std64(&bytes));
        let found = detect(&text);
        let svg = found.first().is_some_and(|a| a.format == AssetFormat::Svg);
        assert_eq!(svg, is_svg, "{prolog:?}: {found:?}");
    }
}

#[test]
fn svg_qualified_names_follow_the_jdk_parser() {
    let ns = SVG_NAMESPACE;
    let cases = [
        // A colon in first place is part of the name.
        (format!("<svg xmlns='{ns}' :b='1'/>"), true),
        (format!("<svg xmlns='{ns}' :='1'/>"), true),
        (format!("<svg xmlns='{ns}' :b='1' b='2'/>"), true),
        (format!("<svg xmlns='{ns}' :xmlns='1'/>"), true),
        (format!("<svg xmlns='{ns}' :b='1' :b='2'/>"), false),
        (format!("<:svg xmlns='{ns}'/>"), false),
        (format!("<svg :xmlns='{ns}'/>"), false),
        // After it, the first colon splits the prefix, which must be bound.
        (format!("<svg xmlns='{ns}' :a:b='1'/>"), false),
        (format!("<svg xmlns='{ns}' ::b='1'/>"), false),
        (format!("<svg xmlns='{ns}' b:='1'/>"), false),
        (format!("<svg xmlns='{ns}' a::b='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:b:c='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:1='1' xmlns:a='u'/>"), false),
        (format!("<svg xmlns='{ns}' a:b='1' xmlns:a='u'/>"), true),
        (format!("<svg: xmlns='{ns}'/>"), false),
        (format!("<a:svg:b xmlns:a='{ns}'/>"), false),
        (format!("<svg xmlns:='{ns}'/>"), false),
        (format!("<\u{e9}:svg xmlns:\u{e9}='{ns}'/>"), true),
    ];
    for (xml, is_svg) in cases {
        let text = format!("{DATA_URI}{}", std64(xml.as_bytes()));
        let svg = detect(&text).first().is_some_and(|a| a.format == AssetFormat::Svg);
        assert_eq!(svg, is_svg, "{xml}");
    }
}

/// Names follow XML 1.0 fifth edition. The JDK applies the fourth
/// edition's tables to XML 1.0 and rejects these; see the differences in
/// docs/normalizers.md.
#[test]
fn svg_names_follow_the_fifth_edition() {
    for name in ["\u{2C30}", "a\u{2C30}", "\u{10000}", "\u{2070}", "\u{FDF0}"] {
        let xml = format!("<svg xmlns='{SVG_NAMESPACE}' {name}='1'/>");
        assert_svg(xml.as_bytes(), "uri");
    }
}

/// Declared encodings other than UTF-8, UTF-16, US-ASCII and ISO-8859-1
/// are not read, so the payload is XML but not SVG; the JDK reads them.
#[test]
fn svg_other_declared_encodings_are_not_read() {
    for encoding in ["windows-1252", "ISO-8859-2", "Shift_JIS", "EBCDIC-CP-US"] {
        let xml = format!("<?xml version='1.0' encoding='{encoding}'?><svg xmlns='{SVG_NAMESPACE}'/>");
        assert_identified(&AssetDetector::new(), xml.as_bytes(), "xml", "application/xml", "uri");
    }
}

/// The scan stops at the end of the root start tag, so bytes after it do
/// not matter. The JDK decodes ahead and rejects a four-byte UTF-8
/// sequence above U+10FFFF within about 8 KB after the tag.
#[test]
fn svg_bytes_after_the_root_tag_do_not_matter() {
    let root = format!("<svg xmlns='{SVG_NAMESPACE}'/>");
    for tail in [&b"\xF4\x90\x80\x80"[..], b"\xF6\xB4\xB6\xAB", b"\xFF\xFF", b"\0"] {
        assert_svg(&cat(root.as_bytes(), tail), "uri");
    }
}
