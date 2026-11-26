//! The layer's record checks, the fold, the annotator, near misses and the
//! manual's examples: OpenNLP's `EmbeddedAssetTest`, `AssetFolderTest`,
//! `AssetAnnotatorTest`, `AssetFalsePositiveTest` and
//! `AssetsManualExampleTest`.

use super::*;

// ---------------------------------------------------------------------------
// EmbeddedAssetTest. The record checks of `EmbeddedAsset`'s constructor
// are `Assets::check` here; null and blank strings cannot occur (the
// format is an enum) and a length cannot be negative (`u64`).

/// `EmbeddedAssetTest.valid`: span 4..12, payload 8..12.
pub(super) fn valid() -> Asset {
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

pub(super) fn annotated_text() -> String {
    format!("Before the image. {} After the image.", std64(&png(5, 7)))
}

pub(super) fn two_asset_text() -> String {
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
pub(super) fn all_on() -> AssetDetector {
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
