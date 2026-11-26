//! False positives with every addition on, UTF-8 offsets, and the three
//! additions: hex dumps, base64 with no recognized signature, and
//! quoted-printable.

use super::layer::all_on;
use super::*;

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

pub(super) fn hex_detector() -> AssetDetector {
    AssetDetector { hex_dumps: true, ..AssetDetector::new() }
}

/// `bytes` as lower-case hex pairs, each written with `prefix`, joined by
/// `separator`.
pub(super) fn hex(bytes: &[u8], prefix: &str, separator: &str) -> String {
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

pub(super) fn unknown_detector() -> AssetDetector {
    AssetDetector { report_unknown_base64: true, ..AssetDetector::new() }
}

/// Bytes with no signature, from a fixed generator.
pub(super) fn random_bytes(seed: u64, n: usize) -> Vec<u8> {
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

pub(super) fn soft_detector() -> AssetDetector {
    AssetDetector { soft_line_breaks: true, ..AssetDetector::new() }
}

/// `encoded` broken with a soft line break every `width` characters.
pub(super) fn soft_wrapped(encoded: &str, width: usize, line_break: &str) -> String {
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
