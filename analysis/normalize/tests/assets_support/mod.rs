//! Fixtures, encoders and checks for the embedded-asset tests: the header
//! fixtures and transports of OpenNLP's `AssetTestSupport`, the layer and
//! fold checks every output goes through, and a deterministic generator.
//! The encoders are written here from RFC 4648 and RFC 2045, apart from
//! the code under test.

#![allow(dead_code)]

use turbo_analysis::{Alignment, Annotator, Document, Ix, Key, Layer, Provenance, Span, View};
use turbo_normalize::{Asset, AssetDetector, AssetFold, normalize};
use turbo_normalize_layers::{Assets, PlaceholderKind};

pub const STANDARD: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
pub const URL_SAFE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

/// Base64 of `bytes` in `alphabet`, with or without `=` padding.
pub fn base64(bytes: &[u8], alphabet: &[u8; 64], pad: bool) -> String {
    let mut out = String::new();
    let (groups, rest) = bytes.as_chunks::<3>();
    for g in groups {
        let n = u32::from(g[0]) << 16 | u32::from(g[1]) << 8 | u32::from(g[2]);
        for shift in [18, 12, 6, 0] {
            out.push(char::from(alphabet[(n >> shift & 63) as usize]));
        }
    }
    if !rest.is_empty() {
        let n = u32::from(rest[0]) << 16 | rest.get(1).map_or(0, |&b| u32::from(b) << 8);
        let digits = rest.len() + 1;
        for shift in [18, 12, 6, 0].into_iter().take(digits) {
            out.push(char::from(alphabet[(n >> shift & 63) as usize]));
        }
        if pad {
            out.push_str(&"=="[..3 - digits + 1]);
        }
    }
    out
}

/// Standard padded base64, `Base64.getEncoder()`.
pub fn std64(bytes: &[u8]) -> String {
    base64(bytes, STANDARD, true)
}

/// URL-safe base64 without padding, `Base64.getUrlEncoder().withoutPadding()`.
pub fn url64(bytes: &[u8]) -> String {
    base64(bytes, URL_SAFE, false)
}

/// Standard base64 broken into lines of `width` characters joined by
/// `separator`, with none after the last line: `Base64.getMimeEncoder`.
pub fn mime(bytes: &[u8], width: usize, separator: &str) -> String {
    let flat = std64(bytes);
    let lines: Vec<&str> = flat.as_bytes().chunks(width).map(|l| std::str::from_utf8(l).unwrap()).collect();
    lines.join(separator)
}

pub const DATA_URI: &str = "data:;base64,";

/// The fixture in one of `AssetTestSupport`'s transports.
pub fn encode(bytes: &[u8], transport: &str) -> String {
    match transport {
        "standard" => std64(bytes),
        "url" => url64(bytes),
        "mime64" => mime(bytes, 64, "\n"),
        "mime76" => mime(bytes, 76, "\r\n"),
        "uri" => format!("{DATA_URI}{}", std64(bytes)),
        _ => panic!("transport is unsupported"),
    }
}

pub const TRANSPORTS: [&str; 5] = ["standard", "url", "mime64", "mime76", "uri"];

/// An SVG circle with an explicit namespace.
pub fn svg() -> Vec<u8> {
    b"<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 10'><circle cx='5' cy='5' r='4'/></svg>".to_vec()
}

/// A PEM envelope around four placeholder characters.
pub fn pem(label: &str) -> Vec<u8> {
    format!("-----BEGIN {label}-----\nAAAA\n-----END {label}-----\n").into_bytes()
}

/// DEX magic, a version and a NUL, zero-filled or cut to `length`.
pub fn dex(version: &str, length: usize) -> Vec<u8> {
    let mut b = format!("dex\n{version}\0").into_bytes();
    b.resize(length, 0);
    b
}

/// A 45-byte PNG prefix with an IHDR chunk.
pub fn png(width: i32, height: i32) -> Vec<u8> {
    let mut b = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    b.extend_from_slice(&13i32.to_be_bytes());
    b.extend_from_slice(b"IHDR");
    b.extend_from_slice(&width.to_be_bytes());
    b.extend_from_slice(&height.to_be_bytes());
    b.extend_from_slice(&[8, 6, 0, 0, 0]);
    b.resize(45, 0);
    b
}

/// A 30-byte GIF prefix with a logical screen size.
pub fn gif(width: u16, height: u16) -> Vec<u8> {
    let mut b = b"GIF89a".to_vec();
    b.extend_from_slice(&width.to_le_bytes());
    b.extend_from_slice(&height.to_le_bytes());
    b.resize(30, 0);
    b
}

/// `prefix` zero-filled to at least `length` bytes.
pub fn padded(prefix: &[u8], length: usize) -> Vec<u8> {
    let mut b = prefix.to_vec();
    if b.len() < length {
        b.resize(length, 0);
    }
    b
}

/// Two byte strings joined.
pub fn cat(a: &[u8], b: &[u8]) -> Vec<u8> {
    [a, b].concat()
}

pub fn covered(text: &str, span: Span) -> &str {
    &text[span.range()]
}

/// The assets of `text` with OpenNLP's detector, after the layer checks.
pub fn detect(text: &str) -> Vec<Asset> {
    detect_with(&AssetDetector::new(), text)
}

/// The assets `detector` finds in `text`. Every call also builds the
/// layer through a document, which runs `Layer::check` and the document's
/// bounds and boundary checks, and checks each row against the asset.
pub fn detect_with(detector: &AssetDetector, text: &str) -> Vec<Asset> {
    let found = detector.detect(text);
    let layer = check_layer(detector, text);
    assert_eq!(layer.spans.len(), found.len());
    for (i, a) in found.iter().enumerate() {
        assert_eq!(layer.spans.get(Ix::new(i as u32)), a.span);
        assert_eq!(layer.payload[i], a.payload);
        assert_eq!(layer.media_type(i), a.media_type(text));
        assert_eq!(layer.format_name(i), a.format_name(text));
        assert_eq!(layer.decoded_len[i], a.decoded_len);
        assert!(text.is_char_boundary(a.span.start as usize) && text.is_char_boundary(a.span.end as usize));
        assert!(a.span.contains(a.payload));
        if let Ok(bytes) = a.decode(text) {
            assert_eq!(bytes.len() as u64, a.decoded_len, "decoded length of {:?}", covered(text, a.payload));
        }
    }
    found
}

/// Runs the detector as an annotator over a document holding `text` and
/// adds its layer, which checks it.
pub fn check_layer(detector: &AssetDetector, text: &str) -> Assets {
    let mut doc = Document::new(text);
    let view = doc.get::<View>().unwrap().clone();
    let (layer,) = detector.annotate((&view,)).unwrap();
    layer.check().unwrap();
    let (copy,) = detector.annotate((&view,)).unwrap();
    doc.add(&Key::<Assets>::DEFAULT, copy, Provenance::by("test")).unwrap();
    layer
}

/// The one asset found in `text`.
pub fn one(text: &str) -> Asset {
    let found = detect(text);
    assert_eq!(found.len(), 1, "assets of {text:?}: {found:?}");
    found[0]
}

/// `AssetTestSupport.assertIdentified`: the fixture in `transport`, after
/// `Attachment: [`, is one asset of the given format and media type whose
/// span is the encoded text, whose payload leaves out a data URI's prefix,
/// and which decodes to the fixture.
pub fn assert_identified(detector: &AssetDetector, bytes: &[u8], format: &str, media_type: &str, transport: &str) {
    let encoded = encode(bytes, transport);
    let prefix = "Attachment: [";
    let payload_offset = if transport == "uri" { DATA_URI.len() } else { 0 };
    let text = format!("{prefix}{encoded}]");
    let found = detect_with(detector, &text);
    assert_eq!(found.len(), 1, "{transport} {format}: {found:?}");
    let a = found[0];
    assert_eq!(a.format_name(&text), format, "{transport}");
    assert_eq!(a.media_type(&text), media_type, "{transport}");
    assert_eq!(a.decoded_len, bytes.len() as u64);
    let end = text.len() as u32 - 1;
    assert_eq!(a.span, Span::new(prefix.len() as u32, end));
    assert_eq!(a.payload, Span::new((prefix.len() + payload_offset) as u32, end));
    assert_eq!(a.decode(&text).unwrap(), bytes);
}

/// `assert_identified` in every transport.
pub fn assert_identified_everywhere(bytes: &[u8], format: &str, media_type: &str) {
    for transport in TRANSPORTS {
        assert_identified(&AssetDetector::new(), bytes, format, media_type, transport);
    }
}

/// `AssetTestSupport.assertUnrecognized`: neither the standard base64 nor
/// an untyped data URI of `bytes` is an asset.
pub fn assert_unrecognized(detector: &AssetDetector, bytes: &[u8]) {
    let encoded = std64(bytes);
    assert!(detect_with(detector, &encoded).is_empty(), "{encoded}");
    assert!(detect_with(detector, &format!("{DATA_URI}{encoded}")).is_empty(), "{encoded}");
}

/// Folds `text` with `fold`, checks the alignment both ways and the
/// placeholders, and returns the output.
pub fn fold_checked(fold: &AssetFold, text: &str) -> String {
    let (out, a) = normalize(fold, text);
    let assets = fold.detector().detect(text);
    check_fold_alignment(text, &out, &a, &assets);
    let (view_out, rows) = placeholders(fold, text);
    assert_eq!(view_out, out);
    let spans: Vec<Span> = assets.iter().map(|x| x.span).collect();
    assert_eq!(rows, spans);
    out
}

/// The alignment of a fold: the edits rebuild the output from the source,
/// every output byte outside an edit maps back to the same source byte,
/// every edit's parent is one asset or a run of assets that touch, and
/// each asset maps forward to the caption of its edit and back again.
pub fn check_fold_alignment(src: &str, out: &str, a: &Alignment, assets: &[Asset]) {
    let edits = a.edits();
    let mut rebuilt = String::new();
    let (mut p, mut c) = (0usize, 0usize);
    let mut next = 0;
    for e in edits {
        let (ps, pe, cs, ce) =
            (e.parent.start as usize, e.parent.end as usize, e.child.start as usize, e.child.end as usize);
        assert!(ps >= p && cs >= c, "edits out of order in {src:?}: {edits:?}");
        assert_eq!(&src[p..ps], &out[c..cs], "kept text before {e:?}");
        rebuilt.push_str(&src[p..ps]);
        rebuilt.push_str(&out[cs..ce]);
        // The edit is a run of touching assets.
        assert_eq!(assets[next].span.start as usize, ps, "edit {e:?} does not start at an asset");
        let mut end = assets[next].span.end as usize;
        next += 1;
        while end < pe {
            assert_eq!(assets[next].span.start as usize, end, "assets of {e:?} do not touch");
            end = assets[next].span.end as usize;
            next += 1;
        }
        assert_eq!(end, pe, "edit {e:?} ends inside an asset");
        (p, c) = (pe, ce);
    }
    assert_eq!(next, assets.len(), "an asset has no edit");
    rebuilt.push_str(&src[p..]);
    assert_eq!(rebuilt, out);
    for x in assets {
        let edit = edits.iter().find(|e| e.parent.contains(x.span)).unwrap();
        assert_eq!(a.span_to_child(x.span), edit.child, "forward from {x:?}");
        // An empty caption is a deletion. Its empty output span maps back
        // to the empty span just before the deleted asset, not to the asset.
        if edit.child.is_empty() {
            let before = Span::new(edit.parent.start, edit.parent.start);
            assert_eq!(a.span_to_parent(edit.child), before, "back to {x:?}");
        } else {
            assert_eq!(a.span_to_parent(edit.child), edit.parent, "back to {x:?}");
        }
    }
    for (i, ch) in out.char_indices() {
        let span = Span::new(i as u32, (i + ch.len_utf8()) as u32);
        let parent = a.span_to_parent(span);
        match edits.iter().find(|e| e.child.contains(span)) {
            Some(e) => assert_eq!(parent, e.parent),
            None => assert_eq!(&src[parent.range()], &out[span.range()]),
        }
    }
    assert_eq!(a.span_to_child(Span::new(0, src.len() as u32)), Span::new(0, out.len() as u32));
}

/// Runs the fold as an annotator through a document: checks the
/// placeholder layer and that each placeholder is an `Asset` row in the
/// view read, and returns the output text and the placeholder spans.
pub fn placeholders(fold: &AssetFold, src: &str) -> (String, Vec<Span>) {
    let mut doc = Document::new(src);
    let view = doc.get::<View>().unwrap().clone();
    let (out_view, found) = fold.annotate((&view,)).unwrap();
    found.check().unwrap();
    assert_eq!(found.spans.view(), view.id());
    let mut spans = Vec::new();
    for (ix, s) in found.spans.iter() {
        assert_eq!(found.kind[ix.get()], PlaceholderKind::Asset);
        spans.push(s);
    }
    let out = out_view.text().to_string();
    let p = Provenance::by("test");
    doc.add(&Key::<View>::named("folded"), out_view, p.clone()).unwrap();
    doc.add(&Key::DEFAULT, found, p).unwrap();
    (out, spans)
}

/// xorshift64*, for reproducible generated inputs.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.max(1))
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    pub fn bytes(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| self.next() as u8).collect()
    }

    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> &'a T {
        &items[self.below(items.len())]
    }
}

/// Decodes base64 in either alphabet, skipping line breaks and stopping
/// at padding. Only for well-formed test input.
pub fn unbase64(s: &str) -> Vec<u8> {
    let mut out = Vec::new();
    let (mut bits, mut held) = (0u32, 0);
    for c in s.bytes().take_while(|&c| c != b'=') {
        let value = match c {
            b'\r' | b'\n' => continue,
            b'-' => 62,
            b'_' => 63,
            _ => STANDARD.iter().position(|&x| x == c).expect("a base64 character") as u32,
        };
        bits = bits << 6 | value;
        held += 6;
        if held >= 8 {
            held -= 8;
            out.push((bits >> held) as u8);
        }
    }
    out
}
