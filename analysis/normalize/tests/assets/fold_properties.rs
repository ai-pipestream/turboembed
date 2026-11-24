//! The fold's alignment, and property loops over generated text.

use super::additions::{hex, hex_detector, random_bytes, soft_detector, soft_wrapped, unknown_detector};
use super::layer::{all_on, annotated_text, two_asset_text, valid};
use super::*;

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
