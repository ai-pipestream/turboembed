//! The normalize layers across the document graph boundary: they
//! round-trip through bytes on source and derived views, and
//! corrupted payloads (unknown discriminants, split UTF-8 spans,
//! scores and asset rows the layers themselves refuse) are refused.

use turbo_analysis::export::{Codecs, DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Alignment, Document, Key, Provenance, Span, Spans, View, ViewBuilder, ViewId};
use turbo_analysis_layers::Strs;
use turbo_normalize_layers::export::register_codecs;
use turbo_normalize_layers::{
    ArtifactKind, Artifacts, AssetEncoding, AssetFormat, Assets, Corrections, Emoji, EmojiKind, Noise, NoiseSeverity,
    PlaceholderKind, Placeholders,
};

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    register_codecs(&mut codecs);
    codecs
}

fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

fn spans<L>(view: ViewId, pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(view);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

const TEXT: &str = "hi http://x.io 😀 end";

fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Placeholders { spans: spans(ViewId::SOURCE, &[(3, 14)]), kind: vec![PlaceholderKind::Url] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Emoji {
            spans: spans(ViewId::SOURCE, &[(15, 19)]),
            kind: vec![EmojiKind::Basic],
            name: strs(&["grinning face"]),
            region: strs(&[""]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Artifacts { spans: spans(ViewId::SOURCE, &[(20, 23)]), kind: vec![ArtifactKind::Mojibake] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Noise { spans: spans(ViewId::SOURCE, &[(20, 23)]), severity: vec![NoiseSeverity::Damaged], score: vec![0.5] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Corrections { spans: spans(ViewId::SOURCE, &[(20, 23)]), suggestion: strs(&["end."]), edit_distance: vec![1] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Assets {
            spans: spans(ViewId::SOURCE, &[(3, 14), (20, 23)]),
            payload: vec![Span::new(5, 10), Span::new(20, 22)],
            format: vec![AssetFormat::Png, AssetFormat::Declared],
            declared_type: strs(&["", "text/plain"]),
            encoding: vec![AssetEncoding::Base64, AssetEncoding::Hex],
            decoded_len: vec![3, 1],
            width: vec![None, Some(2)],
            height: vec![None, Some(3)],
        },
        by,
    )
    .unwrap();
    doc
}

#[test]
fn normalize_layers_round_trip() {
    let graph = export(&document(), &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.get::<Placeholders>().unwrap().kind, [PlaceholderKind::Url]);
    let emoji = back.get::<Emoji>().unwrap();
    assert_eq!(emoji.name.get(0), "grinning face");
    assert_eq!(&back.source()[15..19], "😀");
    assert_eq!(back.get::<Artifacts>().unwrap().kind, [ArtifactKind::Mojibake]);
    assert_eq!(back.get::<Noise>().unwrap().score, [0.5]);
    assert_eq!(back.get::<Corrections>().unwrap().suggestion.get(0), "end.");
    let assets = back.get::<Assets>().unwrap();
    assert_eq!(assets.format, [AssetFormat::Png, AssetFormat::Declared]);
    assert_eq!(assets.media_type(1), "text/plain");
    assert_eq!(assets.payload[0], Span::new(5, 10));
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn placeholders_on_a_derived_view_round_trip() {
    let mut doc = Document::decoded(
        b"Caf\xe9 moo.".to_vec(),
        "Café moo.",
        Alignment::from_edits(vec![turbo_analysis::text::Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }])
            .unwrap(),
    )
    .unwrap();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(5);
    builder.replace(6, "");
    let (clean, alignment) = builder.finish();
    doc.add(&Key::<View>::named("clean"), View::derived(&source, clean, alignment), Provenance::by("test")).unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    doc.add(
        &Key::DEFAULT,
        Placeholders { spans: spans(clean_id, &[(5, 8)]), kind: vec![PlaceholderKind::Hashtag] },
        Provenance::by("test"),
    )
    .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let placeholders = back.get::<Placeholders>().unwrap();
    assert_eq!(placeholders.spans.view(), back.get_key(&Key::<View>::named("clean")).unwrap().id());
    assert_eq!(&back.text(placeholders.spans.view())[5..8], "moo");
}

/// The span column encoding, written out longhand so corrupted
/// payloads can be built without an export context.
fn write_spans(w: &mut Writer, spans: &[Span]) {
    w.str("").unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

fn splice(graph: &mut DocumentGraph, layer: &str, body: Vec<u8>) {
    let at = graph.layers.iter().position(|l| l.layer == layer).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = 1;
}

fn assets_body(payload: Span, name: &str, media_type: &str) -> Vec<u8> {
    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(3, 14)]);
    w.usize(1).unwrap();
    w.span(payload);
    w.usize(1).unwrap();
    w.u8(1);
    w.str(name).unwrap();
    w.str(media_type).unwrap();
    w.usize(1).unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u8(0);
    w.usize(1).unwrap();
    w.u64(3);
    w.usize(1).unwrap();
    w.u8(0);
    w.usize(1).unwrap();
    w.u8(0);
    w.finish()
}

#[test]
fn corrupted_normalize_payloads_are_refused() {
    let base = export(&document(), &codecs()).unwrap();

    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(0, 2)]);
    w.usize(1).unwrap();
    w.u8(9);
    let mut graph = base.clone();
    splice(&mut graph, "turbo:placeholders", w.finish());
    assert!(import(&graph, &codecs()).is_err(), "an unknown placeholder kind imported");

    // A span ending inside the emoji's four bytes.
    let mut w = Writer::new();
    write_spans(&mut w, &[Span::new(15, 17)]);
    w.usize(1).unwrap();
    w.u8(0);
    w.usize(1).unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.str("").unwrap();
    let mut graph = base.clone();
    splice(&mut graph, "turbo:emoji", w.finish());
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("UTF-8"), "{detail}"),
        Ok(_) => panic!("a span splitting a character imported"),
        Err(other) => panic!("a split span failed as {other}"),
    }

    for score in [0.0, f64::NAN, 1.5] {
        let mut w = Writer::new();
        write_spans(&mut w, &[Span::new(0, 2)]);
        w.usize(1).unwrap();
        w.u8(1);
        w.usize(1).unwrap();
        w.f64(score);
        let mut graph = base.clone();
        splice(&mut graph, "turbo:noise", w.finish());
        assert!(import(&graph, &codecs()).is_err(), "a noise score of {score} imported");
    }

    let mut graph = base.clone();
    splice(&mut graph, "turbo:assets", assets_body(Span::new(10, 20), "png", "image/png"));
    assert!(import(&graph, &codecs()).is_err(), "a payload outside its asset span imported");

    let mut graph = base.clone();
    splice(&mut graph, "turbo:assets", assets_body(Span::new(5, 10), "nope", "application/x-nope"));
    assert!(import(&graph, &codecs()).is_err(), "an unknown asset format imported");

    let mut graph = base.clone();
    splice(&mut graph, "turbo:assets", assets_body(Span::new(16, 17), "png", "image/png"));
    assert!(import(&graph, &codecs()).is_err(), "a payload splitting a character imported");

    let mut graph = base;
    splice(&mut graph, "turbo:placeholders", vec![0]);
    assert!(import(&graph, &codecs()).is_err(), "a truncated placeholders payload imported");
}
