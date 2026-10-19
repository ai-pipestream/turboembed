//! The source encoding layer across the document graph boundary:
//! it round-trips through bytes, and corrupted payloads (unknown
//! discriminants, confidences the layer itself refuses) are refused
//! with typed errors.

use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import};
use turbo_analysis::{Document, Key, Provenance};
use turbo_encoding_layers::export::codecs;
use turbo_encoding_layers::{Encoding, EncodingEvidence, SourceEncoding};

fn body(encoding: u8, confidence: f32, evidence: u8) -> Vec<u8> {
    let mut w = Writer::new();
    w.u8(encoding);
    w.u32(confidence.to_bits());
    w.u8(evidence);
    w.finish()
}

fn encoding_at(encoding: Encoding) -> u8 {
    Encoding::ALL.iter().position(|e| *e == encoding).unwrap() as u8
}

fn splice(graph: &mut DocumentGraph, body: Vec<u8>) {
    let at = graph.layers.iter().position(|l| l.layer == "turbo:encoding").unwrap();
    graph.layers[at].body = Some(body);
}

#[test]
fn source_encoding_round_trips() {
    let mut doc = Document::new("café");
    doc.add(
        &Key::DEFAULT,
        SourceEncoding { encoding: Encoding::Windows1252, confidence: 0.42, evidence: EncodingEvidence::Detected },
        Provenance::by("test"),
    )
    .unwrap();
    doc.add(
        &Key::named("declared"),
        SourceEncoding { encoding: Encoding::Utf8, confidence: 1.0, evidence: EncodingEvidence::ValidUtf8 },
        Provenance::by("test"),
    )
    .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(
        *back.get::<SourceEncoding>().unwrap(),
        SourceEncoding { encoding: Encoding::Windows1252, confidence: 0.42, evidence: EncodingEvidence::Detected }
    );
    assert_eq!(back.get_key(&Key::<SourceEncoding>::named("declared")).unwrap().evidence, EncodingEvidence::ValidUtf8);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn corrupted_source_encodings_are_refused() {
    let mut doc = Document::new("café");
    doc.add(
        &Key::DEFAULT,
        SourceEncoding { encoding: Encoding::Utf8, confidence: 1.0, evidence: EncodingEvidence::ValidUtf8 },
        Provenance::by("test"),
    )
    .unwrap();
    let base = export(&doc, &codecs()).unwrap();
    let utf8 = encoding_at(Encoding::Utf8);
    let win1252 = encoding_at(Encoding::Windows1252);

    for (what, payload) in [
        ("unknown encoding", body(40, 1.0, 2)),
        ("unknown evidence", body(utf8, 1.0, 5)),
        ("NaN confidence", body(win1252, f32::NAN, 4)),
        ("confidence above one", body(win1252, 1.5, 4)),
        ("confidence below zero", body(win1252, -0.1, 4)),
        // A BOM settles only UTF-8 and the UTF-16 encodings, at 1.
        ("BOM evidence for a legacy encoding", body(win1252, 1.0, 0)),
        ("BOM evidence below full confidence", body(utf8, 0.5, 0)),
        ("truncated", Vec::new()),
        ("trailing byte", {
            let mut b = body(utf8, 1.0, 2);
            b.push(0);
            b
        }),
    ] {
        let mut graph = base.clone();
        splice(&mut graph, payload);
        assert!(
            matches!(
                import(&graph, &codecs()).err(),
                Some(ExportError::Malformed(_)) | Some(ExportError::InvalidLayer { .. })
            ),
            "{what} imported"
        );
    }
}
