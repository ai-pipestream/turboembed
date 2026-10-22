use super::*;
use std::sync::Arc;
use turbo::sentencepiece::{Scratch, SentencePiece};
use turbo_analysis::{Analyzer, Bound, Document, Key, Layer, ModelRef, Provenance, Spans, View};
use turbo_analysis_layers::{Entities, Tokens};

// Tiny vocabulary with the committed SentencePiece fixture's normalization
// table, so format/control characters exercise the real normalization path.
fn tokenizer() -> Vec<u8> {
    fn varint(bytes: &[u8], pos: &mut usize) -> usize {
        let mut out = 0;
        for shift in (0..64).step_by(7) {
            let b = bytes[*pos];
            *pos += 1;
            out |= usize::from(b & 127) << shift;
            if b < 128 {
                return out;
            }
        }
        panic!("fixture varint");
    }
    let mut out = Vec::new();
    for (name, kind) in [("<unk>", 2), ("<s>", 3), ("</s>", 3), ("▁", 1)] {
        let mut piece = vec![10, name.len() as u8];
        piece.extend_from_slice(name.as_bytes());
        piece.extend_from_slice(&[24, kind]);
        out.extend_from_slice(&[10, piece.len() as u8]);
        out.extend(piece);
    }
    let bytes = include_bytes!("../../../testdata/sentencepiece/t5.model");
    let mut pos = 0;
    while pos < bytes.len() {
        let start = pos;
        let key = varint(bytes, &mut pos);
        assert_eq!(key & 7, 2);
        let len = varint(bytes, &mut pos);
        pos += len;
        if key >> 3 == 3 {
            out.extend_from_slice(&bytes[start..pos]);
            return out;
        }
    }
    panic!("fixture normalizer missing");
}

fn model() -> Arc<Model> {
    let encoder = turbo::deberta::Encoder::from_safetensors(
        ec(),
        include_bytes!("../../../core/tests/fixtures/deberta/weights.safetensors"),
    )
    .unwrap();
    Arc::new(
        Model::new(
            encoder,
            config(),
            include_bytes!("../tests/fixtures/heads.safetensors"),
            &tokenizer(),
            ModelRef { id: "fixture:gliner".into(), revision: "test".into(), sha256: "".into() },
        )
        .unwrap(),
    )
}

#[test]
fn empty_documents_produce_empty_entities_and_decisions() {
    let model = model();
    let schema = ExtractionSchema::new(vec!["person".into()], 0.5, Decode::Flat).unwrap();
    assert!(model.pack(&[], &schema).unwrap_err().contains("max_tokens"));
    let pipeline = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .step(Bound::new(GlinerAnnotator::new(model, schema)))
        .build()
        .unwrap();
    for text in ["", " \t\n", "\u{2003}"] {
        let mut doc = Document::new(text);
        let spans = Spans::new(doc.get::<View>().unwrap().id());
        doc.add(&Key::DEFAULT, Tokens { spans, sentence: None }, Provenance::by("fixture:tokens")).unwrap();
        let mut docs = vec![doc];
        for result in pipeline.run(&mut docs) {
            result.unwrap();
        }
        assert_eq!(docs[0].get::<Entities>().unwrap().len(), 0);
        assert_eq!(docs[0].get::<Decisions>().unwrap().len(), 0);
        GlinerAnnotator::evidence(&docs[0], Key::DEFAULT, Key::DEFAULT).unwrap().verify(&docs[0]).unwrap();
    }
}

#[test]
fn zero_piece_tokens_keep_word_positions() {
    let sp = SentencePiece::parse(&tokenizer()).unwrap();
    let model = model();
    let schema = ExtractionSchema::new(vec!["person".into()], 0.5, Decode::Flat).unwrap();
    for control in ["\u{200B}", "\u{FEFF}", "\u{200D}", "\u{0007}"] {
        let mut ids = Vec::new();
        sp.encode_ids(control, &mut Scratch::default(), &mut ids);
        assert!(ids.is_empty(), "fixture must erase {control:?}");
        let words = ["Alice", control, "\u{FEFF}", "Smith"];
        let packed = model.pack(&words, &schema).unwrap();
        assert_eq!(packed.words.len(), words.len());
        assert_eq!(packed.ids[packed.words[1]], sp.unk_id());
        assert_eq!(packed.ids[packed.words[2]], sp.unk_id());
        assert_eq!(packed.words[2], packed.words[1] + 1);
        assert_eq!(packed.words[3], packed.words[2] + 1);
        let scores = model.score_packed(&packed, &schema).unwrap();
        assert!(scores.iter().any(|c| c.start == 0 && c.end == 1));
        assert!(scores.iter().any(|c| c.start == 3 && c.end == 4));
    }
}
