use std::path::Path;
use turbo_analysis::{Document, Span, Spans, View, ViewId};
use turbo_analysis_embed::{Input, SubwordAnnotator, VectorAnnotator, VectorOptions};
use turbo_analysis_layers::{Chunks, Sentences};

#[test]
fn discontinuous_pooling_is_refused_instead_of_embedding_the_gap() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let encoder = VectorAnnotator::open(&bundle, &VectorOptions::default()).unwrap();
    let tokenizer = SubwordAnnotator::open(&bundle).unwrap();
    let doc = Document::new("the good bad text");
    let subwords = tokenizer.pieces(doc.get::<View>().unwrap()).unwrap();
    let mut spans = Spans::<Sentences>::new(ViewId::SOURCE);
    spans.try_push_fragments(&[Span::new(0, 3), Span::new(13, 17)]).unwrap();
    let mut chunks = Spans::<Chunks>::new(ViewId::SOURCE);
    chunks.try_push_fragments(&[Span::new(0, 3), Span::new(13, 17)]).unwrap();
    for input in [
        Input { subwords: &subwords, sentences: Some(&spans), chunks: None, levels: None },
        Input { subwords: &subwords, sentences: None, chunks: Some(&chunks), levels: None },
    ] {
        let result = encoder.embed("test", &[input]);
        assert!(result[0].is_err(), "a gap was silently included in pooling");
    }
}
