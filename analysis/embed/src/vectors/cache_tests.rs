use super::*;
use crate::subwords::SubwordAnnotator;
use turbo_analysis::{Document, View};

#[test]
fn retained_states_match_token_and_span_paths_without_reencoding() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let sub = SubwordAnnotator::open(&bundle).unwrap();
    let vectors = VectorAnnotator::open(
        &bundle,
        &VectorOptions {
            encoder: EncoderOptions { max_batch: 3, max_seq: 12, max_ranges: 2, cpu_threads: 1, ..Default::default() },
            ..Default::default()
        },
    )
    .unwrap();
    let doc = Document::new(
        "The good service is here. The food is very good. A long sentence with many words and more words to span several windows in the model.",
    );
    let view = doc.get::<View>().unwrap();
    let pieces = sub.pieces(view).unwrap();
    let mut sents = Spans::new(view.id());
    for (a, b) in [(0, 24), (25, 46), (47, view.text().len() as u32)] {
        sents.push(Span::new(a, b));
    }
    let expected_tokens = vectors.embed_subwords("test", &pieces, Some(&sents)).unwrap();
    let spans = [
        Span::new(4, 8),
        Span::new(25, 46),
        Span::new(55, 100),
        Span::new(0, 0),
        Span::new(0, view.text().len() as u32),
    ];
    let expected: Vectors<Chunks> = vectors.embed_spans("test", &pieces, Some(&sents), view.id(), &spans).unwrap();
    let before = vectors.run_counts();
    let cached = vectors.cache_windows("test", &[(&pieces, &sents), (&pieces, &sents)]).remove(0).unwrap();
    let after = vectors.run_counts();
    assert_eq!(after.1 - before.1, cached.window_count() * 2, "each distinct document window encoded once");
    assert_eq!(after.0 - before.0, (cached.window_count() * 2).div_ceil(3), "windows packed across documents");
    let actual_tokens = vectors.cached_tokens("test", &pieces, &cached).unwrap();
    assert_eq!(actual_tokens.data, expected_tokens.data);
    let actual: Vectors<Chunks> = vectors.cached_spans("test", &pieces, &sents, &cached, &spans).unwrap();
    assert_eq!(actual.data, expected.data);
    assert_eq!(actual.empty, expected.empty);
    assert_eq!(vectors.run_counts(), after, "heads reuse states without running the encoder");
}

#[test]
fn projection_changes_token_vectors_but_not_raw_span_pooling() {
    let bundle = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle");
    let sub = SubwordAnnotator::open(&bundle).unwrap();
    let vectors = VectorAnnotator::open(&bundle, &VectorOptions::default())
        .unwrap()
        .with_projection(
            ProjectionHead::new(32, 7, (0..32 * 7).map(|i| (i as f32).sin()).collect(), Some(vec![0.1; 7])).unwrap(),
        )
        .unwrap();
    let doc = Document::new("The service is good.");
    let view = doc.get::<View>().unwrap();
    let pieces = sub.pieces(view).unwrap();
    let mut sents = Spans::new(view.id());
    sents.push(Span::new(0, view.text().len() as u32));
    let spans = [Span::new(4, 11)];
    let tokens = vectors.embed_subwords("test", &pieces, Some(&sents)).unwrap();
    let pooled: Vectors<Chunks> = vectors.embed_spans("test", &pieces, Some(&sents), view.id(), &spans).unwrap();
    let cached = vectors.cache_windows("test", &[(&pieces, &sents)]).remove(0).unwrap();
    assert_eq!(vectors.cached_tokens("test", &pieces, &cached).unwrap().data, tokens.data);
    let actual: Vectors<Chunks> = vectors.cached_spans("test", &pieces, &sents, &cached, &spans).unwrap();
    assert_eq!(actual.dim, 32);
    assert_eq!(tokens.dim, 7);
    assert_eq!(actual.data, pooled.data);
}
