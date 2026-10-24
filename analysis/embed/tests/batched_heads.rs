//! Real tiny-BERT forward passes, not a substitute encoder.
use std::{path::Path, sync::Arc};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::*;

fn bundle() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata/tiny-bert-bundle")
}
fn categories_equal<L: CategoryScope>(a: &Categories<L>, b: &Categories<L>) {
    assert_eq!(a.category, b.category);
    assert_eq!(a.selected, b.selected);
    assert_eq!(a.selected_end, b.selected_end);
    assert_eq!(a.scores, b.scores);
}
#[test]
fn all_span_heads_batch_real_windows_and_isolate_bad_documents() {
    let sub = SubwordAnnotator::open(&bundle()).unwrap();
    let vectors = Arc::new(
        VectorAnnotator::open(
            &bundle(),
            &VectorOptions {
                encoder: EncoderOptions { max_batch: 3, max_seq: 12, cpu_threads: 1, ..Default::default() },
                ..Default::default()
            },
        )
        .unwrap(),
    );
    let head = ClassifierHead::linear(
        vectors.model().clone(),
        vectors.dim(),
        TagSet::new("sentiment", &["positive", "negative", "neutral"]),
        HeadMode::SingleLabel,
        (0..vectors.dim() * 3).map(|i| (i as f64).sin()).collect(),
        vec![0.1, 0.2, 0.3],
    )
    .unwrap();
    let doc = Document::new("The service is good. The food is bad. The service is good again.");
    let view = doc.get::<View>().unwrap();
    let mut ss = Spans::new(view.id());
    for (a, b) in [(0, 20), (21, 37), (38, 63)] {
        ss.push(Span::new(a, b));
    }
    let sentences = Sentences { spans: ss };
    let (tokens,) = turbo_tokenize::SimpleTokenizer::default().annotate((view, None)).unwrap();
    let (pieces,) = sub.annotate((view, Some(&tokens))).unwrap();
    let mut bad = sub.pieces(view).unwrap();
    bad.model.id.push_str("-different");
    let chunks = Chunks {
        spans: sentences.spans.iter().fold(Spans::new(view.id()), |mut s, (_, span)| {
            s.push(span);
            s
        }),
        level: vec![0; 3],
        parent: vec![None; 3],
        sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2), IxRange::new(2, 3)],
        cut: vec![Cut::Sentence; 3],
        cost: vec![0; 3],
        model_tokens: vec![0; 3],
        over_budget: vec![false; 3],
        prompt_overhead: 0,
    };
    let opinions = Opinions::from_items(
        &tokens,
        vec![Opinion { tokens: IxRange::new(3, 4), polarity: 1.0, intensity: 1.0, negated: false }],
    )
    .unwrap();
    let mut aspects = Aspects::new(TagSet::new("aspect", &["service"]));
    aspects
        .try_push(
            &tokens,
            Some(&sentences),
            Some(&opinions),
            Aspect {
                target: AspectTarget::Term { tokens: IxRange::new(1, 2), sentence: Some(Ix::new(0)) },
                category: Some(Tag(0)),
                polarity: PolarityScores::new(1.0, 0.0, 0.0),
                opinions: &[Ix::new(0)],
            },
        )
        .unwrap();
    aspects
        .try_push(
            &tokens,
            Some(&sentences),
            Some(&opinions),
            Aspect {
                target: AspectTarget::Implicit { sentence: Ix::new(1) },
                category: Some(Tag(0)),
                polarity: PolarityScores::new(0.0, 1.0, 0.0),
                opinions: &[],
            },
        )
        .unwrap();
    macro_rules! check {
        ($head:expr, $good:expr, $bad:expr) => {{
            let h = $head;
            let expected = h.annotate($good).unwrap().0;
            let mut results = h.annotate_batch(vec![$good, $bad, $good]);
            assert!(results.remove(1).is_err());
            for result in results {
                categories_equal(&result.unwrap().0, &expected);
            }
            assert!(h.annotate_batch(Vec::new()).is_empty());
        }};
    }
    check!(SentenceSpanHead::new(vectors.clone(), head.clone()), (&pieces, &sentences), (&bad, &sentences));
    check!(
        ChunkSpanHead::new(vectors.clone(), head.clone()),
        (&pieces, Some(&sentences), &chunks),
        (&bad, Some(&sentences), &chunks)
    );
    check!(
        OpinionSpanHead::new(vectors.clone(), head.clone()),
        (&pieces, &tokens, Some(&sentences), &opinions),
        (&bad, &tokens, Some(&sentences), &opinions)
    );
    check!(
        AspectSpanHead::new(vectors.clone(), head.clone()),
        (&pieces, &tokens, Some(&sentences), &aspects),
        (&bad, &tokens, Some(&sentences), &aspects)
    );
    let mut bad_opinions = opinions.clone();
    bad_opinions.tokens[0] = IxRange::new(1, 1000);
    assert!(
        OpinionSpanHead::new(vectors.clone(), head.clone())
            .annotate((&pieces, &tokens, Some(&sentences), &bad_opinions))
            .is_err()
    );
    aspects.sentence[1] = Some(Ix::new(1000));
    assert!(AspectSpanHead::new(vectors, head).annotate((&pieces, &tokens, Some(&sentences), &aspects)).is_err());
}

#[test]
fn subword_batch_reuses_scratch_without_carrying_offsets_between_views() {
    let sub = SubwordAnnotator::open(&bundle()).unwrap();
    let docs = [Document::new("The good service. ".repeat(30)), Document::new("café 東京"), Document::new("")];
    let invalid = Tokens { spans: Spans::new(ViewId(20)), sentence: None };
    let good: Vec<_> = docs.iter().map(|d| (d.get::<View>().unwrap(), None)).collect();
    let mut batch = good.clone();
    batch.insert(1, (good[0].0, Some(&invalid)));
    let mut got = sub.annotate_batch(batch);
    assert!(got.remove(1).is_err());
    for (input, actual) in good.into_iter().zip(got) {
        let expected = sub.annotate(input).unwrap().0;
        let actual = actual.unwrap().0;
        assert_eq!(expected.ids, actual.ids);
        assert_eq!(expected.spans.columns(), actual.spans.columns());
        assert_eq!(expected.token, actual.token);
        assert_eq!(expected.model, actual.model);
    }
}
