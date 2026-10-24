use super::*;
use turbo_analysis::{Execution, ModelRef};
use turbo_doccat::{DlDoccatError, DlSequenceScorer};

// Count calls while delegating every score to the real BERT bundle.
struct Counted {
    scorer: BundleSequenceScorer,
    calls: AtomicUsize,
}
impl DlSequenceScorer for Counted {
    fn categories(&self) -> &[String] {
        self.scorer.categories()
    }
    fn model_ref(&self) -> ModelRef {
        self.scorer.model_ref()
    }
    fn execution(&self) -> Option<Execution> {
        self.scorer.execution()
    }
    fn max_content_tokens(&self) -> usize {
        self.scorer.max_content_tokens()
    }
    fn score_chunks(&self, chunks: &[String], o: &DlInferenceOptions) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.scorer.score_chunks(chunks, o)
    }
    fn score_subword_windows(&self, windows: &[&[i32]]) -> Result<Vec<Vec<f32>>, DlDoccatError> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.scorer.score_subword_windows(windows)
    }
}

#[test]
fn dl_batches_windows_across_documents_and_span_units() {
    let dir = bert_seq_cls_bundle();
    let sub = SubwordAnnotator::open(&dir.0).unwrap();
    let docs =
        [Document::new("The quick brown fox. The lazy dog."), Document::new("Café naïve RÉSUMÉ."), Document::new("")];
    let subs: Vec<_> = docs.iter().map(|d| sub.pieces(d.get::<View>().unwrap()).unwrap()).collect();
    let mut wrong = sub.pieces(docs[0].get::<View>().unwrap()).unwrap();
    wrong.model.id.push_str("-wrong");
    let sentences: Vec<_> = docs
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let mut spans = Spans::new(ViewId::SOURCE);
            if i == 0 {
                spans.push(Span::new(0, 20));
                spans.push(Span::new(21, d.source().len() as u32));
            } else {
                spans.push(Span::new(0, d.source().len() as u32));
            }
            Sentences { spans }
        })
        .collect();
    for mode in [DlChunkingMode::Subwords, DlChunkingMode::WhitespaceWords] {
        let scorer = BundleSequenceScorer::open(
            &dir.0,
            &EncoderOptions { cpu_threads: 1, max_batch: 3, max_seq: 12, ..Default::default() },
        )
        .unwrap();
        let cat = Arc::new(
            DocumentCategorizerDL::with_options(
                Counted { scorer, calls: AtomicUsize::new(0) },
                DlInferenceOptions {
                    chunking_mode: mode,
                    document_split_size: 4,
                    split_overlap_size: 1,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        let inputs: Vec<_> = docs.iter().zip(&subs).map(|(d, s)| (d.get::<View>().unwrap(), Some(s))).collect();
        let expected: Vec<_> = inputs.iter().map(|&i| cat.annotate(i).unwrap().0).collect();
        let direct = match mode {
            DlChunkingMode::Subwords => cat.categorize_subword_slice(&subs[0].ids).unwrap(),
            DlChunkingMode::WhitespaceWords => cat.categorize_text(docs[0].source()).unwrap(),
        };
        assert_eq!(direct, expected[0].scores);
        let mut batch = inputs.clone();
        batch.insert(1, (inputs[0].0, Some(&wrong)));
        cat.scorer().calls.store(0, Ordering::Relaxed);
        let mut results = cat.annotate_batch(batch);
        assert!(results.remove(1).is_err());
        assert_eq!(cat.scorer().calls.load(Ordering::Relaxed), 1);
        for (r, e) in results.into_iter().zip(expected) {
            let got = r.unwrap().0;
            assert_eq!(got.scores, e.scores);
            assert_eq!(got.category, e.category);
        }
        let spans = DlSpanCategorizer::<_, Sentences>::new(cat.clone());
        let inputs: Vec<_> = inputs.iter().zip(&sentences).map(|(&(v, s), t)| (v, s, t)).collect();
        let expected: Vec<_> = inputs.iter().map(|&i| spans.annotate(i).unwrap().0).collect();
        cat.scorer().calls.store(0, Ordering::Relaxed);
        let results = spans.annotate_batch(inputs);
        assert_eq!(cat.scorer().calls.load(Ordering::Relaxed), 1, "all units in all docs share a scorer call");
        for (r, e) in results.into_iter().zip(expected) {
            let got = r.unwrap().0;
            assert_eq!(got.scores, e.scores);
            assert_eq!(got.category, e.category);
        }
        assert!(spans.annotate_batch(Vec::new()).is_empty());
        assert!(cat.categorize_subword_batches(&[&[]])[0].is_err());
        assert!(cat.categorize_text_batches(&["  "])[0].is_err());
    }
}
