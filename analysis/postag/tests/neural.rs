use turbo_analysis::{Analyzer, Annotator, Document, Error, Ix, Key, Layer, Provenance, Span, Spans, View, ViewId};
use turbo_analysis_layers::{PosTags, Sentences, Tokens};
use turbo_postag::neural::{
    BilstmPosModel, BilstmPosTagger, BilstmPosTrainer, BilstmSettings, FeedforwardPosModel, FeedforwardPosTagger,
    FeedforwardPosTrainer, FeedforwardSettings, MultiTaskSample, NeuralError, PosSample,
};

fn build_tokens(view: ViewId, triples: &[(u32, u32, u32)]) -> Tokens {
    let mut spans = Spans::new(view);
    let mut sentence = Vec::with_capacity(triples.len());
    for &(start, end, sent) in triples {
        spans.push(Span::new(start, end));
        sentence.push(Ix::new(sent));
    }
    Tokens { spans, sentence: Some(sentence) }
}

fn tiny_samples() -> Vec<PosSample> {
    vec![
        PosSample::new(["The", "cat", "sat"], ["DET", "NOUN", "VERB"]).unwrap(),
        PosSample::new(["A", "dog", "ran"], ["DET", "NOUN", "VERB"]).unwrap(),
    ]
}

#[test]
fn bilstm_and_feedforward_annotators_in_pipeline_and_view_validation() {
    let samples = tiny_samples();
    let bilstm_settings =
        BilstmSettings::new(8, 4, 4, 8, 5, 2, 0.02, 5.0, 0.0, 1, 12, 7, 1, 0.0, 0, true, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap();
    let bilstm_model = BilstmPosTrainer::train(&samples, bilstm_settings).unwrap();
    let bilstm_tagger = BilstmPosTagger::new(bilstm_model);

    let ff_settings = FeedforwardSettings::new(8, 12, 10, 2, 0.05, 1e-6, 0.0, 1, 1, 7).unwrap();
    let ff_model = FeedforwardPosTrainer::train(&samples, ff_settings).unwrap();
    let ff_tagger = FeedforwardPosTagger::new(ff_model);

    // Refuse tokens in another view
    let doc_wrong = Document::new("The cat sat");
    let wrong_tokens = build_tokens(ViewId(2), &[(0, 3, 0), (4, 7, 0), (8, 11, 0)]);
    let err_bilstm = bilstm_tagger.annotate((doc_wrong.get::<View>().unwrap(), &wrong_tokens)).err().unwrap();
    assert_eq!(
        err_bilstm,
        Error::Annotator {
            annotator: "turbo-postag:bilstm",
            detail: "the tokens are in view 2, not in view 0 that is tagged".into(),
        }
    );
    let err_ff = ff_tagger.annotate((doc_wrong.get::<View>().unwrap(), &wrong_tokens)).err().unwrap();
    assert_eq!(
        err_ff,
        Error::Annotator {
            annotator: "turbo-postag:feedforward",
            detail: "the tokens are in view 2, not in view 0 that is tagged".into(),
        }
    );

    // Run both in an Analyzer pipeline over two sentences
    let text = "The cat sat A dog ran";
    for (expected_id, use_bilstm) in [("turbo-postag:bilstm", true), ("turbo-postag:feedforward", false)] {
        let mut doc = Document::new(text);
        let view = doc.get::<View>().unwrap().id();
        let mut s = Spans::new(view);
        s.push(Span::new(0, 11));
        s.push(Span::new(12, 21));
        doc.add(&Key::<Sentences>::DEFAULT, Sentences { spans: s }, Provenance::by("test")).unwrap();
        let toks = build_tokens(view, &[(0, 3, 0), (4, 7, 0), (8, 11, 0), (12, 13, 1), (14, 17, 1), (18, 21, 1)]);
        doc.add(&Key::<Tokens>::DEFAULT, toks, Provenance::by("test")).unwrap();

        let analyzer = if use_bilstm {
            Analyzer::builder()
                .given(Key::<Sentences>::DEFAULT)
                .given(Key::<Tokens>::DEFAULT)
                .add(bilstm_tagger.clone())
                .build()
                .unwrap()
        } else {
            Analyzer::builder()
                .given(Key::<Sentences>::DEFAULT)
                .given(Key::<Tokens>::DEFAULT)
                .add(ff_tagger.clone())
                .build()
                .unwrap()
        };
        assert_eq!(analyzer.run(std::slice::from_mut(&mut doc)), [Ok(())]);
        let pos = doc.get::<PosTags>().unwrap();
        assert_eq!(pos.len(), 6);
        let probs = pos.prob.as_ref().unwrap();
        assert_eq!(probs.len(), 6);
        for &p in probs {
            assert!(p > 0.0 && p <= 1.0);
        }
        let prov = doc.provenance(Key::<PosTags>::DEFAULT.id()).unwrap();
        assert_eq!(prov.annotator, expected_id);
    }
}

#[test]
fn bilstm_fine_tuning_dropout_schedule_and_representation_cache() {
    let corpus = vec![
        MultiTaskSample::new(
            ["The", "cat", "sat"],
            ["DET", "NOUN", "VERB"],
            Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
            Some(vec!["Def".into(), "Sing".into(), "Past".into()]),
        )
        .unwrap(),
        MultiTaskSample::new(
            ["A", "dog", "ran"],
            ["DET", "NOUN", "VERB"],
            Some(vec!["DT".into(), "NN".into(), "VBD".into()]),
            Some(vec!["Ind".into(), "Sing".into(), "Past".into()]),
        )
        .unwrap(),
    ];
    let dev = tiny_samples();

    // Train with fine-tuning, word/pretrained/encoder dropout, lr half-life, and validation set
    let settings = BilstmSettings::new(
        6, 4, 4, 6, 4, 1, 0.01, 5.0, 0.1, 1, 10, 17, 1, 0.1, 2, true, 2, 0.1, 0.1, 0.5, 0.25, false,
    )
    .unwrap();
    let vec_fn = |w: &str| Some(vec![w.len() as f32 * 0.1, 0.5]);
    let model =
        BilstmPosTrainer::train_multi_task(&corpus, settings, Some(vec_fn), Some(&["unseen".to_owned()]), Some(&dev))
            .unwrap();

    model.enable_representation_cache();
    let r1 = model.word_representation("The").unwrap();
    let r2 = model.word_representation("cat").unwrap();
    assert_eq!(r1, model.word_representation("The").unwrap());
    assert_eq!(r2, model.word_representation("cat").unwrap());
}

#[test]
fn binary_format_rejects_corrupt_streams() {
    let bad_magic = b"\x00\x0bONLP-BAD-99";
    assert!(matches!(BilstmPosModel::load(&bad_magic[..]), Err(NeuralError::Format(_))));
    assert!(matches!(FeedforwardPosModel::load(&bad_magic[..]), Err(NeuralError::Format(_))));

    // Truncated stream after valid header
    let samples = tiny_samples();
    let ff_settings = FeedforwardSettings::new(4, 6, 1, 2, 0.01, 1e-6, 0.0, 1, 1, 7).unwrap();
    let ff_model = FeedforwardPosTrainer::train(&samples, ff_settings).unwrap();
    let mut buf = Vec::new();
    ff_model.serialize(&mut buf).unwrap();
    assert!(FeedforwardPosModel::load(&buf[..buf.len() / 2]).is_err());
}

#[test]
fn neural_batches_preserve_sentence_boundaries_and_invalid_document_positions() {
    let samples = tiny_samples();
    let settings =
        BilstmSettings::new(8, 4, 4, 8, 5, 2, 0.02, 5.0, 0.0, 1, 12, 7, 1, 0.0, 0, true, 1, 0.0, 0.0, 1.0, 0.0, false)
            .unwrap();
    let model = BilstmPosTrainer::train(&samples, settings).unwrap();
    let ff = FeedforwardPosTrainer::train(
        &samples,
        FeedforwardSettings::new(8, 12, 10, 2, 0.05, 1e-6, 0.0, 1, 1, 7).unwrap(),
    )
    .unwrap();
    let d = Document::new("The cat sat A dog ran");
    let v = d.get::<View>().unwrap();
    let t = build_tokens(ViewId::SOURCE, &[(0, 3, 0), (4, 7, 0), (8, 11, 0), (12, 13, 1), (14, 17, 1), (18, 21, 1)]);
    let short = build_tokens(ViewId::SOURCE, &[(0, 3, 0), (4, 7, 0), (8, 11, 0)]);
    let empty = build_tokens(ViewId::SOURCE, &[]);
    let mut bad = build_tokens(ViewId::SOURCE, &[(0, 3, 0), (4, 7, 0)]);
    bad.sentence = Some(vec![]);
    macro_rules! check {
        ($tagger:expr) => {{
            let tagger = $tagger;
            let expected = [
                tagger.annotate((v, &t)).unwrap().0,
                tagger.annotate((v, &short)).unwrap().0,
                tagger.annotate((v, &empty)).unwrap().0,
            ];
            let mut got = tagger.annotate_batch(vec![(v, &t), (v, &bad), (v, &short), (v, &empty)]);
            assert!(got.remove(1).is_err());
            for (got, want) in got.into_iter().zip(expected) {
                let got = got.unwrap().0;
                assert_eq!(got.tags, want.tags);
                assert_eq!(got.prob, want.prob);
            }
            assert!(tagger.annotate_batch(Vec::new()).is_empty());
        }};
    }
    check!(BilstmPosTagger::new(model.clone()));
    check!(BilstmPosTagger::jvm_parity(model));
    check!(FeedforwardPosTagger::new(ff));
}
