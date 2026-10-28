use super::model::CACHE_ENTRY_LIMIT;
use super::*;
use crate::neural::crf::CrfScoreTable;
use crate::neural::{NeuralError, PosSample};
use turbo_analysis::{Annotator, Document, Error, Span, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;

fn corpus() -> Vec<PosSample> {
    let distinct = vec![
        PosSample::new(["the", "dog", "barks"], ["DT", "NN", "VBZ"]).unwrap(),
        PosSample::new(["the", "cat", "sleeps"], ["DT", "NN", "VBZ"]).unwrap(),
        PosSample::new(["dogs", "bark"], ["NNS", "VBP"]).unwrap(),
        PosSample::new(["she", "eats", "fish"], ["PRP", "VBZ", "NN"]).unwrap(),
    ];
    let mut out = Vec::with_capacity(40 * distinct.len());
    for _ in 0..40 {
        out.extend_from_slice(&distinct);
    }
    out
}

#[test]
fn shapes_and_suffixes_match_java_conformance() {
    // Each row is what Java's FeedforwardPOSContext.shape returns.
    let shape_cases = [
        ("Paris", "*cap*"),
        ("Mr.", "*cap*"),
        ("U.S.", "*allcaps*"),
        ("3.5", "*digit*"),
        ("iPhone", "*lower*"),
        ("ALL-CAPS", "*allcaps*"),
        ("x", "*lower*"),
        ("X", "*allcaps*"),
        ("hello-world", "*lower*"),
        ("Hello-World", "*cap*"),
        ("McDonald's", "*cap*"),
        ("USA", "*allcaps*"),
        ("2020", "*digit*"),
        ("1,000", "*digit*"),
        ("B2B", "*alnum*"),
        ("3rd", "*alnum*"),
        ("--", "*other*"),
        (".", "*other*"),
        (",", "*other*"),
        ("$", "*other*"),
        ("(", "*other*"),
        ("dog", "*lower*"),
        ("", "*other*"),
        // Titlecase and other letters are not uppercase, so they count as lowercase.
        ("\u{1C5}", "*lower*"),
        ("\u{1C4}\u{1C5}", "*cap*"),
        ("\u{4E2D}\u{6587}", "*lower*"),
        // Non-ASCII decimal digits are digits.
        ("\u{0663}", "*digit*"),
        // Supplementary characters are classified by code point.
        ("\u{10400}\u{10428}", "*cap*"),
        ("\u{10400}\u{10401}", "*allcaps*"),
        ("\u{10428}\u{10429}", "*lower*"),
        ("\u{10428}1", "*alnum*"),
        ("\u{1D400}\u{1D401}\u{1D402}", "*allcaps*"),
        ("\u{1D7CE}", "*digit*"),
        ("\u{1F600}", "*other*"),
        // A circled letter is uppercase but not a letter.
        ("\u{24B6}b", "*cap*"),
    ];
    for (word, expected) in shape_cases {
        assert_eq!(FeedforwardPosContext::shape(word), expected, "{word}");
    }

    // Java takes the last code points.
    let suffix_cases = [
        ("dog", 2, "og"),
        ("dog", 3, "dog"),
        ("a", 2, "a"),
        ("a", 3, "a"),
        ("", 2, ""),
        ("running", 3, "ing"),
        ("a\u{10428}", 2, "a\u{10428}"),
        ("\u{10428}\u{10429}\u{1042A}", 2, "\u{10429}\u{1042A}"),
        ("ab\u{10428}", 2, "b\u{10428}"),
        ("caf\u{E9}", 3, "af\u{E9}"),
    ];
    for (word, len, expected) in suffix_cases {
        assert_eq!(FeedforwardPosContext::suffix(word, len), expected, "{word} {len}");
    }

    // Java lowercases the whole token for the suffixes; only the word
    // lookup passes symbols starting with `*` through unchanged.
    let symbols = FeedforwardPosContext::extract(&["*ABCD", "Mr."], 0, None, None);
    assert_eq!(&symbols[..5], [ABSENT, ABSENT, "*ABCD", "Mr.", ABSENT]);
    assert_eq!(&symbols[5..10], ["cd", "bcd", ABSENT, "*allcaps*", "*cap*"]);
}

#[test]
fn training_inference_and_round_trip_match_java_exact_outputs() {
    let settings = FeedforwardSettings::new(16, 32, 80, 32, 0.05, 0.0, 0.0, 1, 1, 17).unwrap();
    let model = FeedforwardPosTrainer::train(&corpus(), settings).unwrap();
    let tagger = FeedforwardPosTagger::new(model.clone());

    assert_eq!(tagger.tag(&["the", "dog", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
    assert_eq!(tagger.tag(&["she", "eats", "fish"]).unwrap(), vec!["PRP", "VBZ", "NN"]);
    assert_eq!(tagger.tag(&["dog"]).unwrap(), vec!["NN"]);
    assert_eq!(tagger.tag(&["the", "wug", "barks"]).unwrap(), vec!["DT", "NN", "VBZ"]);
    assert_eq!(tagger.tag(&["blorp", "quexination"]).unwrap(), vec!["NNS", "VBZ"]);
    assert!(tagger.tag(&Vec::<&str>::new()).unwrap().is_empty());

    let seqs = tagger.top_k_sequences(&["the", "dog", "barks"]).unwrap();
    assert_eq!(seqs.len(), 1);
    assert_eq!(seqs[0].outcomes(), &["DT", "NN", "VBZ"]);
    for &p in seqs[0].probs() {
        assert!(p > 0.9 && p <= 1.0);
    }

    let mut buf = Vec::new();
    model.serialize(&mut buf).unwrap();
    let reloaded = FeedforwardPosModel::load(&buf[..]).unwrap();
    let reloaded_tagger = FeedforwardPosTagger::new(reloaded);
    assert_eq!(tagger.tag(&["the", "wug", "barks"]).unwrap(), reloaded_tagger.tag(&["the", "wug", "barks"]).unwrap());
}

#[test]
fn pretrained_vectors_and_lexicon_separate_indistinguishable_words() {
    let class_a = ["aazqx", "bazqx", "cazqx", "dazqx"];
    let class_b = ["eezqx", "fezqx", "gezqx", "hezqx"];
    let mut samples = Vec::new();
    for w in class_a {
        samples.push(PosSample::new([w], ["A"]).unwrap());
    }
    for w in class_b {
        samples.push(PosSample::new([w], ["B"]).unwrap());
    }
    let settings = FeedforwardSettings::new(8, 16, 60, 4, 0.1, 1e-8, 0.0, 2, 2, 17).unwrap();
    let vec_fn = |w: &str| Some(if w.as_bytes()[0] <= b'd' { vec![1.0f32, 0.0f32] } else { vec![0.0f32, 1.0f32] });
    let lexicon = vec!["aazqy".to_owned(), "eezqy".to_owned()];
    let model = FeedforwardPosTrainer::train_with_lexicon(&samples, settings, vec_fn, &lexicon).unwrap();
    assert!(model.uses_pretrained_vectors());
    let tagger = FeedforwardPosTagger::new(model.clone());
    for w in class_a {
        assert_eq!(tagger.tag(&[w]).unwrap(), vec!["A"]);
    }
    for w in class_b {
        assert_eq!(tagger.tag(&[w]).unwrap(), vec!["B"]);
    }
    assert_eq!(tagger.tag(&["aazqy"]).unwrap(), vec!["A"]);
    assert_eq!(tagger.tag(&["eezqy"]).unwrap(), vec!["B"]);

    let mut buf = Vec::new();
    model.serialize(&mut buf).unwrap();
    assert_eq!(&buf[2..2 + buf[1] as usize], b"ONLP-FFPT-2");
    let loaded = FeedforwardPosModel::load(&buf[..]).unwrap();
    assert_eq!(
        FeedforwardPosTagger::new(loaded).tag(&["aazqx", "eezqx", "zzzqx"]).unwrap(),
        tagger.tag(&["aazqx", "eezqx", "zzzqx"]).unwrap()
    );
}

#[test]
fn scoring_cache_budget_and_cancellation() {
    let vocab = vec![(UNKNOWN.to_owned(), 0), (ABSENT.to_owned(), 0), ("cat".to_owned(), 0)];
    let emb = vec![vec![1.0f32, CrfScoreTable::scalb(1.0, -24) as f32]];
    let mut hw = vec![vec![0.0f32; 24]];
    hw[0][0] = 1.0;
    hw[0][1] = 1.0;
    let hb = vec![-1.0f32];
    let scale = CrfScoreTable::scalb(1.0, 80) as f32;
    let ow = vec![vec![-scale], vec![scale]];
    let ob = vec![0.0f32, 0.0f32];

    let model = FeedforwardPosModel::new(
        vocab.clone(),
        vocab.clone(),
        vocab.clone(),
        vocab,
        vec!["A".into(), "B".into()],
        2,
        emb,
        hw,
        hb,
        ow,
        ob,
    );
    let ctx = FeedforwardPosContext::extract(&["cat"], 0, None, None);
    let f_ids = model.feature_ids(&ctx);
    let direct = model.score(&f_ids).unwrap();
    model.enable_scoring_cache();
    let cached1 = model.score(&f_ids).unwrap();
    let cached2 = model.score(&f_ids).unwrap();
    assert_eq!(direct, cached1);
    assert_eq!(direct, cached2);

    if let Some(ref c) = *model.cache.read().unwrap() {
        assert!(c.budget() < CACHE_ENTRY_LIMIT);
        assert!(c.cached_entries() > 0);
        c.set_budget(0);
        assert_eq!(c.budget(), 0);
    }
}

/// Builds a one-row model whose single slot adds two products of 2^-53 to
/// a hidden bias of 1.0: summed one by one into the bias they round away,
/// summed from zero first they do not.
fn order_sensitive_model() -> FeedforwardPosModel {
    let vocab = vec![(UNKNOWN.to_owned(), 0), (ABSENT.to_owned(), 0)];
    let tiny = CrfScoreTable::scalb(1.0, -53) as f32;
    let emb = vec![vec![tiny, tiny]];
    let mut hw = vec![vec![0.0f32; 24]];
    hw[0][0] = 1.0;
    hw[0][1] = 1.0;
    let scale = CrfScoreTable::scalb(1.0, 60) as f32;
    FeedforwardPosModel::new(
        vocab.clone(),
        vocab.clone(),
        vocab.clone(),
        vocab,
        vec!["A".into(), "B".into()],
        2,
        emb,
        hw,
        vec![1.0f32],
        vec![vec![-scale], vec![scale]],
        vec![0.0f32, 0.0f32],
    )
}

#[test]
fn scores_do_not_depend_on_the_cache_state() {
    let ctx = FeedforwardPosContext::extract(&["cat"], 0, None, None);

    let uncached = order_sensitive_model();
    let f_ids = uncached.feature_ids(&ctx);
    let direct = uncached.score(&f_ids).unwrap();

    let cached = order_sensitive_model();
    cached.enable_scoring_cache();
    let with_room = cached.score(&f_ids).unwrap();

    let full = order_sensitive_model();
    full.enable_scoring_cache();
    if let Some(ref c) = *full.cache.read().unwrap() {
        c.set_budget(0);
    }
    let without_room = full.score(&f_ids).unwrap();

    assert_eq!(direct, with_room);
    assert_eq!(without_room, with_room);
}

#[test]
fn score_and_score_with_pretrained_reject_out_of_bounds_indices_without_panicking() {
    let model = order_sensitive_model();
    let mut bad_features = [0usize; SLOTS];
    bad_features[3] = model.embeddings().len();
    let err = model.score(&bad_features).unwrap_err();
    assert!(
        matches!(&err, NeuralError::InvalidArgument(msg) if msg.contains("feature id at slot 3 out of bounds")),
        "unexpected error: {err:?}"
    );

    model.enable_scoring_cache();
    let err_cached = model.score(&bad_features).unwrap_err();
    assert!(
        matches!(&err_cached, NeuralError::InvalidArgument(msg) if msg.contains("feature id at slot 3 out of bounds")),
        "unexpected cached error: {err_cached:?}"
    );

    // Model with 1 pretrained vector row.
    let vocab = vec![(UNKNOWN.to_owned(), 0), (ABSENT.to_owned(), 0)];
    let pretrained_model = FeedforwardPosModel::with_pretrained(
        vocab.clone(),
        vocab.clone(),
        vocab.clone(),
        vocab,
        vec!["A".into(), "B".into()],
        2,
        vec![vec![0.5f32, 0.5f32]],
        vec![vec![0.0f32; SLOTS * 2 + PRETRAINED_SLOTS * 2]],
        vec![0.0f32],
        vec![vec![1.0f32], vec![-1.0f32]],
        vec![0.0f32, 0.0f32],
        2,
        vec![("cat".into(), 0)],
        vec![vec![1.0f32, 2.0f32]],
    );

    let valid_features = [0usize; SLOTS];
    assert!(pretrained_model.score_with_pretrained(&valid_features, Some(&[-1, 0, -1, -1, -1])).is_ok());

    for bad_rows in [[-2, 0, -1, -1, -1], [isize::MIN, 0, -1, -1, -1], [0, 1, -1, -1, -1]] {
        let err = pretrained_model.score_with_pretrained(&valid_features, Some(&bad_rows)).unwrap_err();
        assert!(
            matches!(&err, NeuralError::InvalidArgument(msg) if msg.contains("pretrained row at slot")),
            "unexpected error for {bad_rows:?}: {err:?}"
        );
    }
}

#[test]
fn feedforward_annotator_rejects_invalid_token_spans_without_panicking() {
    let tagger = FeedforwardPosTagger::new(order_sensitive_model());
    let doc = Document::new("café");
    let view = doc.get::<View>().unwrap();

    for bad_span in [
        Span::new(0, 99),          // past end of text
        Span { start: 3, end: 1 }, // inverted span
        Span::new(0, 4),           // splits 2-byte UTF-8 'é' (bytes 3..5)
    ] {
        let mut spans = Spans::new(ViewId::SOURCE);
        spans.push(bad_span);
        let tokens = Tokens { spans, sentence: None };
        let err = tagger.annotate((view, &tokens)).err().unwrap();
        assert!(
            matches!(&err, Error::Annotator { annotator: FeedforwardPosTagger::ID, detail } if detail.contains("outside the text or off a UTF-8 boundary")),
            "unexpected error for {bad_span:?}: {err:?}"
        );
    }
}
