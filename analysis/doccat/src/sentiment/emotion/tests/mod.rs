//! The emotion annotator on documents built here: the seed lexicon with
//! negation, neutral and emoji, the learned and hybrid stages over
//! vectors given as data, provenance, and the evaluation helper.

use super::*;
use turbo_analysis::{Analyzer, Annotator, Document, Ix, Key, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{EmotionLayer, Sentences, Vectors};

mod head;

/// Sentences at `ranges`, tokens at whitespace and at Unicode punctuation
/// other than an apostrophe.
pub(super) fn doc(text: &str, ranges: &[(u32, u32)]) -> Document {
    let mut d = Document::new(text);
    let mut sentences = Spans::new(ViewId::SOURCE);
    let mut tokens = Spans::new(ViewId::SOURCE);
    let mut sentence = Vec::new();
    for (s, &(a, b)) in ranges.iter().enumerate() {
        sentences.push(Span::new(a, b));
        let mut start = None;
        let mut push = |x: usize, y: usize, tokens: &mut Spans<Tokens>| {
            tokens.push(Span::new(x as u32, y as u32));
            sentence.push(Ix::new(s as u32));
        };
        for (i, c) in text[a as usize..b as usize].char_indices().map(|(i, c)| (i + a as usize, c)) {
            let apostrophe = matches!(c, '\'' | '\u{2019}');
            let cut =
                c.is_whitespace() || (!apostrophe && turbo_unicode::category::general_category(c).is_punctuation());
            if cut {
                if let Some(x) = start.take() {
                    push(x, i, &mut tokens);
                }
                if !c.is_whitespace() {
                    push(i, i + c.len_utf8(), &mut tokens);
                }
            } else if start.is_none() {
                start = Some(i);
            }
        }
        if let Some(x) = start {
            push(x, b as usize, &mut tokens);
        }
    }
    d.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    d.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: Some(sentence) }, Provenance::by("test")).unwrap();
    d
}

/// Runs `a` on `d` and gives each unit's scores and evidence.
pub(super) fn run(a: EmotionAnnotator, d: &mut Document) -> Result<Vec<(EmotionScores, Vec<EmotionEvidence>)>, String> {
    let mut b = Analyzer::builder().given(Key::<Sentences>::DEFAULT).given(Key::<Tokens>::DEFAULT);
    if d.get::<Vectors<Sentences>>().is_some() {
        b = b.given(Key::<Vectors<Sentences>>::DEFAULT);
    }
    let p = b.add(a).build().unwrap();
    if let Some(Err(e)) = p.run(std::slice::from_mut(d)).into_iter().next() {
        return Err(e.to_string());
    }
    let layer = d.get::<EmotionLayer>().unwrap();
    Ok((0..layer.target.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            (layer.scores_of(ix), layer.evidence_of(ix).collect())
        })
        .collect())
}

fn one(text: &str) -> (EmotionScores, Vec<EmotionEvidence>) {
    run(EmotionAnnotator::new(), &mut doc(text, &[(0, text.len() as u32)])).unwrap().remove(0)
}

use EmotionDimension as D;

#[test]
fn cues_score_their_labels_with_their_tokens_as_evidence() {
    // Tokens: I am so happy and grateful today !
    let (s, ev) = one("I am so happy and grateful today!");
    assert!(s.get(D::Joy) >= 0.5 && s.get(D::Gratitude) >= 0.5);
    assert!(s.get(D::Anger) == 0.0 && s.get(D::Neutral) < 0.5);
    assert_eq!(ev, [(D::Gratitude, IxRange::new(5, 6)), (D::Joy, IxRange::new(3, 4))]);
    // Two cues of one label combine by a noisy-or.
    let (s, _) = one("Happy, so glad.");
    assert!((s.get(D::Joy) - (1.0 - 0.2 * 0.25)).abs() < 1e-12, "{}", s.get(D::Joy));
}

#[test]
fn a_negated_cue_does_not_count() {
    for text in ["I am not happy about this.", "Never been less scared.", "We are hardly excited"] {
        let (s, ev) = one(text);
        assert!(s.active(0.5).all(|d| d == D::Neutral), "{text}: {s:?}");
        assert!(ev.is_empty(), "{text}");
    }
    // A negator past the three tokens before the cue does not reach it.
    let (s, _) = one("Not that it matters, I am happy.");
    assert!(s.get(D::Joy) >= 0.5);
}

#[test]
fn typographic_punctuation_and_case_do_not_hide_a_cue() {
    // A curly apostrophe still negates.
    let (s, ev) = one("I don\u{2019}t feel happy.");
    assert!(s.get(D::Joy) < 0.5 && ev.is_empty(), "{s:?}");
    // Ellipsis, curly quotes, inverted and fullwidth marks around a cue.
    for text in ["I am happy\u{2026}", "I am \u{201C}happy\u{201D}", "\u{00BF}Happy?", "I am happy\u{3002}"] {
        assert!(one(text).0.get(D::Joy) >= 0.5, "{text}");
    }
    // The lookup form on its own, where a tokenizer left the marks on.
    assert_eq!(annotate::key("\u{201C}Happy\u{2026}\u{201D}"), "happy");
    assert_eq!(annotate::key("Don\u{2019}t"), "don't");
    assert_eq!(annotate::key("n\u{02BC}t"), "n't");
    assert_eq!(annotate::key("\u{1F602}"), "\u{1F602}");
    assert_eq!(annotate::key("HAPPY"), "happy");
}

#[test]
fn text_without_cues_is_neutral_with_no_evidence() {
    let (s, ev) = one("The meeting is at noon in room four.");
    assert_eq!(s.get(D::Neutral), 1.0);
    assert!(D::ALL.iter().filter(|&&d| d != D::Neutral).all(|&d| s.get(d) == 0.0));
    assert!(ev.is_empty());
}

#[test]
fn an_emoji_is_a_cue() {
    let (s, ev) = one("That joke \u{1F602}");
    assert!(s.get(D::Amusement) >= 0.5);
    assert_eq!(ev, [(D::Amusement, IxRange::new(2, 3))]);
}

/// Fear reads value 0 and joy value 1, weight 6, bias -3; neutral bias 0.
fn head() -> EmotionVectorHead {
    let mut w = vec![0.0f32; 28 * 2];
    w[D::Fear.index() * 2] = 6.0;
    w[D::Joy.index() * 2 + 1] = 6.0;
    let mut b = [-3.0f32; 28];
    b[D::Neutral.index()] = 0.0;
    EmotionVectorHead::new(2, w, b).unwrap()
}

fn model() -> ModelRef {
    ModelRef { id: "test-encoder".into(), revision: "1".into(), sha256: "e".repeat(64) }
}

const TEXT: &str = "I am so happy today. The lights went out at night.";
const RANGES: [(u32, u32); 2] = [(0, 20), (21, 50)];

fn with_vectors(vectors: &[[f32; 2]]) -> Document {
    let mut d = doc(TEXT, &RANGES);
    let data = vectors.iter().flatten().copied().collect();
    d.add(&Key::DEFAULT, Vectors::<Sentences>::new(2, data, Vec::new(), model()), Provenance::by("test")).unwrap();
    d
}

#[test]
fn the_three_stages_score_with_their_own_signals() {
    // The lexicon finds joy in the first sentence; the vectors say the
    // second is fearful.
    let vectors = [[0.0, 0.0], [1.0, 0.0]];
    let h: Arc<dyn EmotionVectorScorer> = Arc::new(head());
    let at = |stage| {
        run(EmotionAnnotator::new().with_vector_scorer(Arc::clone(&h)).with_stage(stage), &mut with_vectors(&vectors))
    };
    let lex = at(EmotionStage::Lexicon).unwrap();
    let learned = at(EmotionStage::Learned).unwrap();
    let hybrid = at(EmotionStage::Hybrid).unwrap();
    assert!(lex[0].0.get(D::Joy) >= 0.5 && lex[1].0.get(D::Fear) < 0.5);
    assert!(learned[0].0.get(D::Joy) < 0.5 && learned[1].0.get(D::Fear) >= 0.5);
    assert_eq!(learned[1].0, head().score(&vectors[1]).unwrap());
    assert!(hybrid[0].0.get(D::Joy) >= 0.5 && hybrid[1].0.get(D::Fear) >= 0.5);
    // The head's positive points at its whole sentence (tokens 6..13);
    // the lexicon's at its cue.
    assert_eq!(learned[1].1, [(D::Fear, IxRange::new(6, 13))]);
    assert_eq!(hybrid[0].1, [(D::Joy, IxRange::new(3, 4))]);
    // Neutral: the head's, capped by the strongest fused emotion.
    assert!(hybrid[1].0.get(D::Neutral) <= 1.0 - hybrid[1].0.get(D::Fear) + 1e-12);
    // The stage is in the model id and the configuration hash.
    let ann = |stage| EmotionAnnotator::new().with_vector_scorer(Arc::clone(&h)).with_stage(stage);
    let ids: Vec<String> = [EmotionStage::Lexicon, EmotionStage::Learned, EmotionStage::Hybrid]
        .into_iter()
        .map(|s| Annotator::model(&ann(s)).unwrap().id)
        .collect();
    assert_eq!(ids, ["goemotions-seed:lexicon", "goemotions-seed:learned", "goemotions-seed:hybrid"]);
    let hashes: Vec<u64> = [EmotionStage::Lexicon, EmotionStage::Learned, EmotionStage::Hybrid]
        .into_iter()
        .map(|s| ann(s).config())
        .collect();
    assert!(hashes[0] != hashes[1] && hashes[1] != hashes[2] && hashes[0] != hashes[2]);
    // A head bound to the encoder names it.
    let bound = EmotionAnnotator::new()
        .with_vector_scorer(Arc::new(head().with_model(model())))
        .with_stage(EmotionStage::Hybrid);
    assert_eq!(Annotator::model(&bound).unwrap().id, "test-encoder:emotion-hybrid");
    // Two heads on one encoder are two references.
    let other = EmotionVectorHead::new(2, vec![0.5; 56], [0.0; 28]).unwrap().with_model(model());
    let bound2 = EmotionAnnotator::new().with_vector_scorer(Arc::new(other)).with_stage(EmotionStage::Hybrid);
    assert_ne!(Annotator::model(&bound).unwrap().revision, Annotator::model(&bound2).unwrap().revision);
}

#[test]
fn the_learned_stages_need_a_head_vectors_and_the_right_encoder() {
    let learned = EmotionAnnotator::new().with_stage(EmotionStage::Learned);
    assert!(run(learned.clone(), &mut with_vectors(&[[0.0, 0.0]; 2])).unwrap_err().contains("need a vector scorer"));
    let with_head = learned.with_vector_scorer(Arc::new(head()));
    let e = run(with_head.clone(), &mut doc(TEXT, &RANGES)).unwrap_err();
    assert!(e.contains("vector for every unit"), "{e}");
    let mut d = doc(TEXT, &RANGES);
    let v = Vectors::<Sentences>::new(2, vec![0.0; 4], vec![Ix::new(1)], model());
    d.add(&Key::DEFAULT, v, Provenance::by("test")).unwrap();
    assert!(run(with_head, &mut d).unwrap_err().contains("unit 1"));
    let other = ModelRef { id: "other-encoder".into(), ..model() };
    let bound =
        EmotionAnnotator::new().with_stage(EmotionStage::Hybrid).with_vector_scorer(Arc::new(head().with_model(other)));
    assert!(run(bound.clone(), &mut with_vectors(&[[0.0, 0.0]; 2])).unwrap_err().contains("other-encoder"));
    // The lexicon stage reads no vectors, so checks no encoder.
    assert!(run(bound.with_stage(EmotionStage::Lexicon), &mut with_vectors(&[[0.0, 0.0]; 2])).is_ok());
    assert!(run(EmotionAnnotator::new().with_threshold(f64::NAN), &mut doc(TEXT, &RANGES)).is_err());
}

#[test]
fn the_layer_round_trips_through_the_document_graph() {
    use turbo_analysis::export::{DocumentGraph, export, import};
    use turbo_analysis_layers::export::codecs;
    let mut d = with_vectors(&[[0.0, 0.0], [1.0, 0.0]]);
    let a = EmotionAnnotator::new().with_vector_scorer(Arc::new(head())).with_stage(EmotionStage::Hybrid);
    let want = run(a, &mut d).unwrap();
    let bytes = export(&d, &codecs()).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs()).unwrap();
    let layer = back.get::<EmotionLayer>().unwrap();
    for (i, (s, ev)) in want.iter().enumerate() {
        let ix = Ix::new(i as u32);
        assert_eq!(&layer.scores_of(ix), s);
        assert_eq!(&layer.evidence_of(ix).collect::<Vec<_>>(), ev);
    }
}

#[test]
fn the_evaluation_counts_each_label() {
    let text = "I am so happy. I am not happy. The bus is late.";
    let mut d = doc(text, &[(0, 14), (15, 30), (31, 47)]);
    run(EmotionAnnotator::new(), &mut d).unwrap();
    let layer = d.get::<EmotionLayer>().unwrap();
    let gold = [
        GoldEmotionUnit { labels: vec![D::Joy] },
        GoldEmotionUnit { labels: vec![D::Disappointment] },
        GoldEmotionUnit { labels: vec![D::Neutral] },
    ];
    let m = evaluate_emotion_corpus(&gold, layer, 0.5).unwrap();
    let joy = m.per_label[D::Joy.index()].1;
    assert_eq!((joy.tp, joy.fp, joy.fn_count), (1, 0, 0));
    let dis = m.per_label[D::Disappointment.index()].1;
    assert_eq!((dis.tp, dis.fn_count, dis.recall), (0, 1, 0.0));
    // Neutral fires on the last two: one right, one wrong.
    let neutral = m.per_label[D::Neutral.index()].1;
    assert_eq!((neutral.tp, neutral.fp), (1, 1));
    assert_eq!((m.active, m.active_with_evidence), (1, 1));
    assert!(evaluate_emotion_corpus(&gold[..2], layer, 0.5).is_err());

    let mut per_label = [0.5f64; 28];
    per_label[D::Joy.index()] = 0.99;
    let m_high = evaluate_emotion_corpus_with_thresholds(&gold, layer, &per_label).unwrap();
    assert_eq!(m_high.per_label[D::Joy.index()].1.tp, 0);

    let mut d_thresh = doc("I am happy.", &[(0, 11)]);
    let res = run(EmotionAnnotator::new().with_thresholds(per_label), &mut d_thresh).unwrap();
    assert!(res[0].1.is_empty());
    let mut bad_thresh = [0.5f64; 28];
    bad_thresh[0] = f64::NAN;
    assert!(run(EmotionAnnotator::new().with_thresholds(bad_thresh), &mut d_thresh).is_err());
    assert!(evaluate_emotion_corpus_with_thresholds(&gold, layer, &bad_thresh).is_err());
}
