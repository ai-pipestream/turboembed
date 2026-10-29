//! `PragmaticsHeadAnnotator` (#324): incongruity, emoji and an embedding
//! head combined.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Categories, Pragmatics, Sentences, Vectors};
use turbo_doccat::sentiment::LexiconSentiment;
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[derive(Debug)]
struct HeadPragmaticsScorer(Arc<ClassifierHead>);

impl turbo_doccat::PragmaticsVectorScorer for HeadPragmaticsScorer {
    fn score_sentence_vector(&self, vec: &[f32]) -> Result<f64, String> {
        let mut probs = vec![0.0f64; self.0.tags().len()];
        self.0.score(vec, &mut probs)?;
        let pos_idx = self
            .0
            .tags()
            .tag("sarcastic")
            .or_else(|| self.0.tags().tag("ironic"))
            .or_else(|| self.0.tags().tag("sarcasm"))
            .unwrap_or(Tag(0))
            .0 as usize;
        Ok(probs[pos_idx].clamp(0.0, 1.0))
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }

    fn model_ref(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }
}

#[test]
fn pragmatics_annotator_combines_incongruity_emoji_and_embedding_head() {
    use turbo_doccat::{PragmaticsAnnotator, PragmaticsVectorScorer};
    use turbo_normalize::emoji::EmojiAnnotate;

    let (sub, vec_ann) = open_cpu(64);

    // Train a sentence-level ClassifierHead on implicit irony examples vs literal sentences.
    let sarcastic_train = [
        "Another 7am status meeting on Saturday morning, my absolute dream.",
        "Nothing says competence like pushing straight to production on Friday night.",
        "Clearly the best use of four engineering weeks was renaming a button.",
    ];
    let literal_train = [
        "We deployed the patch on Tuesday morning after running the test suite.",
        "The team fixed the database indexing bug and reduced query latency.",
        "Our documentation now covers all configuration flags and error codes.",
    ];

    let centroids = LabelCentroids::from_text_examples(
        &vec_ann,
        &sub,
        "pragmatics",
        &[("sarcastic", &sarcastic_train[..]), ("literal", &literal_train[..])],
        CentroidNorm::L2,
        0.1,
    )
    .unwrap();

    let tags = TagSet::new("pragmatics", &["sarcastic", "literal"]);
    let mut train_vecs: Vec<(Vec<f32>, Tag)> = Vec::new();
    for s in &sarcastic_train {
        let doc = make_doc_with_sentences(s, &[(0, s.len() as u32)]);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&vec_ann)))
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
        train_vecs.push((v, Tag(0)));
    }
    for s in &literal_train {
        let doc = make_doc_with_sentences(s, &[(0, s.len() as u32)]);
        let pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&vec_ann)))
            .build()
            .unwrap();
        let mut docs = [doc];
        pipe.run(&mut docs).remove(0).unwrap();
        let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
        train_vecs.push((v, Tag(1)));
    }
    let train_refs: Vec<(&[f32], Tag)> = train_vecs.iter().map(|(v, t)| (v.as_slice(), *t)).collect();
    let head = Arc::new(
        ClassifierHead::fit_single_label(vec_ann.model().clone(), 32, tags, &train_refs, 1e-4, 200, 2.0).unwrap(),
    );

    // Build a 4-sentence document exercising all three #324 signals + literal control:
    //   0. Riloff incongruity: "I just love waiting two hours on hold for customer service."
    //   1. Emoji mismatch: "What a wonderful way to start the morning 😭"
    //   2. Implicit irony (embedding head only, no explicit contrast):
    //      "Another 7am status meeting on Saturday morning, my absolute dream."
    //   3. Literal positive control (polarity untouched, pragmatics near 0):
    //      "We deployed the patch on Tuesday morning after running the test suite."
    let s0 = "I just love waiting two hours on hold for customer service.";
    let s1 = "What a wonderful way to start the morning 😭";
    let s2 = "Another 7am status meeting on Saturday morning, my absolute dream.";
    let s3 = "We deployed the patch on Tuesday morning after running the test suite.";
    let full = format!("{s0} {s1} {s2} {s3}");
    let o0 = (0, s0.len() as u32);
    let o1 = (o0.1 + 1, o0.1 + 1 + s1.len() as u32);
    let o2 = (o1.1 + 1, o1.1 + 1 + s2.len() as u32);
    let o3 = (o2.1 + 1, full.len() as u32);

    let scorer: Arc<dyn PragmaticsVectorScorer> = Arc::new(HeadPragmaticsScorer(Arc::clone(&head)));
    let prag_ann = PragmaticsAnnotator::new().with_vector_scorer(scorer);

    let pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(EmojiAnnotate::default())
        .add(LexiconSentiment::new())
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .step(Bound::new(VectorHeadAnnotator::<Sentences>::new(centroids)).write(0, "prag-centroids"))
        .add(prag_ann)
        .build()
        .unwrap();

    let mut docs = [make_doc_with_sentences(&full, &[o0, o1, o2, o3])];
    pipe.run(&mut docs).remove(0).unwrap();
    let doc = &docs[0];

    let prag = doc.get::<Pragmatics>().unwrap();
    assert_eq!(prag.len(), 4);

    // The centroid head wrote its own Categories<Sentences> instance; it is not
    // what PragmaticsAnnotator read (that was the lexicon polarity at the
    // default instance), so check it on its own: the implicit-irony training
    // sentence lands on "sarcastic" and the literal one on "literal".
    let cent = doc.get_key::<Categories<Sentences>>(&Key::named("prag-centroids")).unwrap();
    assert_eq!(cent.len(), 4);
    assert_eq!(cent.best_category(2), "sarcastic");
    assert_eq!(cent.best_category(3), "literal");

    // Sentence 0: Riloff incongruity -> high score and Some(Incongruity).
    let p0 = prag.get(Ix::new(0));
    assert!(p0.score >= 0.65, "expected high Riloff incongruity score, got {}", p0.score);
    assert!(p0.evidence.is_some(), "expected Some(Incongruity) on sentence 0");

    // Sentence 1: Emoji mismatch -> high score and Some(Incongruity) with negative emoji span.
    let p1 = prag.get(Ix::new(1));
    assert!(p1.score >= 0.65, "expected high emoji mismatch score, got {}", p1.score);
    assert!(p1.evidence.is_some(), "expected Some(Incongruity) on sentence 1");

    // Sentence 2: Implicit irony via embedding head -> high score and `evidence: None`.
    let p2 = prag.get(Ix::new(2));
    assert!(p2.score >= 0.60, "expected high embedding-head score, got {}", p2.score);
    assert!(p2.evidence.is_none(), "expected evidence: None for head-only prediction on sentence 2");

    // Sentence 3: Literal sentence -> low score and `evidence: None`.
    let p3 = prag.get(Ix::new(3));
    assert!(p3.score < 0.35, "expected low score on literal sentence 3, got {}", p3.score);
    assert!(p3.evidence.is_none());

    // When TURBO_TEST_DEVICE or TURBO_TEST_REQUIRE_CUDA=1 is set, verify CPU vs GPU parity.
    let want = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(w) => Some(w),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").as_deref() == Ok("1") => Some("cuda".to_string()),
        Err(_) => None,
    };
    if let Some(want) = want {
        let device = match want.parse() {
            Ok(i) => Device::Index(i),
            Err(_) => Device::Backend(want.clone()),
        };
        let dev_opts = VectorOptions {
            encoder: EncoderOptions { device, max_seq: 64, ..EncoderOptions::default() },
            ..VectorOptions::default()
        };
        let dev_vec = Arc::new(
            VectorAnnotator::open(&bert(), &dev_opts).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={want}: {e}")),
        );
        let dev_scorer: Arc<dyn PragmaticsVectorScorer> = Arc::new(HeadPragmaticsScorer(Arc::clone(&head)));
        let dev_prag_ann = PragmaticsAnnotator::new().with_vector_scorer(dev_scorer);
        let dev_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .add(SimpleTokenizer::default())
            .add(EmojiAnnotate::default())
            .add(LexiconSentiment::new())
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&dev_vec)))
            .add(dev_prag_ann)
            .build()
            .unwrap();
        let mut dev_docs = [make_doc_with_sentences(&full, &[o0, o1, o2, o3])];
        dev_pipe.run(&mut dev_docs).remove(0).unwrap();
        let dev_prag = dev_docs[0].get::<Pragmatics>().unwrap();
        assert_eq!(dev_prag.len(), prag.len());
        for i in 0..prag.len() {
            let ix = Ix::new(i as u32);
            let cpu_item = prag.get(ix);
            let dev_item = dev_prag.get(ix);
            assert_eq!(cpu_item.evidence, dev_item.evidence, "sentence {i} evidence mismatch");
            assert!(
                (cpu_item.score - dev_item.score).abs() <= 1e-5,
                "sentence {i} pragmatics score diff: cpu={} dev={}",
                cpu_item.score,
                dev_item.score
            );
        }
    }
}
