//! `IncivilityHeadAnnotator`: lexicon, embedding head and reply context
//! combined, CPU and device matching.

mod heads_support;

use std::sync::Arc;

use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Incivility, IncivilityLabel, Sentences, Vectors};
use turbo_tokenize::SimpleTokenizer;

use heads_support::*;

#[derive(Debug)]
struct HeadIncivilityScorer(Arc<ClassifierHead>);

impl turbo_doccat::IncivilityVectorScorer for HeadIncivilityScorer {
    fn score_incivility(&self, vec: &[f32]) -> Result<turbo_analysis_layers::IncivilityScores, String> {
        self.0.score_incivility_vector(vec)
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.0.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.0.config_hash()
    }
}

#[derive(Debug)]
struct CentroidIncivilityScorer {
    centroids: Arc<LabelCentroids>,
    bias: f64,
    scale: f64,
}

impl turbo_doccat::IncivilityVectorScorer for CentroidIncivilityScorer {
    fn score_incivility(&self, vec: &[f32]) -> Result<turbo_analysis_layers::IncivilityScores, String> {
        self.centroids.score_incivility_vector(vec, self.bias, self.scale)
    }

    fn model(&self) -> Option<ModelRef> {
        Some(self.centroids.model().clone())
    }

    fn config_hash(&self) -> u64 {
        self.centroids.config_hash() ^ self.bias.to_bits() ^ self.scale.to_bits()
    }
}

#[test]
fn incivility_annotator_combines_lexicon_embedding_head_and_reply_context_cpu_device() {
    use turbo_analysis_layers::{Block, IncivilityScores, IncivilityTarget, Structure};
    use turbo_doccat::{
        DEFAULT_INCIVILITY_THRESHOLD, GoldIncivilityUnit, IncivilityAnnotator, IncivilityVectorScorer,
        evaluate_incivility_corpus,
    };

    let (sub, vec_ann) = open_cpu(64);

    // Project-authored multi-label training examples across all 6 Oasis / HKBU dimensions,
    // including an implicit intolerance sentence ("Outsiders have no right to participate
    // in our civic institutions.") that contains no direct lexicon phrase.
    let train_rows: &[(&str, &[IncivilityLabel])] = &[
        ("You are an incompetent moron and a total clown.", &[IncivilityLabel::Insult]),
        ("This broken release is absolute bullshit.", &[IncivilityLabel::Profanity]),
        ("Watch your back, we will hunt you down.", &[IncivilityLabel::Threat]),
        ("Those people are parasites and subhuman vermin.", &[IncivilityLabel::IdentityAttack]),
        ("Outsiders have no right to participate in our civic institutions.", &[IncivilityLabel::Intolerance]),
        (
            "You fucking scumbag, we will kill you!",
            &[
                IncivilityLabel::Insult,
                IncivilityLabel::Profanity,
                IncivilityLabel::Threat,
                IncivilityLabel::SevereToxicity,
            ],
        ),
        ("The patch fixes the buffer alignment issue in the parser.", &[]),
        ("We respectfully disagree with the proposed schedule change.", &[]),
    ];

    let inciv_names = IncivilityLabel::ALL.map(|l| l.name());
    let inciv_tags = TagSet::new("incivility", &inciv_names);

    let embed_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .build()
        .unwrap();

    let mut labeled_vecs: Vec<(Vec<f32>, [bool; 6])> = Vec::with_capacity(train_rows.len());
    for &(text, labels) in train_rows {
        let mut docs = [make_doc_with_sentences(text, &[(0, text.len() as u32)])];
        embed_pipe.run(&mut docs).remove(0).unwrap();
        let v = docs[0].get::<Vectors<Sentences>>().unwrap().row(Ix::new(0)).to_vec();
        let mut mask = [false; 6];
        for (idx, &l) in IncivilityLabel::ALL.iter().enumerate() {
            mask[idx] = labels.contains(&l);
        }
        labeled_vecs.push((v, mask));
    }

    let head = Arc::new(
        ClassifierHead::fit_multi_label(vec_ann.model().clone(), 32, inciv_tags, &labeled_vecs, 1e-4, 300, 2.5)
            .unwrap(),
    );

    // Also build a 6-label prototype `LabelCentroids` scorer and verify `CentroidIncivilityScorer`.
    let centroids = Arc::new(
        LabelCentroids::from_text_examples(
            &vec_ann,
            &sub,
            "incivility",
            &[
                ("insult", &["You are an incompetent moron and a total clown."][..]),
                ("profanity", &["This broken release is absolute bullshit."][..]),
                ("threat", &["Watch your back, we will hunt you down."][..]),
                ("identity_attack", &["Those people are parasites and subhuman vermin."][..]),
                ("intolerance", &["Outsiders have no right to participate in our civic institutions."][..]),
                ("severe_toxicity", &["You fucking scumbag, we will kill you!"][..]),
            ],
            CentroidNorm::L2,
            0.1,
        )
        .unwrap(),
    );
    let cent_scorer = CentroidIncivilityScorer { centroids, bias: 0.85, scale: 12.0 };
    let cent_scores: IncivilityScores = cent_scorer.score_incivility(&labeled_vecs[4].0).unwrap();
    assert!(
        cent_scores.intolerance >= 0.70,
        "expected high centroid intolerance score on example 4, got {}",
        cent_scores.intolerance
    );

    // Build a multi-sentence document with reply-context structure (`Block::Quote` + `Block::Paragraph`):
    //   0. Quoted parent sentence (neutral)
    //   1. Reply sentence with Opinion-Conflict pattern ("You have no clue what you are talking about.")
    //   2. Implicit intolerance sentence scored by the multi-label `ClassifierHead`
    //   3. Co-occurring Profanity + Insult + Threat -> SevereToxicity
    //   4. Benign civil sentence
    let eval_sents: &[(&str, &[IncivilityLabel])] = &[
        ("The new transit schedule adds two evening buses.", &[]),
        ("You have no clue what you are talking about.", &[IncivilityLabel::Insult]),
        ("Outsiders have no right to participate in our civic institutions.", &[IncivilityLabel::Intolerance]),
        (
            "You fucking scumbag, we will kill you!",
            &[
                IncivilityLabel::Insult,
                IncivilityLabel::Profanity,
                IncivilityLabel::Threat,
                IncivilityLabel::SevereToxicity,
            ],
        ),
        ("The patch fixes the buffer alignment issue in the parser.", &[]),
    ];

    let mut full_text = String::new();
    let mut sent_spans = Vec::new();
    let mut gold = Vec::new();
    for &(s, labels) in eval_sents {
        if !full_text.is_empty() {
            full_text.push(' ');
        }
        let start = full_text.len() as u32;
        full_text.push_str(s);
        let end = full_text.len() as u32;
        sent_spans.push((start, end));
        gold.push(GoldIncivilityUnit { positive_labels: labels.to_vec() });
    }

    let make_eval_doc = || {
        let mut doc = make_doc_with_sentences(&full_text, &sent_spans);
        let mut b_spans = Spans::new(ViewId::SOURCE);
        for &(s, e) in &sent_spans {
            b_spans.push(Span::new(s, e));
        }
        let structure = Structure {
            spans: b_spans,
            kind: vec![Block::Quote, Block::Paragraph, Block::Paragraph, Block::Paragraph, Block::Paragraph],
            parent: vec![None; 5],
            depth: vec![1, 0, 0, 0, 0],
        };
        doc.add(&Key::<Structure>::DEFAULT, structure, Provenance::by("test")).unwrap();
        doc
    };

    let scorer: Arc<dyn IncivilityVectorScorer> = Arc::new(HeadIncivilityScorer(Arc::clone(&head)));
    let inciv_ann = IncivilityAnnotator::new().with_vector_scorer(scorer).with_reply_context(true);

    let cpu_pipe = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Structure>::DEFAULT)
        .add(SimpleTokenizer::default())
        .add(open_subwords())
        .add(SentenceVectors(Arc::clone(&vec_ann)))
        .add(inciv_ann)
        .build()
        .unwrap();

    let mut cpu_docs = [make_eval_doc()];
    cpu_pipe.run(&mut cpu_docs).remove(0).unwrap();
    let cpu_doc = &cpu_docs[0];

    let prov = cpu_doc.provenance(Key::<Incivility>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, "turbo-doccat:incivility");
    assert!(prov.model.as_ref().unwrap().id.ends_with("+lexicon:reply-context"));

    let cpu_inciv = cpu_doc.get::<Incivility>().unwrap();
    assert_eq!(cpu_inciv.len(), 5);
    for i in 0..5 {
        assert_eq!(cpu_inciv.target[i], IncivilityTarget::Sentence(Ix::new(i as u32)));
    }

    let metrics = evaluate_incivility_corpus(&gold, cpu_inciv, DEFAULT_INCIVILITY_THRESHOLD);
    println!(
        "Incivility regression fixture (5 sentences): macro_P={:.3}, macro_R={:.3}, macro_F1={:.3}, micro_F1={:.3}, evidence={}/{}",
        metrics.macro_precision,
        metrics.macro_recall,
        metrics.macro_f1,
        metrics.micro_f1,
        metrics.positive_with_evidence,
        metrics.total_positive_predictions,
    );
    assert_eq!(
        metrics.positive_with_evidence, metrics.total_positive_predictions,
        "every positive label prediction must carry a non-empty token evidence span"
    );
    assert!(metrics.micro_f1 >= 0.95, "expected micro_f1 >= 0.95, got {}", metrics.micro_f1);

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
        let dev_scorer: Arc<dyn IncivilityVectorScorer> = Arc::new(HeadIncivilityScorer(Arc::clone(&head)));
        let dev_ann = IncivilityAnnotator::new().with_vector_scorer(dev_scorer).with_reply_context(true);
        let dev_pipe = Analyzer::builder()
            .given(Key::<Sentences>::DEFAULT)
            .given(Key::<Structure>::DEFAULT)
            .add(SimpleTokenizer::default())
            .add(open_subwords())
            .add(SentenceVectors(Arc::clone(&dev_vec)))
            .add(dev_ann)
            .build()
            .unwrap();
        let mut dev_docs = [make_eval_doc()];
        dev_pipe.run(&mut dev_docs).remove(0).unwrap();
        let dev_inciv = dev_docs[0].get::<Incivility>().unwrap();
        assert_eq!(dev_inciv.len(), cpu_inciv.len());
        for i in 0..cpu_inciv.len() {
            let ix = Ix::new(i as u32);
            let cs = cpu_inciv.scores(ix);
            let ds = dev_inciv.scores(ix);
            for &lbl in &IncivilityLabel::ALL {
                assert!(
                    (cs.get(lbl) - ds.get(lbl)).abs() <= 1e-5,
                    "sentence {i} {lbl} score diff: cpu={} dev={}",
                    cs.get(lbl),
                    ds.get(lbl)
                );
            }
            let cev: Vec<_> = cpu_inciv.evidence_of(ix).collect();
            let dev: Vec<_> = dev_inciv.evidence_of(ix).collect();
            assert_eq!(cev, dev, "sentence {i} evidence spans mismatch between CPU and device");
        }
    }
}
