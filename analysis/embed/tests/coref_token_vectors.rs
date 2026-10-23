//! Integration tests for `BundleTokenVectors` (`#960`):
//!
//! 1. Alignment and pooling parity between `TokenVectors::vectors`,
//!    `BundleTokenVectors::embed_document_tokens`, and `pool_subword_vectors`
//!    (`SubwordAnnotator` + `SubwordVectors` `#709`), including multi-piece
//!    words, overlapping BPE spans, and `ProjectionHead`.
//! 2. End-to-end training (`train_ranking`) and CoNLL evaluation (`CorefScorer`:
//!    MUC, B-cubed, CEAF-m, CEAF-e, and CoNLL average) comparing:
//!    - surface/syntactic ranking baseline (no vectors),
//!    - static word vectors (`Word2VecTrainer` / `WordVectors`),
//!    - contextual token vectors (`BundleTokenVectors` / `TokenVectors`), and
//!    - combined static + contextual vectors.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::head::{Linear, ProjectionHead};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Entities, PhraseChunks, PosTags, Sentences, Subwords, Tokens, Vectors};
use turbo_coref::scorer::{CorefScorer, CorefScores, partition};
use turbo_coref::train::{GoldMention, TrainingDocument, train_ranking};
use turbo_coref::word2vec::{Word2VecArch, Word2VecParameters, Word2VecTrainer};
use turbo_coref::{CorefAnnotator, CorefInput, DEFAULT_THRESHOLD, TokenVectors, WordVectors};

const PERSON: &[&str] = &["person"];
const NEUTRAL: &[&str] = &["organization", "location"];

fn testdata() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../testdata")
}

fn bert_bundle() -> PathBuf {
    testdata().join("tiny-bert-bundle")
}

fn build_tagset(name: &str, labels: &[&str]) -> (Arc<TagSet>, Vec<Tag>) {
    let mut unique: Vec<&str> = Vec::new();
    for &l in labels {
        if !unique.contains(&l) {
            unique.push(l);
        }
    }
    let set = TagSet::new(name, &unique);
    let tags = labels.iter().map(|&l| set.tag(l).unwrap()).collect();
    (set, tags)
}

fn token_range(tokens: &Spans<Tokens>, span: Span) -> IxRange<Tokens> {
    let (_, t_ends) = tokens.columns();
    let (t_starts, _) = tokens.columns();
    let first = t_ends.iter().position(|&e| e > span.start).unwrap_or(tokens.len());
    let end = t_starts.iter().position(|&s| s >= span.end).unwrap_or(tokens.len());
    IxRange::new(first as u32, end.max(first) as u32)
}

/// Builds a `Document` with `Sentences`, `Tokens`, `PosTags`, `Entities`, and
/// `PhraseChunks` from a sequence of sentences (`&[&[(&str, &str)]]`), entity
/// phrases (`&[(&str, &str)]`), and NP chunk phrases (`&[(&str, &str)]`).
struct BuiltDoc {
    text: String,
    sentences: Sentences,
    tokens: Tokens,
    tags: PosTags,
    entities: Entities,
    chunks: PhraseChunks,
    gold: Vec<GoldMention>,
}

impl BuiltDoc {
    fn build(
        sents: &[&[(&str, &str)]],
        entity_marks: &[(&str, &str)],
        chunk_marks: &[(&str, &str)],
        gold_marks: &[(&str, usize)],
    ) -> Self {
        let mut text = String::new();
        let mut s_spans = Spans::new(ViewId::SOURCE);
        let mut t_spans = Spans::new(ViewId::SOURCE);
        let mut t_sent = Vec::new();
        let mut pos_labels = Vec::new();

        for (s_idx, &sent) in sents.iter().enumerate() {
            if !text.is_empty() {
                text.push(' ');
            }
            let s_start = text.len() as u32;
            for (w_idx, &(word, pos)) in sent.iter().enumerate() {
                if w_idx > 0 {
                    text.push(' ');
                }
                let w_start = text.len() as u32;
                text.push_str(word);
                let w_end = text.len() as u32;
                t_spans.push(Span::new(w_start, w_end));
                t_sent.push(Ix::new(s_idx as u32));
                pos_labels.push(pos);
            }
            let s_end = text.len() as u32;
            s_spans.push(Span::new(s_start, s_end));
        }

        let (tagset, tags) = build_tagset("penn", &pos_labels);

        let mut e_spans = Spans::new(ViewId::SOURCE);
        let mut e_tokens = Vec::new();
        let mut e_labels = Vec::new();
        let mut search_from = 0usize;
        for &(phrase, kind) in entity_marks {
            let pos = text[search_from..]
                .find(phrase)
                .unwrap_or_else(|| panic!("entity {phrase:?} not found after {search_from} in {text:?}"))
                + search_from;
            let span = Span::new(pos as u32, (pos + phrase.len()) as u32);
            e_spans.push(span);
            e_tokens.push(token_range(&t_spans, span));
            e_labels.push(kind);
            search_from = pos + 1;
        }
        let (e_kinds, e_kind) = build_tagset("entity", &e_labels);

        let mut c_spans = Spans::new(ViewId::SOURCE);
        let mut c_tokens = Vec::new();
        let mut c_labels = Vec::new();
        let mut chunk_from = 0usize;
        for &(phrase, kind) in chunk_marks {
            let pos = text[chunk_from..]
                .find(phrase)
                .unwrap_or_else(|| panic!("chunk {phrase:?} not found after {chunk_from} in {text:?}"))
                + chunk_from;
            let span = Span::new(pos as u32, (pos + phrase.len()) as u32);
            c_spans.push(span);
            c_tokens.push(token_range(&t_spans, span));
            c_labels.push(kind);
            chunk_from = pos + 1;
        }
        let (c_kinds, c_kind) = build_tagset("chunk", &c_labels);

        let mut gold = Vec::new();
        let mut gold_from = 0usize;
        for &(phrase, chain) in gold_marks {
            let pos = text[gold_from..]
                .find(phrase)
                .unwrap_or_else(|| panic!("gold {phrase:?} not found after {gold_from} in {text:?}"))
                + gold_from;
            let end = pos + phrase.len();
            gold.push(GoldMention { span: Span::new(pos as u32, end as u32), chain });
            gold_from = pos + 1;
        }

        let e_len = e_spans.len();
        BuiltDoc {
            text,
            sentences: Sentences { spans: s_spans },
            tokens: Tokens { spans: t_spans, sentence: Some(t_sent) },
            tags: PosTags { tagset, tags, prob: None },
            entities: Entities {
                spans: e_spans,
                tokens: e_tokens,
                kinds: e_kinds,
                kind: e_kind,
                score: vec![1.0; e_len],
            },
            chunks: PhraseChunks { spans: c_spans, tokens: c_tokens, kinds: c_kinds, kind: c_kind },
            gold,
        }
    }

    fn input(&self) -> CorefInput<'_> {
        CorefInput {
            text: &self.text,
            view: ViewId::SOURCE,
            sentences: &self.sentences,
            tokens: &self.tokens,
            tags: &self.tags,
            entities: &self.entities,
            chunks: Some(&self.chunks),
            phrases: None,
            speakers: None,
        }
    }

    fn gold_partition(&self) -> Vec<HashSet<(u32, u32)>> {
        let max_chain = self.gold.iter().map(|g| g.chain).max().unwrap_or(0);
        let mut chains = vec![HashSet::new(); max_chain + 1];
        for g in &self.gold {
            chains[g.chain].insert((g.span.start, g.span.end));
        }
        chains.into_iter().filter(|c| !c.is_empty()).collect()
    }

    fn sentence_tokens(&self) -> Vec<Vec<&str>> {
        let mut out = Vec::new();
        for (s_ix, _) in self.sentences.spans.iter() {
            let r = self.tokens.of(s_ix);
            let words: Vec<&str> = (r.start.get()..r.end.get())
                .map(|t| {
                    let sp = self.tokens.spans.get(Ix::new(t as u32));
                    &self.text[sp.range()]
                })
                .collect();
            out.push(words);
        }
        out
    }
}

#[test]
fn bundle_token_vectors_align_multi_piece_words_and_match_subwords_layer() {
    let dir = bert_bundle();
    let tv = BundleTokenVectors::open(&dir, &VectorOptions::default()).unwrap();
    assert_eq!(tv.dimension(), 32);
    assert_ne!(tv.config_hash(), 0);

    // Sentence with multi-piece WordPiece tokens ("Embeddings", "contextualized", "representations").
    let sentence = ["Embeddings", "give", "contextualized", "token", "representations", "."];
    let vecs = tv.vectors(&sentence);
    assert_eq!(vecs.len(), sentence.len());
    for (i, v) in vecs.iter().enumerate() {
        assert_eq!(v.len(), 32, "token {i} dimension");
        assert!(v.iter().all(|x| x.is_finite()), "token {i} finite");
        let norm: f64 = v.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "token {i} expected unit norm, got {norm}");
    }

    // Build an explicit Document with Sentences, Tokens, Subwords, and Vectors<Subwords> (#709)
    // and verify that `embed_document_tokens` and `pool_subword_vectors` produce identical vectors.
    let joined = sentence.join(" ");
    let mut doc = Document::new(joined.clone());
    let mut s_spans = Spans::new(ViewId::SOURCE);
    s_spans.push(Span::new(0, joined.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("test")).unwrap();

    let mut t_spans = Spans::new(ViewId::SOURCE);
    let mut offset = 0u32;
    for (i, &w) in sentence.iter().enumerate() {
        if i > 0 {
            offset += 1;
        }
        let end = offset + w.len() as u32;
        t_spans.push(Span::new(offset, end));
        offset = end;
    }
    doc.add(
        &Key::DEFAULT,
        Tokens { spans: t_spans, sentence: Some(vec![Ix::new(0); sentence.len()]) },
        Provenance::by("test"),
    )
    .unwrap();

    let sub_annotator = SubwordAnnotator::open(&dir).unwrap();
    let pipeline = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(sub_annotator)
        .add(SubwordVectors(tv.annotator().clone()))
        .build()
        .unwrap();
    let mut docs = [doc];
    pipeline.run(&mut docs).remove(0).unwrap();
    let [doc] = docs;

    let tokens_layer = doc.get::<Tokens>().unwrap();
    let subwords_layer = doc.get::<Subwords>().unwrap();
    let subword_vecs_layer = doc.get::<Vectors<Subwords>>().unwrap();
    assert!(
        subwords_layer.len() > tokens_layer.len(),
        "expected multi-piece words to produce more subwords ({}) than word tokens ({})",
        subwords_layer.len(),
        tokens_layer.len()
    );
    assert!(subwords_layer.token.is_some(), "SubwordAnnotator must populate Subwords.token");

    let from_doc = tv.embed_document_tokens(tokens_layer, subwords_layer, None).unwrap();
    let from_layer = pool_subword_vectors(tokens_layer, subwords_layer, subword_vecs_layer, true).unwrap();
    assert_eq!(vecs, from_doc);
    assert_eq!(vecs, from_layer);

    // Attaching a ProjectionHead changes both dimension() and config_hash().
    let proj = ProjectionHead {
        dim: 8,
        hidden: 32,
        has_bias: false,
        linear: Linear::new(32, 8, vec![0.125f32; 8 * 32], vec![0.0f32; 8]).unwrap(),
    };
    let proj_annotator = VectorAnnotator::open(&dir, &VectorOptions::default()).unwrap().with_projection(proj).unwrap();
    let proj_tv = BundleTokenVectors::new(Arc::new(proj_annotator));
    assert_eq!(proj_tv.dimension(), 8);
    assert_ne!(proj_tv.config_hash(), tv.config_hash());
    let proj_vecs = proj_tv.vectors(&sentence);
    assert_eq!(proj_vecs.len(), sentence.len());
    assert!(proj_vecs.iter().all(|v| v.len() == 8 && v.iter().all(|x| x.is_finite())));
}

#[test]
fn pool_word_tokens_handles_shared_bpe_spans_and_validates_inputs() {
    let mut t_spans = Spans::<Tokens>::new(ViewId::SOURCE);
    // "it" (0..2), "'s" (2..4), "done" (5..9)
    t_spans.push(Span::new(0, 2));
    t_spans.push(Span::new(2, 4));
    t_spans.push(Span::new(5, 9));

    // A single byte-level piece covering "it's" (0..4), and a second covering "done" (5..9).
    let mut p_spans = Spans::<Subwords>::new(ViewId::SOURCE);
    p_spans.push(Span::new(0, 4));
    p_spans.push(Span::new(5, 9));
    let subwords = Subwords {
        token: align(&p_spans, &t_spans),
        spans: p_spans,
        ids: vec![10, 20],
        model: ModelRef { id: "test".into(), revision: "1".into(), sha256: "00".into() },
    };
    // Piece 0 has [1.0, 0.0], piece 1 has [0.0, 1.0].
    let raw = [1.0f32, 0.0, 0.0, 1.0];
    let pooled = pool_word_tokens(&t_spans, &subwords, &raw, 2, false).unwrap();
    assert_eq!(pooled.len(), 3);
    // Both "it" and "'s" overlap piece 0 (0..4), so both receive [1.0, 0.0].
    assert_eq!(pooled[0], vec![1.0, 0.0]);
    assert_eq!(pooled[1], vec![1.0, 0.0]);
    assert_eq!(pooled[2], vec![0.0, 1.0]);

    // Nearest-piece fallback when a token's characters were dropped by the tokenizer:
    // token 1 (4..5) and token 3 (10..11) overlap no piece and receive piece 1 (`a.min(len - 1)`).
    let mut gapped = Spans::<Tokens>::new(ViewId::SOURCE);
    for (s, e) in [(0, 4), (4, 5), (5, 9), (10, 11)] {
        gapped.push(Span::new(s, e));
    }
    let gapped_sub = Subwords {
        spans: subwords.spans.clone(),
        ids: subwords.ids.clone(),
        model: subwords.model.clone(),
        token: align(&subwords.spans, &gapped),
    };
    assert_eq!(
        pool_word_tokens(&gapped, &gapped_sub, &raw, 2, false).unwrap(),
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.0, 1.0], vec![0.0, 1.0]],
    );

    // Validation errors: zero dim, mismatched buffer length, mixed views, or malformed token mappings.
    assert!(pool_word_tokens(&t_spans, &subwords, &raw, 0, false).is_err());
    assert!(pool_word_tokens(&t_spans, &subwords, &[1.0, 0.0], 2, false).is_err());
    let mut other_view = Spans::<Tokens>::new(ViewId(1));
    other_view.push(Span::new(0, 2));
    assert!(pool_word_tokens(&other_view, &subwords, &raw, 2, false).is_err());
    for bad_token in
        [vec![Ix::new(0), Ix::new(0), Ix::new(1)], vec![Ix::new(1), Ix::new(0)], vec![Ix::new(0), Ix::new(9)]]
    {
        let bad = Subwords { spans: subwords.spans.clone(), ids: subwords.ids.clone(), model: subwords.model.clone(), token: Some(bad_token) };
        assert!(pool_word_tokens(&t_spans, &bad, &raw, 2, false).is_err());
    }
}

/// Builds a coreference benchmark where each document introduces two candidate
/// entities of the same entity type in the first two sentences, followed by a
/// nominal anaphor in the third sentence whose modifier disambiguates which
/// antecedent it corefers with (`BundleTokenVectors` pools over the full
/// mention span while static `WordVectors` only sees the shared head noun).
fn benchmark_splits() -> (Vec<BuiltDoc>, Vec<BuiltDoc>) {
    let specs: &[(&str, &str, &str, &str, &str, usize)] = &[
        ("Apex Cloud", "cloud", "Vertex Bio", "bio", "division", 0),
        ("Apex Cloud", "cloud", "Vertex Bio", "bio", "division", 1),
        ("Solaris Grid", "solar", "Lunar Harbor", "lunar", "group", 0),
        ("Solaris Grid", "solar", "Lunar Harbor", "lunar", "group", 1),
        ("Nordic Rail", "rail", "Alpine Freight", "freight", "unit", 0),
        ("Nordic Rail", "rail", "Alpine Freight", "freight", "unit", 1),
        ("Pacific Optics", "optics", "Atlantic Sonar", "sonar", "center", 0),
        ("Pacific Optics", "optics", "Atlantic Sonar", "sonar", "center", 1),
        ("Cobalt Motors", "cobalt", "Quartz Robotics", "quartz", "division", 0),
        ("Cobalt Motors", "cobalt", "Quartz Robotics", "quartz", "division", 1),
        ("Harbor Logistics", "harbor", "Summit Aviation", "summit", "group", 0),
        ("Harbor Logistics", "harbor", "Summit Aviation", "summit", "group", 1),
    ];

    let make_doc =
        |ant1: &str, mod1: &str, ant2: &str, mod2: &str, head: &str, target: usize, verb: &str| -> BuiltDoc {
            let w1: Vec<&str> = ant1.split(' ').collect();
            let w2: Vec<&str> = ant2.split(' ').collect();
            let chosen_mod = if target == 0 { mod1 } else { mod2 };
            let anaphor_np = format!("The {chosen_mod} {head}");
            let s1 = [(w1[0], "NNP"), (w1[1], "NNP"), ("built", "VBD"), (mod1, "NN"), ("systems", "NNS"), (".", ".")];
            let s2 = [(w2[0], "NNP"), (w2[1], "NNP"), ("tested", "VBD"), (mod2, "NN"), ("sensors", "NNS"), (".", ".")];
            let s3 = [("The", "DT"), (chosen_mod, "NN"), (head, "NN"), (verb, "VBD"), ("rapidly", "RB"), (".", ".")];
            let gold_marks: Vec<(&str, usize)> = if target == 0 {
                vec![(ant1, 0), (ant2, 1), (&anaphor_np, 0)]
            } else {
                vec![(ant1, 0), (ant2, 1), (&anaphor_np, 1)]
            };
            BuiltDoc::build(
                &[&s1, &s2, &s3],
                &[(ant1, "organization"), (ant2, "organization")],
                &[(ant1, "NP"), (ant2, "NP"), (&anaphor_np, "NP")],
                &gold_marks,
            )
        };

    let mut train_docs = Vec::new();
    let mut test_docs = Vec::new();
    for &(a1, m1, a2, m2, head, target) in specs {
        train_docs.push(make_doc(a1, m1, a2, m2, head, target, "expanded"));
        train_docs.push(make_doc(a1, m1, a2, m2, head, target, "grew"));
        test_docs.push(make_doc(a1, m1, a2, m2, head, target, "succeeded"));
    }
    (train_docs, test_docs)
}

fn evaluate_split(annotator: &CorefAnnotator, docs: &[BuiltDoc]) -> CorefScores {
    let mut scorer = CorefScorer::new();
    for doc in docs {
        let (mentions, chains, _) = annotator.resolve(&doc.input()).unwrap();
        let response = partition(&mentions, &chains);
        let key = doc.gold_partition();
        scorer.add(&key, &response).unwrap();
    }
    scorer.scores()
}

fn run_ranking_comparison(bundle_dir: &Path, label: &str) -> (CorefScores, CorefScores, CorefScores, CorefScores) {
    let (train_docs, test_docs) = benchmark_splits();
    let train_inputs: Vec<TrainingDocument<'_>> =
        train_docs.iter().map(|d| TrainingDocument { input: d.input(), gold: d.gold.clone() }).collect();

    // 1. Train static Word2Vec vectors on the training split sentences only.
    let mut sentences_owned: Vec<Vec<String>> = Vec::new();
    for d in &train_docs {
        for s in d.sentence_tokens() {
            sentences_owned.push(s.into_iter().map(str::to_owned).collect());
        }
    }
    let w2v_params = Word2VecParameters {
        arch: Word2VecArch::SkipGram,
        dimension: 32,
        min_count: 1,
        epochs: 25,
        window: 4,
        seed: 42,
        ..Word2VecParameters::default()
    };
    let w2v: Arc<dyn WordVectors> =
        Arc::new(Word2VecTrainer::new(w2v_params).unwrap().train(&sentences_owned).unwrap());
    let token_vecs: Arc<dyn TokenVectors> =
        Arc::new(BundleTokenVectors::open(bundle_dir, &VectorOptions::default()).unwrap());

    let train_and_score = |w: Option<Arc<dyn WordVectors>>, t: Option<Arc<dyn TokenVectors>>| -> CorefScores {
        let extractor = CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, w.clone(), t.clone()).unwrap();
        let model = Arc::new(train_ranking("eng", &train_inputs, 20, 0.05, 0.001, &extractor).unwrap());
        let annotator = CorefAnnotator::new(PERSON, NEUTRAL, Some(model), DEFAULT_THRESHOLD, w, t).unwrap();
        evaluate_split(&annotator, &test_docs)
    };

    let base_scores = train_and_score(None, None);
    let static_scores = train_and_score(Some(w2v.clone()), None);
    let ctx_scores = train_and_score(None, Some(token_vecs.clone()));
    let both_scores = train_and_score(Some(w2v), Some(token_vecs));
    for (arm, s) in [
        ("surface", &base_scores),
        ("static_word2vec", &static_scores),
        ("contextual_token_vectors", &ctx_scores),
        ("combined", &both_scores),
    ] {
        println!(
            "[{label}] {arm}: CoNLL={:.4} (MUC={:.4}, B3={:.4}, CEAFe={:.4})",
            s.conll(),
            s.muc().f1(),
            s.b_cubed().f1(),
            s.ceaf_e().f1()
        );
    }
    (base_scores, static_scores, ctx_scores, both_scores)
}

#[test]
fn ranking_model_with_bundle_token_vectors_outperforms_static_word_vectors() {
    let (base, static_w2v, ctx, both) = run_ranking_comparison(&bert_bundle(), "tiny-bert-bundle");
    assert!(
        ctx.conll() > static_w2v.conll(),
        "expected contextual BundleTokenVectors CoNLL F1 ({:.4}) > static Word2Vec CoNLL F1 ({:.4}) (surface={:.4})",
        ctx.conll(),
        static_w2v.conll(),
        base.conll()
    );
    assert!(
        both.conll() >= ctx.conll(),
        "expected combined CoNLL F1 ({:.4}) >= contextual CoNLL F1 ({:.4})",
        both.conll(),
        ctx.conll()
    );

    if let Ok(real_dir) = std::env::var("TURBO_TEST_COREF_BUNDLE") {
        let (_, real_static, real_ctx, real_both) = run_ranking_comparison(Path::new(&real_dir), "real-bundle");
        assert!(
            real_ctx.conll() > real_static.conll(),
            "expected real-bundle contextual CoNLL F1 ({:.4}) > static CoNLL F1 ({:.4})",
            real_ctx.conll(),
            real_static.conll()
        );
        assert!(real_both.conll() >= real_ctx.conll());
    }
}
