//! BundleTokenVectors integration: alignment, pooling, projection and native
//! coreference pair features. Archive-trained ranker comparisons are retired;
//! these tests do not establish replacement coreference accuracy.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use turbo::head::{Linear, ProjectionHead};
use turbo_analysis::*;
use turbo_analysis_embed::*;
use turbo_analysis_layers::{Entities, PhraseChunks, PosTags, Sentences, Subwords, Tokens, Vectors};
use turbo_coref::train::{GoldMention, TrainingDocument, pairs};
use turbo_coref::{CorefAnnotator, CorefInput, DEFAULT_THRESHOLD, TokenVectors};

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
        let bad = Subwords {
            spans: subwords.spans.clone(),
            ids: subwords.ids.clone(),
            model: subwords.model.clone(),
            token: Some(bad_token),
        };
        assert!(pool_word_tokens(&t_spans, &bad, &raw, 2, false).is_err());
    }
}

#[test]
fn real_bundle_vectors_reach_coreference_pair_features() {
    let bundle = Arc::new(BundleTokenVectors::open(&bert_bundle(), &VectorOptions::default()).unwrap());
    let s1 = [("Acme", "NNP"), ("grew", "VBD"), (".", ".")];
    let s2 = [("The", "DT"), ("firm", "NN"), ("hired", "VBD"), (".", ".")];
    let doc = BuiltDoc::build(
        &[&s1, &s2],
        &[("Acme", "organization")],
        &[("Acme", "NP"), ("The firm", "NP")],
        &[("Acme", 0), ("The firm", 0)],
    );
    let training = TrainingDocument { input: doc.input(), gold: doc.gold.clone() };
    let plain = pairs(&training, &CorefAnnotator::default()).unwrap();
    assert_eq!(plain.len(), 1);
    assert!(plain[0].values().is_none());
    let annotator = CorefAnnotator::new(PERSON, NEUTRAL, None, DEFAULT_THRESHOLD, None, Some(bundle.clone())).unwrap();
    let contextual = pairs(&training, &annotator).unwrap();
    assert_eq!(contextual.len(), plain.len());
    assert_eq!(contextual[0].outcome(), plain[0].outcome());
    let names: Vec<String> = contextual[0].context().iter().map(|name| name.to_string_lossy()).collect();
    let values = contextual[0].values().unwrap();
    assert_eq!(names.len(), values.len());
    assert!(values.iter().all(|value| value.is_finite()));
    let value = |name: &str| values[names.iter().position(|n| n == name).unwrap()];
    // Compare the public producer output to the pair feature contract:
    // Acme's token vector versus the mean of The and firm in their sentence.
    let antecedent = bundle.vectors(&["Acme", "grew", "."]);
    let anaphor = bundle.vectors(&["The", "firm", "hired", "."]);
    for d in 0..bundle.dimension() {
        let u = ((f64::from(anaphor[0][d]) + f64::from(anaphor[1][d])) / 2.0) as f32;
        let v = antecedent[0][d];
        let product = (f64::from(u) * f64::from(v)) as f32;
        let difference = (f64::from(u) - f64::from(v)).abs() as f32;
        assert!((value(&format!("vp{d}")) - product).abs() < 1e-6);
        assert!((value(&format!("vd{d}")) - difference).abs() < 1e-6);
    }
    assert_eq!(names.iter().filter(|n| n.starts_with("vp")).count(), bundle.dimension());
    assert_eq!(names.iter().filter(|n| n.starts_with("vd")).count(), bundle.dimension());
}
