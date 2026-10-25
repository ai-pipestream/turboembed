//! The analysis result layer over gRPC with a real client against the
//! server in-process on the CPU: every result kind crosses the wire and
//! back unchanged from its in-process layer, and the capability listing
//! follows LoadBundle and UnloadBundle. Nothing is mocked.

mod common;

use std::sync::Arc;

use tonic::Code;
use turbo_analysis::export::export;
use turbo_analysis::{Document, Edges, Ix, Key, Layer, Provenance, Span, Spans, TagSet, View, ViewId};
use turbo_analysis_embed::{Answers, Reranked};
use turbo_analysis_layers::{
    Abbreviations, Chunks, Entities, Sentences, Subwords, TokenVectorWindows, Tokens, Vectors,
};
use turbo_kserve::offsets::OffsetMode;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::Wire;
use turbo_kserve::results::{
    document::codecs, extraction::Extraction, rerank::RerankScores, sparse::SparseTermsWire, tokens::TokenVectors,
};

use common::*;

#[tokio::test]
async fn sparse_terms_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let t = sparse_terms();
    let msg = <turbo_analysis_layers::SparseTerms<View> as Wire>::to_msg(&t);
    let back = c.round_trip_sparse_terms(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg, "the answer is the request");
    let decoded = turbo_analysis_layers::SparseTerms::from_msg(&back).unwrap();
    assert_eq!(decoded.model, t.model);
    assert_eq!(decoded.term, t.term);
    assert_eq!(decoded.weight, t.weight);
    assert_eq!(decoded.span, t.span);
    assert_eq!(decoded.occ, t.occ);
    assert_eq!(decoded.occ_at, t.occ_at);
}

#[tokio::test]
async fn token_vectors_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let (v, w) = token_vectors();
    let t = TokenVectors::new(&v, &w);
    let msg = t.to_msg();
    let back = c.round_trip_token_vectors(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = TokenVectors::from_msg(&back).unwrap();
    assert_eq!(decoded.dim, 3);
    assert_eq!(decoded.data, t.data);
    assert_eq!(decoded.empty, [1]);
    assert_eq!(decoded.view, 0);
    assert_eq!(decoded.windows, t.windows);
    assert_eq!(decoded.vectors().unwrap().data, v.data);
    assert_eq!(decoded.windows_layer(4).unwrap().spans.columns(), w.spans.columns());
}

#[tokio::test]
async fn rerank_scores_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let r = reranked();
    let m = RerankScores::new(&r);
    let msg = m.to_msg();
    let back = c.round_trip_rerank_scores(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = RerankScores::from_msg(&back).unwrap();
    assert_eq!(decoded.model, r.model);
    assert_eq!(decoded.source, "turbo:chunks");
    assert_eq!(decoded.instance, "candidates");
    assert_eq!(decoded.items, r.items.iter().map(|i| i.get() as u32).collect::<Vec<_>>());
    assert_eq!(decoded.scores, r.scores);
    // Named instances of one layer keep their identity on the wire.
    for name in ["shortlist-a", "shortlist-b"] {
        let r = Reranked {
            source: Key::<turbo_analysis_layers::Chunks>::named(name),
            items: vec![Ix::new(0)],
            scores: vec![1.0],
            model: model_ref(),
        };
        let decoded = RerankScores::from_msg(&RerankScores::new(&r).to_msg()).unwrap();
        assert_eq!(decoded.instance, name);
    }
}

#[tokio::test]
async fn extraction_answers_cross_the_wire_unchanged() {
    let (_s, mut c) = serve().await;
    let a = answers();
    let x = Extraction::new(&a, model_ref());
    let msg = x.to_msg();
    let back = c.round_trip_extraction(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = Extraction::from_msg(&back).unwrap();
    assert_eq!(decoded.model, x.model);
    assert_eq!(decoded.spans, x.spans);
    assert_eq!(decoded.scores, x.scores);
    assert_eq!(decoded.probabilities, x.probabilities);
    assert_eq!(decoded.answers().unwrap().spans.columns(), a.spans.columns());
}

#[tokio::test]
async fn discontinuous_answers_keep_their_fragments_over_the_wire() {
    let (_s, mut c) = serve().await;
    // An answer in two separated pieces: the envelope [0, 10), the
    // fragments [0, 2) and [8, 10). Without the fragment columns the
    // gap would decode as answer text.
    let mut spans = turbo_analysis::Spans::new(ViewId::SOURCE);
    spans.push_fragments(&[Span::new(0, 2), Span::new(8, 10)]);
    let a = Answers { spans, scores: vec![6.0], probabilities: vec![0.8], null_score: -1.0, null_probability: 0.2 };
    a.check().unwrap();
    let x = Extraction::new(&a, model_ref());
    assert_eq!(x.spans, [Span::new(0, 10)]);
    assert_eq!(x.fragments, [Span::new(0, 2), Span::new(8, 10)]);
    let msg = x.to_msg();
    let back = c.round_trip_extraction(msg.clone()).await.unwrap().into_inner();
    assert_eq!(back, msg);
    let decoded = Extraction::from_msg(&back).unwrap();
    assert_eq!(decoded.fragments, [Span::new(0, 2), Span::new(8, 10)]);
    assert_eq!(decoded.frag_at, [2]);
    let layer = decoded.answers().unwrap();
    assert_eq!(layer.spans.fragments(Ix::new(0)).collect::<Vec<_>>(), [Span::new(0, 2), Span::new(8, 10)]);
}

#[tokio::test]
async fn the_capability_listing_matches_the_loaded_bundle() {
    let (_s, mut c) = serve().await;
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    let b = the_one_bundle(&list);
    let m = manifest();
    assert_eq!(b.name, NAME);
    assert_eq!(b.model.as_ref().unwrap().id, m["model"]["id"]);
    assert_eq!(b.model.as_ref().unwrap().revision, m["model"]["revision"]);
    assert!(!b.model.as_ref().unwrap().sha256.is_empty());
    assert_eq!(b.task, "TASK_EMBED");
    assert_eq!(b.roles, ["PROMPT_QUERY"], "the bundle carries a query prefix only");
    assert_eq!(b.widths, [m["embed"]["dim"].as_u64().unwrap() as u32]);
    assert_eq!(b.context_limit, m["embed"]["max_seq"].as_u64().unwrap() as u32);
    assert_eq!(b.batch_limit, m["embed"]["max_batch"].as_u64().unwrap() as u32);
    assert!(b.schema.is_none(), "no merged task reads an extraction schema yet");
    let limits = list.limits.as_ref().unwrap();
    assert!(limits.max_sparse_entries > 0 && limits.max_sparse_occurrences >= limits.max_sparse_entries);
    assert!(limits.max_token_vector_bytes > 0 && limits.max_rerank_items > 0 && limits.max_answers > 0);
    assert!(limits.max_answer_fragments >= limits.max_answers && limits.max_windows > 0);
    let server = list.server.as_ref().unwrap();
    assert!(server.max_sessions_per_model > 0 && server.max_sessions_per_model <= 64);
    assert!(server.max_loaded_models > 0);
}

#[tokio::test]
async fn the_listing_follows_load_and_unload() {
    let (s, mut c) = serve().await;
    // The same bundle under a second name, as one bundle at two tiers would be.
    let bundle = tiny().to_str().unwrap().to_string();
    let req = load_req("second", &bundle, 1, 0);
    let loaded = c.load_bundle(req).await.unwrap().into_inner();
    let b = loaded.capabilities.as_ref().unwrap();
    assert_eq!(b.name, "second");
    assert_eq!(b.widths, [32]);
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(), [NAME, "second"]);

    // The new name serves inference too.
    let mut k = kserve(&s).await;
    assert_eq!(embed_shape(&mut k, "second", &[b"hello"]).await, [1, 32]);

    // Unload: the listing shrinks and the name is refused as never served.
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
    let gone = k.model_infer(texts_req("second", &[b"hello"])).await.expect_err("gone");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));
    let gone = c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.expect_err("not served");
    assert_eq!(refusal(&gone), (Code::NotFound, 1280));

    // The freed name serves again.
    c.load_bundle(load_req("second", &bundle, 1, 0)).await.unwrap();
    c.unload_bundle(UnloadBundleRequest { name: "second".into() }).await.unwrap();

    // A name that is taken, and requests that cannot mean anything, refuse
    // by name; a bundle that cannot load leaves nothing served.
    let e = c.load_bundle(load_req(NAME, &bundle, 1, 0)).await.expect_err("the name is taken");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
    for bad in [
        load_req("x", "", 1, 0),
        load_req("x", &bundle, 0, 0),
        load_req("x", &bundle, 1, 9),
        load_req("x", "/no/such/bundle", 1, 0),
    ] {
        c.load_bundle(bad).await.expect_err("refused");
    }
    let list = c.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(list.bundles.len(), 1);
}

#[tokio::test]
async fn sparse_terms_token_vectors_and_extraction_round_trip_in_all_offset_modes() {
    let (_s, mut c) = serve().await;
    // "a" (0..1, u16 0..1, cp 0..1) + e-acute (1..3, u16 1..2, cp 1..2) +
    // grinning face (3..7, u16 2..4, cp 2..3) + CJK (7..10, u16 4..5, cp 3..4) +
    // "!" (10..11, u16 5..6, cp 4..5).
    const TEXT: &str = "a\u{e9}\u{1f600}\u{4e2d}!";

    let make_sparse = || {
        let mut sparse = turbo_analysis_layers::SparseTerms::<View>::new(model_ref());
        sparse.push(2, 0.5, Some(Span::new(1, 3)));
        sparse.push_occurrences(&[Span::new(1, 3)]);
        sparse.push(7, 1.25, Some(Span::new(3, 10)));
        sparse.push_occurrences(&[Span::new(3, 7), Span::new(7, 10)]);
        sparse.unit();
        sparse.check().unwrap();
        sparse
    };
    let sparse_ref = make_sparse();

    let vec_data: Vec<f32> = (0..6).map(|i| (i + 1) as f32 * 0.5).collect();
    let v = Vectors::<Subwords>::new(3, vec_data, Vec::new(), model_ref());
    let mut win_spans = Spans::new(ViewId::SOURCE);
    win_spans.push(Span::new(0, 7));
    win_spans.push(Span::new(7, 11));
    let w = TokenVectorWindows { spans: win_spans };

    let mut ans_spans = Spans::new(ViewId::SOURCE);
    ans_spans.push_fragments(&[Span::new(1, 3), Span::new(7, 10)]);
    ans_spans.push(Span::new(3, 7));
    let a = Answers {
        spans: ans_spans,
        scores: vec![5.0, 3.0],
        probabilities: vec![0.6, 0.3],
        null_score: -1.0,
        null_probability: 0.1,
    };
    a.check().unwrap();

    for (mode, expected_emoji, expected_cjk) in [
        (OffsetMode::Utf8, (3, 7), (7, 10)),
        (OffsetMode::Utf16, (2, 4), (4, 5)),
        (OffsetMode::CodePoint, (2, 3), (3, 4)),
    ] {
        let s_msg = SparseTermsWire::new(make_sparse(), TEXT, mode).to_msg();
        assert_eq!((s_msg.span_start[1], s_msg.span_end[1]), (expected_emoji.0, expected_cjk.1));
        assert_eq!((s_msg.occ_start[1], s_msg.occ_end[1]), expected_emoji);
        assert_eq!((s_msg.occ_start[2], s_msg.occ_end[2]), expected_cjk);
        let s_back = c.round_trip_sparse_terms(s_msg.clone()).await.unwrap().into_inner();
        assert_eq!(s_back, s_msg);
        let s_dec = SparseTermsWire::from_msg(&s_back).unwrap();
        assert_eq!(s_dec.terms.span, sparse_ref.span);
        assert_eq!(s_dec.terms.occ, sparse_ref.occ);

        let tv_msg = TokenVectors::new(&v, &w).with_offsets(TEXT, mode).to_msg();
        assert_eq!((tv_msg.window_start[0], tv_msg.window_end[0]), (0, expected_emoji.1));
        let tv_back = c.round_trip_token_vectors(tv_msg.clone()).await.unwrap().into_inner();
        assert_eq!(tv_back, tv_msg);
        let tv_dec = TokenVectors::from_msg(&tv_back).unwrap();
        assert_eq!(tv_dec.windows, [Span::new(0, 7), Span::new(7, 11)]);

        let ex_msg = Extraction::new(&a, model_ref()).with_offsets(TEXT, mode).to_msg();
        assert_eq!((ex_msg.span_start[1], ex_msg.span_end[1]), expected_emoji);
        assert_eq!((ex_msg.frag_start[1], ex_msg.frag_end[1]), expected_cjk);
        let ex_back = c.round_trip_extraction(ex_msg.clone()).await.unwrap().into_inner();
        assert_eq!(ex_back, ex_msg);
        let ex_dec = Extraction::from_msg(&ex_back).unwrap();
        assert_eq!(ex_dec.spans, [Span::new(1, 10), Span::new(3, 7)]);
        assert_eq!(ex_dec.fragments, [Span::new(1, 3), Span::new(7, 10), Span::new(3, 7)]);
    }

    // Mid-surrogate (offset 3 in UTF-16 is inside the emoji surrogate pair 2..4) is refused.
    let mut bad_sparse = SparseTermsWire::new(make_sparse(), TEXT, OffsetMode::Utf16).to_msg();
    bad_sparse.span_start[1] = 3;
    let e = c.round_trip_sparse_terms(bad_sparse).await.expect_err("mid-surrogate sparse span");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));

    let mut bad_tv = TokenVectors::new(&v, &w).with_offsets(TEXT, OffsetMode::Utf16).to_msg();
    bad_tv.window_end[0] = 3;
    let e = c.round_trip_token_vectors(bad_tv).await.expect_err("mid-surrogate window");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));

    let mut bad_ex = Extraction::new(&a, model_ref()).with_offsets(TEXT, OffsetMode::Utf16).to_msg();
    bad_ex.span_start[1] = 3;
    let e = c.round_trip_extraction(bad_ex).await.expect_err("mid-surrogate answer");
    assert_eq!(refusal(&e), (Code::InvalidArgument, 256));
}

#[tokio::test]
async fn round_trip_document_graph_supports_all_registered_workspace_crates() {
    assert!(std::ptr::eq(codecs(), codecs()), "codecs registry is cached in OnceLock");
    let (_s, mut c) = serve().await;
    let mut doc = Document::new("Hello world");
    let p = || Provenance::by("test");

    let mut sents = Sentences { spans: Spans::new(ViewId::SOURCE) };
    sents.spans.push(Span::new(0, 11));
    doc.add(&Key::<Sentences>::DEFAULT, sents, p()).unwrap();

    let mut toks = Tokens { spans: Spans::new(ViewId::SOURCE), sentence: Some(vec![Ix::new(0), Ix::new(0)]) };
    toks.spans.push(Span::new(0, 5));
    toks.spans.push(Span::new(6, 11));
    doc.add(&Key::<Tokens>::DEFAULT, toks, p()).unwrap();

    let ent_tags = TagSet::new("ner", &["PERSON"]);
    let mut ent_spans = Spans::new(ViewId::SOURCE);
    ent_spans.push(Span::new(0, 5));
    let ents = Entities {
        spans: ent_spans,
        tokens: vec![turbo_analysis::IxRange::new(0, 1)],
        kind: vec![turbo_analysis::Tag(0)],
        kinds: ent_tags,
        score: vec![0.95],
    };
    doc.add(&Key::<Entities>::DEFAULT, ents, p()).unwrap();

    doc.add(&Key::<Abbreviations>::DEFAULT, Abbreviations::new(ViewId::SOURCE), p()).unwrap();
    doc.add(
        &Key::<Edges<Abbreviations, Abbreviations>>::DEFAULT,
        Edges::new(TagSet::new("abbrev", &["definition"]), Vec::new()),
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_chunker::terms::Vocabulary>::DEFAULT,
        turbo_chunker::terms::Vocabulary { text: turbo_analysis_layers::Strs::default() },
        p(),
    )
    .unwrap();
    doc.add(&Key::<turbo_coref::Mentions>::DEFAULT, turbo_coref::Mentions::new(ViewId::SOURCE), p()).unwrap();
    doc.add(
        &Key::<turbo_coref::Speakers>::DEFAULT,
        turbo_coref::Speakers {
            spans: Spans::new(ViewId::SOURCE),
            speaker: Vec::new(),
            speakers: TagSet::new("speakers", &["alice"]),
        },
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_encoding_layers::SourceEncoding>::DEFAULT,
        turbo_encoding_layers::SourceEncoding {
            encoding: turbo_encoding_layers::Encoding::Utf8,
            confidence: 1.0,
            evidence: turbo_encoding_layers::EncodingEvidence::ValidUtf8,
        },
        p(),
    )
    .unwrap();
    let mut linked = turbo_entity_linker::LinkedEntities::with_capacity(1);
    linked.kb_id.push(None);
    linked.provider_id.push(String::new());
    linked.canonical_name.push(String::new());
    linked.matched_alias.push(String::new());
    linked.score.push(0.0);
    linked.nil_reason.push(Some(turbo_entity_linker::NilReason::NoCandidates));
    linked.nil_cluster.push(Some(0));
    linked.candidates.push(Vec::new());
    doc.add(&Key::<turbo_entity_linker::LinkedEntities>::DEFAULT, linked, p()).unwrap();
    doc.add(&Key::<turbo_geo::Locations>::DEFAULT, turbo_geo::Locations::new(ViewId::SOURCE), p()).unwrap();
    doc.add(
        &Key::<turbo_gliner::Decisions>::DEFAULT,
        turbo_gliner::Decisions { threshold: 0.5, logits: vec![0.9] },
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_glossary_layers::GlossaryTerms>::DEFAULT,
        turbo_glossary_layers::GlossaryTerms::new(ViewId::SOURCE, Arc::new(turbo_glossary_layers::Glossary::new())),
        p(),
    )
    .unwrap();
    let mut sketches = turbo_group::Sketches::<Sentences>::empty(3);
    sketches.words.push(2);
    sketches.exact.push(1);
    sketches.shape.push(1);
    sketches.simhash.push(1);
    sketches.minhash.resize(64, 1);
    doc.add(&Key::<turbo_group::Sketches<Sentences>>::DEFAULT, sketches, p()).unwrap();
    doc.add(&Key::<turbo_keywords::Keywords>::DEFAULT, turbo_keywords::Keywords::new(ViewId::SOURCE), p()).unwrap();
    doc.add(
        &Key::<turbo_normalize_layers::Placeholders>::DEFAULT,
        turbo_normalize_layers::Placeholders::new(ViewId::SOURCE),
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_relation::Relations>::DEFAULT,
        turbo_relation::Relations {
            spans: Spans::new(ViewId::SOURCE),
            subject: Vec::new(),
            object: Vec::new(),
            kind: Vec::new(),
            kinds: TagSet::new("relations", &["works_at"]),
            score: Vec::new(),
        },
        p(),
    )
    .unwrap();
    let empty_tags = TagSet::new("records", &["record"]);
    doc.add(
        &Key::<turbo_schema_extract::Records>::DEFAULT,
        turbo_schema_extract::Records {
            spans: Spans::new(ViewId::SOURCE),
            record: Vec::new(),
            path: Vec::new(),
            score: Vec::new(),
            logit: Vec::new(),
            support_end: Vec::new(),
            support_source: Vec::new(),
            support_row: Vec::new(),
            record_type: Vec::new(),
            instance: Vec::new(),
            choice_record: Vec::new(),
            choice_path: Vec::new(),
            choice: Vec::new(),
            choice_score: Vec::new(),
            choice_logit: Vec::new(),
            types: empty_tags.clone(),
            paths: empty_tags.clone(),
            choices: empty_tags,
        },
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_stopword_layers::Stopwords>::DEFAULT,
        turbo_stopword_layers::Stopwords {
            language: turbo_stopword_layers::StopwordLanguage::English,
            stop: vec![false, false],
        },
        p(),
    )
    .unwrap();
    doc.add(
        &Key::<turbo_tokenize_layers::WordTypes>::DEFAULT,
        turbo_tokenize_layers::WordTypes {
            kind: vec![turbo_tokenize_layers::WordType::Alphanumeric, turbo_tokenize_layers::WordType::Alphanumeric],
        },
        p(),
    )
    .unwrap();
    let _ = Chunks::ID;

    let graph_bytes = export(&doc, codecs()).unwrap().to_bytes().unwrap();
    let resp = c
        .round_trip_document_graph(DocumentGraphRequest { graph: graph_bytes.clone().into() })
        .await
        .unwrap()
        .into_inner();
    assert_eq!(resp.graph, graph_bytes, "re-exported DocumentGraph bytes are byte-identical");
    assert_eq!(resp.source_text, "Hello world");
    assert!(resp.layers.iter().all(|l| l.status == LayerWireStatus::Ok as i32));
    assert!(resp.layers.len() >= 18);
}
