//! A StreamingTraining corpus job end to end on the tiny BERT bundle: a
//! vocabulary learned from the streamed documents, a static model
//! distilled with its terms and served, and an index of the documents
//! that a search reads; the start's checks, each naming the part of the
//! job it refuses; the stream's own rules; and an upload that is
//! abandoned or goes idle leaving nothing, with the slot free.

mod common;
#[path = "training_stream/fixture.rs"]
mod fixture;

use std::fs;

use fixture::*;

#[tokio::test]
async fn a_corpus_becomes_a_vocabulary_a_served_model_and_a_searchable_index() {
    let d = scratch("full");
    let (s, mut c) = server(&d, ALL).await;
    let updates = job(&mut c, corpus("corpus-static", "corpus-index")).await.unwrap();
    use StreamingStage::*;
    assert_eq!(
        stages(&updates),
        [Admitted, Analyzing, LearningVocabulary, DistillingModel, Indexing, Completed],
        "{updates:?}"
    );
    let progress = |stage| {
        updates.iter().rev().find_map(|u| match u {
            Update::Progress(p) if p.stage() == stage => Some(p),
            _ => None,
        })
    };
    let analyzed = progress(Analyzing).unwrap();
    assert_eq!((analyzed.completed_items, analyzed.total_items), (3, 3));
    // neuroimaging, turboembed, scans and the are seen twice.
    assert_eq!(progress(LearningVocabulary).unwrap().completed_items, 4);
    let distilled: Vec<TrainStage> = updates
        .iter()
        .filter_map(|u| match u {
            Update::Progress(p) => p.distillation.as_ref().map(|d| d.stage()),
            _ => None,
        })
        .collect();
    assert_eq!(distilled.first(), Some(&TrainStage::Kept), "{distilled:?}");
    assert_eq!(distilled[distilled.len() - 2..], [TrainStage::Sealed, TrainStage::Loaded], "{distilled:?}");
    let indexed = progress(Indexing).unwrap();
    assert_eq!((indexed.completed_items, indexed.total_items), (3, 3));

    let Some(Update::Corpus(kept)) = updates.last() else { panic!("{updates:?}") };
    assert!(kept.vocabulary_created);
    assert_eq!((kept.vocabulary_terms, kept.vocabulary_ngrams), (4, 0));
    let trained = kept.model.as_ref().unwrap();
    assert_eq!(trained.added_terms, ["neuroimaging", "turboembed"]);
    assert_eq!(trained.vocabulary_artifact_id, kept.vocabulary_artifact_id);
    assert_eq!(trained.vocabulary_sha256, kept.vocabulary_sha256);
    let m = trained.model.as_ref().unwrap();
    assert!(m.verified && m.served, "{m:?}");
    assert_eq!((m.name.as_str(), m.dims, m.learned_terms), ("corpus-static", 16, 2));
    assert_eq!((kept.index_name.as_str(), kept.indexed_documents), ("corpus-index", 3));
    assert!(kept.indexed_chunks >= 3, "{kept:?}");

    // All three are where their own services keep them.
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!(vocabularies, std::slice::from_ref(&kept.vocabulary_artifact_id));
    assert_eq!(models, ["corpus-static"], "no staging is left");
    assert_eq!(indexes, ["corpus-index"]);
    let mut k = common::kserve(&s).await;
    assert_eq!(common::embed_shape(&mut k, "corpus-static", &[b"neuroimaging scans"]).await, [1, 16]);

    // The index is in the model's space, every chunk has its vector, and
    // a search reads the documents.
    let mut x = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let summary = x.list_indexes(ListIndexesRequest {}).await.unwrap().into_inner().indexes.remove(0);
    let space = summary.space.as_ref().unwrap();
    assert_eq!(space.dim, 16);
    assert_eq!(space.model.as_ref().unwrap().sha256, m.manifest_sha256);
    assert_eq!((summary.document_count, u64::from(summary.chunk_count)), (3, kept.indexed_chunks));
    assert_eq!(summary.dense_row_count, summary.chunk_count);
    let query = SearchRequest {
        index_name: "corpus-index".into(),
        natural_query: "neuroimaging".into(),
        top_k: 3,
        ..Default::default()
    };
    let hits = x.search(query).await.unwrap().into_inner().hits;
    let mut found: Vec<&str> = hits.iter().map(|h| h.doc_id.as_str()).collect();
    found.sort();
    assert_eq!(found, ["a", "b"], "{hits:?}");

    // Both names are taken now, and the vocabulary is the store's.
    let e = job(&mut c, corpus("corpus-static", "other")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "2".into()), "{e:?}");
    let e = job(&mut c, corpus("other", "corpus-index")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "3".into()), "{e:?}");
    let updates = job(&mut c, corpus("second-static", "second-index")).await.unwrap();
    let Some(Update::Corpus(second)) = updates.last() else { panic!("{updates:?}") };
    assert!(!second.vocabulary_created);
    assert_eq!(second.vocabulary_artifact_id, kept.vocabulary_artifact_id);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_corpus_job_needs_admin_a_training_directory_and_a_vocabulary_store() {
    let d = scratch("opt-in");
    let unsupported = (Code::Unimplemented, TURBO_E_UNSUPPORTED);
    let (s, mut c) = server(&d, Dirs { admin: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, unsupported, "{e:?}");
    assert!(e.message().contains("--admin"), "{e:?}");
    s.stop().await;
    let (s, mut c) = server(&d, Dirs { training: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, unsupported, "{e:?}");
    assert!(e.message().contains("--training-dir"), "{e:?}");
    s.stop().await;
    let (s, mut c) = server(&d, Dirs { vocabulary: false, ..ALL }).await;
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e), ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "1".into()), "{e:?}");
    assert!(e.message().contains("--vocabulary-dir"), "{e:?}");
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_bad_start_is_refused_for_the_part_it_names_before_any_work() {
    let d = scratch("checks");
    let (s, mut c) = server(&d, ALL).await;
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    let not_found = (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND);
    let good = || corpus("x", "x");
    let with_vocabulary = |v: CorpusVocabulary| CorpusTraining { vocabulary: Some(v), ..good() };
    let with_model = |f: &dyn Fn(&mut TrainStaticModelRequest)| {
        let mut t = good();
        f(t.model.as_mut().unwrap());
        t
    };
    let with_index = |ix: CorpusIndex| CorpusTraining { index: Some(ix), ..good() };
    let named = |ix: CorpusIndex| CorpusIndex { index_name: "x".into(), ..ix };
    let cases = [
        (with_vocabulary(CorpusVocabulary { language: "a\nb".into(), ..vocabulary() }), invalid, "1"),
        (with_vocabulary(CorpusVocabulary { include_ngrams: true, min_ngram_size: 1, ..vocabulary() }), invalid, "1"),
        (
            with_vocabulary(CorpusVocabulary { dictionary_artifact_id: "dict-none".into(), ..vocabulary() }),
            not_found,
            "1",
        ),
        (CorpusTraining { model: None, ..good() }, invalid, "2"),
        (with_model(&|m| m.vocabulary_artifact_id = "vocab-none".into()), invalid, "2"),
        (with_model(&|m| m.name = "../escape".into()), invalid, "2"),
        (with_model(&|m| m.teacher = "nobody".into()), not_found, "2"),
        (with_model(&|m| m.pca_dims = 33), invalid, "2"),
        (CorpusTraining { index: None, ..good() }, invalid, "3"),
        (with_index(CorpusIndex::default()), invalid, "3"),
        (with_index(named(CorpusIndex { storage: 7, ..Default::default() })), invalid, "3"),
        (with_index(named(CorpusIndex { collection_name: "a\tb".into(), ..Default::default() })), invalid, "3"),
        (
            with_index(named(CorpusIndex {
                storage: CorpusIndexStorage::TurboQuant as i32,
                quant_bits: 9,
                ..Default::default()
            })),
            invalid,
            "3",
        ),
        // 12 values are no whole number of TurboQuant's blocks of 8.
        (
            CorpusTraining {
                index: Some(named(CorpusIndex {
                    storage: CorpusIndexStorage::TurboQuant as i32,
                    ..Default::default()
                })),
                ..with_model(&|m| m.pca_dims = 12)
            },
            invalid,
            "3",
        ),
    ];
    for (t, want, field) in cases {
        let e = job(&mut c, t.clone()).await.unwrap_err();
        assert_eq!(refusal(&e), (want, field.to_owned()), "{t:?}: {e:?}");
    }

    // The stream's own rules, field 0: the start first and once, a job
    // named, documents and not samples, each with an id of its own and a
    // text, and at least one.
    let stream_cases: Vec<(Vec<StreamingTrainingRequest>, &str)> = vec![
        (vec![], "before its start"),
        (documents(), "first message must be the start"),
        (vec![part(Part::Start(StreamingTrainingStart { job: None }))], "corpus job or a component job"),
        (vec![start(good()), document("a", "one"), start(good())], "second start"),
        (vec![start(good()), part(Part::Samples(b"1\tx".to_vec()))], "not samples"),
        (vec![start(good()), document("", "one")], "doc_id"),
        (vec![start(good()), document("a", "")], "no text"),
        (vec![start(good()), document("a", "one"), document("a", "two")], "second document"),
        (vec![start(good())], "no documents"),
    ];
    for (requests, why) in stream_cases {
        let e = run(&mut c, requests).await.unwrap_err();
        assert_eq!(refusal(&e), (invalid, "0".into()), "{why}: {e:?}");
        assert!(e.message().contains(why), "{why}: {e:?}");
    }
    assert_eq!(held(&s, &d).await, (vec![], vec![], vec![]));
    // Every refusal gave the slot back.
    job(&mut c, good()).await.unwrap();
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn an_abandoned_or_idle_upload_leaves_nothing_and_frees_the_slot() {
    let d = scratch("cancel");
    let (s, mut c) = server(&d, ALL).await;
    // The client drops the answer and its upload after two documents.
    let (tx, stream) = open(&mut c, corpus("gone", "gone")).await;
    tx.send(document("a", DOCS[0].1)).await.unwrap();
    tx.send(document("b", DOCS[1].1)).await.unwrap();
    drop((tx, stream));
    let updates = when_free(&mut c, corpus("after", "after")).await;
    assert!(matches!(updates.last(), Some(Update::Corpus(_))), "{updates:?}");
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!((vocabularies.len(), models, indexes), (1, vec!["after".to_owned()], vec!["after".to_owned()]));

    // An upload that sends nothing for the idle limit is dropped.
    let (tx, mut stream) = open(&mut c, corpus("idle", "idle")).await;
    tx.send(document("a", DOCS[0].1)).await.unwrap();
    let e = loop {
        match stream.message().await {
            Ok(Some(_)) => {}
            Ok(None) => panic!("the idle upload ended without a status"),
            Err(e) => break e,
        }
    };
    assert_eq!(refusal(&e), ((Code::OutOfRange, TURBO_E_CAPACITY), "0".into()), "{e:?}");
    drop(tx);
    let (vocabularies, models, indexes) = held(&s, &d).await;
    assert_eq!((vocabularies.len(), models, indexes), (1, vec!["after".to_owned()], vec!["after".to_owned()]));
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}
