//! How a StreamingTraining corpus job is kept or undone, on a real
//! server: a job that fails at its last step removes its model and never
//! showed its vocabulary; the index's name is checked against indexes and
//! aliases when the job publishes, not only when it starts; and a
//! vocabulary another call learned, before the job or while it runs, is
//! not the job's to remove.

mod common;
#[path = "training_stream/fixture.rs"]
mod fixture;

use std::fs;

use fixture::*;
use turbo_kserve::proto::search::SetAliasRequest;

type Search = TurboSearchServiceClient<Channel>;

/// An index in some other model's space, to take a name with.
async fn take(x: &mut Search, name: &str) {
    let spec = IndexSpec {
        index_name: name.into(),
        space: Some(SpaceSpec {
            model: Some(ModelIdentity { id: "other".into(), revision: "1".into(), sha256: "0".repeat(64) }),
            dim: 8,
            normalization: 1,
        }),
        ..Default::default()
    };
    x.create_index(CreateIndexRequest { spec: Some(spec), load_from_path: String::new() }).await.unwrap();
}

/// Sends the documents, ends the upload, and reads the job to its end:
/// its updates, and the status it failed with.
async fn finish(
    tx: mpsc::Sender<StreamingTrainingRequest>,
    stream: &mut tonic::Streaming<StreamingTrainingUpdate>,
) -> (Vec<Update>, Option<tonic::Status>) {
    for r in documents() {
        tx.send(r).await.unwrap();
    }
    drop(tx);
    let mut updates = Vec::new();
    loop {
        match stream.message().await {
            Ok(Some(u)) => updates.push(u.update.unwrap()),
            Ok(None) => return (updates, None),
            Err(e) => return (updates, Some(e)),
        }
    }
}

/// LearnVocabulary of the job's documents with the job's settings: the
/// same bytes, so the same artifact.
async fn learn(s: &Server) -> String {
    let mut v = TurboVocabularyServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let header = LearnVocabularyHeader { lowercase: true, min_term_frequency: 2, ..Default::default() };
    let frames: Vec<_> = std::iter::once(Frame::Header(header))
        .chain(DOCS.iter().map(|(_, t)| Frame::Document(LearnVocabularyDocument { text: (*t).into() })))
        .map(|f| LearnVocabularyRequest { frame: Some(f) })
        .collect();
    v.learn_vocabulary(tokio_stream::iter(frames)).await.unwrap().into_inner().artifact_id
}

const TAKEN: ((Code, i32), &str) = ((Code::FailedPrecondition, TURBO_E_INVALID_STATE), "3");

fn failed(e: &Option<tonic::Status>) -> ((Code, i32), String) {
    refusal(e.as_ref().expect("the job fails"))
}

#[tokio::test]
async fn a_job_that_fails_at_its_last_step_removes_its_model_and_never_showed_its_vocabulary() {
    let d = scratch("rollback");
    let (s, mut c) = server(&d, ALL).await;
    let mut x = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    // The index name is free at the start and taken before the job
    // publishes: the model is made and served, then removed.
    let (tx, mut stream) = open(&mut c, corpus("rolled-back", "late")).await;
    // While it holds the slot another job is busy.
    let e = job(&mut c, corpus("x", "x")).await.unwrap_err();
    assert_eq!(refusal(&e).0, (Code::Unavailable, TURBO_E_BUSY), "{e:?}");
    take(&mut x, "late").await;
    for r in documents() {
        tx.send(r).await.unwrap();
    }
    drop(tx);
    // The vocabulary is learned, and the store does not hold it: this job
    // will not complete, so at no time from here does a listing show it.
    let mut updates = Vec::new();
    let e = loop {
        match stream.message().await {
            Ok(Some(u)) => {
                let u = u.update.unwrap();
                if stages(std::slice::from_ref(&u)) == [StreamingStage::LearningVocabulary] {
                    assert!(held(&s, &d).await.0.is_empty(), "a vocabulary is stored before its job is kept");
                }
                updates.push(u);
            }
            Ok(None) => panic!("the job ended without its failure: {updates:?}"),
            Err(e) => break e,
        }
    };
    assert_eq!(refusal(&e), (TAKEN.0, TAKEN.1.into()), "{e:?}");
    // It got as far as the index: the model was served on the way.
    use StreamingStage::*;
    assert_eq!(stages(&updates), [Analyzing, LearningVocabulary, DistillingModel, Indexing], "{updates:?}");
    assert_eq!(held(&s, &d).await, (vec![], vec![], vec!["late".to_owned()]));
    let mut k = common::kserve(&s).await;
    let e = k.model_infer(common::texts_req("rolled-back", &[b"hello"])).await.unwrap_err();
    assert_eq!(e.code(), Code::NotFound, "{e:?}");
    // The names the failed job asked for are free again.
    let updates = job(&mut c, corpus("rolled-back", "second")).await.unwrap();
    let Some(Update::Corpus(kept)) = updates.last() else { panic!("{updates:?}") };
    assert!(kept.vocabulary_created);
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn an_alias_that_takes_the_name_while_the_job_runs_fails_it() {
    let d = scratch("alias");
    let (s, mut c) = server(&d, ALL).await;
    let mut x = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    take(&mut x, "other").await;
    // "claimed" is free when the job starts and an alias of another index
    // when it publishes. Published, the job's index would be listed and
    // never searched: a search of the name goes to the alias's index.
    let (tx, mut stream) = open(&mut c, corpus("aliased", "claimed")).await;
    x.set_alias(SetAliasRequest { alias: "claimed".into(), index_name: "other".into() }).await.unwrap();
    let (updates, e) = finish(tx, &mut stream).await;
    assert_eq!(failed(&e), (TAKEN.0, TAKEN.1.into()), "{updates:?} {e:?}");
    assert!(e.unwrap().message().contains("alias"));
    assert_eq!(held(&s, &d).await, (vec![], vec![], vec!["other".to_owned()]));
    let query = SearchRequest { index_name: "claimed".into(), natural_query: "scans".into(), ..Default::default() };
    assert_eq!(x.search(query).await.unwrap().into_inner().resolved_index, "other");
    // An alias in the way at the start is refused before any work.
    let e = job(&mut c, corpus("aliased", "claimed")).await.unwrap_err();
    assert_eq!(refusal(&e), (TAKEN.0, TAKEN.1.into()), "{e:?}");
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}

#[tokio::test]
async fn a_vocabulary_another_call_learned_outlives_a_failed_job() {
    let d = scratch("adopted");
    let (s, mut c) = server(&d, ALL).await;
    let mut x = TurboSearchServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    // Learned by LearnVocabulary while the job runs: the job fails, and
    // the other call's vocabulary is still there.
    let (tx, mut stream) = open(&mut c, corpus("failing", "late")).await;
    take(&mut x, "late").await;
    let learned = learn(&s).await;
    let (updates, e) = finish(tx, &mut stream).await;
    assert_eq!(failed(&e), (TAKEN.0, TAKEN.1.into()), "{updates:?} {e:?}");
    assert_eq!(held(&s, &d).await, (vec![learned.clone()], vec![], vec!["late".to_owned()]));

    // Learned before the job: the same, and a job that completes names it
    // without having stored it.
    let (tx, mut stream) = open(&mut c, corpus("failing", "later")).await;
    take(&mut x, "later").await;
    let (updates, e) = finish(tx, &mut stream).await;
    assert_eq!(failed(&e), (TAKEN.0, TAKEN.1.into()), "{updates:?} {e:?}");
    assert_eq!(held(&s, &d).await, (vec![learned.clone()], vec![], vec!["late".to_owned(), "later".to_owned()]));
    let updates = job(&mut c, corpus("failing", "third")).await.unwrap();
    let Some(Update::Corpus(kept)) = updates.last() else { panic!("{updates:?}") };
    assert_eq!((kept.vocabulary_artifact_id.as_str(), kept.vocabulary_created), (learned.as_str(), false));
    s.stop().await;
    fs::remove_dir_all(d).unwrap();
}
