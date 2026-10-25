//! LearnVocabulary against a served store: documents streamed in, the
//! vocabulary stored, listed and downloaded with its digest, the same
//! documents giving the same artifact, and a seed dictionary's headwords
//! kept.

mod vocab;

use tonic::Code;
use turbo_kserve::api::{TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use turbo_kserve::proto::vocabulary::vocabulary_artifact_chunk::Frame;
use turbo_kserve::proto::vocabulary::*;
use vocab::*;

const DOCS: [&str; 3] = [
    "The New York office opened. The office in New York is large.",
    "A new office opened in Boston, and the Boston office is small.",
    "New York and Boston each have an office.",
];

fn learning() -> LearnVocabularyHeader {
    LearnVocabularyHeader {
        min_term_frequency: 2,
        max_terms: 0,
        lowercase: true,
        filter_stopwords: true,
        include_ngrams: true,
        min_ngram_size: 2,
        max_ngram_size: 3,
        min_pmi: 1.0,
        dictionary_artifact_id: String::new(),
        language: "en".into(),
    }
}

/// The vocabulary's rows after its two comment lines and column header.
fn rows(bytes: &[u8]) -> Vec<Vec<String>> {
    let text = std::str::from_utf8(bytes).unwrap();
    text.lines().skip(3).map(|l| l.split('\t').map(str::to_owned).collect()).collect()
}

#[tokio::test]
async fn a_learned_vocabulary_is_stored_and_downloads_with_its_digest() {
    let (s, mut c) = start(true, Some(dir("learn"))).await;
    let v = learn(&mut c, learning(), &DOCS).await.unwrap();
    assert_eq!((v.documents, v.language.as_str()), (3, "en"));
    let (head, bytes, sizes) = download(&mut c, &v.artifact_id).await.unwrap();
    assert_eq!(head, Frame::Vocabulary(v.clone()));
    assert_eq!((bytes.len() as u64, sha(&bytes)), (v.byte_size, v.sha256.clone()));
    assert_eq!(sizes.len(), 1);
    let rows = rows(&bytes);
    assert_eq!(rows.len() as u64, v.term_count + v.ngram_count);
    let find = |t: &str| rows.iter().find(|r| r[0] == t).cloned();
    // office: five times in three documents, the most frequent term.
    assert_eq!(rows[0][0..4], ["office", "term", "5", "3"]);
    assert!(find("the").is_none() && find("in").is_none(), "stopwords filtered");
    assert!(find("large").is_none(), "seen once, under min_term_frequency");
    let ny = find("new york").expect("new york");
    assert_eq!(ny[1..4], ["ngram", "3", "2"]);
    assert!(ny[5].parse::<f64>().unwrap() >= 1.0);
    // "york office" is a pair once, under the threshold, and "opened the"
    // would cross a full stop.
    assert!(find("york office").is_none() && find("opened the").is_none());
    assert_eq!(
        c.list_vocabularies(ListVocabulariesRequest {}).await.unwrap().into_inner().vocabularies,
        std::slice::from_ref(&v)
    );
    // The same documents in the same order are the same artifact.
    assert_eq!(learn(&mut c, learning(), &DOCS).await.unwrap(), v);
    c.delete_vocabulary_artifact(DeleteVocabularyArtifactRequest { artifact_id: v.artifact_id.clone() }).await.unwrap();
    let e = download(&mut c, &v.artifact_id).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    s.stop().await;
}

#[tokio::test]
async fn a_seed_dictionary_keeps_its_headwords() {
    let (s, mut c) = start(true, Some(dir("seed"))).await;
    let seed = import(&mut c, "headword-definition-tsv", "en", b"Boston\ta city\nharbour\ta port\nNew York\ta city\n")
        .await
        .unwrap();
    let h = LearnVocabularyHeader { dictionary_artifact_id: seed.artifact_id.clone(), ..learning() };
    let v = learn(&mut c, h, &DOCS).await.unwrap();
    assert_eq!(v.seed_dictionary_artifact_id, seed.artifact_id);
    let (_, bytes, _) = download(&mut c, &v.artifact_id).await.unwrap();
    let rows = rows(&bytes);
    let find = |t: &str| rows.iter().find(|r| r[0] == t).cloned().unwrap_or_else(|| panic!("{t}"));
    assert_eq!(find("boston")[6], "1");
    assert_eq!(find("new york")[6], "1");
    // Never in the corpus: kept with no counts.
    assert_eq!(find("harbour")[1..4], ["term", "0", "0"]);
    assert_eq!(find("office")[6], "0");
    s.stop().await;
}

#[tokio::test]
async fn refused_learning_stores_nothing() {
    let (s, mut c) = start(true, Some(dir("learn-refused"))).await;
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    for h in [
        LearnVocabularyHeader { min_ngram_size: 1, ..learning() },
        LearnVocabularyHeader { max_ngram_size: 9, ..learning() },
        LearnVocabularyHeader { language: "xx".into(), ..learning() },
        LearnVocabularyHeader { min_pmi: f64::INFINITY, ..learning() },
    ] {
        let e = learn(&mut c, h.clone(), &DOCS).await.unwrap_err();
        assert_eq!(refusal(&e), invalid, "{h:?}");
    }
    let e = learn(&mut c, learning(), &[]).await.unwrap_err();
    assert_eq!(refusal(&e), invalid);
    let h = LearnVocabularyHeader { dictionary_artifact_id: "dict-none".into(), ..learning() };
    let e = learn(&mut c, h, &DOCS).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    let mut f = learn_frames(learning(), &DOCS);
    f.push(f[0].clone());
    let e = c.learn_vocabulary(tokio_stream::iter(f)).await.unwrap_err();
    assert_eq!(refusal(&e), invalid);
    assert!(c.list_vocabularies(ListVocabulariesRequest {}).await.unwrap().into_inner().vocabularies.is_empty());
    s.stop().await;
}
