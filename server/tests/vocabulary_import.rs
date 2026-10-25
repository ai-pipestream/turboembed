//! ImportDictionary, the listings, DownloadVocabulary and
//! DeleteVocabularyArtifact against a served vocabulary store: every
//! importable format read and counted by its analysis reader, malformed
//! uploads refused with nothing stored, and the store's files surviving a
//! restart.

mod vocab;

use tonic::Code;
use turbo_glossary::{Case, Glossary, MappedGlossaryWriter, TbxDialect, write_tbx};
use turbo_kserve::api::{
    TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_CAPACITY, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_STATE, TURBO_E_UNSUPPORTED,
};
use turbo_kserve::proto::vocabulary::vocabulary_artifact_chunk::Frame;
use turbo_kserve::proto::vocabulary::*;
use turbo_kserve::vocabulary::{DOWNLOAD_CHUNK_BYTES, MAX_DICTIONARY_UPLOAD_BYTES};
use vocab::*;

const LMF: &str = "<?xml version=\"1.0\"?>\n\
    <LexicalResource><Lexicon id=\"t\" label=\"t\" language=\"en\" version=\"1\">\
    <LexicalEntry id=\"t-cat-n\"><Lemma writtenForm=\"cat\" partOfSpeech=\"n\"/>\
    <Sense id=\"t-cat-n-1\" synset=\"t-1\"/></LexicalEntry>\
    <LexicalEntry id=\"t-tom-n\"><Lemma writtenForm=\"tom cat\" partOfSpeech=\"n\"/>\
    <Sense id=\"t-tom-n-1\" synset=\"t-1\"/></LexicalEntry>\
    <Synset id=\"t-1\" partOfSpeech=\"n\"><Definition>a feline</Definition></Synset>\
    </Lexicon></LexicalResource>";

fn glossary() -> Glossary {
    let mut g = Glossary::new();
    g.push("g1", "New York").unwrap();
    g.push("g2", "Big Apple").unwrap();
    g
}

#[tokio::test]
async fn formats_and_limits_are_listed() {
    let (s, mut c) = start(false, None).await;
    let r = c.list_dictionary_formats(ListDictionaryFormatsRequest {}).await.unwrap().into_inner();
    let ids: Vec<(&str, bool)> = r.formats.iter().map(|f| (f.format_id.as_str(), f.importable)).collect();
    assert_eq!(ids.len(), 11);
    assert!(ids.contains(&("glossary-tbx", true)) && ids.contains(&("hunspell-aff-dic", false)));
    assert!(r.formats.iter().find(|f| f.format_id == "glossary-tbx").unwrap().needs_language);
    let l = r.limits.unwrap();
    assert_eq!((l.store, l.admin), (false, false));
    assert_eq!((l.max_dictionary_upload_bytes, l.download_chunk_bytes), (MAX_DICTIONARY_UPLOAD_BYTES, 65536));
    // Without a store there is nothing to list or download.
    assert!(c.list_dictionaries(ListDictionariesRequest {}).await.unwrap().into_inner().dictionaries.is_empty());
    let e = download(&mut c, "dict-x").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    s.stop().await;
}

#[tokio::test]
async fn writes_need_admin_and_a_store() {
    let (s, mut c) = start(false, Some(dir("no-admin"))).await;
    let e = import(&mut c, "plain-word-list", "", b"cat\n").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    let e = c.delete_vocabulary_artifact(DeleteVocabularyArtifactRequest { artifact_id: "x".into() }).await;
    assert_eq!(refusal(&e.unwrap_err()), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    let e = learn(&mut c, LearnVocabularyHeader::default(), &["a cat"]).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));
    s.stop().await;
    let (s, mut c) = start(true, None).await;
    let e = import(&mut c, "plain-word-list", "", b"cat\n").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::FailedPrecondition, TURBO_E_INVALID_STATE));
    s.stop().await;
}

#[tokio::test]
async fn each_importable_format_is_read_and_counted() {
    let d = dir("formats");
    let (s, mut c) = start(true, Some(d.clone())).await;
    let mut tbx = String::new();
    write_tbx(&glossary(), "en", TbxDialect::Martif, &mut tbx).unwrap();
    let mapped = MappedGlossaryWriter::write(&glossary(), Case::Sensitive).unwrap();
    let cases: Vec<(&str, &str, Vec<u8>, u64)> = vec![
        ("glossary-csv", "", b"g1,New York\ng2,\"Big, Apple\"\n".to_vec(), 2),
        ("glossary-tbx", "en", tbx.into_bytes(), 2),
        ("glossary-mapped", "", mapped, 2),
        ("plain-word-list", "", b"alpha\n\nbeta\r\ngamma".to_vec(), 3),
        ("headword-definition-tsv", "en", b"cat\ta feline\ndog\ta canine\n".to_vec(), 2),
        (
            "opennlp-dictionary-xml",
            "",
            b"<dictionary case_sensitive=\"false\"><entry><token>New</token><token>York</token></entry>\
              <entry><token>Paris</token></entry></dictionary>"
                .to_vec(),
            2,
        ),
        (
            "pos-tag-dictionary-xml",
            "",
            b"<dictionary><entry tags=\"NN VB\"><token>run</token></entry></dictionary>".to_vec(),
            1,
        ),
        ("lemmatizer-dictionary", "", b"cats\tNNS\tcat\nran\tVBD\trun\n".to_vec(), 2),
        ("wordnet-lmf-xml", "en", LMF.as_bytes().to_vec(), 2),
    ];
    for (format, language, bytes, entries) in &cases {
        let got = import(&mut c, format, language, bytes).await.unwrap_or_else(|e| panic!("{format}: {e:?}"));
        assert_eq!(got.format_id, *format);
        assert_eq!((got.entry_count, got.byte_size), (*entries, bytes.len() as u64), "{format}");
        assert_eq!(got.sha256, sha(bytes));
        assert_eq!(got.language, *language);
        let (head, back, _) = download(&mut c, &got.artifact_id).await.unwrap();
        assert_eq!(head, Frame::Dictionary(got.clone()));
        assert_eq!(&back, bytes, "{format}");
    }
    let listed = c.list_dictionaries(ListDictionariesRequest {}).await.unwrap().into_inner().dictionaries;
    assert_eq!(listed.len(), cases.len());
    assert!(listed.windows(2).all(|w| w[0].artifact_id < w[1].artifact_id));
    // A glossary read from text is stored with its compiled automaton.
    let images = std::fs::read_dir(d.join("dictionaries")).unwrap();
    let images = images.filter(|e| e.as_ref().unwrap().path().extension().is_some_and(|x| x == "image")).count();
    assert_eq!(images, 2);
    s.stop().await;
}

#[tokio::test]
async fn malformed_uploads_are_refused_and_nothing_is_stored() {
    let d = dir("malformed");
    let (s, mut c) = start(true, Some(d.clone())).await;
    let invalid = (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT);
    for (format, language, bytes) in [
        ("glossary-csv", "", &b"g1,\"New York\n"[..]),
        ("glossary-tbx", "en", b"<martif><text><body><termEntry>"),
        ("glossary-mapped", "", b"not an image at all, no"),
        ("plain-word-list", "", b"\n \n"),
        ("plain-word-list", "", b"caf\xC3"),
        ("headword-definition-tsv", "", b"cat a feline\n"),
        ("opennlp-dictionary-xml", "", b"<dictionary><entry><token>x</token>"),
        (
            "pos-tag-dictionary-xml",
            "",
            b"<dictionary><entry tags=\"NN\"><token>a</token><token>b</token></entry></dictionary>",
        ),
        ("lemmatizer-dictionary", "", b"cats\tNNS\n"),
        ("wordnet-lmf-xml", "", b"<LexicalResource><Lexicon id=\"t\">"),
        ("no-such-format", "", b"x"),
    ] {
        let e = import(&mut c, format, language, bytes).await.unwrap_err();
        assert_eq!(refusal(&e), invalid, "{format} {bytes:?}: {e:?}");
    }
    let e = import(&mut c, "glossary-tbx", "", b"<martif/>").await.unwrap_err();
    assert_eq!(refusal(&e), invalid);
    let e = import(&mut c, "hunspell-aff-dic", "", b"SET UTF-8\n").await.unwrap_err();
    assert_eq!(refusal(&e), (Code::Unimplemented, TURBO_E_UNSUPPORTED));

    let words = b"cat\ndog\n";
    let bad_headers = [
        ImportDictionaryHeader { sha256: sha(b"other"), ..header("plain-word-list", "", words) },
        ImportDictionaryHeader { sha256: "ABC".into(), ..header("plain-word-list", "", words) },
        ImportDictionaryHeader { byte_size: 100, ..header("plain-word-list", "", words) },
        ImportDictionaryHeader { byte_size: 3, ..header("plain-word-list", "", words) },
        ImportDictionaryHeader { name: "a\nb".into(), ..header("plain-word-list", "", words) },
    ];
    for h in bad_headers {
        let e = import_with(&mut c, h.clone(), words).await.unwrap_err();
        assert_eq!(refusal(&e), invalid, "{h:?}");
    }
    let past =
        ImportDictionaryHeader { byte_size: MAX_DICTIONARY_UPLOAD_BYTES + 1, ..header("plain-word-list", "", words) };
    let e = import_with(&mut c, past, words).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::OutOfRange, TURBO_E_CAPACITY));
    // A stream that opens with bytes, or sends a second header.
    let mut f = frames(header("plain-word-list", "", words), words);
    f.swap(0, 1);
    let e = c.import_dictionary(tokio_stream::iter(f)).await.unwrap_err();
    assert_eq!(refusal(&e), invalid);
    let mut f = frames(header("plain-word-list", "", words), words);
    f.push(f[0].clone());
    let e = c.import_dictionary(tokio_stream::iter(f)).await.unwrap_err();
    assert_eq!(refusal(&e), invalid);

    assert!(c.list_dictionaries(ListDictionariesRequest {}).await.unwrap().into_inner().dictionaries.is_empty());
    assert_eq!(std::fs::read_dir(d.join("dictionaries")).unwrap().count(), 0);
    s.stop().await;
}

#[tokio::test]
async fn one_upload_is_one_artifact_downloaded_in_chunks_until_deleted() {
    let d = dir("chunks");
    let (s, mut c) = start(true, Some(d.clone())).await;
    let words: String = (0..30_000).map(|i| format!("word{i}\n")).collect();
    let a = import(&mut c, "plain-word-list", "en", words.as_bytes()).await.unwrap();
    let mut renamed = header("plain-word-list", "en", words.as_bytes());
    renamed.name = "the same bytes again".into();
    let b = import_with(&mut c, renamed, words.as_bytes()).await.unwrap();
    assert_eq!(a, b, "the first upload's descriptor, name included");
    assert_eq!(a.entry_count, 30_000);
    let (_, back, sizes) = download(&mut c, &a.artifact_id).await.unwrap();
    assert_eq!(back, words.as_bytes());
    assert_eq!(sizes.len(), words.len().div_ceil(DOWNLOAD_CHUNK_BYTES));
    assert!(sizes.iter().all(|&n| n > 0 && n <= DOWNLOAD_CHUNK_BYTES));
    assert_eq!((back.len() as u64, sha(&back)), (a.byte_size, a.sha256.clone()));

    let del = |id: &str| DeleteVocabularyArtifactRequest { artifact_id: id.into() };
    c.delete_vocabulary_artifact(del(&a.artifact_id)).await.unwrap();
    assert!(c.list_dictionaries(ListDictionariesRequest {}).await.unwrap().into_inner().dictionaries.is_empty());
    assert_eq!(std::fs::read_dir(d.join("dictionaries")).unwrap().count(), 0);
    let e = download(&mut c, &a.artifact_id).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    let e = c.delete_vocabulary_artifact(del(&a.artifact_id)).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    let e = c.delete_vocabulary_artifact(del("../dictionaries")).await.unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    s.stop().await;
}

#[tokio::test]
async fn the_store_survives_a_restart_and_drops_cut_writes() {
    let d = dir("restart");
    let (s, mut c) = start(true, Some(d.clone())).await;
    let a = import(&mut c, "headword-definition-tsv", "en", b"cat\ta feline\n").await.unwrap();
    s.stop().await;
    // What a write cut short leaves: a temporary file and no descriptor.
    let stray = d.join("dictionaries").join(".tmp-1-0");
    std::fs::write(&stray, b"half").unwrap();
    let (s, mut c) = start(false, Some(d.clone())).await;
    assert_eq!(
        c.list_dictionaries(ListDictionariesRequest {}).await.unwrap().into_inner().dictionaries,
        std::slice::from_ref(&a)
    );
    assert!(!stray.exists());
    let (_, back, _) = download(&mut c, &a.artifact_id).await.unwrap();
    assert_eq!(back, b"cat\ta feline\n");
    // A stored file changed behind the store's back is refused, not served.
    std::fs::write(d.join("dictionaries").join(format!("{}.data", a.artifact_id)), b"dog\ta canine\n").unwrap();
    let e = download(&mut c, &a.artifact_id).await.unwrap_err();
    assert_eq!(e.code(), Code::Internal);
    s.stop().await;
}
