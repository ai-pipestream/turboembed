//! End-to-end gRPC tests for zero-copy suggest lexicons and context-aware
//! query spellcheck on `TurboVocabularyService` (issue #2341).

mod vocab;

use tonic::Code;
use turbo_kserve::api::{TURBO_E_BUNDLE_NOT_FOUND, TURBO_E_INVALID_ARGUMENT};
use turbo_kserve::proto::vocabulary::build_suggest_lexicon_request::Frame as BuildFrame;
use turbo_kserve::proto::vocabulary::vocabulary_artifact_chunk::Frame as ChunkFrame;
use turbo_kserve::proto::vocabulary::*;
use turbo_suggest::{ImageBuf, MappedSurfaceLexicon};
use vocab::*;

#[tokio::test]
async fn suggest_lexicon_builds_mmaps_downloads_and_completes_stemmed_and_diacritic_terms() {
    let (s, mut c) = start(true, Some(dir("suggest"))).await;

    let frames = vec![
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Header(BuildSuggestLexiconHeader {
                language: "en".into(),
                top_k_prefix: 8,
                top_k_stem: 8,
                count_doc_freq: true,
            })),
        },
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Document(LearnVocabularyDocument {
                text: "Courtesy of the diplomatic office. Courtesy matters.".into(),
            })),
        },
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Document(LearnVocabularyDocument {
                text: "Courtesy was extended to every delegation.".into(),
            })),
        },
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Document(LearnVocabularyDocument {
                text: "Diplomatic courtesies were observed.".into(),
            })),
        },
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Observation(SuggestObservation {
                surface: "Café".into(),
                stem: "cafe".into(),
                count: 25,
            })),
        },
        BuildSuggestLexiconRequest {
            frame: Some(BuildFrame::Observation(SuggestObservation {
                surface: "cafeteria".into(),
                stem: "cafeteria".into(),
                count: 10,
            })),
        },
    ];

    let desc = c.build_suggest_lexicon(tokio_stream::iter(frames)).await.unwrap().into_inner();
    assert_eq!(desc.language, "en");
    assert!(desc.entry_count >= 5 && desc.stem_count >= 4);

    let listed = c.list_suggest_lexicons(ListSuggestLexiconsRequest {}).await.unwrap().into_inner().lexicons;
    assert_eq!(listed, std::slice::from_ref(&desc));

    // DownloadVocabulary returns the raw TSUGLEX1 image with matching digest.
    let (head, bytes, _) = download(&mut c, &desc.artifact_id).await.unwrap();
    assert_eq!(head, ChunkFrame::SuggestLexicon(desc.clone()));
    assert_eq!((bytes.len() as u64, sha(&bytes)), (desc.byte_size, desc.sha256.clone()));
    let img = ImageBuf::copy_from(&bytes);
    let local_lex = MappedSurfaceLexicon::from_bytes(img.bytes()).unwrap();
    assert_eq!(local_lex.best_surface_for_stem("courtesi"), Some("courtesy"));

    // Prefix "courtes" and full word "courtesy" both resolve human surfaces + stem "courtesi".
    let pref = c
        .suggest_prefix(SuggestPrefixRequest {
            artifact_id: desc.artifact_id.clone(),
            prefix: "courtes".into(),
            limit: 5,
        })
        .await
        .unwrap()
        .into_inner()
        .completions;
    assert_eq!(pref.len(), 2);
    assert_eq!((pref[0].surface.as_str(), pref[0].stem.as_str(), pref[0].count), ("courtesy", "courtesi", 2));
    assert_eq!((pref[1].surface.as_str(), pref[1].stem.as_str(), pref[1].count), ("courtesies", "courtesi", 1));

    let full = c
        .suggest_prefix(SuggestPrefixRequest {
            artifact_id: desc.artifact_id.clone(),
            prefix: "courtesy".into(),
            limit: 5,
        })
        .await
        .unwrap()
        .into_inner()
        .completions;
    assert_eq!(full.len(), 1);
    assert_eq!((full[0].surface.as_str(), full[0].stem.as_str()), ("courtesy", "courtesi"));

    // Diacritic folding over gRPC.
    let cafe = c
        .suggest_prefix(SuggestPrefixRequest { artifact_id: desc.artifact_id.clone(), prefix: "cafe".into(), limit: 5 })
        .await
        .unwrap()
        .into_inner()
        .completions;
    assert_eq!(cafe[0].surface, "café");
    assert_eq!(cafe[1].surface, "cafeteria");

    // Stem-to-surface lookup over gRPC.
    let stems = c
        .lookup_stem_surfaces(LookupStemSurfacesRequest {
            artifact_id: desc.artifact_id.clone(),
            stem: "courtesi".into(),
            limit: 5,
        })
        .await
        .unwrap()
        .into_inner()
        .surfaces;
    assert_eq!(stems.len(), 2);
    assert_eq!((stems[0].surface.as_str(), stems[1].surface.as_str()), ("courtesy", "courtesies"));

    c.delete_vocabulary_artifact(DeleteVocabularyArtifactRequest { artifact_id: desc.artifact_id.clone() })
        .await
        .unwrap();
    let e = c
        .suggest_prefix(SuggestPrefixRequest { artifact_id: desc.artifact_id, prefix: "courtes".into(), limit: 5 })
        .await
        .unwrap_err();
    assert_eq!(refusal(&e), (Code::NotFound, TURBO_E_BUNDLE_NOT_FOUND));
    s.stop().await;
}

#[tokio::test]
async fn spellcheck_query_uses_learned_vocabulary_bigrams_and_projects_all_offset_modes() {
    let (s, mut c) = start(true, Some(dir("spellcheck"))).await;

    let mut docs: Vec<&str> = Vec::new();
    docs.extend(std::iter::repeat_n("The United States Supreme Court issued a unanimous opinion.", 30));
    docs.extend(std::iter::repeat_n("Push the grocery shopping cart to the checkout lane.", 50));
    docs.extend(std::iter::repeat_n("Old notes mention a cort unit in passing.", 4));

    let h = LearnVocabularyHeader {
        min_term_frequency: 2,
        max_terms: 0,
        lowercase: true,
        filter_stopwords: false,
        include_ngrams: true,
        min_ngram_size: 2,
        max_ngram_size: 3,
        min_pmi: -10.0,
        dictionary_artifact_id: String::new(),
        language: "en".into(),
    };
    let vocab = learn(&mut c, h, &docs).await.unwrap();

    // "🚀 " is 4 UTF-8 bytes + 1 space = 5 bytes; 2 UTF-16 units + 1 space = 3 UTF-16 units; 1 CP + 1 space = 2 CPs.
    // "united states " is 14 ASCII chars.
    // So "supreem" starts at UTF-8 19..26, UTF-16 17..24, CodePoint 16..23.
    let query = "🚀 united states supreem cort \"supreem cort\"";
    for (mode, exp_span) in [
        (SpellcheckOffsetMode::Utf8, (19, 26)),
        (SpellcheckOffsetMode::Utf16, (17, 24)),
        (SpellcheckOffsetMode::CodePoint, (16, 23)),
    ] {
        let resp = c
            .spellcheck_query(SpellcheckQueryRequest {
                vocabulary_artifact_id: vocab.artifact_id.clone(),
                query: query.into(),
                offsets: mode as i32,
                protected_spans: Vec::new(),
                max_edit_distance: 2,
            })
            .await
            .unwrap()
            .into_inner();
        assert!(resp.did_change);
        assert_eq!(resp.corrected_query, "🚀 united states supreme court \"supreem cort\"");
        assert_eq!(resp.edits.len(), 2);
        let sp = resp.edits[0].span.unwrap();
        assert_eq!((sp.start, sp.end), exp_span, "{mode:?}");
        assert_eq!(resp.edits[1].replacement, "court");
    }

    // Mid-surrogate UTF-16 protected span is refused with TURBO_E_INVALID_ARGUMENT.
    let bad = c
        .spellcheck_query(SpellcheckQueryRequest {
            vocabulary_artifact_id: vocab.artifact_id,
            query: query.into(),
            offsets: SpellcheckOffsetMode::Utf16 as i32,
            protected_spans: vec![SpellcheckSpan { start: 1, end: 3 }],
            max_edit_distance: 2,
        })
        .await
        .unwrap_err();
    assert_eq!(refusal(&bad), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT));
    s.stop().await;
}
