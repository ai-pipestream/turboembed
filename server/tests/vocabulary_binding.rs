//! Binding stored glossary and stopword artifacts from the vocabulary store
//! into `AnalyzeStream`, `AnalyzeDocumentProgressive`, and `LoadBundle`
//! without a restart (issue `#2013`).

#[path = "gliner_progressive/fixture.rs"]
mod gliner_fixture;
mod vocab;

use tokio_stream::StreamExt;
use tonic::Code;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Key, Layer, ViewId};
use turbo_analysis_layers::{Entities, Tokens};
use turbo_glossary_layers::GlossaryTerms;
use turbo_keywords::Keywords;
use turbo_kserve::api::TURBO_E_INVALID_ARGUMENT;
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::analyze_stream_request::Kind;
use turbo_kserve::proto::result::analyze_stream_response::Outcome;
use turbo_kserve::proto::result::{
    AnalyzeDocument, AnalyzeOutput, AnalyzePipelineRequest, AnalyzeStreamHeader, AnalyzeStreamRequest,
    CapabilitiesRequest, DocumentGraphResponse, InspectDocumentGraphRequest, LoadBundleRequest, ModelRef, OffsetMode,
    OpenSpanSchema, ProcessingReceipt, VocabularyArtifactRef, VocabularyBinding,
};
use turbo_kserve::proto::vocabulary::{DeleteVocabularyArtifactRequest, LearnVocabularyHeader};
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_stopword::Stopwords;
use vocab::{Client as VocabClient, dir, header, import as upload, import_with, learn, refusal};

type AnalysisClient = AnalysisServiceClient<tonic::transport::Channel>;

const TEXT: &str = "Quixotl and Zebulon build Zephyr engines.";

fn tiny_bundle_path() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle")
}

fn model_cfg(name: &str) -> ModelConfig {
    ModelConfig {
        bundle: tiny_bundle_path().to_str().unwrap().into(),
        name: Some(name.into()),
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn start_server(vocabulary: Option<std::path::PathBuf>, preload: bool) -> (Server, AnalysisClient, VocabClient) {
    let models = if preload { vec![model_cfg("base")] } else { Vec::new() };
    let serve = Serve { admin: true, vocabulary, ..Serve::default() };
    let s = Server::start("127.0.0.1:0".parse().unwrap(), models, MAX, serve).await.unwrap();
    if preload {
        s.load().await.unwrap();
    }
    let addr = format!("http://{}", s.local_addr());
    let ac = AnalysisServiceClient::connect(addr.clone()).await.unwrap().max_decoding_message_size(MAX);
    let vc = VocabClient::connect(addr).await.unwrap().max_decoding_message_size(MAX);
    (s, ac, vc)
}

fn art_ref(id: &str, sha256: &str) -> VocabularyArtifactRef {
    VocabularyArtifactRef { artifact_id: id.into(), sha256: sha256.into() }
}

fn sw_bind(id: &str, sha256: &str) -> VocabularyBinding {
    VocabularyBinding { stopwords: Some(art_ref(id, sha256)), glossary: None }
}

fn gloss_bind(id: &str, sha256: &str) -> VocabularyBinding {
    VocabularyBinding { stopwords: None, glossary: Some(art_ref(id, sha256)) }
}

async fn run_progressive_req(
    ac: &mut AnalysisClient,
    req: AnalyzePipelineRequest,
) -> (DocumentGraphResponse, ProcessingReceipt, Document) {
    let mut stream = ac.analyze_document_progressive(req).await.unwrap().into_inner();
    let (mut complete, mut receipt) = (None, None);
    while let Some(ev) = stream.next().await {
        match ev.unwrap().event {
            Some(Event::Complete(c)) => complete = Some(c),
            Some(Event::Receipt(r)) => receipt = Some(r),
            _ => {}
        }
    }
    let (complete, receipt) = (complete.expect("Complete event"), receipt.expect("Receipt event"));
    let doc = import(&DocumentGraph::from_bytes(&complete.graph).unwrap(), codecs()).unwrap();
    (complete, receipt, doc)
}

async fn run_progressive(
    ac: &mut AnalysisClient,
    model: &str,
    text: &str,
    vocabulary: Option<VocabularyBinding>,
) -> (DocumentGraphResponse, ProcessingReceipt, Document) {
    let req = AnalyzePipelineRequest {
        doc_id: "doc-vocab".into(),
        text: text.into(),
        offsets: OffsetMode::Utf8 as i32,
        profile: "default".into(),
        model: model.into(),
        vocabulary,
        ..Default::default()
    };
    run_progressive_req(ac, req).await
}

async fn run_stream_graph(
    ac: &mut AnalysisClient,
    model: &str,
    text: &str,
    vocabulary: Option<VocabularyBinding>,
) -> (DocumentGraphResponse, Document) {
    let doc = AnalyzeDocument {
        id: "stream-doc-1".into(),
        model: model.into(),
        text: text.as_bytes().to_vec(),
        output: AnalyzeOutput::DocumentGraph as i32,
        offsets: OffsetMode::Utf8 as i32,
        vocabulary,
        ..Default::default()
    };
    let requests = vec![
        AnalyzeStreamRequest { kind: Some(Kind::Header(AnalyzeStreamHeader::default())) },
        AnalyzeStreamRequest { kind: Some(Kind::Document(doc)) },
    ];
    let mut stream = ac.analyze_stream(tokio_stream::iter(requests)).await.unwrap().into_inner();
    let resp = stream.next().await.unwrap().unwrap();
    assert_eq!(resp.id, "stream-doc-1");
    let Some(Outcome::DocumentGraph(dg)) = resp.outcome else { panic!("expected DocumentGraph") };
    let doc = import(&DocumentGraph::from_bytes(&dg.graph).unwrap(), codecs()).unwrap();
    (dg, doc)
}

fn flagged_tokens(doc: &Document) -> Vec<String> {
    let (tokens, stop, text) =
        (doc.get::<Tokens>().unwrap(), doc.get::<Stopwords>().unwrap(), doc.text(ViewId::SOURCE));
    tokens
        .spans
        .iter()
        .zip(&stop.stop)
        .filter(|&(_, &is_stop)| is_stop)
        .map(|((_, sp), _)| text[sp.start as usize..sp.end as usize].to_owned())
        .collect()
}

fn matched_glossary(doc: &Document) -> Vec<(String, String)> {
    let (terms, text) = (doc.get::<GlossaryTerms>().expect("turbo:glossary-terms"), doc.text(ViewId::SOURCE));
    terms
        .spans
        .iter()
        .zip(&terms.entry)
        .map(|((_, sp), &ix)| {
            (terms.glossary.get(ix).id.to_owned(), text[sp.start as usize..sp.end as usize].to_owned())
        })
        .collect()
}

async fn assert_prov(
    ac: &mut AnalysisClient,
    (resp, doc, receipt): (&DocumentGraphResponse, &Document, Option<&ProcessingReceipt>),
    (layer_id, step_id): (&str, &str),
    want: &ModelRef,
) {
    let info = resp.layers.iter().find(|l| l.layer == layer_id).unwrap_or_else(|| panic!("missing layer {layer_id}"));
    assert_eq!(info.model.as_ref(), Some(want), "LayerInfo.model mismatch for {layer_id}");

    let typed = ac
        .inspect_document_graph(InspectDocumentGraphRequest { graph: resp.graph.to_vec(), ..Default::default() })
        .await
        .unwrap()
        .into_inner();
    let typed_layer =
        typed.layers.iter().find(|l| l.id == layer_id).unwrap_or_else(|| panic!("missing typed layer {layer_id}"));
    let model_ref_id = typed_layer.model_refs.first().expect("typed layer model_ref");
    let model_prov = typed
        .models
        .iter()
        .find(|m| &m.model_ref == model_ref_id)
        .unwrap_or_else(|| panic!("missing ModelProvenance {model_ref_id}"));
    assert_eq!(model_prov.bundle.as_ref(), Some(want), "TypedDocumentResult ModelProvenance.bundle mismatch");

    let doc_model = match layer_id {
        "turbo:stopwords" => doc.provenance(Key::<Stopwords>::DEFAULT.id()).unwrap().model.as_ref(),
        "turbo:keywords" => doc.provenance(Key::<Keywords>::DEFAULT.id()).unwrap().model.as_ref(),
        "turbo:glossary-terms" => doc.provenance(Key::<GlossaryTerms>::DEFAULT.id()).unwrap().model.as_ref(),
        other => panic!("unexpected layer {other}"),
    };
    let doc_model = doc_model.expect("Document layer Provenance.model");
    assert_eq!((&doc_model.id, &doc_model.revision, &doc_model.sha256), (&want.id, &want.revision, &want.sha256));

    if let Some(r) = receipt {
        let lr = r.layers.iter().find(|l| l.layer == layer_id).unwrap_or_else(|| panic!("missing receipt {layer_id}"));
        assert!(lr.executed.iter().any(|d| d.id == step_id), "missing detector {step_id} in receipt");
        assert!(!lr.provenance_digest.as_ref().expect("provenance_digest").is_empty());
    }
}

#[tokio::test]
async fn imported_stopword_list_changes_next_request_without_restart_and_records_provenance() {
    let store_dir = dir("bind-stopwords");
    let (server, mut ac, mut vc) = start_server(Some(store_dir.clone()), true).await;

    let (base_prog, _, base_doc) = run_progressive(&mut ac, "", TEXT, None).await;
    assert_eq!(flagged_tokens(&base_doc), vec!["and".to_owned()]);
    let base_sw_info = base_prog.layers.iter().find(|l| l.layer == "turbo:stopwords").unwrap();
    assert!(base_sw_info.model.is_none());
    let (_, base_stream_doc) = run_stream_graph(&mut ac, "base", TEXT, None).await;
    assert_eq!(flagged_tokens(&base_stream_doc), vec!["and".to_owned()]);

    let desc = upload(&mut vc, "plain-word-list", "en", b"quixotl\nzebulon\n").await.unwrap();
    let binding = VocabularyBinding { stopwords: Some(art_ref(&desc.artifact_id, &desc.sha256)), glossary: None };
    let want_model =
        ModelRef { id: desc.artifact_id.clone(), revision: "plain-word-list".into(), sha256: desc.sha256.clone() };

    let (bound_prog, bound_receipt, bound_doc) = run_progressive(&mut ac, "", TEXT, Some(binding.clone())).await;
    assert_eq!(flagged_tokens(&bound_doc), vec!["Quixotl".to_owned(), "Zebulon".to_owned()]);
    let bound_sw_info = bound_prog.layers.iter().find(|l| l.layer == "turbo:stopwords").unwrap();
    assert_ne!(bound_sw_info.config_hash, base_sw_info.config_hash);
    let p_ctx = (&bound_prog, &bound_doc, Some(&bound_receipt));
    assert_prov(&mut ac, p_ctx, ("turbo:stopwords", "turbo-stopword:bundled"), &want_model).await;
    assert_prov(&mut ac, p_ctx, ("turbo:keywords", "turbo-keywords:yake"), &want_model).await;
    let kws = bound_doc.get::<Keywords>().expect("turbo:keywords");
    for i in 0..kws.len() {
        let phrase = kws.phrase.get(i);
        assert!(phrase != "quixotl" && phrase != "zebulon", "stopword leaked into keywords: {phrase}");
    }

    let (stream_resp, stream_doc) = run_stream_graph(&mut ac, "base", TEXT, Some(binding.clone())).await;
    assert_eq!(flagged_tokens(&stream_doc), vec!["Quixotl".to_owned(), "Zebulon".to_owned()]);
    let s_ctx = (&stream_resp, &stream_doc, None);
    assert_prov(&mut ac, s_ctx, ("turbo:stopwords", "turbo-stopword:bundled"), &want_model).await;
    assert_prov(&mut ac, s_ctx, ("turbo:keywords", "turbo-keywords:yake"), &want_model).await;
    // Cache hit skips disk read; deleting the artifact evicts it from index and cache.
    std::fs::remove_file(store_dir.join("dictionaries").join(format!("{}.data", desc.artifact_id))).unwrap();
    let (_, _, cached_doc) = run_progressive(&mut ac, "", TEXT, Some(binding.clone())).await;
    assert_eq!(flagged_tokens(&cached_doc), vec!["Quixotl".to_owned(), "Zebulon".to_owned()]);
    vc.delete_vocabulary_artifact(DeleteVocabularyArtifactRequest { artifact_id: desc.artifact_id }).await.unwrap();
    let del_req = AnalyzePipelineRequest { vocabulary: Some(binding), ..Default::default() };
    let del_err = ac.analyze_document_progressive(del_req).await.err().unwrap();
    assert_eq!(refusal(&del_err), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT));
    server.stop().await;
}

#[tokio::test]
async fn imported_glossary_and_learned_vocabulary_bind_with_provenance() {
    let store_dir = dir("bind-glossary-learned");
    let (server, mut ac, mut vc) = start_server(Some(store_dir), true).await;

    let csv_bytes = b"CONCEPT_ZEB,Zebulon\nCONCEPT_ZEP,Zephyr engines\n";
    let gloss_desc = upload(&mut vc, "glossary-csv", "en", csv_bytes).await.unwrap();
    let learn_hdr =
        LearnVocabularyHeader { language: "en".into(), min_term_frequency: 1, lowercase: true, ..Default::default() };
    let learned_desc = learn(&mut vc, learn_hdr, &["quixotl build", "quixotl build"]).await.unwrap();

    let binding = VocabularyBinding {
        stopwords: Some(art_ref(&learned_desc.artifact_id, &learned_desc.sha256)),
        glossary: Some(art_ref(&gloss_desc.artifact_id, &gloss_desc.sha256)),
    };
    let want_sw_model = ModelRef {
        id: learned_desc.artifact_id.clone(),
        revision: "turbo-vocabulary-1".into(),
        sha256: learned_desc.sha256.clone(),
    };
    let want_gloss_model =
        ModelRef { id: gloss_desc.artifact_id.clone(), revision: "glossary-csv".into(), sha256: gloss_desc.sha256 };
    let want_glossary_matches =
        vec![("CONCEPT_ZEB".to_owned(), "Zebulon".to_owned()), ("CONCEPT_ZEP".to_owned(), "Zephyr engines".to_owned())];

    let (prog_resp, prog_receipt, prog_doc) = run_progressive(&mut ac, "", TEXT, Some(binding.clone())).await;
    assert_eq!(flagged_tokens(&prog_doc), vec!["Quixotl".to_owned(), "build".to_owned()]);
    assert_eq!(matched_glossary(&prog_doc), want_glossary_matches);
    let p_ctx = (&prog_resp, &prog_doc, Some(&prog_receipt));
    assert_prov(&mut ac, p_ctx, ("turbo:stopwords", "turbo-stopword:bundled"), &want_sw_model).await;
    assert_prov(&mut ac, p_ctx, ("turbo:glossary-terms", "turbo-glossary:annotator"), &want_gloss_model).await;

    let (stream_resp, stream_doc) = run_stream_graph(&mut ac, "base", TEXT, Some(binding)).await;
    assert_eq!(flagged_tokens(&stream_doc), vec!["Quixotl".to_owned(), "build".to_owned()]);
    assert_eq!(matched_glossary(&stream_doc), want_glossary_matches);
    let s_ctx = (&stream_resp, &stream_doc, None);
    assert_prov(&mut ac, s_ctx, ("turbo:glossary-terms", "turbo-glossary:annotator"), &want_gloss_model).await;

    // Identical CSV bytes imported with has_header_row=false and has_header_row=true
    // in one AnalyzeStream batch keep distinct matches and provenance (#2036).
    let same_csv = b"id,term\nz,Zebulon\n";
    let no_hdr = import_with(&mut vc, header("glossary-csv", "en", same_csv), same_csv).await.unwrap();
    let mut hdr_meta = header("glossary-csv", "en", same_csv);
    hdr_meta.has_header_row = true;
    let with_hdr = import_with(&mut vc, hdr_meta, same_csv).await.unwrap();
    assert_eq!(no_hdr.sha256, with_hdr.sha256);
    assert_ne!(no_hdr.artifact_id, with_hdr.artifact_id);
    let doc_req = |id: &str, v: VocabularyBinding| AnalyzeStreamRequest {
        kind: Some(Kind::Document(AnalyzeDocument {
            id: id.into(),
            model: "base".into(),
            text: b"term and Zebulon".to_vec(),
            output: AnalyzeOutput::DocumentGraph as i32,
            vocabulary: Some(v),
            ..Default::default()
        })),
    };
    let batch_hdr = AnalyzeStreamHeader { order: 0, max_inflight: 0, batch_linger_ms: 50 };
    let batch_reqs = vec![
        AnalyzeStreamRequest { kind: Some(Kind::Header(batch_hdr)) },
        doc_req("b1", gloss_bind(&no_hdr.artifact_id, &no_hdr.sha256)),
        doc_req("b2", gloss_bind(&with_hdr.artifact_id, &with_hdr.sha256)),
    ];
    let mut b_stream = ac.analyze_stream(tokio_stream::iter(batch_reqs)).await.unwrap().into_inner();
    let (r1, r2) = (b_stream.next().await.unwrap().unwrap(), b_stream.next().await.unwrap().unwrap());
    let (Some(Outcome::DocumentGraph(dg1)), Some(Outcome::DocumentGraph(dg2))) = (r1.outcome, r2.outcome) else {
        panic!("expected two DocumentGraph outcomes");
    };
    let doc1 = import(&DocumentGraph::from_bytes(&dg1.graph).unwrap(), codecs()).unwrap();
    let doc2 = import(&DocumentGraph::from_bytes(&dg2.graph).unwrap(), codecs()).unwrap();
    assert_eq!(
        matched_glossary(&doc1),
        vec![("id".to_owned(), "term".to_owned()), ("z".to_owned(), "Zebulon".to_owned())]
    );
    assert_eq!(matched_glossary(&doc2), vec![("z".to_owned(), "Zebulon".to_owned())]);
    let l1 = dg1.layers.iter().find(|l| l.layer == "turbo:glossary-terms").unwrap();
    let l2 = dg2.layers.iter().find(|l| l.layer == "turbo:glossary-terms").unwrap();
    assert_eq!(l1.model.as_ref().unwrap().id, no_hdr.artifact_id);
    assert_eq!(l2.model.as_ref().unwrap().id, with_hdr.artifact_id);
    server.stop().await;
}

#[tokio::test]
async fn load_bundle_binds_default_vocabulary_and_request_overrides_it() {
    let store_dir = dir("bind-load-bundle");
    let (server, mut ac, mut vc) = start_server(Some(store_dir), false).await;

    let sw_a = upload(&mut vc, "plain-word-list", "en", b"quixotl\n").await.unwrap();
    let sw_b = upload(&mut vc, "plain-word-list", "en", b"zebulon\n").await.unwrap();
    let gloss_a = upload(&mut vc, "glossary-csv", "en", b"G_ZEP,Zephyr\n").await.unwrap();

    let bundle_vocab = VocabularyBinding {
        stopwords: Some(art_ref(&sw_a.artifact_id, &sw_a.sha256)),
        glossary: Some(art_ref(&gloss_a.artifact_id, &gloss_a.sha256)),
    };
    let load_req = |name: &str, bundle: std::path::PathBuf| LoadBundleRequest {
        name: name.into(),
        bundle: bundle.to_str().unwrap().into(),
        sessions: 1,
        vocabulary: Some(bundle_vocab.clone()),
        ..Default::default()
    };
    let loaded = ac.load_bundle(load_req("bound-model", tiny_bundle_path())).await.unwrap().into_inner();
    assert_eq!(loaded.capabilities.as_ref().unwrap().vocabulary, Some(bundle_vocab.clone()));
    let caps = ac.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(caps.bundles[0].vocabulary, Some(bundle_vocab.clone()));

    // Model-free progressive request does not inherit from an unrelated named bundle (#2036).
    let (_, _, free_doc) = run_progressive(&mut ac, "", TEXT, None).await;
    assert_eq!(flagged_tokens(&free_doc), vec!["and".to_owned()]);
    assert!(free_doc.get::<GlossaryTerms>().is_none());

    let (_, _, named_prog_doc) = run_progressive(&mut ac, "bound-model", TEXT, None).await;
    assert_eq!(flagged_tokens(&named_prog_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(matched_glossary(&named_prog_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);

    let (_, default_doc) = run_stream_graph(&mut ac, "bound-model", TEXT, None).await;
    assert_eq!(flagged_tokens(&default_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(matched_glossary(&default_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);

    let override_vocab =
        VocabularyBinding { stopwords: Some(art_ref(&sw_b.artifact_id, &sw_b.sha256)), glossary: None };
    let (_, over_doc) = run_stream_graph(&mut ac, "bound-model", TEXT, Some(override_vocab)).await;
    assert_eq!(flagged_tokens(&over_doc), vec!["Zebulon".to_owned()]);
    assert_eq!(matched_glossary(&over_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);

    // GLiNER bundles also advertise and apply their bound default vocabulary (#2036).
    let gliner_loaded = ac.load_bundle(load_req("bound-gliner", gliner_fixture::package())).await.unwrap().into_inner();
    assert_eq!(gliner_loaded.capabilities.as_ref().unwrap().vocabulary, Some(bundle_vocab.clone()));
    let caps = ac.capabilities(CapabilitiesRequest {}).await.unwrap().into_inner();
    assert_eq!(caps.bundles.iter().find(|b| b.name == "bound-gliner").unwrap().vocabulary, Some(bundle_vocab.clone()));
    let gliner_req = |vocabulary: Option<VocabularyBinding>| AnalyzePipelineRequest {
        doc_id: "gliner-vocab".into(),
        text: TEXT.into(),
        model: "bound-gliner".into(),
        steps: vec!["turbo-tokenize:uax29-untyped".into(), "turbo-stopword:bundled".into(), "turbo:gliner".into()],
        open_span: Some(OpenSpanSchema { labels: vec!["person".into()], threshold: Some(0.0), ..Default::default() }),
        vocabulary,
        ..Default::default()
    };
    let (_, _, gliner_prof_doc) = run_progressive(&mut ac, "bound-gliner", TEXT, None).await;
    assert_eq!(flagged_tokens(&gliner_prof_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(matched_glossary(&gliner_prof_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);
    let (_, _, gliner_default_doc) = run_progressive_req(&mut ac, gliner_req(None)).await;
    let (_, _, gliner_explicit_doc) = run_progressive_req(&mut ac, gliner_req(Some(bundle_vocab))).await;
    assert_eq!(flagged_tokens(&gliner_default_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(flagged_tokens(&gliner_explicit_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(matched_glossary(&gliner_default_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);
    assert!(!gliner_default_doc.get::<Entities>().unwrap().spans.is_empty());
    let (_, gliner_stream_doc) = run_stream_graph(&mut ac, "bound-gliner", TEXT, None).await;
    assert_eq!(flagged_tokens(&gliner_stream_doc), vec!["Quixotl".to_owned()]);
    assert_eq!(matched_glossary(&gliner_stream_doc), vec![("G_ZEP".to_owned(), "Zephyr".to_owned())]);
    server.stop().await;
}

#[tokio::test]
async fn wrong_digest_unknown_id_wrong_format_and_no_store_are_refused_with_typed_status() {
    let store_dir = dir("bind-refusals");
    let (server, mut ac, mut vc) = start_server(Some(store_dir), true).await;

    let secret_word = "UltraSecretStopwordToken";
    let sw = upload(&mut vc, "plain-word-list", "en", format!("{secret_word}\n").as_bytes()).await.unwrap();
    let map = upload(&mut vc, "headword-definition-tsv", "en", b"k\tv\n").await.unwrap();
    let wrong_sha = format!("{}{}", if sw.sha256.starts_with('0') { "1" } else { "0" }, &sw.sha256[1..]);

    let cases: Vec<(&str, VocabularyBinding, &str)> = vec![
        ("wrong digest", sw_bind(&sw.artifact_id, &wrong_sha), "does not have this sha256"),
        ("bad hex digest", sw_bind(&sw.artifact_id, "not-a-sha256"), "64 lowercase hexadecimal digits"),
        ("unknown artifact", sw_bind("missing-artifact", &sw.sha256), "not registered"),
        ("path traversal id", sw_bind("../etc/passwd", &sw.sha256), "looks like a file path"),
        ("wrong format for stopwords", sw_bind(&map.artifact_id, &map.sha256), "headword-definition-tsv is not taken"),
        ("wrong format for glossary", gloss_bind(&sw.artifact_id, &sw.sha256), "plain-word-list is not taken here"),
    ];

    for (label, vocab, want_msg) in cases {
        let prog_req = AnalyzePipelineRequest {
            doc_id: "bad".into(),
            text: TEXT.into(),
            profile: "default".into(),
            vocabulary: Some(vocab.clone()),
            ..Default::default()
        };
        let prog_err = ac.analyze_document_progressive(prog_req).await.err().unwrap();
        assert_eq!(refusal(&prog_err), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{label}");
        assert!(prog_err.message().contains(want_msg) && !prog_err.message().contains(secret_word), "{label}");

        let stream_reqs = vec![
            AnalyzeStreamRequest { kind: Some(Kind::Header(AnalyzeStreamHeader::default())) },
            AnalyzeStreamRequest {
                kind: Some(Kind::Document(AnalyzeDocument {
                    id: "bad-stream".into(),
                    model: "base".into(),
                    text: TEXT.as_bytes().to_vec(),
                    output: AnalyzeOutput::DocumentGraph as i32,
                    vocabulary: Some(vocab.clone()),
                    ..Default::default()
                })),
            },
        ];
        let mut st = ac.analyze_stream(tokio_stream::iter(stream_reqs)).await.unwrap().into_inner();
        let Some(Outcome::Failure(fail)) = st.next().await.unwrap().unwrap().outcome else { panic!("{label}") };
        assert_eq!(fail.turbo_code, TURBO_E_INVALID_ARGUMENT, "{label}");
        assert!(fail.message.contains(want_msg) && !fail.message.contains(secret_word), "{label}");

        let load_req = LoadBundleRequest {
            name: "bad-bundle".into(),
            bundle: tiny_bundle_path().to_str().unwrap().into(),
            sessions: 1,
            vocabulary: Some(vocab),
            ..Default::default()
        };
        let load_err = ac.load_bundle(load_req).await.expect_err(label);
        assert_eq!(refusal(&load_err), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT), "{label}");
        assert!(load_err.message().contains(want_msg) && !load_err.message().contains(secret_word), "{label}");
    }
    server.stop().await;

    let (no_vocab_server, mut no_vocab_ac, _) = start_server(None, false).await;
    let no_store_req = AnalyzePipelineRequest {
        doc_id: "no-store".into(),
        text: TEXT.into(),
        vocabulary: Some(sw_bind(&sw.artifact_id, &sw.sha256)),
        ..Default::default()
    };
    let err = no_vocab_ac.analyze_document_progressive(no_store_req).await.expect_err("expected refusal");
    assert_eq!(refusal(&err), (Code::InvalidArgument, TURBO_E_INVALID_ARGUMENT));
    assert!(err.message().contains("the server has no registry for it"), "got: {}", err.message());
    no_vocab_server.stop().await;
}
