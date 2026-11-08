//! Redaction policy dictionaries served from the vocabulary store (#1882
//! part C), end to end over gRPC: a policy that names its terms and its
//! map table by `PiiDictionaryRef` redacts exactly as the same entries
//! inline, a named policy resolves its references at start and a bad one
//! fails start, and a refused reference names the reference, never the
//! dictionary's content.

#[path = "pii_policy/common.rs"]
mod common;
mod vocab;

use common::*;
use pb::pii_rewrite_strategy::Strategy;
use prost::Message;
use tokio_stream::StreamExt;
use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{Document, Key};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::pii_policy::PolicyFiles;
use turbo_kserve::proto::result as pb;
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_pii::PiiEdits;
use turbo_pii::rewrite::REDACTED;
use vocab::{dir, import as upload};

const NAME: &str = "Zebulon";
const EMAIL: &str = "jane@example.com";
const TEXT: &str = "Ask Zebulon or mail jane@example.com today.";
const WORDS: &str = "Zebulon\nQuixotl\n";
const MAP: &str = "jane@example.com\tJ. Doe\n";

fn model() -> ModelConfig {
    let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../testdata/tiny-bert-bundle");
    ModelConfig {
        bundle: bundle.to_str().unwrap().into(),
        name: None,
        device: Device::Index(0),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    }
}

async fn start(serve: Serve) -> Result<Server, String> {
    Server::start("127.0.0.1:0".parse().unwrap(), vec![model()], MAX, serve).await
}

fn dictionary(id: &str, sha256: &str) -> pb::PiiDictionaryRef {
    pb::PiiDictionaryRef { artifact_id: id.into(), sha256: sha256.into() }
}

/// Codenames from `terms` are a custom kind masked whole; emails take the
/// map `table`.
fn redaction(id: &str, terms: pb::pii_term_source::Source, table: pb::pii_map_strategy::Table) -> pb::RedactionPolicy {
    let map = Strategy::Map(Box::new(pb::PiiMapStrategy {
        table: Some(table),
        case_sensitive: false,
        target: pb::PiiMapTarget::NormalizedOrSurface as i32,
        fallback: None,
    }));
    let mut p = policy(vec![rule("email", builtin(pb::PiiKindId::Email), vec![entry(map)])]);
    p.id = id.into();
    p.custom_kinds = vec!["codename".into()];
    p.deny = vec![pb::PiiDenyList {
        rule_id: "codenames".into(),
        kind: custom(0),
        terms: Some(pb::PiiTermSource { source: Some(terms) }),
        ..Default::default()
    }];
    p.default_strategy = Some(pb::PiiDefaultStrategy {
        strategy: Some(pb::pii_default_strategy::Strategy::Mask(pb::PiiMaskStrategy::default())),
    });
    p
}

fn inline() -> pb::RedactionPolicy {
    let terms = pb::pii_term_source::Source::Terms(pb::PiiTerms { terms: vec!["Zebulon".into(), "Quixotl".into()] });
    let table = pb::pii_map_strategy::Table::Entries(pb::PiiMapEntries {
        entries: [(EMAIL.to_owned(), "J. Doe".to_owned())].into_iter().collect(),
    });
    redaction("inline", terms, table)
}

/// The two stored dictionaries, by the ids and digests the store gave.
#[derive(Clone)]
struct Dicts {
    words: pb::PiiDictionaryRef,
    map: pb::PiiDictionaryRef,
}

fn by_ref(id: &str, d: &Dicts) -> pb::RedactionPolicy {
    let terms = pb::pii_term_source::Source::Dictionary(d.words.clone());
    redaction(id, terms, pb::pii_map_strategy::Table::Dictionary(d.map.clone()))
}

/// A store holding the word list and the map, written through a served
/// ImportDictionary and reopened by each later server on its directory.
async fn seeded(name: &str) -> (std::path::PathBuf, Dicts) {
    let d = dir(name);
    let s = start(Serve { admin: true, vocabulary: Some(d.clone()), ..Serve::default() }).await.unwrap();
    let mut c = vocab::Client::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let w = upload(&mut c, "plain-word-list", "", WORDS.as_bytes()).await.unwrap();
    let m = upload(&mut c, "headword-definition-tsv", "", MAP.as_bytes()).await.unwrap();
    s.stop().await;
    (d, Dicts { words: dictionary(&w.artifact_id, &w.sha256), map: dictionary(&m.artifact_id, &m.sha256) })
}

fn request(sel: pb::pii_redaction_selection::Policy) -> pb::AnalyzePipelineRequest {
    pb::AnalyzePipelineRequest {
        doc_id: "dict".into(),
        text: TEXT.into(),
        offsets: pb::OffsetMode::Utf8 as i32,
        steps: vec!["turbo-pii:pii".into(), "turbo-pii:redact".into()],
        pii_output: Some(pb::PiiOutputOptions { full_text: true, rewritten_text: true, ..Default::default() }),
        pii_redaction: Some(pb::PiiRedactionSelection { policy: Some(sel), dry_run: false }),
        ..Default::default()
    }
}

/// The redacted view's text and each edit's span, rule id and strategy.
async fn redact(s: &Server, sel: pb::pii_redaction_selection::Policy) -> Result<(String, Vec<String>), tonic::Status> {
    let mut c = AnalysisServiceClient::connect(format!("http://{}", s.local_addr())).await.unwrap();
    let mut stream = c.analyze_document_progressive(request(sel)).await?.into_inner();
    let mut graph = None;
    while let Some(ev) = stream.next().await {
        if let Some(Event::Complete(done)) = ev?.event {
            graph = Some(done.graph);
        }
    }
    let doc: Document = import(&DocumentGraph::from_bytes(&graph.expect("complete")).unwrap(), codecs()).unwrap();
    let view = doc.get_key(&REDACTED).expect("the redacted view").text().to_owned();
    let edits = doc.get_key(&Key::<PiiEdits>::named(REDACTED.name)).expect("its edits");
    let rows = edits.edits().map(|e| format!("{:?} {} {:?}", e.span, e.rule_id, e.strategy)).collect();
    Ok((view, rows))
}

#[tokio::test]
async fn a_dictionary_policy_redacts_as_the_same_entries_inline() {
    let (d, dicts) = seeded("same").await;
    let s = start(Serve { vocabulary: Some(d), ..Serve::default() }).await.unwrap();
    let a = redact(&s, pb::pii_redaction_selection::Policy::Inline(inline())).await.unwrap();
    let b = redact(&s, pb::pii_redaction_selection::Policy::Inline(by_ref("p", &dicts))).await.unwrap();
    assert_eq!(a, b);
    assert!(!a.0.contains(NAME) && !a.0.contains(EMAIL) && a.0.contains("J. Doe"), "{}", a.0);
    // The codename from the word list is masked by the default; the email
    // is mapped by the email rule.
    assert_eq!(a.1.len(), 2, "{:?}", a.1);
    assert!(a.1[0].ends_with("  Mask") && a.1[1].ends_with(" email Map"), "{:?}", a.1);
    s.stop().await;
}

#[tokio::test]
async fn a_named_policy_resolves_its_dictionaries_at_start_and_a_bad_ref_fails_start() {
    let (d, dicts) = seeded("named").await;
    let files = PolicyFiles(vec![by_ref("named", &dicts).encode_to_vec()]);
    let s = start(Serve { vocabulary: Some(d.clone()), pii_policies: files, ..Serve::default() }).await.unwrap();
    let named = redact(&s, pb::pii_redaction_selection::Policy::Name("named".into())).await.unwrap();
    let inline = redact(&s, pb::pii_redaction_selection::Policy::Inline(inline())).await.unwrap();
    assert_eq!(named, inline);
    s.stop().await;
    let mut mismatch = dicts.clone();
    mismatch.words.sha256 = turbo_maxent::sha256(b"other").to_hex();
    let cases = [
        (by_ref("bad", &mismatch), Some(d.clone()), &dicts.words.artifact_id, "does not have this sha256"),
        (by_ref("bad", &dicts), None, &dicts.map.artifact_id, "no registry"),
    ];
    for (bad, vocabulary, id, why) in cases {
        let files = PolicyFiles(vec![bad.encode_to_vec()]);
        let e = start(Serve { vocabulary, pii_policies: files, ..Serve::default() }).await.err().expect("refused");
        assert!(e.contains("pii policy 0") && e.contains(id.as_str()) && e.contains(why), "{e}");
        assert!(!e.contains(NAME) && !e.contains("Quixotl"), "{e}");
    }
}

#[tokio::test]
async fn an_inline_bad_reference_is_refused_naming_the_ref_only() {
    let (d, dicts) = seeded("refused").await;
    let s = start(Serve { vocabulary: Some(d), ..Serve::default() }).await.unwrap();
    let other = turbo_maxent::sha256(b"other").to_hex();
    let (mut words, mut map) = (dicts.clone(), dicts.clone());
    words.words.sha256 = other.clone();
    map.map.sha256 = other;
    // The map table named as a term source: the wrong format.
    let wrong_format = redaction(
        "p",
        pb::pii_term_source::Source::Dictionary(dicts.map.clone()),
        pb::pii_map_strategy::Table::Dictionary(dicts.map.clone()),
    );
    let cases = [
        (by_ref("p", &words), &dicts.words.artifact_id, "does not have this sha256"),
        (by_ref("p", &map), &dicts.map.artifact_id, "does not have this sha256"),
        (wrong_format, &dicts.map.artifact_id, "headword-definition-tsv is not taken here"),
    ];
    for (p, id, why) in cases {
        let e = redact(&s, pb::pii_redaction_selection::Policy::Inline(p)).await.unwrap_err();
        assert_eq!(e.code(), tonic::Code::InvalidArgument, "{e:?}");
        assert!(e.message().contains(id.as_str()) && e.message().contains(why), "{}", e.message());
        assert!(!e.message().contains(NAME) && !e.message().contains("J. Doe"), "{}", e.message());
    }
    let mut unknown = dicts.clone();
    unknown.words.artifact_id = "dict-none".into();
    let unknown = by_ref("p", &unknown);
    let e = redact(&s, pb::pii_redaction_selection::Policy::Inline(unknown)).await.unwrap_err();
    assert!(e.message().contains("dict-none") && e.message().contains("not registered"), "{}", e.message());
    s.stop().await;
}
