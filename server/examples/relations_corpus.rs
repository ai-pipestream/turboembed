//! Scores the served relation steps on the Re-DocRED test split (#1952).
//!
//! A real server is started on a loopback port with the trained relation
//! head's bundle and a GLiNER2 package, and every test document goes
//! through `AnalyzeDocumentProgressive`: its words as `Tokens`, every gold
//! entity mention as one `Entities` row typed by the release's entity type,
//! sent as the request's initial graph, and one step, either
//! `turbo-tagger:relations` or `turbo:zero-shot-relations`. Each mention
//! relation the step writes is read back as its two mentions' entities
//! (Re-DocRED's `vertexSet` clusters), so a document's prediction is a set
//! of (head entity, tail entity, property) triples, scored as the
//! in-process recipe scores them (`turbo_trainer_recipes::redocred_eval`):
//! a triple counts when its entities, direction and property are gold's.
//!
//! ```sh
//! cargo run --release -p turbo-kserve --example relations_corpus -- \
//!     --corpus Re-DocRED --head-bundle relation-bundle --gliner2 gliner2-package \
//!     --scope document --out receipt.json
//! ```
//!
//! With `--glirel PACKAGE --glirel-spm SPM`, GLiREL is scored too, in
//! process: no serving step runs it ([`glirel`]).
//!
//! GLiNER2 reads label names, not property ids, and takes a bounded label
//! set, so it is given the ten properties commonest in training under
//! their Wikidata English labels ([`SUBSET`]); both systems are also scored
//! on that subset alone, gold and predictions restricted to it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use serde_json::{Value, json};
use tokio_stream::StreamExt as _;
use turbo_analysis::export::{DocumentGraph, export, import};
use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_kserve::config::{DEFAULT_MAX_MESSAGE_BYTES as MAX, Device, ModelConfig};
use turbo_kserve::proto::result::analysis_service_client::AnalysisServiceClient;
use turbo_kserve::proto::result::analyze_progressive_event::Event;
use turbo_kserve::proto::result::*;
use turbo_kserve::results::document::codecs;
use turbo_kserve::{Serve, Server};
use turbo_relation::Relations;

#[path = "relations_corpus/glirel.rs"]
mod glirel;

type Fail = Box<dyn std::error::Error>;
type Triples = BTreeSet<(u32, u32, String)>;
type Client = AnalysisServiceClient<tonic::transport::Channel>;

/// The release's entity types, as the head was trained on them.
const TYPES: [&str; 6] = ["LOC", "MISC", "NUM", "ORG", "PER", "TIME"];
/// The ten properties commonest in the training split, with their Wikidata
/// English labels, as GLiNER2 is given them.
const SUBSET: [(&str, &str); 10] = [
    ("P131", "located in the administrative territorial entity"),
    ("P17", "country"),
    ("P27", "country of citizenship"),
    ("P150", "contains the administrative territorial entity"),
    ("P800", "notable work"),
    ("P527", "has part(s)"),
    ("P361", "part of"),
    ("P175", "performer"),
    ("P577", "publication date"),
    ("P463", "member of"),
];
const DRAWS: usize = 1000;

struct Doc {
    title: String,
    text: String,
    words: Vec<(u32, u32)>,
    /// Each sentence's first word and word count.
    sentences: Vec<(usize, usize)>,
    /// Each mention: its first word, one past its last, its entity and type.
    mentions: Vec<(usize, usize, u32, usize)>,
    gold: Triples,
}

fn read(path: &Path) -> Result<(Vec<Doc>, usize), Fail> {
    let raw: Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let (mut docs, mut dropped) = (Vec::new(), 0usize);
    for d in raw.as_array().ok_or("not an array")? {
        let (mut text, mut words, mut sentences, mut starts) = (String::new(), Vec::new(), Vec::new(), Vec::new());
        for s in d["sents"].as_array().ok_or("sents")? {
            starts.push(words.len());
            let first = words.len();
            for w in s.as_array().ok_or("a sentence")? {
                if !text.is_empty() {
                    text.push(' ');
                }
                let a = text.len() as u32;
                text.push_str(w.as_str().ok_or("a word")?);
                words.push((a, text.len() as u32));
            }
            sentences.push((first, words.len() - first));
        }
        let mut seen = BTreeSet::new();
        let mut mentions = Vec::new();
        for (e, cluster) in d["vertexSet"].as_array().ok_or("vertexSet")?.iter().enumerate() {
            for m in cluster.as_array().ok_or("an entity")? {
                let sent = m["sent_id"].as_u64().ok_or("sent_id")? as usize;
                let pos = m["pos"].as_array().ok_or("pos")?;
                let (a, b) = (pos[0].as_u64().ok_or("pos")? as usize, pos[1].as_u64().ok_or("pos")? as usize);
                let kind = TYPES.iter().position(|t| m["type"] == *t).ok_or("an entity type")?;
                let (a, b) = (starts[sent] + a, starts[sent] + b);
                // One row a span: a span two entities share keeps its first.
                if a < b && seen.insert((a, b)) {
                    mentions.push((a, b, e as u32, kind));
                } else {
                    dropped += 1;
                }
            }
        }
        mentions.sort_unstable();
        let gold = d["labels"]
            .as_array()
            .ok_or("labels")?
            .iter()
            .map(|l| Ok((l["h"].as_u64().ok_or("h")? as u32, l["t"].as_u64().ok_or("t")? as u32, str_of(&l["r"])?)))
            .collect::<Result<Triples, Fail>>()?;
        docs.push(Doc { title: str_of(&d["title"])?, text, words, sentences, mentions, gold });
    }
    Ok((docs, dropped))
}

fn str_of(v: &Value) -> Result<String, Fail> {
    Ok(v.as_str().ok_or("a string")?.to_owned())
}

/// The request's initial graph: the source, its tokens, its sentences
/// when `sentences`, and one entity row a mention.
fn graph(d: &Doc, sentences: bool) -> Result<Vec<u8>, Fail> {
    Ok(export(&doc_of(d, sentences)?, codecs())?.to_bytes()?)
}

/// The initial graph as a document.
fn doc_of(d: &Doc, sentences: bool) -> Result<Document, Fail> {
    let mut doc = Document::new(d.text.as_str());
    let mut tokens = Spans::new(ViewId::SOURCE);
    for &(a, b) in &d.words {
        tokens.push(Span::new(a, b));
    }
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: None }, Provenance::by("re-docred"))?;
    if sentences {
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(first, n) in &d.sentences {
            spans.push(Span::new(d.words[first].0, d.words[first + n - 1].1));
        }
        doc.add(&Key::DEFAULT, Sentences { spans }, Provenance::by("re-docred"))?;
    }
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _, _) in &d.mentions {
        spans.push(Span::new(d.words[a].0, d.words[b - 1].1));
    }
    let n = d.mentions.len();
    let entities = Entities {
        spans,
        tokens: d.mentions.iter().map(|&(a, b, _, _)| IxRange::new(a as u32, b as u32)).collect(),
        kind: d.mentions.iter().map(|m| Tag(m.3 as u16)).collect(),
        kinds: Arc::clone(&TagSet::new("ner", &TYPES)),
        score: vec![1.0; n],
    };
    doc.add(&Key::DEFAULT, entities, Provenance::by("re-docred"))?;
    Ok(doc)
}

async fn analyze(client: &mut Client, req: AnalyzePipelineRequest) -> Result<Document, Fail> {
    let mut stream = client.analyze_document_progressive(req).await?.into_inner();
    let (mut complete, mut failed) = (None, None);
    while let Some(ev) = stream.next().await {
        match ev?.event {
            Some(Event::Complete(c)) => complete = Some(c),
            Some(Event::StepFailed(f)) => failed = failed.or(f.failure),
            _ => {}
        }
    }
    // A refused step is reported by its code and message, which name the
    // limit and both counts and never the text.
    if let Some(f) = failed {
        return Err(format!("code {}: {}", f.turbo_code, f.message).into());
    }
    let c = complete.ok_or("no Complete event")?;
    if let Some(bad) = c.layers.iter().find(|l| l.status != LayerWireStatus::Ok as i32) {
        return Err(format!("layer {} failed: {:?}", bad.layer, bad.error).into());
    }
    Ok(import(&DocumentGraph::from_bytes(&c.graph)?, codecs())?)
}

/// A step's relations as entity triples, property ids for labels.
fn triples(d: &Doc, doc: &Document, names: &BTreeMap<String, String>) -> Result<Triples, Fail> {
    let Some(r) = doc.get::<Relations>() else { return Ok(Triples::new()) };
    let mut out = Triples::new();
    for row in 0..r.spans.len() {
        let (s, o, kind, _) = r.mention(Ix::new(row as u32));
        let (h, t) = (d.mentions[s.get()].2, d.mentions[o.get()].2);
        let p = names.get(kind).cloned().unwrap_or_else(|| kind.to_owned());
        if h != t {
            out.insert((h, t, p));
        }
    }
    Ok(out)
}

fn request(
    d: &Doc,
    i: usize,
    sentences: bool,
    zero_shot: bool,
    threshold: f32,
) -> Result<AnalyzePipelineRequest, Fail> {
    let mut req = AnalyzePipelineRequest {
        doc_id: format!("re-docred-test-{i}"),
        text: d.text.clone(),
        offsets: OffsetMode::Utf8 as i32,
        initial_graph: graph(d, sentences)?,
        ..Default::default()
    };
    if zero_shot {
        req.steps = vec!["turbo:zero-shot-relations".into()];
        req.relation_model = "gliner2".into();
        req.relations = Some(OpenRelationSchema {
            labels: SUBSET
                .iter()
                .map(|(_, name)| OpenRelationLabel { name: (*name).into(), ..Default::default() })
                .collect(),
            threshold: Some(threshold),
            within_sentence: sentences,
        });
    } else {
        req.steps = vec!["turbo-tagger:relations".into()];
        req.model = "head".into();
    }
    Ok(req)
}

#[derive(Clone, Copy, Default)]
struct Tally {
    right: usize,
    found: usize,
    gold: usize,
}

impl Tally {
    fn add(&mut self, o: Tally) {
        (self.right, self.found, self.gold) = (self.right + o.right, self.found + o.found, self.gold + o.gold);
    }
    fn f1(self) -> f64 {
        if self.found + self.gold == 0 { 0.0 } else { 2.0 * self.right as f64 / (self.found + self.gold) as f64 }
    }
    fn json(self) -> Value {
        let ratio = |n: usize, d: usize| if d == 0 { 0.0 } else { n as f64 / d as f64 };
        json!({ "gold": self.gold, "found": self.found, "right": self.right,
            "precision": round(ratio(self.right, self.found)), "recall": round(ratio(self.right, self.gold)),
            "f1": round(self.f1()) })
    }
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

fn tallies(docs: &[Doc], pred: &[Triples], keep: impl Fn(&str) -> bool) -> Vec<Tally> {
    docs.iter()
        .zip(pred)
        .map(|(d, p)| {
            let g: Triples = d.gold.iter().filter(|x| keep(&x.2)).cloned().collect();
            let p: Triples = p.iter().filter(|x| keep(&x.2)).cloned().collect();
            Tally { right: g.intersection(&p).count(), found: p.len(), gold: g.len() }
        })
        .collect()
}

fn sum(t: &[Tally], draw: impl Iterator<Item = usize>) -> Tally {
    let mut s = Tally::default();
    draw.for_each(|i| s.add(t[i]));
    s
}

/// `other`'s F1 minus `base`'s over paired whole-document resamples
/// (SplitMix64 from seed 1952), 2.5th to 97.5th percentile.
fn interval(base: &[Tally], other: &[Tally]) -> Value {
    let (n, mut state) = (base.len(), 1952u64);
    let mut next = || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    };
    let mut d: Vec<f64> = (0..DRAWS)
        .map(|_| {
            let draw: Vec<usize> = (0..n).map(|_| next()).collect();
            sum(other, draw.iter().copied()).f1() - sum(base, draw.iter().copied()).f1()
        })
        .collect();
    d.sort_by(f64::total_cmp);
    let delta = sum(other, 0..n).f1() - sum(base, 0..n).f1();
    json!({ "delta_f1": round(delta), "low": round(d[DRAWS / 40]), "high": round(d[DRAWS - 1 - DRAWS / 40]) })
}

fn sha256(path: &Path) -> Result<String, Fail> {
    Ok(turbo_sha256::sha256(&std::fs::read(path)?).to_hex())
}

#[tokio::main]
async fn main() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |n: &str| args.iter().position(|a| a == n).and_then(|i| args.get(i + 1)).cloned();
    let corpus = PathBuf::from(flag("--corpus").ok_or("--corpus DIR")?);
    let head = PathBuf::from(flag("--head-bundle").ok_or("--head-bundle DIR")?);
    let gliner2 = PathBuf::from(flag("--gliner2").ok_or("--gliner2 DIR")?);
    let out = PathBuf::from(flag("--out").ok_or("--out FILE")?);
    let sentences = match flag("--scope").as_deref().unwrap_or("document") {
        "document" => false,
        "sentence" => true,
        s => return Err(format!("--scope {s}: document or sentence").into()),
    };
    let threshold: f32 = flag("--threshold").map(|t| t.parse()).transpose()?.unwrap_or(0.5);
    let limit: Option<usize> = flag("--limit").map(|t| t.parse()).transpose()?;
    let test = corpus.join("data/test_revised.json");
    let (mut docs, dropped) = read(&test)?;
    docs.truncate(limit.unwrap_or(docs.len()));

    let rt = turbo_kserve::api::Runtime::create().map_err(|e| format!("{e:?}"))?;
    let cpu = (0..16)
        .find(|&i| rt.device_info(i).map(|d| d.kind == turbo::TURBO_DEVICE_CPU).unwrap_or(false))
        .ok_or("no CPU device")?;
    let name = rt.device_info(cpu).map_err(|e| format!("{e:?}"))?.name;
    let device: String = name.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect();
    let head_model = ModelConfig {
        bundle: head.to_string_lossy().into_owned(),
        name: Some("head".into()),
        device: Device::Index(cpu),
        precision: 0,
        max_batch: 0,
        max_seq: 0,
        sessions: 1,
    };
    let server =
        Server::start("127.0.0.1:0".parse()?, vec![head_model], MAX, Serve { admin: true, ..Serve::default() }).await?;
    server.load().await?;
    let mut client = AnalysisServiceClient::connect(format!("http://{}", server.local_addr()))
        .await?
        .max_decoding_message_size(MAX)
        .max_encoding_message_size(MAX);
    client
        .load_bundle(LoadBundleRequest {
            name: "gliner2".into(),
            bundle: gliner2.to_string_lossy().into_owned(),
            device: cpu,
            sessions: 1,
            ..Default::default()
        })
        .await?;
    let caps = client.capabilities(CapabilitiesRequest {}).await?.into_inner();
    let models: Vec<Value> = caps
        .bundles
        .iter()
        .map(|b| json!({ "name": b.name, "task": b.task, "sha256": b.model.as_ref().map(|m| m.sha256.clone()) }))
        .collect();

    let names: BTreeMap<String, String> = SUBSET.iter().map(|(p, n)| ((*n).to_owned(), (*p).to_owned())).collect();
    let mut runs = Vec::new();
    for (system, zero_shot) in [("turbo-tagger:relations", false), ("turbo:zero-shot-relations", true)] {
        let started = Instant::now();
        let (mut pred, mut refused) = (Vec::with_capacity(docs.len()), Vec::new());
        for (i, d) in docs.iter().enumerate() {
            // A document the step refuses is scored as finding nothing, and
            // listed with what the server said.
            match analyze(&mut client, request(d, i, sentences, zero_shot, threshold)?).await {
                Ok(doc) => pred.push(triples(d, &doc, &names)?),
                Err(e) => {
                    refused.push(json!({ "document": i, "title": d.title, "error": e.to_string() }));
                    pred.push(Triples::new());
                }
            }
            if (i + 1) % 100 == 0 {
                eprintln!("{system}: {}/{} documents, {:.0} s", i + 1, docs.len(), started.elapsed().as_secs_f64());
            }
        }
        runs.push((system, pred, refused, started.elapsed().as_secs_f64()));
    }
    server.stop().await;
    let mut models = models;
    if let Some(package) = flag("--glirel") {
        let spm = PathBuf::from(flag("--glirel-spm").ok_or("--glirel-spm FILE")?);
        let (analyzer, digest) = glirel::analyzer(Path::new(&package), &spm, sentences, threshold)?;
        models.push(json!({ "name": "glirel", "task": "in process", "sha256": digest }));
        let started = Instant::now();
        let (mut pred, mut refused) = (Vec::with_capacity(docs.len()), Vec::new());
        for (i, d) in docs.iter().enumerate() {
            match glirel::run(&analyzer, doc_of(d, sentences)?) {
                Ok(doc) => pred.push(triples(d, &doc, &names)?),
                Err(e) => {
                    refused.push(json!({ "document": i, "title": d.title, "error": e.to_string() }));
                    pred.push(Triples::new());
                }
            }
        }
        runs.push(("glirel (in process)", pred, refused, started.elapsed().as_secs_f64()));
    }

    let in_subset = |p: &str| SUBSET.iter().any(|(id, _)| *id == p);
    let base = tallies(&docs, &runs[0].1, in_subset);
    let mut systems = Vec::new();
    for (system, pred, refused, seconds) in &runs {
        let all = tallies(&docs, pred, |_| true);
        let subset = tallies(&docs, pred, in_subset);
        let mut per_label = serde_json::Map::new();
        for (id, name) in SUBSET {
            let t = tallies(&docs, pred, |p| p == id);
            let mut row = sum(&t, 0..t.len()).json();
            row["label"] = json!(name);
            per_label.insert(id.into(), row);
        }
        let digest = turbo_sha256::sha256(serde_json::to_string(&pred_rows(pred))?.as_bytes()).to_hex();
        systems.push(json!({
            "step": system, "seconds": round(*seconds), "predictions_sha256": digest,
            "documents_refused": refused.len(), "refused": refused,
            "all_labels": sum(&all, 0..all.len()).json(),
            "subset": sum(&subset, 0..subset.len()).json(),
            "subset_vs_head": interval(&base, &subset),
            "subset_per_label": per_label,
        }));
    }
    let receipt = json!({
        "issue": "#1952",
        "corpus": { "file": "data/test_revised.json", "sha256": sha256(&test)?, "documents": docs.len(),
            "gold_triples": docs.iter().map(|d| d.gold.len()).sum::<usize>(),
            "mentions": docs.iter().map(|d| d.mentions.len()).sum::<usize>(),
            "mentions_dropped_shared_or_empty_span": dropped },
        "scope": if sentences { "sentence" } else { "document" },
        "zero_shot_threshold": threshold,
        "subset": SUBSET.iter().map(|(p, n)| json!([p, n])).collect::<Vec<_>>(),
        "device": device,
        "served_models": models,
        "bootstrap": { "unit": "document", "draws": DRAWS, "seed": 1952 },
        "systems": systems,
    });
    std::fs::write(&out, serde_json::to_string_pretty(&receipt)? + "\n")?;
    println!("{}", serde_json::to_string_pretty(&receipt["systems"])?);
    Ok(())
}

/// The predictions in a stable form for their digest.
fn pred_rows(pred: &[Triples]) -> Vec<Vec<(u32, u32, String)>> {
    pred.iter().map(|p| p.iter().cloned().collect()).collect()
}
