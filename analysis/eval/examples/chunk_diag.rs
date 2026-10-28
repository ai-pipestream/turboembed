//! Where two chunkers' chunks differ on one set, per judged document:
//! the document's tokens and sentences, each chunker's chunk count and
//! chunk tokens, and for a set with evidence spans, how many of the
//! document's spans each chunker cuts through. One JSON line per
//! (query, relevant document), for joining with two records' per-query
//! nDCG@10 (#172).
//!
//! Usage: chunk_diag --bundle <dir> --cache <dir> --set <name>
//!        --a <chunker spec> --b <chunker spec>
//! where a spec is `turbo:<max>:<target>`, `windows:<n>` or `fixed:<n>`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use serde_json::json;
use turbo_analysis_embed::{Device, Precision};
use turbo_analysis_layers::Sentences;
use turbo_chunker::{Budget, Costs};
use turbo_eval::analysis::{Analysis, Sentencer};
use turbo_eval::chunkers::{Chunker, FixedTokens, SentenceWindows, Signals, TurboChunker};
use turbo_eval::embedder::{Embedder, Role, TurboEmbedder};
use turbo_eval::sets;
use turbo_eval::spans::Span;
use turbo_structure::Syntax;

fn chunker(spec: &str, e: &dyn Embedder) -> Box<dyn Chunker> {
    let p: Vec<&str> = spec.split(':').collect();
    let n = |i: usize| p[i].parse::<u32>().unwrap();
    match p[0] {
        "turbo" => Box::new(TurboChunker {
            budget: Budget { max: n(1), target: n(2), size_weight: 1000, overhead: e.overhead(Role::Document) },
            costs: Costs::default(),
            signals: Signals::Structure,
            cohesion: None,
            clauses: p.get(3) == Some(&"clauses"),
        }),
        "windows" => Box::new(SentenceWindows::new(n(1)).unwrap()),
        "fixed" => Box::new(FixedTokens::new(n(1), 0).unwrap()),
        _ => panic!("{spec}: turbo:<max>:<target>, windows:<n> or fixed:<n>"),
    }
}

/// How many spans a chunking cuts through: a span no chunk holds whole.
fn split(chunks: &[Span], spans: &[Span]) -> usize {
    spans.iter().filter(|s| !chunks.iter().any(|c| c.start <= s.start && s.end <= c.end)).count()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut o = BTreeMap::new();
    for kv in args.chunks(2) {
        o.insert(kv[0].clone(), kv[1].clone());
    }
    let bundle = PathBuf::from(&o["--bundle"]);
    let mut e = TurboEmbedder::open(&bundle, Device::Cpu, Precision::Model).unwrap();
    let set = sets::find(&o["--set"]).unwrap().load(&PathBuf::from(&o["--cache"])).unwrap();
    let analysis = Analysis { sentences: Sentencer::rules("en"), structure: Some(Syntax::Plain) };
    let (a, b) = (chunker(&o["--a"], &e), chunker(&o["--b"], &e));
    let docs: BTreeMap<&str, usize> = set.docs.iter().enumerate().map(|(i, d)| (d.id.as_str(), i)).collect();
    let mut cache: BTreeMap<usize, serde_json::Value> = BTreeMap::new();
    for q in &set.queries {
        for (doc_id, &i) in &docs {
            if set.qrels.grade(&q.id, doc_id) <= 0 {
                continue;
            }
            let d = &set.docs[i];
            let row = cache.entry(i).or_insert_with(|| {
                let doc = analysis.analyze(&d.text, &e).unwrap();
                let sentences = doc.get::<Sentences>().map_or(0, |s| s.spans.len());
                let ca = a.chunk(&doc, &mut e).unwrap();
                let cb = b.chunk(&doc, &mut e).unwrap();
                let toks = |c: &[Span]| -> Vec<u64> {
                    c.iter().map(|s| e.count(&d.text[s.start..s.end], Role::Document)).collect()
                };
                json!({
                    "tokens": e.count(&d.text, Role::Document),
                    "sentences": sentences,
                    "a_tokens": toks(&ca), "b_tokens": toks(&cb),
                    "a_spans": ca.iter().map(|s| [s.start, s.end]).collect::<Vec<_>>(),
                    "b_spans": cb.iter().map(|s| [s.start, s.end]).collect::<Vec<_>>(),
                })
            });
            let mut out = json!({ "query": q.id, "doc": doc_id, "doc_row": row });
            if let Some(sp) = &set.spans {
                let spans = sp.spans(&q.id, doc_id);
                let span = |k: &str| -> Vec<Span> {
                    row[k]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|v| Span { start: v[0].as_u64().unwrap() as usize, end: v[1].as_u64().unwrap() as usize })
                        .collect()
                };
                out["evidence"] = json!(spans.len());
                out["a_split"] = json!(split(&span("a_spans"), spans));
                out["b_split"] = json!(split(&span("b_spans"), spans));
            }
            println!("{out}");
        }
    }
}
