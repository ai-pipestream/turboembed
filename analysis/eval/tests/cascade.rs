//! The static-to-contextual cascade (#712) over cached BEIR sets:
//! static-only, contextual-only, and `turbo_chunker::Cascade` at
//! several shortlist sizes, scored as document retrieval (a
//! document's score is its best unit's score; every arm uses the
//! same pool of all `Levels` units) with recall@k and nDCG@10.
//! Latency columns time `Cascade::search` only (after
//! `Cascade::new`), summed over one cascade per document, so they
//! are not the single-index latency evidence (the `turbo-chunker`
//! fixture is); the latency-relevant column is contextual units
//! scored per query. Encoders are deterministic synthetic
//! mechanism encoders, so records measure the cascade mechanism
//! over real texts and qrels, not trained model quality. Receipt:
//!
//! TURBO_EVAL_CACHE=<BEIR_CACHE> cargo test --locked --release \
//!   -p turbo-eval --test cascade cascade_benchmark_receipt -- --ignored --nocapture
use std::marker::PhantomData;
use std::path::PathBuf;
use std::time::Instant;

use turbo_analysis::*;
use turbo_analysis_layers::{Levels, Sentences, Subwords, Tokens, Vectors};
use turbo_chunker::levels::LevelBuilder;
use turbo_chunker::terms::{LocalCollection, TermCounter, TermIndexer, TermStats, Vocabulary};
use turbo_chunker::{
    Budget, Cascade, CascadeConfig, Chunker, Costs, LexicalSpace, QueryVector, StructureBoundaries, VectorSpace,
};
use turbo_eval::metrics::{Hit, ndcg, recall};
use turbo_eval::read::beir;
use turbo_eval::set::LoadedSet;
use turbo_structure::{StructureAnnotator, Syntax};

const DIM: usize = 256;
const DIM_C: usize = 768;

fn model(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "synthetic-1".into(), sha256: "0".into() }
}

fn fnv(token: &str) -> u64 {
    let mut h = 0xcbf29ce484222325_u64;
    for b in token.as_bytes() {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn words(text: &str) -> Vec<(Span, String)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, c) in text.char_indices().chain(std::iter::once((text.len(), ' '))) {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push((Span::new(s as u32, i as u32), text[s..i].to_lowercase()));
        }
    }
    out
}

fn token_vec(token: &str) -> [f32; DIM] {
    let h = fnv(token);
    let mut v = [0.0f32; DIM];
    v[(h % DIM as u64) as usize] += if h >> 63 == 0 { 1.0 } else { -1.0 };
    v[((h >> 8) % DIM as u64) as usize] += if (h >> 62) & 1 == 0 { 0.5 } else { -0.5 };
    v
}

/// Wider contextual token vector: three salted hash blocks.
fn token_vec_wide(token: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM_C];
    for (block, salt) in ["", "\u{1}", "\u{2}"].into_iter().enumerate() {
        let h = fnv(&format!("{salt}{token}"));
        let base = block * DIM;
        v[base + (h % DIM as u64) as usize] += if h >> 63 == 0 { 1.0 } else { -1.0 };
        v[base + ((h >> 8) % DIM as u64) as usize] += if (h >> 62) & 1 == 0 { 0.5 } else { -0.5 };
        v[base + ((h >> 16) % DIM as u64) as usize] += if (h >> 61) & 1 == 0 { 0.25 } else { -0.25 };
    }
    v
}

fn static_vec(tokens: &[String]) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM];
    for t in tokens {
        let tv = token_vec(t);
        for (a, b) in v.iter_mut().zip(tv) {
            *a += b.abs();
        }
    }
    v
}

fn contextual_tokens(tokens: &[String]) -> Vec<Vec<f32>> {
    let base: Vec<Vec<f32>> = tokens.iter().map(|t| token_vec_wide(t)).collect();
    (0..base.len())
        .map(|i| {
            let mut v = vec![0.0f32; DIM_C];
            for (j, w) in [(i.wrapping_sub(1), 0.25f32), (i, 0.5), (i + 1, 0.25)] {
                if let Some(b) = base.get(j) {
                    for (a, x) in v.iter_mut().zip(b) {
                        *a += w * x;
                    }
                }
            }
            v
        })
        .collect()
}

fn mean_vec(rows: &[Vec<f32>]) -> Vec<f32> {
    let mut v = vec![0.0f32; DIM_C];
    for r in rows {
        for (a, b) in v.iter_mut().zip(r) {
            *a += b;
        }
    }
    v
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x).powi(2);
        nb += f64::from(*y).powi(2);
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    ((dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0) + 1.0) / 2.0
}

fn sentence_spans(text: &str) -> Vec<Span> {
    let mut out = Vec::new();
    let mut start = 0usize;
    let bytes = text.as_bytes();
    for i in 0..bytes.len() {
        if matches!(bytes[i], b'.' | b'!' | b'?') && (i + 1 == bytes.len() || bytes[i + 1].is_ascii_whitespace()) {
            out.push(Span::new(start as u32, (i + 1) as u32));
            start = i + 1;
            while start < bytes.len() && bytes[start].is_ascii_whitespace() {
                start += 1;
            }
        }
    }
    if start < text.len() {
        out.push(Span::new(start as u32, text.len() as u32));
    }
    if out.is_empty() {
        out.push(Span::new(0, text.len() as u32));
    }
    out
}

struct Prepared {
    id: String,
    doc: Document,
    statics: Vectors<Levels>,
    contextuals: Vectors<Levels>,
    units: Vec<Ix<Levels>>,
}

fn prepare(id: &str, text: &str) -> Prepared {
    let words = words(text);
    let mut doc = Document::new(text);
    let mut s_spans = Spans::new(ViewId::SOURCE);
    for s in sentence_spans(text) {
        s_spans.push(s);
    }
    let mut w_spans = Spans::new(ViewId::SOURCE);
    let mut pieces = Spans::new(ViewId::SOURCE);
    for (s, _) in &words {
        w_spans.push(*s);
        pieces.push(*s);
    }
    let n = words.len();
    doc.add(&Key::DEFAULT, Sentences { spans: s_spans }, Provenance::by("study")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords { spans: pieces, ids: vec![0; n], model: model("study/static"), token: None },
        Provenance::by("study"),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: w_spans, sentence: None }, Provenance::by("study")).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .add(StructureAnnotator::new(Syntax::Markdown))
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 40, target: 40, size_weight: 1000, overhead: 0 } })
        .add(LevelBuilder)
        .add(TermIndexer)
        .add(TermCounter::<Levels>::new())
        .build()
        .unwrap();
    let mut docs = vec![std::mem::replace(&mut doc, Document::new(""))];
    assert!(analyzer.run(&mut docs)[0].is_ok());
    let doc = docs.pop().unwrap();
    let levels = doc.get::<Levels>().unwrap();
    let ctx = contextual_tokens(&words.iter().map(|(_, w)| w.clone()).collect::<Vec<_>>());
    let unit_vecs = |contextual: bool| -> Vec<f32> {
        let mut data = Vec::new();
        for i in 0..levels.len() as u32 {
            let unit = Ix::new(i);
            let toks: Vec<usize> = words
                .iter()
                .enumerate()
                .filter(|(_, (s, _))| levels.spans.fragments(unit).any(|f| f.start <= s.start && s.end <= f.end))
                .map(|(j, _)| j)
                .collect();
            if contextual {
                data.extend(mean_vec(&toks.iter().map(|&j| ctx[j].clone()).collect::<Vec<_>>()));
            } else {
                data.extend(static_vec(&toks.iter().map(|&j| words[j].1.clone()).collect::<Vec<_>>()));
            }
        }
        data
    };
    let statics = Vectors {
        dim: DIM as u32,
        data: unit_vecs(false),
        empty: vec![],
        model: model("study/static"),
        _of: PhantomData,
    };
    let contextuals = Vectors {
        dim: DIM_C as u32,
        data: unit_vecs(true),
        empty: vec![],
        model: model("study/contextual"),
        _of: PhantomData,
    };
    let units: Vec<Ix<Levels>> = (0..levels.len() as u32).map(Ix::new).collect();
    Prepared { id: id.into(), doc, statics, contextuals, units }
}

fn subset(set: &mut LoadedSet, max_docs: usize, max_queries: usize) {
    set.queries.sort_by(|a, b| a.id.cmp(&b.id));
    set.queries.retain(|q| set.qrels.scored_queries().any(|s| s == q.id));
    set.queries.truncate(max_queries);
    let keep_q: Vec<&str> = set.queries.iter().map(|q| q.id.as_str()).collect();
    let relevant: std::collections::BTreeSet<String> = set
        .docs
        .iter()
        .filter(|d| keep_q.iter().any(|q| set.qrels.grade(q, &d.id) > 0))
        .map(|d| d.id.clone())
        .collect();
    let (mut kept, mut distractors): (Vec<_>, Vec<_>) = set.docs.drain(..).partition(|d| relevant.contains(&d.id));
    distractors.truncate(max_docs.saturating_sub(kept.len()));
    kept.extend(distractors);
    set.docs = kept;
}

/// (recall@1, recall@5, recall@10, recall@100, nDCG@10) for one query.
type Metrics = (f64, f64, f64, f64, f64);

struct Arms {
    static_only: Vec<Hit>,
    contextual_only: Vec<Hit>,
    cascade: Vec<(usize, Vec<Hit>, usize)>,
}

struct QueryIn {
    terms: Vec<String>,
    qs: Vec<f32>,
    qc: Vec<f32>,
}

fn query_in(query: &str) -> QueryIn {
    let qtoks: Vec<String> = words(query).into_iter().map(|(_, w)| w).collect();
    QueryIn { qs: static_vec(&qtoks), qc: mean_vec(&contextual_tokens(&qtoks)), terms: qtoks }
}

/// Accumulates one document's hit into every query's arm lists.
fn run_doc(p: &Prepared, queries: &[QueryIn], arms: &mut [Arms], times: &mut [Vec<f64>]) {
    let (sm, cm) = (model("study/static"), model("study/contextual"));
    let levels = p.doc.get::<Levels>().unwrap();
    let stats = p.doc.get::<TermStats<Levels>>().unwrap();
    let vocab = p.doc.get::<Vocabulary>().unwrap();
    let collection = LocalCollection::new(stats, vocab);
    let best =
        |rows: &Vectors<Levels>, q: &[f32]| p.units.iter().map(|&u| cosine(rows.row(u), q)).fold(0.0f64, f64::max);
    for (qi, q) in queries.iter().enumerate() {
        let qterms: Vec<&str> = q.terms.iter().map(String::as_str).collect();
        let t = Instant::now();
        arms[qi].static_only.push(Hit { doc: p.id.clone(), score: best(&p.statics, &q.qs) });
        times[qi][0] += t.elapsed().as_secs_f64() * 1e6;
        let t = Instant::now();
        arms[qi].contextual_only.push(Hit { doc: p.id.clone(), score: best(&p.contextuals, &q.qc) });
        times[qi][1] += t.elapsed().as_secs_f64() * 1e6;
        for (k, (depth, hits, scored)) in arms[qi].cascade.iter_mut().enumerate() {
            let cascade = Cascade::new(
                levels,
                CascadeConfig { candidates: *depth, top_k: 10, ..CascadeConfig::default() },
                Some(VectorSpace { rows: &p.statics, query: QueryVector { model: &sm, vector: &q.qs } }),
                Some(VectorSpace { rows: &p.contextuals, query: QueryVector { model: &cm, vector: &q.qc } }),
                Some(LexicalSpace { stats, vocabulary: vocab, collection: &collection }),
            )
            .unwrap();
            let t = Instant::now();
            let out = cascade.search(&qterms);
            times[qi][2 + k] += t.elapsed().as_secs_f64() * 1e6;
            *scored += out.contextual_scored;
            let score = out.hits.iter().find_map(|h| h.contextual_score).unwrap_or(0.0);
            hits.push(Hit { doc: p.id.clone(), score });
        }
    }
}

fn ranked(mut hits: Vec<Hit>, query_id: &str) -> Vec<Hit> {
    hits.retain(|h| h.doc != query_id);
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.doc.cmp(&b.doc)));
    hits
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied().unwrap_or(0.0)
}

fn study(set: &LoadedSet, depths: &[usize]) -> String {
    let prepared: Vec<Prepared> = set
        .docs
        .iter()
        .map(|d| {
            let text = match &d.title {
                Some(t) if !t.is_empty() => format!("{t}\n\n{}", d.text),
                _ => d.text.clone(),
            };
            prepare(&d.id, &text)
        })
        .collect();
    let total_units: usize = prepared.iter().map(|p| p.units.len()).sum();
    let mut out = format!(
        "### {} ({} docs, {} level units, {} queries)\n\n",
        set.name,
        prepared.len(),
        total_units,
        set.queries.len()
    );
    out.push_str("| Arm | C | Recall@1 | Recall@5 | Recall@10 | Recall@100 | nDCG@10 | Search p50 (us) | Contextual scored/query |\n|---|---|---|---|---|---|---|---|---|\n");
    let n_arms = 2 + depths.len();
    let mut per_arm: Vec<Vec<Metrics>> = vec![Vec::new(); n_arms];
    let mut lat: Vec<Vec<f64>> = vec![Vec::new(); n_arms];
    let mut scored: Vec<f64> = vec![0.0; depths.len()];
    let inputs: Vec<QueryIn> = set.queries.iter().map(|q| query_in(&q.text)).collect();
    let mut arms: Vec<Arms> = set
        .queries
        .iter()
        .map(|_| Arms {
            static_only: Vec::new(),
            contextual_only: Vec::new(),
            cascade: depths.iter().map(|&c| (c, Vec::new(), 0)).collect(),
        })
        .collect();
    let mut qtimes: Vec<Vec<f64>> = set.queries.iter().map(|_| vec![0.0; n_arms]).collect();
    for p in &prepared {
        run_doc(p, &inputs, &mut arms, &mut qtimes);
    }
    for (qi, q) in set.queries.iter().enumerate() {
        let arms = &arms[qi];
        let times = &qtimes[qi];
        let all = std::iter::once(&arms.static_only)
            .chain(std::iter::once(&arms.contextual_only))
            .chain(arms.cascade.iter().map(|(_, h, _)| h));
        for (k, hits) in all.enumerate() {
            let r = ranked(hits.clone(), &q.id);
            per_arm[k].push((
                recall(&r, &set.qrels, &q.id, 1),
                recall(&r, &set.qrels, &q.id, 5),
                recall(&r, &set.qrels, &q.id, 10),
                recall(&r, &set.qrels, &q.id, 100),
                ndcg(&r, &set.qrels, &q.id, 10),
            ));
            lat[k].push(times[k]);
        }
        for (k, (_, _, s)) in arms.cascade.iter().enumerate() {
            scored[k] += *s as f64;
        }
    }
    let names: Vec<String> = std::iter::once("Static only (static vectors, full scan)".to_string())
        .chain(std::iter::once("Contextual only".to_string()))
        .chain(depths.iter().map(|c| format!("Cascade C={c}")))
        .collect();
    let cs: Vec<String> = std::iter::once("-".to_string())
        .chain(std::iter::once("-".to_string()))
        .chain(depths.iter().map(|c| c.to_string()))
        .collect();
    for k in 0..n_arms {
        let n = per_arm[k].len() as f64;
        let m = |f: &dyn Fn(&Metrics) -> f64| per_arm[k].iter().map(f).sum::<f64>() / n;
        let scored_cell = if k == 0 {
            "0".to_string()
        } else if k == 1 {
            total_units.to_string()
        } else {
            format!("{:.0}", scored[k - 2] / n)
        };
        out.push_str(&format!(
            "| {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.1} | {} |\n",
            names[k],
            cs[k],
            m(&|r| r.0),
            m(&|r| r.1),
            m(&|r| r.2),
            m(&|r| r.3),
            m(&|r| r.4),
            median(lat[k].clone()),
            scored_cell
        ));
    }
    out
}

fn synthetic_set() -> LoadedSet {
    use turbo_eval::metrics::{Qrel, Qrels};
    use turbo_eval::set::{Doc, Query};
    let docs: Vec<Doc> = (0..12)
        .map(|i| Doc {
            id: format!("d{i}"),
            title: None,
            text: format!("Cats hunt mice at dawn. Topic {i} filler words here. Dogs bark loudly."),
        })
        .collect();
    let queries = vec![Query { id: "q0".into(), text: "cats hunt mice".into() }];
    let mut qrels = std::collections::BTreeMap::new();
    qrels.insert(("q0".to_string(), "d3".to_string()), 1);
    LoadedSet {
        name: "synthetic".into(),
        sha256: "0".repeat(64),
        docs,
        queries,
        qrels: Qrels::new(
            qrels.into_iter().map(|((query, doc), grade)| Qrel { query, doc, grade }).collect::<Vec<Qrel>>(),
        )
        .expect("valid qrels"),
        spans: None,
        unplaced: Vec::new(),
        absent: Vec::new(),
    }
}

#[test]
fn cascade_matches_contextual_at_full_pool_on_synthetic_set() {
    let set = synthetic_set();
    let table = study(&set, &[2, 64]);
    assert!(table.contains("Cascade C=64"), "{table}");
}

#[test]
fn cascade_at_full_pool_equals_contextual_only_scores() {
    let text = (0..4)
        .map(|i| format!("{} ", format!("topic{i} ").repeat(30)).trim_end().to_string() + ".")
        .collect::<Vec<_>>()
        .join(" ");
    let p = prepare("d0", &text);
    assert!(p.units.len() > 4, "document unit is in the shared pool: {}", p.units.len());
    let inputs = vec![query_in("topic0 topic1 topic2 topic3")];
    let mut arms =
        vec![Arms { static_only: Vec::new(), contextual_only: Vec::new(), cascade: vec![(64, Vec::new(), 0)] }];
    let mut times = vec![vec![0.0; 3]];
    run_doc(&p, &inputs, &mut arms, &mut times);
    let contextual = arms[0].contextual_only[0].score;
    let cascade = arms[0].cascade[0].1[0].score;
    assert!((contextual - cascade).abs() < 1e-9, "contextual {contextual} vs cascade {cascade}");
    assert_eq!(arms[0].cascade[0].2, p.units.len());
}

#[test]
#[ignore = "benchmark receipt generator for docs/analysis.md; run explicitly with --release --ignored"]
fn cascade_benchmark_receipt() {
    let root = PathBuf::from(std::env::var("TURBO_EVAL_CACHE").expect("TURBO_EVAL_CACHE"));
    for (name, max_docs, max_queries) in
        [("scifact", 1_500usize, 100usize), ("nfcorpus", 1_500, 100), ("arguana", 1_500, 100), ("qasper", 800, 50)]
    {
        let dir = root.join(name);
        if !dir.join("corpus.jsonl").is_file() {
            println!("### {name}: not in BEIR jsonl layout in this cache, skipped\n");
            continue;
        }
        let mut set = beir::read(name, &dir).expect("load cached BEIR set");
        subset(&mut set, max_docs, max_queries);
        println!("{}", study(&set, &[5, 10, 25, 50, 64]));
    }
    for (name, max_docs) in [("gutenqa", 300usize), ("conteb-covid-qa", 800), ("conteb-squad", 800), ("hicbench", 800)]
    {
        match turbo_eval::sets::find(name).and_then(|s| s.load(&root)) {
            Ok(mut set) => {
                subset(&mut set, max_docs, 50);
                println!("{}", study(&set, &[5, 10, 25, 50, 64]));
            }
            Err(e) => println!("### {name}: not loadable from this cache ({e})\n"),
        }
    }
}
