//! The #1731 gate: exact-span F1 of the rule `date`, `time`, `money` and
//! `percent` entities against gold, beside peer predictions on the same text.
//!
//! Two sets. `masc` is the MASC 3.0.0 held-out documents of the recipes'
//! split that carry a named entity layer, scored on `date` against MASC's
//! own date annotations. `reviewed` is the project-authored reviewed set of
//! `tests/support/rule_entities_corpus.rs`, scored on all four types.
//!
//! ```sh
//! # Write the documents a peer reads: one JSON object per line, {"id", "text"}.
//! cargo run --release -p turbo-query --example rule_entities_gate -- export --set masc --masc MASC-3.0.0 --out masc.jsonl
//! # Score the rules and each peer file: {"id", "entities": [{"start", "end", "label"}]},
//! # UTF-8 byte offsets into the exported text.
//! cargo run --release -p turbo-query --example rule_entities_gate -- score --set masc --masc MASC-3.0.0 \
//!     --peer spacy-trf=masc.spacy-trf.jsonl --out receipt.json
//! ```
//!
//! Every count is a document's set of distinct `(start, end, label)` spans,
//! matched exactly. Intervals are paired bootstrap resamples of documents,
//! the same draws for every system, of each peer's F1 minus the rules'.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use serde_json::{Value, json};
use turbo_analysis::{Analyzer, Document, Ix, Key, Layer};
use turbo_analysis_layers::{DateTime, Entities, NumberNotation};
use turbo_formats::masc::MascCorpus;
use turbo_grammar::TimeAnnotator;
use turbo_namefind::{
    ConsensusEntityMerger, DATE_ENTITY, EntityMergePolicy, MONEY_ENTITY, PERCENT_ENTITY, RULE_ENTITY_KINDS,
    RuleEntityProjector, TIME_ENTITY, TimeEntityFilter,
};
use turbo_numeric::{MoneyExtractor, QuantityExtractor, TemporalExtractor};
use turbo_tokenize::SimpleTokenizer;
use turbo_train_recipes::masc::{self, RELEASE, SPLIT};

include!("../tests/support/rule_entities_corpus.rs");

type Spans = BTreeSet<(u32, u32, String)>;
type Fail = Box<dyn std::error::Error>;

const DRAWS: usize = 1000;
const SEED: u64 = 1731;

struct GoldSet {
    name: &'static str,
    kinds: Vec<&'static str>,
    docs: Vec<(String, String, Spans)>,
    facts: Value,
}

fn masc_set(dir: &Path) -> Result<GoldSet, Fail> {
    let digest = masc::check_corpus(dir, RELEASE, false)?;
    let split = SPLIT.apply(MascCorpus::read(dir)?, dir)?;
    let mut docs = Vec::new();
    let (mut discontinuous, mut unplaced, mut kinds) = (0usize, 0usize, BTreeMap::<String, usize>::new());
    for (doc, path) in split.held_out.documents().iter().zip(&split.held_out_paths) {
        if !doc.has_named_entities() {
            continue;
        }
        let tokens = doc.document_tokens().ok_or("a named entity document without Penn tokens")?;
        let spans = doc.token_spans(tokens)?;
        let mut gold = Spans::new();
        for e in tokens.entities() {
            let kind = e.kind.clone().unwrap_or_default();
            *kinds.entry(kind.clone()).or_default() += 1;
            if kind != DATE_ENTITY {
                continue;
            }
            if !e.is_contiguous() || e.start >= e.end {
                discontinuous += 1;
                continue;
            }
            gold.insert((spans[e.start as usize].start, spans[e.end as usize - 1].end, kind));
        }
        unplaced += doc.unplaced_entities().iter().filter(|u| u.kind.as_deref() == Some(DATE_ENTITY)).count();
        docs.push((path.clone(), doc.text().to_owned(), gold));
    }
    let facts = json!({
        "corpus": masc::NAME,
        "corpus_sha256": digest.sha256,
        "corpus_files": digest.files,
        "split": format!("held out at {} modulo {}, Twitter genre excluded", SPLIT.held_out_at(), SPLIT.every()),
        "documents_with_entities": docs.len(),
        "gold_entity_kinds": kinds,
        "gold_date_entities_discontinuous_or_empty": discontinuous,
        "gold_date_entities_unplaced_in_sentences": unplaced,
    });
    Ok(GoldSet { name: "masc", kinds: vec![DATE_ENTITY], docs, facts })
}

fn reviewed_set() -> Result<GoldSet, Fail> {
    let mut docs = Vec::new();
    for (i, g) in GOLD_CORPUS.iter().enumerate() {
        let (mut from, mut gold) = (0usize, Spans::new());
        for &(mention, kind) in g.expected {
            let at = g.text[from..].find(mention).ok_or_else(|| format!("{mention:?} not in document {i}"))? + from;
            from = at + mention.len();
            gold.insert((at as u32, from as u32, kind.to_owned()));
        }
        docs.push((format!("reviewed-{i}"), g.text.to_owned(), gold));
    }
    let facts = json!({ "source": "analysis/query/tests/support/rule_entities_corpus.rs", "documents": docs.len() });
    Ok(GoldSet { name: "reviewed", kinds: RULE_ENTITY_KINDS.to_vec(), docs, facts })
}

/// The pipeline of `tests/rule_entities_gold.rs`.
fn rules(texts: &[&str]) -> Result<Vec<Spans>, Fail> {
    let analyzer = Analyzer::builder()
        .add(SimpleTokenizer::default())
        .add(TemporalExtractor::new())
        .add(TimeAnnotator::english().with_reference(DateTime::parse("2026-07-14T00:00:00").ok_or("reference date")?))
        .add(MoneyExtractor::new(NumberNotation::LatinUs))
        .add(QuantityExtractor::new(NumberNotation::LatinUs))
        .step(RuleEntityProjector::money(Key::DEFAULT, Key::named("money")))
        .step(RuleEntityProjector::quantities(Key::DEFAULT, Key::named("percent")))
        .step(RuleEntityProjector::temporals(Key::DEFAULT, Key::named("temporal")))
        .step(RuleEntityProjector::times_with_filter(
            Key::DEFAULT,
            Key::named("times"),
            TimeEntityFilter::ClockTimesOnly,
        ))
        .step(ConsensusEntityMerger::new(
            vec![Key::named("money"), Key::named("percent"), Key::named("temporal"), Key::named("times")],
            Key::DEFAULT,
            EntityMergePolicy::PriorityFallback,
        ))
        .build()?;
    let mut docs: Vec<Document> = texts.iter().map(|t| Document::new(*t)).collect();
    for r in analyzer.run(&mut docs) {
        r?;
    }
    docs.iter()
        .map(|d| {
            let e = d.get::<Entities>().ok_or("no entities layer")?;
            Ok((0..e.len())
                .map(|i| {
                    let s = e.spans.get(Ix::new(i as u32));
                    (s.start, s.end, e.kinds.label(e.kind[i]).to_owned())
                })
                .collect())
        })
        .collect()
}

/// A peer span offset. One past `u32::MAX` is refused, not wrapped into a
/// span that may pass the text checks below.
fn offset(e: &Value, key: &str) -> Result<u32, Fail> {
    let n = e[key].as_u64().ok_or_else(|| format!("{key} is not a whole number"))?;
    Ok(u32::try_from(n).map_err(|_| format!("{key} {n} is past u32::MAX"))?)
}

fn peer(path: &Path, set: &GoldSet) -> Result<Vec<Spans>, Fail> {
    let mut by_id = BTreeMap::new();
    for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()) {
        let v: Value = serde_json::from_str(line)?;
        let id = v["id"].as_str().ok_or("peer line without id")?.to_owned();
        let mut spans = Spans::new();
        for e in v["entities"].as_array().ok_or("peer line without entities")? {
            let (s, t) = (offset(e, "start")?, offset(e, "end")?);
            spans.insert((s, t, e["label"].as_str().ok_or("label")?.to_owned()));
        }
        if by_id.insert(id.clone(), spans).is_some() {
            return Err(format!("{}: {id} twice", path.display()).into());
        }
    }
    if by_id.len() != set.docs.len() {
        return Err(format!("{}: {} documents, the set has {}", path.display(), by_id.len(), set.docs.len()).into());
    }
    set.docs
        .iter()
        .map(|(id, text, _)| {
            let spans = by_id.remove(id).ok_or_else(|| format!("{}: no {id}", path.display()))?;
            match spans.iter().find(|(s, e, _)| {
                s >= e
                    || *e as usize > text.len()
                    || !text.is_char_boundary(*s as usize)
                    || !text.is_char_boundary(*e as usize)
            }) {
                Some(bad) => Err(format!("{}: {id}: bad span {bad:?}", path.display()).into()),
                None => Ok(spans),
            }
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct Count {
    tp: usize,
    fp: usize,
    fn_: usize,
}

impl Count {
    fn of(gold: &Spans, pred: &Spans, kind: Option<&str>) -> Count {
        let keep = |x: &&(u32, u32, String)| kind.is_none_or(|k| x.2 == k);
        let g: BTreeSet<_> = gold.iter().filter(keep).collect();
        let p: BTreeSet<_> = pred.iter().filter(keep).collect();
        let tp = g.intersection(&p).count();
        Count { tp, fp: p.len() - tp, fn_: g.len() - tp }
    }
    fn add(&mut self, o: Count) {
        (self.tp, self.fp, self.fn_) = (self.tp + o.tp, self.fp + o.fp, self.fn_ + o.fn_);
    }
    fn f1(self) -> f64 {
        let d = 2 * self.tp + self.fp + self.fn_;
        if d == 0 { 1.0 } else { 2.0 * self.tp as f64 / d as f64 }
    }
    fn json(self) -> Value {
        let ratio = |n: usize, d: usize| if d == 0 { 1.0 } else { n as f64 / d as f64 };
        json!({
            "tp": self.tp, "fp": self.fp, "fn": self.fn_,
            "precision": round(ratio(self.tp, self.tp + self.fp)),
            "recall": round(ratio(self.tp, self.tp + self.fn_)),
            "f1": round(self.f1()),
        })
    }
}

fn round(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

/// Per-document counts of one system for one kind (`None`: every kind of the set).
fn counts(set: &GoldSet, pred: &[Spans], kind: Option<&str>) -> Vec<Count> {
    let filter = |s: &Spans| -> Spans { s.iter().filter(|x| set.kinds.contains(&x.2.as_str())).cloned().collect() };
    set.docs.iter().zip(pred).map(|((_, _, g), p)| Count::of(g, &filter(p), kind)).collect()
}

fn total(c: &[Count], draw: impl Iterator<Item = usize>) -> Count {
    let mut t = Count::default();
    draw.for_each(|i| t.add(c[i]));
    t
}

/// SplitMix64, so the draws are the same on every machine.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn interval(base: &[Count], other: &[Count]) -> Value {
    let n = base.len();
    let mut state = SEED;
    let mut deltas: Vec<f64> = (0..DRAWS)
        .map(|_| {
            let draw: Vec<usize> = (0..n).map(|_| (next(&mut state) % n as u64) as usize).collect();
            total(other, draw.iter().copied()).f1() - total(base, draw.iter().copied()).f1()
        })
        .collect();
    deltas.sort_by(f64::total_cmp);
    let delta = total(other, 0..n).f1() - total(base, 0..n).f1();
    json!({ "delta_f1": round(delta), "low": round(deltas[DRAWS / 40]), "high": round(deltas[DRAWS - 1 - DRAWS / 40]) })
}

fn score(set: &GoldSet, peers: &[(String, PathBuf)], dump: Option<&Path>) -> Result<Value, Fail> {
    let texts: Vec<&str> = set.docs.iter().map(|(_, t, _)| t.as_str()).collect();
    let started = std::time::Instant::now();
    let rule = rules(&texts)?;
    let rule_seconds = started.elapsed().as_secs_f64();
    let mut systems = vec![("rules".to_owned(), rule, None)];
    for (name, path) in peers {
        let digest = turbo::bundle::sha256_hex(&std::fs::read(path)?);
        systems.push((name.clone(), peer(path, set)?, Some(digest)));
    }
    if let Some(dump) = dump {
        write_misses(set, &systems, dump)?;
    }
    let mut kinds: Vec<Option<&str>> = set.kinds.iter().map(|k| Some(*k)).collect();
    if set.kinds.len() > 1 {
        kinds.push(None);
    }
    let mut rows = Vec::new();
    for (name, pred, digest) in &systems {
        let mut by = serde_json::Map::new();
        for &kind in &kinds {
            let c = counts(set, pred, kind);
            let mut row = total(&c, 0..c.len()).json();
            if name != "rules" {
                row["vs_rules"] = interval(&counts(set, &systems[0].1, kind), &c);
            }
            by.insert(kind.unwrap_or("micro").to_owned(), row);
        }
        rows.push(json!({ "system": name, "predictions_sha256": digest, "scores": by }));
    }
    let gold: usize = set.docs.iter().map(|(_, _, g)| g.len()).sum();
    Ok(json!({
        "gate": "#1731",
        "set": set.name,
        "facts": set.facts,
        "gold_spans": gold,
        "match": "exact UTF-8 byte span and type, distinct spans per document",
        "bootstrap": { "unit": "document", "draws": DRAWS, "seed": SEED, "interval": "2.5th to 97.5th percentile" },
        "rules_seconds": round(rule_seconds),
        "systems": rows,
    }))
}

/// Each system's misses per document, as text, for reading the errors.
fn write_misses(set: &GoldSet, systems: &[(String, Vec<Spans>, Option<String>)], out: &Path) -> Result<(), Fail> {
    let mut lines = Vec::new();
    for (name, pred, _) in systems {
        for ((id, text, gold), p) in set.docs.iter().zip(pred) {
            let p: Spans = p.iter().filter(|x| set.kinds.contains(&x.2.as_str())).cloned().collect();
            let show = |s: &BTreeSet<&(u32, u32, String)>| -> Vec<Value> {
                s.iter().map(|(a, b, k)| json!([a, b, k, &text[*a as usize..*b as usize]])).collect()
            };
            let missed: BTreeSet<_> = gold.difference(&p).collect();
            let extra: BTreeSet<_> = p.difference(gold).collect();
            lines.push(json!({ "system": name, "id": id, "missed": show(&missed), "extra": show(&extra) }).to_string());
        }
    }
    std::fs::write(out, lines.join("\n") + "\n")?;
    Ok(())
}

fn main() -> Result<(), Fail> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let flag = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
    let set = match flag("--set").as_deref() {
        Some("masc") => masc_set(Path::new(&flag("--masc").ok_or("--masc DIR")?))?,
        Some("reviewed") => reviewed_set()?,
        _ => return Err("--set masc|reviewed".into()),
    };
    let out = PathBuf::from(flag("--out").ok_or("--out FILE")?);
    match args.first().map(String::as_str) {
        Some("export") => {
            let lines: Vec<String> =
                set.docs.iter().map(|(id, text, _)| json!({ "id": id, "text": text }).to_string()).collect();
            std::fs::write(&out, lines.join("\n") + "\n")?;
        }
        Some("score") => {
            let mut peers = Vec::new();
            for (i, a) in args.iter().enumerate() {
                if a == "--peer" {
                    let spec = args.get(i + 1).ok_or("--peer NAME=FILE")?;
                    let (name, file) = spec.split_once('=').ok_or("--peer NAME=FILE")?;
                    peers.push((name.to_owned(), PathBuf::from(file)));
                }
            }
            let receipt = score(&set, &peers, flag("--dump").as_deref().map(Path::new))?;
            std::fs::write(&out, serde_json::to_string_pretty(&receipt)? + "\n")?;
            println!("{}", serde_json::to_string_pretty(&receipt["systems"])?);
        }
        _ => return Err("export | score".into()),
    }
    eprintln!("{} documents written to {}", set.docs.len(), out.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_offset_past_u32_is_refused_not_wrapped() {
        let wrapped = 4_294_967_296u64 + 3;
        for key in ["start", "end"] {
            let e = serde_json::json!({ "start": 3, "end": 9, "label": "date" });
            assert_eq!(offset(&e, key).unwrap(), if key == "start" { 3 } else { 9 });
            let mut bad = e.clone();
            bad[key] = Value::from(wrapped);
            let err = offset(&bad, key).unwrap_err().to_string();
            assert!(err.contains("past u32::MAX"), "{err}");
            bad[key] = Value::from(-1);
            assert!(offset(&bad, key).is_err());
        }
        let max = serde_json::json!({ "start": u64::from(u32::MAX) });
        assert_eq!(offset(&max, "start").unwrap(), u32::MAX);
    }
}
