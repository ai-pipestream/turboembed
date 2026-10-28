//! Entity scoring for the analysis benchmark (WNUT-17 and MultiCoNER
//! inputs): a SHA-pinned gold file of `token<TAB>BIO-label` lines
//! with blank lines between sentences, one predicted label per gold
//! token from each peer, exact-span F1 with a per-sentence bootstrap.
//! Peers run on the gold tokenization, as these benchmarks are
//! conventionally run; that conditioning is written into every
//! record, and label-set mapping is the peer runner's job, documented
//! beside its record. Records keep counts only, never the texts.

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde::Serialize;

use super::{BOOTSTRAP_SEED, DEFAULT_RESAMPLES, MetricScore};
use crate::Result;
use crate::model_gate::{Counts, Measure};
use crate::sha::{sha256, to_hex};

/// One gold sentence: tokens with their BIO labels.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NerSentence {
    pub tokens: Vec<String>,
    pub labels: Vec<String>,
}

fn check_label(label: &str, at: String) -> Result<()> {
    if label == "O" {
        return Ok(());
    }
    match label.split_once('-') {
        Some(("B" | "I", kind)) if !kind.is_empty() => Ok(()),
        _ => Err(format!("{at}: label {label:?} is not BIO")),
    }
}

/// Reads gold entity lines: `token<TAB>label`, blank line between
/// sentences. Comment lines starting with `#` are skipped, as the
/// MultiCoNER files carry sentence headers there.
pub fn parse_ner_gold(text: &str) -> Result<Vec<NerSentence>> {
    let mut out = Vec::new();
    let mut current = NerSentence { tokens: Vec::new(), labels: Vec::new() };
    for (i, line) in text.lines().enumerate() {
        if line.starts_with('#') {
            continue;
        }
        if line.trim().is_empty() {
            if !current.tokens.is_empty() {
                out.push(std::mem::replace(&mut current, NerSentence { tokens: Vec::new(), labels: Vec::new() }));
            }
            continue;
        }
        let (token, label) =
            line.split_once('\t').ok_or_else(|| format!("line {}: expected token<TAB>label", i + 1))?;
        if token.is_empty() {
            return Err(format!("line {}: empty token", i + 1));
        }
        check_label(label, format!("line {}", i + 1))?;
        current.tokens.push(token.to_string());
        current.labels.push(label.to_string());
    }
    if !current.tokens.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        return Err("no entity gold sentences".into());
    }
    Ok(out)
}

/// Reads predicted labels in gold sentence and token shape: one label
/// per line (a TSV first field is also accepted), blank lines between
/// sentences. The shape must match the gold file exactly.
pub fn parse_ner_pred(text: &str, gold: &[NerSentence]) -> Result<Vec<Vec<String>>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            continue;
        }
        let label = line.split('\t').next().unwrap_or_default().trim();
        check_label(label, format!("prediction line {}", i + 1))?;
        current.push(label.to_string());
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.len() != gold.len() {
        return Err(format!("{} predicted sentences for {} gold sentences", out.len(), gold.len()));
    }
    for (i, (pred, gold_sentence)) in out.iter().zip(gold).enumerate() {
        if pred.len() != gold_sentence.tokens.len() {
            return Err(format!(
                "sentence {i}: {} predicted labels for {} gold tokens",
                pred.len(),
                gold_sentence.tokens.len()
            ));
        }
    }
    Ok(out)
}

/// Entity spans as (start, end exclusive, type). Decoding is the
/// lenient common one: `B-X` starts a span, `I-X` continues a span of
/// the same type and otherwise starts one, `O` ends any span.
fn spans(labels: &[String]) -> BTreeSet<(usize, usize, &str)> {
    let mut out = BTreeSet::new();
    let mut open: Option<(usize, &str)> = None;
    for (i, label) in labels.iter().enumerate() {
        let (prefix, kind) = label.split_once('-').unwrap_or(("O", ""));
        match prefix {
            "B" => {
                if let Some((start, kind)) = open.take() {
                    out.insert((start, i, kind));
                }
                open = Some((i, kind));
            }
            "I" => match open {
                Some((_, current)) if current == kind => {}
                _ => {
                    if let Some((start, kind)) = open.take() {
                        out.insert((start, i, kind));
                    }
                    open = Some((i, kind));
                }
            },
            _ => {
                if let Some((start, kind)) = open.take() {
                    out.insert((start, i, kind));
                }
            }
        }
    }
    if let Some((start, kind)) = open {
        out.insert((start, labels.len(), kind));
    }
    out
}

/// Exact-span F1 for one entity type.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TypeScore {
    pub kind: String,
    pub correct: u64,
    pub selected: u64,
    pub target: u64,
    pub score: f64,
}

/// One system's entity scores on one gold file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NerSystem {
    pub system: String,
    pub entities: MetricScore,
    pub per_type: Vec<TypeScore>,
}

/// Scores predicted BIO labels by exact span match per sentence.
/// Bootstrap units are sentences.
pub fn score_ner(system: &str, gold: &[NerSentence], pred: &[Vec<String>], resamples: u32) -> NerSystem {
    let mut units = Vec::with_capacity(gold.len());
    let mut by_kind: BTreeMap<&str, Counts> = BTreeMap::new();
    for (gold_sentence, pred_sentence) in gold.iter().zip(pred) {
        let gold_spans = spans(&gold_sentence.labels);
        let pred_spans = spans(pred_sentence);
        units.push(Counts {
            true_positives: gold_spans.intersection(&pred_spans).count() as u64,
            selected: pred_spans.len() as u64,
            target: gold_spans.len() as u64,
        });
        for span in &pred_spans {
            by_kind.entry(span.2).or_default().selected += 1;
        }
        for span in &gold_spans {
            by_kind.entry(span.2).or_default().target += 1;
        }
        for span in gold_spans.intersection(&pred_spans) {
            by_kind.entry(span.2).or_default().true_positives += 1;
        }
    }
    let pooled = units.iter().fold(Counts::default(), |mut total, u| {
        total.true_positives += u.true_positives;
        total.selected += u.selected;
        total.target += u.target;
        total
    });
    NerSystem {
        system: system.to_string(),
        entities: MetricScore::of(pooled, Measure::FMeasure, Some(&units), resamples),
        per_type: by_kind
            .into_iter()
            .map(|(kind, counts)| TypeScore {
                kind: kind.to_string(),
                correct: counts.true_positives,
                selected: counts.selected,
                target: counts.target,
                score: counts.score(Measure::FMeasure),
            })
            .collect(),
    }
}

/// One committed entity record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NerRecord {
    pub version: u32,
    pub gold_file: String,
    pub gold_sha256: String,
    pub sentences: usize,
    pub tokens: usize,
    pub resamples: u32,
    pub seed: i64,
    pub conditioning: Vec<String>,
    pub systems: Vec<NerSystem>,
}

/// The entity form of `turbo-eval analysis`: `--ner-gold` in place of
/// `--gold`, `--ner-pred` per peer prediction file.
pub fn ner_command(args: &[String]) -> Result<()> {
    let mut gold_path = None;
    let mut want_sha = None;
    let mut out = None;
    let mut preds: Vec<(String, String)> = Vec::new();
    let mut resamples = DEFAULT_RESAMPLES;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let value =
            |it: &mut std::slice::Iter<'_, String>| it.next().cloned().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--ner-gold" => gold_path = Some(value(&mut it)?),
            "--sha256" => want_sha = Some(value(&mut it)?),
            "--out" => out = Some(value(&mut it)?),
            "--resamples" => resamples = value(&mut it)?.parse().map_err(|_| "--resamples is not a count")?,
            "--ner-pred" => {
                let v = value(&mut it)?;
                let (name, path) = v.split_once('=').ok_or("--ner-pred needs NAME=PATH")?;
                preds.push((name.to_string(), path.to_string()));
            }
            other => return Err(format!("analysis (entities): unknown option {other}")),
        }
    }
    let gold_path = gold_path.ok_or("analysis (entities) needs --ner-gold <file.tsv>")?;
    let want_sha = want_sha.ok_or("analysis (entities) needs --sha256 <hex> pinning the gold file")?;
    let out = out.ok_or("analysis (entities) needs --out <record.json>")?;
    if preds.is_empty() {
        return Err("analysis (entities) needs at least one --ner-pred".into());
    }
    if Path::new(&out).exists() {
        return Err(format!("{out} exists; refusing to overwrite a record"));
    }
    let gold_bytes = std::fs::read(&gold_path).map_err(|e| format!("{gold_path}: {e}"))?;
    let digest = to_hex(&sha256(&gold_bytes));
    if digest != want_sha.to_ascii_lowercase() {
        return Err(format!("{gold_path}: sha256 {digest}, pinned {want_sha}"));
    }
    let gold_text = String::from_utf8(gold_bytes).map_err(|e| format!("{gold_path}: {e}"))?;
    let gold = parse_ner_gold(&gold_text)?;

    let mut systems = Vec::new();
    for (name, path) in &preds {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let pred = parse_ner_pred(&text, &gold)?;
        systems.push(score_ner(name, &gold, &pred, resamples));
    }

    let record = NerRecord {
        version: 1,
        gold_file: Path::new(&gold_path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        gold_sha256: digest,
        sentences: gold.len(),
        tokens: gold.iter().map(|s| s.tokens.len()).sum(),
        resamples,
        seed: BOOTSTRAP_SEED,
        conditioning: vec![
            "entities: exact-span F1 from BIO labels; lenient decoding (an I- with no open span of its type starts one); paired bootstrap resampling sentences".into(),
            "tokenization: gold tokens are given to every system; a prediction must label every gold token".into(),
            "label mapping: each peer runner maps its label set to the gold labels and documents the map beside its record".into(),
        ],
        systems,
    };
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&out, bytes).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out}");
    Ok(())
}
