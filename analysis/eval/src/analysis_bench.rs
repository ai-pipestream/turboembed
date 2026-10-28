//! The peer analysis benchmark mode (`turbo-eval analysis`, #1521).
//!
//! It scores CoNLL-U predictions against a SHA-pinned CoNLL-U test
//! file: sentence boundaries by token position, token forms, and, where
//! a sentence's predicted tokens align with the gold tokens, UPOS,
//! XPOS, feats, lemma, UAS and LAS. The same scoring serves our own
//! systems and peer systems (spaCy, Stanza, Trankit, wtpsplit) once a
//! peer run has written its predictions as CoNLL-U; the peers are
//! measurement inputs here, never dependencies of this crate.
//!
//! It also scores our sentence tiers directly: each document's gold
//! `# text` lines are joined with spaces, split by the chosen
//! [`Sentencer`], and scored by boundary F1 in joined coordinates. That
//! conditioning is recorded in every record, as is the token-aligned
//! conditioning of the field metrics: a sentence whose predicted token
//! forms or count differ from gold contributes zero correct field
//! values and is counted under `token_mismatch_sentences`.
//!
//! Entities and language are not scored by this first mode: CoNLL-U
//! carries neither. Their records land with the MultiCoNER/WNUT and
//! FLORES+ inputs; a system block names them `not_run` rather than
//! inventing a score.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_sat::SatOptions;

use crate::Result;
use crate::analysis::Sentencer;
use crate::model_gate::{Counts, Measure};
use crate::random::JavaRandom;
use crate::sha::{sha256, to_hex};
use crate::significance::percentile;

/// The fixed bootstrap seed of every analysis record.
pub const BOOTSTRAP_SEED: i64 = 42;
/// The default number of paired bootstrap resamples per metric.
pub const DEFAULT_RESAMPLES: u32 = 1000;

/// One CoNLL-U token row (word lines only; multiword-token ranges and
/// empty nodes are skipped by the parser).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConlluToken {
    pub form: String,
    pub lemma: String,
    pub upos: String,
    pub xpos: String,
    pub feats: String,
    pub head: Option<i64>,
    pub deprel: String,
}

/// One CoNLL-U sentence: its `# text` line when present, and its tokens.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ConlluSentence {
    pub text: Option<String>,
    pub tokens: Vec<ConlluToken>,
    /// Whether a `# newdoc` comment starts a document at this sentence.
    pub new_doc: bool,
}

/// Reads CoNLL-U sentences. Lines with a range (`1-2`) or decimal
/// (`3.1`) id are not word lines and are skipped; a word line must have
/// the ten tab-separated fields.
pub fn parse_conllu(text: &str) -> Result<Vec<ConlluSentence>> {
    let mut out = Vec::new();
    let mut current = ConlluSentence::default();
    let mut started = false;
    for (line_no, line) in text.lines().enumerate() {
        let line_no = line_no + 1;
        if line.is_empty() {
            if started {
                out.push(std::mem::take(&mut current));
                started = false;
            }
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            started = true;
            if let Some(t) = rest.trim().strip_prefix("text =") {
                current.text = Some(t.trim().to_string());
            }
            if rest.trim().starts_with("newdoc") {
                current.new_doc = true;
            }
            continue;
        }
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() != 10 {
            return Err(format!("line {line_no}: expected 10 CoNLL-U fields, got {}", fields.len()));
        }
        if fields[0].contains('-') || fields[0].contains('.') {
            continue;
        }
        if fields[0].parse::<u32>().is_err() {
            return Err(format!("line {line_no}: token id {:?} is not a positive integer", fields[0]));
        }
        started = true;
        current.tokens.push(ConlluToken {
            form: fields[1].to_string(),
            lemma: fields[2].to_string(),
            upos: fields[3].to_string(),
            xpos: fields[4].to_string(),
            feats: fields[5].to_string(),
            head: fields[6].parse::<i64>().ok(),
            deprel: fields[7].to_string(),
        });
    }
    if started {
        out.push(current);
    }
    if out.is_empty() {
        return Err("no CoNLL-U sentences".into());
    }
    Ok(out)
}

/// One metric of one system: pooled counts, the score in points, and a
/// bootstrap interval when per-sentence units were resampled.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MetricScore {
    pub correct: u64,
    pub selected: u64,
    pub target: u64,
    pub score: f64,
    pub low: Option<f64>,
    pub high: Option<f64>,
}

impl MetricScore {
    fn of(counts: Counts, measure: Measure, units: Option<&[Counts]>, resamples: u32) -> MetricScore {
        let (low, high) = match units {
            Some(units) if !units.is_empty() && resamples > 0 => {
                let (lo, hi) = bootstrap(units, measure, resamples);
                (Some(lo), Some(hi))
            }
            _ => (None, None),
        };
        MetricScore {
            correct: counts.true_positives,
            selected: counts.selected,
            target: counts.target,
            score: counts.score(measure),
            low,
            high,
        }
    }
}

/// Pooled bootstrap over per-sentence counts, with the crate's seeded
/// draws. Each draw scores the summed counts, never a mean of scores.
fn bootstrap(units: &[Counts], measure: Measure, resamples: u32) -> (f64, f64) {
    let n = units.len();
    let mut random = JavaRandom::new(BOOTSTRAP_SEED);
    let mut scores = Vec::with_capacity(resamples as usize);
    for _ in 0..resamples {
        let mut total = Counts::default();
        for _ in 0..n {
            let u = units[random.next_int_bounded(n as i32) as usize];
            total.true_positives += u.true_positives;
            total.selected += u.selected;
            total.target += u.target;
        }
        scores.push(total.score(measure));
    }
    scores.sort_by(f64::total_cmp);
    (percentile(&scores, 0.025), percentile(&scores, 0.975))
}

/// One system's scores on one gold file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SystemScores {
    pub system: String,
    pub sentences: MetricScore,
    pub tokens: Option<MetricScore>,
    pub upos: Option<MetricScore>,
    pub xpos: Option<MetricScore>,
    pub feats: Option<MetricScore>,
    pub lemma: Option<MetricScore>,
    pub uas: Option<MetricScore>,
    pub las: Option<MetricScore>,
    pub entities: Option<MetricScore>,
    pub language: Option<MetricScore>,
    pub not_run: Vec<String>,
    pub token_mismatch_sentences: usize,
    pub throughput_sentences_per_second: Option<f64>,
}

fn boundaries(sentences: &[ConlluSentence]) -> BTreeSet<usize> {
    let mut at = BTreeSet::new();
    let mut seen = 0usize;
    for s in &sentences[..sentences.len().saturating_sub(1)] {
        seen += s.tokens.len();
        at.insert(seen);
    }
    at
}

fn sentence_counts(gold: &[ConlluSentence], pred: &[ConlluSentence]) -> Counts {
    let gold_at = boundaries(gold);
    let pred_at = boundaries(pred);
    Counts {
        true_positives: gold_at.intersection(&pred_at).count() as u64,
        selected: pred_at.len() as u64,
        target: gold_at.len() as u64,
    }
}

/// Scores predicted CoNLL-U sentences against gold. Sentence
/// boundaries are compared by flattened token position; field metrics
/// are token-aligned per paired sentence, as the module docs state.
pub fn score_conllu(system: &str, gold: &[ConlluSentence], pred: &[ConlluSentence], resamples: u32) -> SystemScores {
    let mut mismatch = 0usize;
    let mut units: [Vec<Counts>; 7] = Default::default();
    for (i, gold_sentence) in gold.iter().enumerate() {
        let mut per = [Counts::default(); 7];
        let gold_tokens = &gold_sentence.tokens;
        per[0].target = gold_tokens.len() as u64;
        for field in per.iter_mut().skip(1) {
            field.selected = gold_tokens.len() as u64;
            field.target = gold_tokens.len() as u64;
        }
        match pred.get(i) {
            Some(pred_sentence) => {
                let pred_tokens = &pred_sentence.tokens;
                per[0].selected = pred_tokens.len() as u64;
                let aligned = pred_tokens.len() == gold_tokens.len()
                    && pred_tokens.iter().zip(gold_tokens).all(|(p, g)| p.form == g.form);
                if aligned {
                    per[0].true_positives = gold_tokens.len() as u64;
                    for (p, g) in pred_tokens.iter().zip(gold_tokens) {
                        per[1].true_positives += (p.upos == g.upos) as u64;
                        per[2].true_positives += (p.xpos == g.xpos) as u64;
                        per[3].true_positives += (p.feats == g.feats) as u64;
                        per[4].true_positives += (p.lemma == g.lemma) as u64;
                        per[5].true_positives += (p.head == g.head) as u64;
                        per[6].true_positives += (p.head == g.head && p.deprel == g.deprel) as u64;
                    }
                } else {
                    mismatch += 1;
                    per[0].true_positives =
                        pred_tokens.iter().zip(gold_tokens).filter(|(p, g)| p.form == g.form).count() as u64;
                }
            }
            None => mismatch += 1,
        }
        for (slot, counts) in units.iter_mut().zip(per) {
            slot.push(counts);
        }
    }
    let pooled = |slot: &[Counts]| {
        slot.iter().fold(Counts::default(), |mut total, u| {
            total.true_positives += u.true_positives;
            total.selected += u.selected;
            total.target += u.target;
            total
        })
    };
    let accuracy =
        |idx: usize| Some(MetricScore::of(pooled(&units[idx]), Measure::Accuracy, Some(&units[idx]), resamples));
    SystemScores {
        system: system.to_string(),
        sentences: MetricScore::of(sentence_counts(gold, pred), Measure::FMeasure, None, resamples),
        tokens: Some(MetricScore::of(pooled(&units[0]), Measure::FMeasure, Some(&units[0]), resamples)),
        upos: accuracy(1),
        xpos: accuracy(2),
        feats: accuracy(3),
        lemma: accuracy(4),
        uas: accuracy(5),
        las: accuracy(6),
        entities: None,
        language: None,
        not_run: vec![
            "entities: CoNLL-U carries no entity layer".into(),
            "language: scored on FLORES+ inputs, not CoNLL-U".into(),
        ],
        token_mismatch_sentences: mismatch,
        throughput_sentences_per_second: None,
    }
}

/// Scores a sentence tier over each document's joined `# text` lines
/// by boundary F1 in joined byte coordinates, and times the split.
pub fn score_sentencer(system: &str, gold: &[ConlluSentence], sentencer: &Sentencer) -> Result<SystemScores> {
    let mut gold_texts = Vec::with_capacity(gold.len());
    for (i, s) in gold.iter().enumerate() {
        gold_texts.push(s.text.as_deref().ok_or_else(|| format!("gold sentence {i} has no # text line"))?);
    }
    // Split each CoNLL-U document separately: joining the whole test
    // file would let bracket and quote state leak across documents.
    let mut documents: Vec<Vec<&str>> = Vec::new();
    for (sentence, text) in gold.iter().zip(&gold_texts) {
        if sentence.new_doc || documents.is_empty() {
            documents.push(Vec::new());
        }
        documents.last_mut().expect("a document group").push(text);
    }
    let started = Instant::now();
    let mut counts = Counts::default();
    for document in &documents {
        let joined = document.join(" ");
        let spans = sentencer.spans(&joined)?;
        let mut gold_at = BTreeSet::new();
        let mut at = 0usize;
        for text in &document[..document.len().saturating_sub(1)] {
            at += text.len();
            gold_at.insert(at as u32);
            at += 1;
        }
        let rows: Vec<_> = spans.iter().collect();
        let pred_at: BTreeSet<u32> = rows[..rows.len().saturating_sub(1)].iter().map(|(_, s)| s.end).collect();
        counts.true_positives += gold_at.intersection(&pred_at).count() as u64;
        counts.selected += pred_at.len() as u64;
        counts.target += gold_at.len() as u64;
    }
    let elapsed = started.elapsed().as_secs_f64();
    Ok(SystemScores {
        system: system.to_string(),
        sentences: MetricScore::of(counts, Measure::FMeasure, None, 0),
        tokens: None,
        upos: None,
        xpos: None,
        feats: None,
        lemma: None,
        uas: None,
        las: None,
        entities: None,
        language: None,
        not_run: vec!["tokens/fields: a sentence tier writes no token fields".into()],
        token_mismatch_sentences: 0,
        throughput_sentences_per_second: (elapsed > 0.0).then_some(gold.len() as f64 / elapsed),
    })
}

/// One committed analysis record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalysisRecord {
    pub version: u32,
    pub gold_file: String,
    pub gold_sha256: String,
    pub sentences: usize,
    pub tokens: usize,
    pub resamples: u32,
    pub seed: i64,
    pub conditioning: Vec<String>,
    pub systems: Vec<SystemScores>,
}

/// The `turbo-eval analysis` command: score CoNLL-U predictions and
/// our sentence tiers against one SHA-pinned gold file.
pub fn command(args: &[String]) -> Result<()> {
    let mut gold_path = None;
    let mut want_sha = None;
    let mut out = None;
    let mut preds: Vec<(String, String)> = Vec::new();
    let mut system_name = "turbo".to_string();
    let mut sentences = None;
    let mut sat_bundle = None;
    let mut resamples = DEFAULT_RESAMPLES;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let value =
            |it: &mut std::slice::Iter<'_, String>| it.next().cloned().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--gold" => gold_path = Some(value(&mut it)?),
            "--sha256" => want_sha = Some(value(&mut it)?),
            "--out" => out = Some(value(&mut it)?),
            "--system" => system_name = value(&mut it)?,
            "--sentences" => sentences = Some(value(&mut it)?),
            "--sat-bundle" => sat_bundle = Some(value(&mut it)?),
            "--resamples" => resamples = value(&mut it)?.parse().map_err(|_| "--resamples is not a count")?,
            "--pred" => {
                let v = value(&mut it)?;
                let (name, path) = v.split_once('=').ok_or("--pred needs NAME=PATH")?;
                preds.push((name.to_string(), path.to_string()));
            }
            other => return Err(format!("analysis: unknown option {other}")),
        }
    }
    let gold_path = gold_path.ok_or("analysis needs --gold <file.conllu>")?;
    let want_sha = want_sha.ok_or("analysis needs --sha256 <hex> pinning the gold file")?;
    let out = out.ok_or("analysis needs --out <record.json>")?;
    if Path::new(&out).exists() {
        return Err(format!("{out} exists; refusing to overwrite a record"));
    }
    let gold_bytes = std::fs::read(&gold_path).map_err(|e| format!("{gold_path}: {e}"))?;
    let digest = to_hex(&sha256(&gold_bytes));
    if digest != want_sha.to_ascii_lowercase() {
        return Err(format!("{gold_path}: sha256 {digest}, pinned {want_sha}"));
    }
    let gold_text = String::from_utf8(gold_bytes).map_err(|e| format!("{gold_path}: {e}"))?;
    let gold = parse_conllu(&gold_text)?;

    let mut systems = Vec::new();
    for (name, path) in &preds {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let pred = parse_conllu(&text)?;
        systems.push(score_conllu(name, &gold, &pred, resamples));
    }
    if let Some(tier) = &sentences {
        let sentencer = match tier.as_str() {
            "rule" | "rules" => Sentencer::rules("en"),
            "newline" => Sentencer::Newline,
            "sat" => {
                let bundle = sat_bundle.as_deref().ok_or("--sentences sat needs --sat-bundle <dir>")?;
                let options = SatOptions {
                    encoder: EncoderOptions { device: Device::Cpu, precision: Precision::Model, ..Default::default() },
                    ..SatOptions::default()
                };
                Sentencer::sat(Path::new(bundle), &options)?
            }
            other => return Err(format!("--sentences {other}: rule, newline or sat")),
        };
        systems.push(score_sentencer(&system_name, &gold, &sentencer)?);
    }
    if systems.is_empty() {
        return Err("analysis needs at least one --pred or --sentences system".into());
    }

    let record = AnalysisRecord {
        version: 1,
        gold_file: Path::new(&gold_path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        gold_sha256: digest,
        sentences: gold.len(),
        tokens: gold.iter().map(|s| s.tokens.len()).sum(),
        resamples,
        seed: BOOTSTRAP_SEED,
        conditioning: vec![
            "sentences: boundary F1 by flattened token position for CoNLL-U predictions; boundary F1 in per-document space-joined # text coordinates for sentence tiers".into(),
            "fields: token-aligned per gold sentence; a token-form or count mismatch scores zero field values for that sentence and is counted".into(),
            "uas/las: punctuation included".into(),
            "bootstrap: per-sentence pooled resampling for token-aligned metrics".into(),
        ],
        systems,
    };
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&out, bytes).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out}");
    Ok(())
}
