//! The peer analysis benchmark mode (`turbo-eval analysis`, #1521).
//!
//! It scores CoNLL-U predictions against a SHA-pinned CoNLL-U test
//! file. Predicted and gold token streams are aligned by token form
//! (a greedy walk with bounded resync, like the official UD scorer's
//! word alignment but on forms rather than character offsets, since a
//! peer prediction need not carry offsets); sentence boundaries are
//! then compared on aligned tokens, and UPOS, XPOS, feats, lemma, UAS,
//! and LAS count over gold tokens, an unaligned gold token scoring as
//! wrong on every field. The same scoring serves our own systems and
//! peer systems (spaCy, Stanza, Trankit, wtpsplit) once a peer run has
//! written its predictions as CoNLL-U; the peers are measurement
//! inputs here, never dependencies of this crate.
//!
//! It also scores our sentence tiers directly: each document's gold
//! `# text` lines are joined with spaces, split by the chosen
//! [`Sentencer`], and scored by boundary F1 in joined coordinates. A
//! gold sentence with a token the alignment could not match is counted
//! under `token_mismatch_sentences`.
//!
//! Entities are not scored by this mode yet: their records land with
//! the MultiCoNER/WNUT inputs, and a CoNLL-U system block names them
//! `not_run` rather than inventing a score. Language is scored by the
//! `--lang-gold` form of the same command (see `language`).

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use serde::Serialize;
use turbo_analysis_embed::{Device, EncoderOptions, Precision};
use turbo_sat::SatOptions;

use crate::Result;
use crate::analysis::Sentencer;

mod align;
mod language;
use crate::model_gate::{Counts, Measure};
use crate::random::JavaRandom;
use crate::sha::{sha256, to_hex};
use crate::significance::percentile;
pub use language::{LangGold, parse_lang_gold, parse_lang_pred, score_language, script_fallback_predictions};

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

use align::{add_boundary_counts, align_forms, flatten, joined_documents};
pub use align::{parse_sentence_list, score_sentence_list};

/// Scores predicted CoNLL-U sentences against gold: the token streams
/// are form-aligned, sentence boundaries and fields are scored on the
/// aligned pairs, and field denominators are gold tokens. Bootstrap
/// units are gold documents.
pub fn score_conllu(system: &str, gold: &[ConlluSentence], pred: &[ConlluSentence], resamples: u32) -> SystemScores {
    let gold_all = flatten(gold);
    let (gold_flat, gold_docs, gold_sents) = (&gold_all.tokens, &gold_all.documents, &gold_all.sentences);
    let pred_all = flatten(pred);
    let pred_flat = &pred_all.tokens;
    let pairs = align_forms(gold_flat, pred_flat);
    let mut gold_to_pred: Vec<Option<usize>> = vec![None; gold_flat.len()];
    let mut pred_to_gold: Vec<Option<usize>> = vec![None; pred_flat.len()];
    for &(g, p) in &pairs {
        gold_to_pred[g] = Some(p);
        pred_to_gold[p] = Some(g);
    }
    // Sentence boundaries: a gold sentence-final token (other than the
    // file's last) whose aligned partner is also sentence-final.
    let mut sent_counts = Counts::default();
    for (i, t) in gold_flat.iter().enumerate() {
        if t.sentence_end && i + 1 < gold_flat.len() {
            sent_counts.target += 1;
            if let Some(p) = gold_to_pred[i] {
                sent_counts.true_positives += pred_flat[p].sentence_end as u64;
            }
        }
    }
    sent_counts.selected =
        pred_flat.iter().enumerate().filter(|(i, t)| t.sentence_end && *i + 1 < pred_flat.len()).count() as u64;

    let doc_of_gold = |g: usize| gold_docs.partition_point(|&(_, end)| end <= g);
    let mut units: Vec<[Counts; 7]> = vec![[Counts::default(); 7]; gold_docs.len()];
    // Predicted tokens join the gold document of the nearest
    // following-or-preceding aligned pair for per-document counts; the
    // pooled totals do not depend on this attribution.
    let mut pred_doc = vec![0usize; pred_flat.len()];
    let mut current = 0usize;
    for (p, slot) in pred_doc.iter_mut().enumerate() {
        if let Some(g) = pred_to_gold[p] {
            current = doc_of_gold(g);
        }
        *slot = current;
    }
    for (doc, &(start, end)) in gold_docs.iter().enumerate() {
        units[doc][0].target = (end - start) as u64;
        for field in units[doc].iter_mut().skip(1) {
            field.selected = (end - start) as u64;
            field.target = (end - start) as u64;
        }
    }
    for &doc in &pred_doc {
        units[doc][0].selected += 1;
    }
    for &(g, p) in &pairs {
        let unit = &mut units[doc_of_gold(g)];
        let (gt, pt) = (gold_flat[g].token, pred_flat[p].token);
        unit[0].true_positives += 1;
        unit[1].true_positives += (pt.upos == gt.upos) as u64;
        unit[2].true_positives += (pt.xpos == gt.xpos) as u64;
        unit[3].true_positives += (pt.feats == gt.feats) as u64;
        unit[4].true_positives += (pt.lemma == gt.lemma) as u64;
        let pred_head = pred_flat[p].head.map(|head| head.and_then(|hp| pred_to_gold[hp]));
        let right_head = pred_head.is_some() && pred_head == gold_flat[g].head;
        unit[5].true_positives += right_head as u64;
        unit[6].true_positives += (right_head && pt.deprel == gt.deprel) as u64;
    }
    let mismatch = gold_sents.iter().filter(|&&(s, e)| (s..e).any(|g| gold_to_pred[g].is_none())).count();
    let pooled = |idx: usize| {
        units.iter().fold(Counts::default(), |mut total, u| {
            total.true_positives += u[idx].true_positives;
            total.selected += u[idx].selected;
            total.target += u[idx].target;
            total
        })
    };
    let slot = |idx: usize| units.iter().map(|u| u[idx]).collect::<Vec<_>>();
    let accuracy = |idx: usize| Some(MetricScore::of(pooled(idx), Measure::Accuracy, Some(&slot(idx)), resamples));
    SystemScores {
        system: system.to_string(),
        sentences: MetricScore::of(sent_counts, Measure::FMeasure, None, resamples),
        tokens: Some(MetricScore::of(pooled(0), Measure::FMeasure, Some(&slot(0)), resamples)),
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
    let documents = joined_documents(gold)?;
    let started = Instant::now();
    let mut counts = Counts::default();
    for (joined, gold_at) in &documents {
        let spans = sentencer.spans(joined)?;
        let rows: Vec<_> = spans.iter().collect();
        let pred_at: BTreeSet<u32> = rows[..rows.len().saturating_sub(1)].iter().map(|(_, s)| s.end).collect();
        add_boundary_counts(&mut counts, gold_at, &pred_at);
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
    if args.iter().any(|a| a == "--lang-gold") {
        return language::language_command(args);
    }
    let mut gold_path = None;
    let mut want_sha = None;
    let mut out = None;
    let mut preds: Vec<(String, String)> = Vec::new();
    let mut sents: Vec<(String, String)> = Vec::new();
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
            "--sents" => {
                let v = value(&mut it)?;
                let (name, path) = v.split_once('=').ok_or("--sents needs NAME=PATH")?;
                sents.push((name.to_string(), path.to_string()));
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
    for (name, path) in &sents {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        systems.push(score_sentence_list(name, &gold, &parse_sentence_list(&text))?);
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
        return Err("analysis needs at least one --pred, --sents or --sentences system".into());
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
            "sentences: boundary F1 on form-aligned tokens for CoNLL-U predictions; boundary F1 in per-document space-joined # text coordinates for sentence tiers".into(),
            "fields: gold denominators; gold and predicted token streams aligned by form with bounded resync; an unaligned gold token scores zero on every field; gold sentences containing one are counted".into(),
            "uas/las: punctuation included".into(),
            "bootstrap: per-document pooled resampling for token-aligned metrics".into(),
        ],
        systems,
    };
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&out, bytes).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out}");
    Ok(())
}
