//! Language scoring for the analysis benchmark (FLORES+ inputs):
//! one gold TSV of `code<TAB>text` lines, one predicted code per line
//! from each peer, accuracy with a per-text bootstrap, and our
//! script-based fallback as the first turbo system. The gold file is
//! an evaluation input pinned by SHA-256 at the command line; records
//! keep counts only, never the texts.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Serialize;
use turbo_langspan::scripts::{group, written_in};
use turbo_unicode::Script;

use super::{BOOTSTRAP_SEED, DEFAULT_RESAMPLES, MetricScore};
use crate::Result;
use crate::model_gate::{Counts, Measure};
use crate::sha::{sha256, to_hex};

/// One gold text with its ISO 639-3 language code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LangGold {
    pub code: String,
    pub text: String,
}

/// Reads gold language lines: `code<TAB>text`, one text per line.
pub fn parse_lang_gold(text: &str) -> Result<Vec<LangGold>> {
    let mut out = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let (code, rest) = line.split_once('\t').ok_or_else(|| format!("line {}: expected code<TAB>text", i + 1))?;
        if code.is_empty() || code.contains(char::is_whitespace) {
            return Err(format!("line {}: bad language code {code:?}", i + 1));
        }
        if rest.trim().is_empty() {
            return Err(format!("line {}: empty text", i + 1));
        }
        out.push(LangGold { code: code.to_string(), text: rest.to_string() });
    }
    if out.is_empty() {
        return Err("no language gold lines".into());
    }
    Ok(out)
}

/// Reads predicted codes, one per gold line (a TSV first field is also
/// accepted, so a peer may append its confidence after a tab).
pub fn parse_lang_pred(text: &str, expected: usize) -> Result<Vec<String>> {
    let mut out = Vec::with_capacity(expected);
    for (i, line) in text.lines().enumerate() {
        let code = line.split('\t').next().unwrap_or_default().trim();
        if code.is_empty() {
            return Err(format!("prediction line {}: empty code", i + 1));
        }
        out.push(code.to_string());
    }
    if out.len() != expected {
        return Err(format!("{} predicted codes for {expected} gold texts", out.len()));
    }
    Ok(out)
}

/// Accuracy for one gold language.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LanguageScore {
    pub code: String,
    pub correct: u64,
    pub total: u64,
    pub score: f64,
}

/// One system's language scores on one gold file.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LanguageSystem {
    pub system: String,
    pub language: MetricScore,
    pub per_language: Vec<LanguageScore>,
}

/// Scores predicted codes against gold. Bootstrap units are texts, so
/// the interval is over the pooled accuracy, not a mean of languages.
pub fn score_language(system: &str, gold: &[LangGold], pred: &[String], resamples: u32) -> LanguageSystem {
    let units: Vec<Counts> = gold
        .iter()
        .zip(pred)
        .map(|(g, p)| Counts { true_positives: (g.code == *p) as u64, selected: 1, target: 1 })
        .collect();
    let pooled = units.iter().fold(Counts::default(), |mut total, u| {
        total.true_positives += u.true_positives;
        total.selected += u.selected;
        total.target += u.target;
        total
    });
    let mut by_code: BTreeMap<&str, (u64, u64)> = BTreeMap::new();
    for (g, p) in gold.iter().zip(pred) {
        let entry = by_code.entry(g.code.as_str()).or_default();
        entry.0 += (g.code == *p) as u64;
        entry.1 += 1;
    }
    LanguageSystem {
        system: system.to_string(),
        language: MetricScore::of(pooled, Measure::Accuracy, Some(&units), resamples),
        per_language: by_code
            .into_iter()
            .map(|(code, (correct, total))| LanguageScore {
                code: code.to_string(),
                correct,
                total,
                score: 100.0 * correct as f64 / total as f64,
            })
            .collect(),
    }
}

/// The table in `turbo-langspan` keys Chinese as `cmn` and Arabic as
/// `ara`; FLORES+ labels them `zho` and `arb`. The alias is only for
/// the script lookup; predictions keep the gold code.
fn table_code(code: &str) -> &str {
    match code {
        "zho" => "cmn",
        "arb" => "ara",
        other => other,
    }
}

/// Our script-based fallback: a text is the one language of the gold
/// label set written in its dominant script group, and `und` when the
/// script is shared, tied, unknown to the table, or absent. It is a
/// closed-set baseline: it can only ever name a language whose script
/// identifies it on its own.
pub fn script_fallback_predictions(gold: &[LangGold]) -> Vec<String> {
    let mut labels: Vec<&str> = gold.iter().map(|g| g.code.as_str()).collect();
    labels.sort_unstable();
    labels.dedup();
    gold.iter()
        .map(|g| {
            let mut counts: Vec<(Script, usize)> = Vec::new();
            for c in g.text.chars().filter(|c| c.is_alphabetic()) {
                if let Some(group) = group(turbo_unicode::script::script(c)) {
                    match counts.iter_mut().find(|(s, _)| *s == group) {
                        Some((_, n)) => *n += 1,
                        None => counts.push((group, 1)),
                    }
                }
            }
            counts.sort_by_key(|&(_, n)| std::cmp::Reverse(n));
            let dominant = match counts.as_slice() {
                [(group, _)] => Some(*group),
                [(group, a), (_, b), ..] if a > b => Some(*group),
                _ => None,
            };
            let Some(dominant) = dominant else { return "und".to_string() };
            let candidates: Vec<&str> = labels
                .iter()
                .copied()
                .filter(|code| written_in(table_code(code)).is_some_and(|scripts| scripts.contains(&dominant)))
                .collect();
            match candidates.as_slice() {
                [only] => only.to_string(),
                _ => "und".to_string(),
            }
        })
        .collect()
}

/// One committed language record.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LanguageRecord {
    pub version: u32,
    pub gold_file: String,
    pub gold_sha256: String,
    pub texts: usize,
    pub languages: Vec<String>,
    pub resamples: u32,
    pub seed: i64,
    pub conditioning: Vec<String>,
    pub systems: Vec<LanguageSystem>,
}

/// The language form of `turbo-eval analysis`: `--lang-gold` in place
/// of `--gold`, `--lang-pred` per peer, `--script-fallback` for ours.
pub fn language_command(args: &[String]) -> Result<()> {
    let mut gold_path = None;
    let mut want_sha = None;
    let mut out = None;
    let mut preds: Vec<(String, String)> = Vec::new();
    let mut system_name = "turbo-script-fallback".to_string();
    let mut script_fallback = false;
    let mut resamples = DEFAULT_RESAMPLES;
    let mut it = args.iter();
    while let Some(arg) = it.next() {
        let value =
            |it: &mut std::slice::Iter<'_, String>| it.next().cloned().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--lang-gold" => gold_path = Some(value(&mut it)?),
            "--sha256" => want_sha = Some(value(&mut it)?),
            "--out" => out = Some(value(&mut it)?),
            "--system" => system_name = value(&mut it)?,
            "--script-fallback" => script_fallback = true,
            "--resamples" => resamples = value(&mut it)?.parse().map_err(|_| "--resamples is not a count")?,
            "--lang-pred" => {
                let v = value(&mut it)?;
                let (name, path) = v.split_once('=').ok_or("--lang-pred needs NAME=PATH")?;
                preds.push((name.to_string(), path.to_string()));
            }
            other => return Err(format!("analysis (language): unknown option {other}")),
        }
    }
    let gold_path = gold_path.ok_or("analysis (language) needs --lang-gold <file.tsv>")?;
    let want_sha = want_sha.ok_or("analysis (language) needs --sha256 <hex> pinning the gold file")?;
    let out = out.ok_or("analysis (language) needs --out <record.json>")?;
    if Path::new(&out).exists() {
        return Err(format!("{out} exists; refusing to overwrite a record"));
    }
    let gold_bytes = std::fs::read(&gold_path).map_err(|e| format!("{gold_path}: {e}"))?;
    let digest = to_hex(&sha256(&gold_bytes));
    if digest != want_sha.to_ascii_lowercase() {
        return Err(format!("{gold_path}: sha256 {digest}, pinned {want_sha}"));
    }
    let gold_text = String::from_utf8(gold_bytes).map_err(|e| format!("{gold_path}: {e}"))?;
    let gold = parse_lang_gold(&gold_text)?;

    let mut systems = Vec::new();
    for (name, path) in &preds {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let pred = parse_lang_pred(&text, gold.len())?;
        systems.push(score_language(name, &gold, &pred, resamples));
    }
    if script_fallback {
        systems.push(score_language(&system_name, &gold, &script_fallback_predictions(&gold), resamples));
    }
    if systems.is_empty() {
        return Err("analysis (language) needs at least one --lang-pred or --script-fallback".into());
    }

    let mut languages: Vec<String> = gold.iter().map(|g| g.code.clone()).collect();
    languages.sort_unstable();
    languages.dedup();
    let record = LanguageRecord {
        version: 1,
        gold_file: Path::new(&gold_path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        gold_sha256: digest,
        texts: gold.len(),
        languages,
        resamples,
        seed: BOOTSTRAP_SEED,
        conditioning: vec![
            "language: accuracy over texts, paired bootstrap resampling texts".into(),
            "script fallback: closed label set of the gold file; dominant script group by letter count; a tied, shared or unknown script predicts und".into(),
        ],
        systems,
    };
    let bytes = serde_json::to_vec_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&out, bytes).map_err(|e| format!("{out}: {e}"))?;
    println!("wrote {out}");
    Ok(())
}
