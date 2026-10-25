//! SQuAD 2.0 exact match and F1 for an extractive QA bundle
//! (docs/extractive-qa.md, "Accuracy").
//!
//! The scoring follows the official SQuAD 2.0 evaluation script: an answer
//! and each gold answer are normalized (lowercased, ASCII punctuation
//! removed, the words "a", "an" and "the" removed, whitespace collapsed);
//! exact match is equality after normalization, F1 the token overlap; a
//! question with no gold answer counts only an empty prediction as right;
//! a question's score is its best over its gold answers. Lowercasing is
//! Java's root-locale mapping and whitespace Python's shared predicate,
//! both from `turbo_unicode`; the lowercase mappings agree on the dev
//! set's text.
//!
//! A prediction is the best span and its margin over the no-answer score.
//! With threshold `t`, the bundle answers when the margin is above `t`
//! and abstains otherwise. `best_threshold` is the official script's
//! best-threshold search, over the margins in place of its
//! no-answer probabilities, for exact match and for F1.

use std::collections::HashMap;

use turbo_unicode::category::general_category;
use turbo_unicode::whitespace::is_python_space;

use crate::Result;

/// One question of a SQuAD 2.0 file.
#[derive(Clone, Debug, PartialEq)]
pub struct Example {
    pub id: String,
    pub question: String,
    pub context: String,
    /// The gold answers' texts; empty for an unanswerable question.
    pub answers: Vec<String>,
}

/// Every question of a SQuAD 2.0 JSON file, in file order.
pub fn read(text: &str) -> Result<Vec<Example>> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("squad: {e}"))?;
    let mut out = Vec::new();
    for article in v["data"].as_array().ok_or("squad: no data array")? {
        for p in article["paragraphs"].as_array().ok_or("squad: an article with no paragraphs")? {
            let context = p["context"].as_str().ok_or("squad: a paragraph with no context")?;
            for qa in p["qas"].as_array().ok_or("squad: a paragraph with no qas")? {
                let s = |k: &str| qa[k].as_str().map(str::to_owned).ok_or_else(|| format!("squad: a qa with no {k}"));
                let answers = qa["answers"].as_array().ok_or("squad: a qa with no answers")?;
                let answers: Vec<String> = answers
                    .iter()
                    .filter_map(|a| a["text"].as_str())
                    .filter(|t| !normalize(t).is_empty())
                    .map(str::to_owned)
                    .collect();
                out.push(Example { id: s("id")?, question: s("question")?, context: context.to_owned(), answers });
            }
        }
    }
    Ok(out)
}

/// The official script's `normalize_answer`.
pub fn normalize(s: &str) -> String {
    let mut lower = String::new();
    turbo_unicode::java::to_lowercase(s, &mut lower);
    let no_punct: String = lower.chars().filter(|c| !c.is_ascii_punctuation()).collect();
    // Articles are whole words: runs of letters, numbers and underscores.
    let word = |c: char| c == '_' || general_category(c).is_letter() || general_category(c).is_number();
    let mut no_articles = String::with_capacity(no_punct.len());
    let mut rest = no_punct.as_str();
    while let Some(c) = rest.chars().next() {
        let len = if word(c) { rest.find(|c| !word(c)).unwrap_or(rest.len()) } else { c.len_utf8() };
        let piece = &rest[..len];
        no_articles.push_str(if matches!(piece, "a" | "an" | "the") { " " } else { piece });
        rest = &rest[len..];
    }
    tokens(&no_articles).join(" ")
}

/// `s` split as Python's `str.split()` splits.
fn tokens(s: &str) -> Vec<&str> {
    s.split(is_python_space).filter(|t| !t.is_empty()).collect()
}

/// Exact match after normalization.
pub fn exact(prediction: &str, gold: &str) -> f64 {
    f64::from(u8::from(normalize(prediction) == normalize(gold)))
}

/// The token F1 of the normalized answers; 1 when both are empty, 0 when
/// one is.
pub fn f1(prediction: &str, gold: &str) -> f64 {
    let (p, g) = (normalize(prediction), normalize(gold));
    let (p, g) = (tokens(&p), tokens(&g));
    if p.is_empty() || g.is_empty() {
        return f64::from(u8::from(p == g));
    }
    let mut counts: HashMap<&str, i64> = HashMap::new();
    for t in &g {
        *counts.entry(t).or_default() += 1;
    }
    let mut same = 0;
    for t in &p {
        if let Some(n) = counts.get_mut(t)
            && *n > 0
        {
            *n -= 1;
            same += 1;
        }
    }
    if same == 0 {
        return 0.0;
    }
    let (precision, recall) = (same as f64 / p.len() as f64, same as f64 / g.len() as f64);
    2.0 * precision * recall / (precision + recall)
}

/// A question's exact match and F1 for `prediction` ("" to abstain): the
/// best over its gold answers, or against "" when it has none.
pub fn question_scores(e: &Example, prediction: &str) -> (f64, f64) {
    if e.answers.is_empty() {
        let s = f64::from(u8::from(normalize(prediction).is_empty()));
        return (s, s);
    }
    e.answers.iter().fold((0f64, 0f64), |(x, f), g| (x.max(exact(prediction, g)), f.max(f1(prediction, g))))
}

/// Percentages over a set of questions, and the counts.
#[derive(Clone, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scores {
    pub exact: f64,
    pub f1: f64,
    pub total: u32,
    pub has_ans_exact: f64,
    pub has_ans_f1: f64,
    pub has_ans_total: u32,
    pub no_ans_exact: f64,
    pub no_ans_f1: f64,
    pub no_ans_total: u32,
}

/// The bundle's best span and its margin over the no-answer score.
#[derive(Clone, Debug, PartialEq)]
pub struct Prediction {
    pub text: String,
    pub margin: f64,
}

/// The scores when the bundle answers above `threshold` and abstains
/// otherwise.
pub fn score(examples: &[Example], predictions: &[Prediction], threshold: f64) -> Scores {
    let mut s = Scores::default();
    for (e, p) in examples.iter().zip(predictions) {
        let answer = if p.margin > threshold { p.text.as_str() } else { "" };
        let (x, f) = question_scores(e, answer);
        s.exact += x;
        s.f1 += f;
        s.total += 1;
        if e.answers.is_empty() {
            (s.no_ans_exact, s.no_ans_f1, s.no_ans_total) = (s.no_ans_exact + x, s.no_ans_f1 + f, s.no_ans_total + 1);
        } else {
            (s.has_ans_exact, s.has_ans_f1, s.has_ans_total) =
                (s.has_ans_exact + x, s.has_ans_f1 + f, s.has_ans_total + 1);
        }
    }
    let pct = |v: f64, n: u32| if n == 0 { 0.0 } else { 100.0 * v / f64::from(n) };
    Scores {
        exact: pct(s.exact, s.total),
        f1: pct(s.f1, s.total),
        has_ans_exact: pct(s.has_ans_exact, s.has_ans_total),
        has_ans_f1: pct(s.has_ans_f1, s.has_ans_total),
        no_ans_exact: pct(s.no_ans_exact, s.no_ans_total),
        no_ans_f1: pct(s.no_ans_f1, s.no_ans_total),
        ..s
    }
}

/// The threshold with the best `metric`, and its scores: the official
/// script's `find_best_thresh`. It starts from abstaining on every
/// question, answers one more margin at a time from the highest down to
/// answering every question, and keeps a threshold only when it scores
/// strictly better, so a tie goes to the fewer answers.
pub fn best_threshold(examples: &[Example], predictions: &[Prediction], metric: fn(&Scores) -> f64) -> (f64, Scores) {
    let mut margins: Vec<f64> = predictions.iter().map(|p| p.margin).filter(|m| m.is_finite()).collect();
    margins.sort_by(|a, b| b.total_cmp(a));
    margins.dedup();
    let top = margins.first().copied().unwrap_or(0.0);
    let mut best = (top, score(examples, predictions, top));
    for t in margins.into_iter().skip(1).chain([f64::NEG_INFINITY]) {
        let s = score(examples, predictions, t);
        if metric(&s) > metric(&best.1) {
            best = (t, s);
        }
    }
    best
}

/// What `turbo-eval squad` writes.
#[derive(Clone, Debug, serde::Serialize)]
pub struct SquadRecord {
    pub model_id: String,
    pub revision: String,
    pub manifest_sha256: String,
    pub device: String,
    pub precision: String,
    pub dev_sha256: String,
    /// The scores answering above a margin of 0, the decoder's default.
    pub at_zero: Scores,
    /// The thresholds with the best exact match and the best F1 over the
    /// margins, each with its scores, as the official script reports
    /// `best_exact` and `best_f1` with their thresholds.
    pub best_exact_threshold: f64,
    pub at_best_exact: Scores,
    pub best_f1_threshold: f64,
    pub at_best_f1: Scores,
    pub seconds: f64,
}

/// `turbo-eval squad --bundle <dir> --dev <file> [--device <backend>]
/// [--precision exact|model|fastest] [--batch <n>] [--max-batch <n>] --out <file>`.
pub fn command(args: &[String]) -> Result<()> {
    use turbo::qa::{Answer, DecodeOptions, ExtractiveQa, QaOptions, decode};
    let get =
        |flag: &str| -> Option<String> { args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).cloned() };
    let bundle = get("--bundle").ok_or("squad: --bundle is required")?;
    let dev = get("--dev").ok_or("squad: --dev is required")?;
    let out = get("--out").ok_or("squad: --out is required")?;
    let device = get("--device").unwrap_or_else(|| "cpu".into());
    let precision_name = get("--precision").unwrap_or_else(|| "exact".into());
    let precision = match precision_name.as_str() {
        "model" => turbo::TURBO_PRECISION_MODEL,
        "fastest" => turbo::TURBO_PRECISION_FASTEST,
        "exact" => turbo::TURBO_PRECISION_EXACT,
        p => return Err(format!("squad: --precision {p}: model, fastest or exact")),
    };
    let number = |v: Option<String>, d: u32| v.map_or(Ok(d), |v| v.parse().map_err(|e| format!("squad: {v}: {e}")));
    let batch = number(get("--batch"), 16)?.max(1) as usize;
    let max_batch = number(get("--max-batch"), 0)?;
    let bytes = std::fs::read(&dev).map_err(|e| format!("{dev}: {e}"))?;
    let examples = read(std::str::from_utf8(&bytes).map_err(|e| format!("{dev}: {e}"))?)?;
    let o = QaOptions { backend: Some(device.clone()), precision, cpu_threads: 0, max_batch };
    let mut qa = ExtractiveQa::open_with(std::path::Path::new(&bundle), &o).map_err(|e| e.message)?;
    let started = std::time::Instant::now();
    let decode_all = DecodeOptions { null_threshold: -1e30, ..Default::default() };
    let mut predictions = Vec::with_capacity(examples.len());
    for chunk in examples.chunks(batch) {
        let pairs: Vec<(&str, &str)> = chunk.iter().map(|e| (e.question.as_str(), e.context.as_str())).collect();
        for (e, logits) in chunk.iter().zip(qa.logits_many(&pairs).map_err(|e| e.message)?) {
            predictions.push(match decode(&e.context, &logits, decode_all).map_err(|e| e.message)? {
                Answer::Spans { answers, null_score, .. } if !answers.is_empty() => Prediction {
                    text: e.context[answers[0].bytes.clone()].to_owned(),
                    margin: answers[0].score - null_score,
                },
                _ => Prediction { text: String::new(), margin: f64::NEG_INFINITY },
            });
        }
        eprint!("\r{} of {}", predictions.len(), examples.len());
    }
    eprintln!();
    let seconds = started.elapsed().as_secs_f64();
    let (best_exact_threshold, at_best_exact) = best_threshold(&examples, &predictions, |s| s.exact);
    let (best_f1_threshold, at_best_f1) = best_threshold(&examples, &predictions, |s| s.f1);
    let record = SquadRecord {
        model_id: qa.model_id.clone(),
        revision: qa.revision.clone(),
        manifest_sha256: qa.manifest_sha256.clone(),
        device,
        precision: precision_name,
        dev_sha256: turbo::bundle::sha256_hex(&bytes),
        at_zero: score(&examples, &predictions, 0.0),
        best_exact_threshold,
        at_best_exact,
        best_f1_threshold,
        at_best_f1,
        seconds,
    };
    let text = serde_json::to_string_pretty(&record).map_err(|e| e.to_string())?;
    std::fs::write(&out, text + "\n").map_err(|e| format!("{out}: {e}"))?;
    eprintln!(
        "EM {:.2} F1 {:.2} at 0; best EM {:.2} at {:.3}; best F1 {:.2} at {:.3}",
        record.at_zero.exact,
        record.at_zero.f1,
        record.at_best_exact.exact,
        best_exact_threshold,
        record.at_best_f1.f1,
        best_f1_threshold
    );
    Ok(())
}

#[cfg(test)]
#[path = "squad_tests.rs"]
mod tests;
