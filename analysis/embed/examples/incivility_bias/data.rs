//! The two evaluation sets, read from the files as published.

use std::fs;
use std::path::Path;

use turbo_analysis_layers::{IncivilityLabel, IncivilityScores};
use turbo_doccat::IdentityTerms;

/// One unit: its text, its label scores (the fraction of raters, or 0 and
/// 1), and the identity subgroups it mentions.
pub struct Unit {
    pub text: String,
    pub targets: IncivilityScores,
    pub subgroups: Vec<String>,
}

/// RFC 4180 records: commas, double quotes with `""` inside, and line
/// breaks inside quoted fields.
pub fn csv_records(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => quoted = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            ',' => record.push(std::mem::take(&mut field)),
            '\r' if chars.peek() == Some(&'\n') => {}
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            _ => field.push(c),
        }
    }
    if quoted {
        return Err("a quoted field runs to the end of the file".into());
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

/// `text` cut by OpenNLP's simple tokenizer, as the pipeline cuts it.
pub fn tokens(text: &str) -> Vec<&str> {
    let mut spans = Vec::new();
    turbo_tokenize::simple_spans(text, turbo_tokenize::TokenizerOptions::default(), &mut spans);
    spans.iter().map(|s| &text[s.start as usize..s.end as usize]).collect()
}

fn read(path: &Path) -> Result<Vec<Vec<String>>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    csv_records(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn column(header: &[String], name: &str) -> Result<usize, String> {
    header.iter().position(|h| h == name).ok_or_else(|| format!("no column {name:?}"))
}

/// The English sentence templates (`en_sentence_templates.csv`) and the
/// neutral identity terms of `words.csv`, each term its own subgroup. A
/// toxic template is an insult; the set has no other label.
pub fn madlibs(dir: &Path) -> Result<(Vec<Unit>, IdentityTerms), String> {
    let words = read(&dir.join("words.csv"))?;
    let (kind, connotation, word) =
        (column(&words[0], "type")?, column(&words[0], "connotation")?, column(&words[0], "word")?);
    let identity: Vec<(&str, &str)> = words[1..]
        .iter()
        .filter(|r| r[kind] == "identity" && r[connotation] == "neutral")
        .map(|r| (r[word].as_str(), r[word].as_str()))
        .collect();
    let terms = IdentityTerms::new(identity)?;
    let rows = read(&dir.join("en_sentence_templates.csv"))?;
    let (toxicity, phrase) = (column(&rows[0], "toxicity")?, column(&rows[0], "phrase")?);
    let mut units = Vec::with_capacity(rows.len() - 1);
    for r in &rows[1..] {
        let mut targets = IncivilityScores::default();
        match r[toxicity].as_str() {
            "toxic" => targets.set(IncivilityLabel::Insult, 1.0),
            "nontoxic" => {}
            t => return Err(format!("a toxicity of {t:?}")),
        }
        let subgroups = terms.find(&tokens(&r[phrase])).into_iter().map(str::to_owned).collect();
        units.push(Unit { text: r[phrase].clone(), targets, subgroups });
    }
    Ok((units, terms))
}

/// The Civil Comments columns each label is fitted to and judged by. The
/// set has no intolerance column, so that label's target is 0.
pub const CIVIL_LABELS: [(IncivilityLabel, Option<&str>); 6] = [
    (IncivilityLabel::Insult, Some("insult")),
    (IncivilityLabel::Profanity, Some("obscene")),
    (IncivilityLabel::Threat, Some("threat")),
    (IncivilityLabel::IdentityAttack, Some("identity_attack")),
    (IncivilityLabel::Intolerance, None),
    (IncivilityLabel::SevereToxicity, Some("severe_toxicity")),
];

/// The identities Borkan et al. (2019) report, each with enough comments
/// in the test split to measure.
pub const CIVIL_IDENTITIES: [&str; 9] = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
];

/// `all_data_with_identities.csv`, split by its `split` column into
/// (train, test). An identity column at or above 0.5 is a mention.
pub fn civil(dir: &Path) -> Result<(Vec<Unit>, Vec<Unit>), String> {
    let rows = read(&dir.join("all_data_with_identities.csv"))?;
    let h = &rows[0];
    let (text, split) = (column(h, "comment_text")?, column(h, "split")?);
    let labels: Vec<(IncivilityLabel, Option<usize>)> = CIVIL_LABELS
        .iter()
        .map(|&(l, c)| Ok((l, c.map(|c| column(h, c)).transpose()?)))
        .collect::<Result<_, String>>()?;
    let ids: Vec<usize> = CIVIL_IDENTITIES.iter().map(|c| column(h, c)).collect::<Result<_, _>>()?;
    let number = |s: &str| -> Result<f64, String> {
        if s.is_empty() { Ok(0.0) } else { s.parse::<f64>().map_err(|e| format!("{s:?}: {e}")) }
    };
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (n, r) in rows[1..].iter().enumerate() {
        if r.len() != h.len() {
            return Err(format!("record {n} has {} fields, the header {}", r.len(), h.len()));
        }
        let mut targets = IncivilityScores::default();
        for &(label, c) in &labels {
            if let Some(c) = c {
                targets.set(label, number(&r[c])?.clamp(0.0, 1.0));
            }
        }
        let mut subgroups = Vec::new();
        for (&c, name) in ids.iter().zip(CIVIL_IDENTITIES) {
            if number(&r[c])? >= 0.5 {
                subgroups.push(name.to_owned());
            }
        }
        let unit = Unit { text: r[text].clone(), targets, subgroups };
        match r[split].as_str() {
            "train" => train.push(unit),
            "test" => test.push(unit),
            "val" => {}
            s => return Err(format!("record {n} has the split {s:?}")),
        }
    }
    Ok((train, test))
}
