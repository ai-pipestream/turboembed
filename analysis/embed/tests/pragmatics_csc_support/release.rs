//! Frozen release verification, separate from training and threshold selection.
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use serde_json::Value;
use turbo_analysis::ModelRef;
use turbo_analysis_embed::ClassifierHead;

use super::corpus::Response;
use super::metrics::Counts;
use super::split::{Partition, SAME_SCENARIO, jaccard, words};

pub fn expected() -> Value {
    serde_json::from_str(include_str!("release.json")).unwrap()
}

pub fn validate_config(config: &Value) -> Result<(), String> {
    if config != &expected()["config"] {
        return Err("CSC release configuration differs from models-sentiment-925-2".into());
    }
    Ok(())
}

pub fn validate_model(model: &ModelRef) -> Result<(), String> {
    let e = expected();
    let m = &e["model"];
    if model.id != m["id"].as_str().unwrap()
        || model.revision != m["revision"].as_str().unwrap()
        || model.sha256 != m["sha256"].as_str().unwrap()
    {
        return Err("CSC release encoder identity differs".into());
    }
    Ok(())
}

pub fn load(dir: &Path, model: &ModelRef) -> Result<(ClassifierHead, f64, f64), String> {
    validate_model(model)?;
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("pragmatics-1.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    validate_config(&config)?;
    let bytes = std::fs::read(dir.join("pragmatics-1.thead")).map_err(|e| e.to_string())?;
    let head = validate_head(&bytes, model)?;
    Ok((head, config["head_threshold"].as_f64().unwrap(), config["combined_threshold"].as_f64().unwrap()))
}

fn validate_head(bytes: &[u8], model: &ModelRef) -> Result<ClassifierHead, String> {
    if turbo::bundle::sha256_hex(bytes) != expected()["head_sha256"].as_str().unwrap() {
        return Err("CSC release head digest differs".into());
    }
    let head = ClassifierHead::from_bytes(bytes)?;
    if head.model() != model {
        return Err("CSC release head uses another encoder".into());
    }
    Ok(head)
}

pub fn validate_source(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    if turbo::bundle::sha256_hex(&bytes) != expected()["source_sha256"].as_str().unwrap() {
        return Err("CSC source digest differs from the pinned release data".into());
    }
    Ok(())
}

pub fn validate_population(rows: &[Response]) -> Result<(), String> {
    let e = expected();
    let p = &e["population"];
    let counts = [
        ("total", rows.len()),
        ("train", rows.iter().filter(|r| r.partition == Partition::Train).count()),
        ("dev", rows.iter().filter(|r| r.partition == Partition::Dev).count()),
        ("test", rows.iter().filter(|r| r.partition == Partition::Test).count()),
        ("test_positive", rows.iter().filter(|r| r.partition == Partition::Test && r.sarcastic).count()),
    ];
    for (name, count) in counts {
        if p[name].as_u64() != Some(count as u64) {
            return Err(format!("CSC release {name} count differs: {count}"));
        }
    }
    Ok(())
}

/// The model sees response text, and responses to one scenario share topic
/// words, so neither a response text nor a scenario may span partitions.
pub fn validate_disjoint_inputs(rows: &[Response]) -> Result<(), String> {
    let mut texts: BTreeMap<&str, BTreeSet<Partition>> = BTreeMap::new();
    let mut contexts: BTreeMap<String, BTreeSet<Partition>> = BTreeMap::new();
    for row in rows {
        texts.entry(&row.text).or_default().insert(row.partition);
        let normalized = row.context_text.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        contexts.entry(normalized).or_default().insert(row.partition);
    }
    let overlap = contexts.values().filter(|p| p.len() > 1).count();
    if overlap > 0 {
        return Err(format!("CSC split leaks {overlap} scenario texts across train/dev/test"));
    }
    let scenarios: Vec<(BTreeSet<String>, &BTreeSet<Partition>)> =
        contexts.iter().map(|(text, p)| (words(text), p)).collect();
    for (i, (a, pa)) in scenarios.iter().enumerate() {
        for (b, pb) in &scenarios[i + 1..] {
            if pa != pb && jaccard(a, b) >= SAME_SCENARIO {
                return Err("CSC split puts reworded variants of one scenario in different partitions".into());
            }
        }
    }
    let overlap = texts.values().filter(|p| p.len() > 1).count();
    if overlap > 0 {
        return Err(format!("CSC split leaks {overlap} response texts across train/dev/test"));
    }
    Ok(())
}

pub fn validate_metrics(actual: Counts, expected: &Value) -> Result<(), String> {
    for (name, count) in [("tp", actual.tp), ("fp", actual.fp), ("tn", actual.tn), ("fn", actual.r#fn)] {
        if expected[name].as_u64() != Some(count as u64) {
            return Err(format!("CSC release {name}: computed {count}, expected {}", expected[name]));
        }
    }
    for (name, metric) in [("precision", actual.precision()), ("recall", actual.recall()), ("f1", actual.f1())] {
        let wanted = expected[name].as_f64().ok_or_else(|| format!("missing release {name}"))?;
        if !metric.is_finite() || !wanted.is_finite() || (metric - wanted).abs() > 1e-12 {
            return Err(format!("CSC release {name}: computed {metric}, expected {wanted}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(context: u32, context_text: &str, text: &str, partition: Partition) -> Response {
        Response { context, context_text: context_text.into(), text: text.into(), sarcastic: false, partition }
    }

    #[test]
    fn scenario_and_text_leaks_fail() {
        let shoes = "Steve bought a pair of shoes after work today.";
        let printer = "You walk into the office and the printer is jammed again.";
        let clean = [row(1, shoes, "Nice.", Partition::Train), row(2, printer, "Great.", Partition::Test)];
        validate_disjoint_inputs(&clean).unwrap();
        let same_text = [row(1, shoes, "Nice.", Partition::Train), row(2, printer, "Nice.", Partition::Test)];
        assert!(validate_disjoint_inputs(&same_text).unwrap_err().contains("response texts"));
        let same_scenario =
            [row(1, shoes, "Nice.", Partition::Train), row(9, &shoes.to_uppercase(), "Great.", Partition::Test)];
        assert!(validate_disjoint_inputs(&same_scenario).unwrap_err().contains("scenario texts"));
        let reworded = "Steve bought an expensive pair of shoes after work.";
        let variant = [row(1, shoes, "Nice.", Partition::Train), row(9, reworded, "Great.", Partition::Test)];
        assert!(validate_disjoint_inputs(&variant).unwrap_err().contains("reworded"));
    }

    #[test]
    #[ignore = "requires the downloaded published CSC head files"]
    fn actual_release_file_accepts_and_corrupt_bytes_fail() {
        let dir = std::env::var("TURBO_PRAGMATICS_HEAD").expect("TURBO_PRAGMATICS_HEAD is required");
        let e = expected();
        let m = &e["model"];
        let model = ModelRef {
            id: m["id"].as_str().unwrap().into(),
            revision: m["revision"].as_str().unwrap().into(),
            sha256: m["sha256"].as_str().unwrap().into(),
        };
        load(Path::new(&dir), &model).unwrap();
        let mut bytes = std::fs::read(Path::new(&dir).join("pragmatics-1.thead")).unwrap();
        bytes[0] ^= 1;
        assert!(validate_head(&bytes, &model).is_err());
    }

    #[test]
    fn published_counts_pass_and_changes_fail() {
        let e = expected();
        for name in ["rules", "head", "combined"] {
            let metric = &e["config"]["evaluation"][name];
            let mut actual = Counts {
                tp: metric["tp"].as_u64().unwrap() as usize,
                fp: metric["fp"].as_u64().unwrap() as usize,
                tn: metric["tn"].as_u64().unwrap() as usize,
                r#fn: metric["fn"].as_u64().unwrap() as usize,
            };
            validate_metrics(actual, metric).unwrap();
            let mut changed = metric.clone();
            changed["f1"] = serde_json::json!(0.99);
            assert!(validate_metrics(actual, &changed).is_err());
            actual.fp += 1;
            assert!(validate_metrics(actual, metric).is_err());
        }
    }

    #[test]
    fn release_threshold_and_encoder_changes_fail() {
        let e = expected();
        validate_config(&e["config"]).unwrap();
        for value in [serde_json::json!(0.5), Value::Null] {
            let mut changed = e["config"].clone();
            changed["combined_threshold"] = value;
            assert!(validate_config(&changed).is_err());
        }
        let mut model = ModelRef {
            id: e["model"]["id"].as_str().unwrap().into(),
            revision: e["model"]["revision"].as_str().unwrap().into(),
            sha256: e["model"]["sha256"].as_str().unwrap().into(),
        };
        validate_model(&model).unwrap();
        model.sha256.push('0');
        assert!(validate_model(&model).is_err());
    }

    #[test]
    #[ignore = "requires the real pinned CSC CSV"]
    fn actual_release_split_has_no_repeated_inputs_or_scenarios() {
        let csv = std::env::var("TURBO_CSC_CSV").expect("TURBO_CSC_CSV is required");
        let path = Path::new(&csv);
        validate_source(path).unwrap();
        let mut rows = super::super::corpus::load(path);
        validate_population(&rows).unwrap();
        validate_disjoint_inputs(&rows).unwrap();
        // The models-sentiment-925-1 split grouped by context id alone.
        for row in &mut rows {
            row.partition = [Partition::Test, Partition::Dev][..]
                .get(row.context as usize % 5)
                .copied()
                .unwrap_or(Partition::Train);
        }
        assert!(validate_disjoint_inputs(&rows).unwrap_err().contains("scenario texts"));
    }
}
