//! Frozen release verification, separate from training and threshold selection.
use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use turbo_analysis::ModelRef;
use turbo_analysis_embed::ClassifierHead;

use super::{corpus::Response, metrics::Counts};

pub fn expected() -> Value {
    serde_json::from_str(include_str!("release.json")).unwrap()
}

pub fn validate_config(config: &Value) -> Result<(), String> {
    if config != &expected()["config"] {
        return Err("CSC release configuration differs from models-sentiment-925-1".into());
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
        ("train", rows.iter().filter(|r| r.context % 5 > 1).count()),
        ("dev", rows.iter().filter(|r| r.context % 5 == 1).count()),
        ("test", rows.iter().filter(|r| r.context % 5 == 0).count()),
        ("test_positive", rows.iter().filter(|r| r.context % 5 == 0 && r.sarcastic).count()),
    ];
    for (name, count) in counts {
        if p[name].as_u64() != Some(count as u64) {
            return Err(format!("CSC release {name} count differs: {count}"));
        }
    }
    Ok(())
}

/// The model sees response text, so context isolation also needs input isolation.
pub fn validate_disjoint_inputs(rows: &[Response]) -> Result<(), String> {
    let mut partitions: BTreeMap<&str, u8> = BTreeMap::new();
    for row in rows {
        let partition = match row.context % 5 {
            0 => 1,
            1 => 2,
            _ => 4,
        };
        *partitions.entry(&row.text).or_default() |= partition;
    }
    let overlap = partitions.values().filter(|bits| bits.count_ones() > 1).count();
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
    #[ignore = "requires the real pinned CSC CSV; current released split is blocked"]
    fn actual_release_split_has_no_repeated_inputs() {
        let csv = std::env::var("TURBO_CSC_CSV").expect("TURBO_CSC_CSV is required");
        let path = Path::new(&csv);
        validate_source(path).unwrap();
        let rows = super::super::corpus::load(path);
        validate_population(&rows).unwrap();
        validate_disjoint_inputs(&rows).unwrap();
    }
}
