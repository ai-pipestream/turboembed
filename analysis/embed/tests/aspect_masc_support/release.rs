//! Frozen expectations from models-sentiment-924-1, not a quality target.
use std::path::Path;

use serde_json::Value;
use turbo_analysis::ModelRef;
use turbo_analysis_embed::ClassifierHead;

use super::{corpus::CorpusDoc, metrics::Metrics};

pub fn expected() -> Value {
    serde_json::from_str(include_str!("release.json")).unwrap()
}

pub fn validate_config(config: &Value) -> Result<(), String> {
    if config != &expected()["config"] {
        return Err("aspect release configuration differs from models-sentiment-924-1".into());
    }
    Ok(())
}

pub fn validate_model(model: &ModelRef) -> Result<(), String> {
    let expected = expected();
    let m = &expected["model"];
    if model.id != m["id"].as_str().unwrap()
        || model.revision != m["revision"].as_str().unwrap()
        || model.sha256 != m["sha256"].as_str().unwrap()
    {
        return Err("aspect release encoder identity differs".into());
    }
    Ok(())
}

pub fn load(dir: &Path, model: &ModelRef) -> Result<(ClassifierHead, ClassifierHead, f64), String> {
    validate_model(model)?;
    let config: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("aspect-sentiment-1.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    validate_config(&config)?;
    let load_head = |name: &str| -> Result<ClassifierHead, String> {
        let bytes = std::fs::read(dir.join(name)).map_err(|e| e.to_string())?;
        validate_head(&bytes, name, model)
    };
    Ok((
        load_head(config["bio_head"].as_str().unwrap())?,
        load_head(config["polarity_head"].as_str().unwrap())?,
        config["bio_threshold"].as_f64().unwrap(),
    ))
}

fn validate_head(bytes: &[u8], name: &str, model: &ModelRef) -> Result<ClassifierHead, String> {
    if turbo::bundle::sha256_hex(bytes) != expected()["head_sha256"][name].as_str().unwrap() {
        return Err(format!("aspect release file digest differs: {name}"));
    }
    let head = ClassifierHead::from_bytes(bytes)?;
    if head.model() != model {
        return Err(format!("aspect release head uses another encoder: {name}"));
    }
    Ok(head)
}

pub fn population_digest(docs: &[CorpusDoc]) -> String {
    fn text(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(docs.len() as u64).to_le_bytes());
    for doc in docs {
        text(&mut bytes, &doc.id);
        text(&mut bytes, doc.view.text());
        bytes.extend_from_slice(&(doc.sentences.spans.len() as u64).to_le_bytes());
        for (_, span) in doc.sentences.spans.iter() {
            bytes.extend_from_slice(&span.start.to_le_bytes());
            bytes.extend_from_slice(&span.end.to_le_bytes());
        }
        bytes.extend_from_slice(&(doc.tokens.spans.len() as u64).to_le_bytes());
        for (_, span) in doc.tokens.spans.iter() {
            bytes.extend_from_slice(&span.start.to_le_bytes());
            bytes.extend_from_slice(&span.end.to_le_bytes());
        }
        bytes.extend_from_slice(&(doc.aspects.len() as u64).to_le_bytes());
        for (range, category, polarity) in &doc.aspects {
            bytes.extend_from_slice(&(range.start.get() as u64).to_le_bytes());
            bytes.extend_from_slice(&(range.end.get() as u64).to_le_bytes());
            text(&mut bytes, category);
            text(&mut bytes, polarity);
        }
        bytes.extend_from_slice(&(doc.opinions.len() as u64).to_le_bytes());
        for (range, polarity) in &doc.opinions {
            bytes.extend_from_slice(&(range.start.get() as u64).to_le_bytes());
            bytes.extend_from_slice(&(range.end.get() as u64).to_le_bytes());
            text(&mut bytes, polarity);
        }
    }
    turbo::bundle::sha256_hex(&bytes)
}

pub fn validate_population(docs: &[CorpusDoc], split: usize) -> Result<(), String> {
    let e = expected();
    let ids: Vec<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    let wanted: Vec<&str> = e["documents"].as_array().unwrap().iter().map(|s| s.as_str().unwrap()).collect();
    if population_digest(docs) != e["population_sha256"].as_str().unwrap() {
        return Err("aspect release input text or annotations differ".into());
    }
    if ids != wanted || split != 56 || docs[split..].iter().map(|d| d.aspects.len()).sum::<usize>() != 127 {
        return Err("aspect release train/test population differs".into());
    }
    Ok(())
}

pub fn validate_metrics(actual: &Metrics, expected: &Value) -> Result<(), String> {
    for (name, count) in [
        ("aspect_tp", actual.aspect_tp),
        ("aspect_fp", actual.aspect_fp),
        ("aspect_fn", actual.aspect_fn),
        ("polarity_correct", actual.polarity_correct),
        ("polarity_total", actual.polarity_total),
    ] {
        if expected[name].as_u64() != Some(count as u64) {
            return Err(format!("aspect release {name}: computed {count}, expected {}", expected[name]));
        }
    }
    for (name, metric) in [("aspect_f1", actual.f1()), ("polarity_accuracy", actual.polarity_accuracy())] {
        let wanted = expected[name].as_f64().ok_or_else(|| format!("missing release {name}"))?;
        if !metric.is_finite() || !wanted.is_finite() || (metric - wanted).abs() > 0.0000005 {
            return Err(format!("aspect release {name}: computed {metric}, expected {wanted}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn release_counts() -> Metrics {
        Metrics { aspect_tp: 4, aspect_fp: 464, aspect_fn: 121, polarity_correct: 4, polarity_total: 4 }
    }

    #[test]
    #[ignore = "requires the real pinned MASC data directory"]
    fn actual_release_population_accepts_and_changed_annotations_fail() {
        let root = std::env::var("TURBO_MASC_ROOT").expect("TURBO_MASC_ROOT is required");
        let mut docs = super::super::corpus::load(Path::new(&root));
        validate_population(&docs, 56).unwrap();
        docs.swap(0, 1);
        assert!(validate_population(&docs, 56).is_err());
        docs.swap(0, 1);
        docs[0].aspects.pop().unwrap();
        assert!(validate_population(&docs, 56).is_err());
    }

    #[test]
    #[ignore = "requires the downloaded published aspect head files"]
    fn actual_release_files_accept_and_corrupt_bytes_fail() {
        let dir = std::env::var("TURBO_ASPECT_HEAD").expect("TURBO_ASPECT_HEAD is required");
        let e = expected();
        let m = &e["model"];
        let model = ModelRef {
            id: m["id"].as_str().unwrap().into(),
            revision: m["revision"].as_str().unwrap().into(),
            sha256: m["sha256"].as_str().unwrap().into(),
        };
        load(Path::new(&dir), &model).unwrap();
        for name in ["aspect-bio-1.thead", "aspect-polarity-1.thead"] {
            let mut bytes = std::fs::read(Path::new(&dir).join(name)).unwrap();
            bytes[0] ^= 1;
            assert!(validate_head(&bytes, name, &model).is_err());
        }
    }

    #[test]
    fn archived_counts_pass_and_prediction_regressions_fail() {
        let e = expected();
        let metrics = &e["config"]["evaluation"];
        validate_metrics(&release_counts(), metrics).unwrap();
        let mut changed = release_counts();
        changed.aspect_fp += 1;
        assert!(validate_metrics(&changed, metrics).is_err());
        let mut changed = metrics.clone();
        changed["aspect_f1"] = serde_json::json!(0.99);
        assert!(validate_metrics(&release_counts(), &changed).is_err());
        validate_metrics(&super::super::metrics::empty(), &e["rules"]).unwrap_err();
    }

    #[test]
    fn configuration_and_encoder_changes_fail_before_inference() {
        let e = expected();
        validate_config(&e["config"]).unwrap();
        for (key, value) in [("bio_threshold", serde_json::json!(0.5)), ("bio_threshold", Value::Null)] {
            let mut altered = e["config"].clone();
            altered[key] = value;
            assert!(validate_config(&altered).is_err());
        }
        let mut m = ModelRef {
            id: e["model"]["id"].as_str().unwrap().into(),
            revision: e["model"]["revision"].as_str().unwrap().into(),
            sha256: e["model"]["sha256"].as_str().unwrap().into(),
        };
        validate_model(&m).unwrap();
        m.revision.push_str("-changed");
        assert!(validate_model(&m).is_err());
    }
}
