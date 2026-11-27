//! #2596: paired development-only fit, with no test data reads.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Seek};
use std::path::Path;

use serde_json::{Value, json};
use turbo::bundle::sha256_hex;
use turbo::head::tagger::{TokenTagger, Work};
use turbo::manifest::HeadSpec;
use turbo_trainer::{Config, Example};
use turbo_trainer_recipes::cache::{CacheRecord, SplitRecord};
use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval::{Metrics, Seen, evaluate};
use turbo_trainer_recipes::nemotron_pii::{context_contrasts, read_pinned_train_dev, shared_names};
use turbo_trainer_recipes::receipt::EncoderRecord;
use turbo_trainer_recipes::{run, safetensors};

const HIDDEN: usize = 768;
const HEAD_SHA: &str = "144017d505e46ed88b51b25443c992518a119d63608563a1ca01adc3e1e48696";
const CACHE_SHA: &str = "aed82a6bcfbb1875a4e40b841983475e8ce13c4fb0b207640b3f6ade5eae76d9";
const PUBLISHER: [&str; 4] = context_contrasts::KINDS;
const LEGACY: [&str; 2] = ["city", "first_name"];
const CONTRAST: [&str; 4] = context_contrasts::KINDS;

fn examples(
    dir: &Path,
    record: &SplitRecord,
    sentences: &[Sentence],
    labels: &BTreeMap<&str, u32>,
    keep: impl Fn(usize) -> bool,
) -> Vec<Example> {
    assert!(matches!(record.name.as_str(), "train" | "dev"), "holdout split is forbidden");
    let mut features = BufReader::new(File::open(dir.join(&record.files[0])).unwrap());
    assert_eq!(features.get_ref().metadata().unwrap().len(), record.bytes_per_file);
    assert_eq!(record.bytes_per_file, record.words * (HIDDEN * 4) as u64);
    let mut bytes = Vec::new();
    let out = sentences
        .iter()
        .enumerate()
        .filter_map(|(i, row)| {
            let n = row.words.len() * HIDDEN * 4;
            if !keep(i) {
                features.seek_relative(n as i64).unwrap();
                return None;
            }
            bytes.resize(n, 0);
            features.read_exact(&mut bytes).unwrap();
            Some(Example {
                features: bytes.as_chunks::<4>().0.iter().map(|f| f32::from_le_bytes(*f)).collect(),
                forms: row.words.clone(),
                labels: row.labels.iter().map(|name| labels[name.as_str()]).collect(),
                language: None,
            })
        })
        .collect();
    assert_eq!(
        features.stream_position().unwrap(),
        sentences.iter().map(|s| s.words.len() * HIDDEN * 4).sum::<usize>() as u64,
        "feature rows must end at the expected offset",
    );
    out
}

fn synthetic(
    dir: &Path,
    split: &str,
    publisher: &EncoderRecord,
    sentences: &[Sentence],
    labels: &BTreeMap<&str, u32>,
) -> Vec<Example> {
    assert!(matches!(split, "train" | "dev"), "holdout split is forbidden");
    let (encoder, rec): (EncoderRecord, SplitRecord) =
        serde_json::from_slice(&fs::read(dir.join(format!("{split}.json"))).unwrap()).unwrap();
    assert_eq!(encoder.manifest_sha256, publisher.manifest_sha256);
    assert_eq!(encoder.layers, publisher.layers);
    assert_eq!(encoder.hidden, HIDDEN);
    assert_eq!(encoder.backend, "cpu");
    assert_eq!(encoder.precision, publisher.precision);
    assert_eq!(rec.name, split);
    assert_eq!(rec.sentences as usize, sentences.len());
    assert_eq!(rec.words as usize, sentences.iter().map(|s| s.words.len()).sum::<usize>());
    examples(dir, &rec, sentences, labels, |_| true)
}

fn score(tagger: &TokenTagger, gold: &[Sentence], encoded: &[Example]) -> Metrics {
    assert_eq!(gold.len(), encoded.len());
    let (mut scores, mut tags, mut work) = (Vec::new(), Vec::new(), Work::default());
    let predicted = encoded
        .iter()
        .map(|row| {
            let forms = row.forms().unwrap();
            let words: Vec<u32> = (0..forms.len() as u32).collect();
            tagger.scores_forms(&row.features, &words, Some(&forms), &mut scores, &mut work).unwrap();
            tagger.decode(&scores, forms.len(), &mut tags).unwrap();
            tags.iter().map(|&tag| tagger.labels()[tag as usize].clone()).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    evaluate(Task::Bio, gold, &predicted, &Seen::default(), None)
}

fn select(metrics: &Metrics, names: &[&str]) -> BTreeMap<String, f64> {
    let Metrics::Spans { all, per_type, .. } = metrics else { panic!("expected PII BIO spans") };
    let mut out = BTreeMap::from([("micro".into(), all.f1)]);
    out.extend(names.iter().map(|&name| {
        let score = &per_type[name];
        assert!(score.gold > 0, "no gold spans for {name}");
        (name.to_owned(), score.f1)
    }));
    out
}

fn gate(control: &[Metrics], variant: &[Metrics]) -> Value {
    let mut checks = Vec::new();
    let mut pass = true;
    for (i, (set, names)) in
        [("publisher", PUBLISHER.as_slice()), ("legacy", LEGACY.as_slice()), ("contrast", CONTRAST.as_slice())]
            .into_iter()
            .enumerate()
    {
        let (before, after) = (select(&control[i], names), select(&variant[i], names));
        let first = if i == 0 { before["micro"] <= after["micro"] } else { true };
        let each = names.iter().all(|&name| before[name] <= after[name]);
        let gain = if i == 2 {
            Some(names.iter().map(|&name| after[name] - before[name]).sum::<f64>() / names.len() as f64)
        } else {
            None
        };
        let passed = first && each && gain.is_none_or(|g| g >= 0.02);
        pass &= passed;
        checks.push(
            json!({"set": set, "control": before, "variant": after, "contrast_macro_gain": gain, "pass": passed}),
        );
    }
    json!({"dev_gate_passed": pass, "sets": checks, "holdout_scored": false})
}

fn main() {
    let args = std::env::args_os().skip(1).collect::<Vec<_>>();
    assert_eq!(
        args.len(),
        4,
        "pii_contrast_fit <publisher-corpus> <original-cache> <original-head.json> <experiment-dir>"
    );
    let publisher = Path::new(&args[0]);
    let cache_dir = Path::new(&args[1]);
    let experiment = Path::new(&args[3]);
    let spec_bytes = fs::read(&args[2]).unwrap();
    assert_eq!(sha256_hex(&spec_bytes), HEAD_SHA);
    let spec: HeadSpec = serde_json::from_slice(&spec_bytes).unwrap();
    assert_eq!(spec.hash_embed.as_ref().unwrap().dim, 128);
    let labels: BTreeMap<_, _> = spec.labels.iter().enumerate().map(|(i, s)| (s.as_str(), i as u32)).collect();
    let cache_bytes = fs::read(cache_dir.join("cache.json")).unwrap();
    assert_eq!(sha256_hex(&cache_bytes), CACHE_SHA);
    let cache: CacheRecord = serde_json::from_slice(&cache_bytes).unwrap();
    assert_eq!(cache.layers, [22]);
    let publisher_record = |name| cache.splits.iter().find(|r| r.name == name).unwrap();
    let (publisher_train, publisher_dev) = read_pinned_train_dev(publisher).unwrap();
    assert_eq!((publisher_train.len(), publisher_dev.len()), (90_000, 10_000));
    // Same sampled original publisher prefix for both arms; keep the cache
    // offsets of original rows even though we only fit one row in four.
    let mut train = examples(cache_dir, publisher_record("train"), &publisher_train, &labels, |i| i % 4 == 0);
    assert_eq!(train.len(), 22_500);
    drop(publisher_train);
    let dev = examples(cache_dir, publisher_record("dev"), &publisher_dev, &labels, |_| true);
    let (legacy_train, legacy_dev) = shared_names::documents().unwrap();
    let (contrast_train, contrast_dev) =
        (context_contrasts::documents(false).unwrap(), context_contrasts::documents(true).unwrap());
    let (legacy, contrast) = (experiment.join("legacy"), experiment.join("contrast"));
    let dev_sets = [
        (&publisher_dev, dev),
        (&legacy_dev, synthetic(&legacy, "dev", &cache.encoder, &legacy_dev, &labels)),
        (&contrast_dev, synthetic(&contrast, "dev", &cache.encoder, &contrast_dev, &labels)),
    ];
    let config = Config {
        seed: 1,
        epochs: 6,
        batch: 128,
        lr: 0.004,
        patience: 4,
        threads: 6,
        verbose: true,
        ..Config::default()
    };
    let mut fitted = Vec::new();
    for (name, directory, rows) in [("control", legacy, legacy_train), ("candidate", contrast, contrast_train)] {
        train.truncate(22_500);
        train.extend(synthetic(&directory, "train", &cache.encoder, &rows, &labels));
        assert_eq!(train.len(), 28_500);
        let (model, fit) = turbo_trainer::fit(&spec, HIDDEN, &train, &dev_sets[0].1, &config).unwrap();
        let tagger = model.to_token_tagger().unwrap();
        let metrics = dev_sets.iter().map(|(gold, features)| score(&tagger, gold, features)).collect::<Vec<_>>();
        let output = experiment.join(name);
        fs::create_dir(&output).unwrap();
        run::write(&output, "head.json", &spec_bytes).unwrap();
        let tensors = run::named(&spec, HIDDEN, model.tensors());
        let file = run::write(&output, "head.safetensors", &safetensors::write_with(&tensors, &[])).unwrap();
        let receipt = json!({
            "encoder": cache.encoder, "split": "dev", "epochs": fit.dev, "best_epoch": fit.best_epoch + 1,
            "seconds": fit.epoch_seconds, "head": file, "dev_metrics": metrics, "training_documents": train.len(),
        });
        run::write(&output, "receipt.json", &serde_json::to_vec_pretty(&receipt).unwrap()).unwrap();
        println!("{name}: selected epoch {}; head {}", fit.best_epoch + 1, file.sha256);
        fitted.push(metrics);
    }
    let outcome = gate(&fitted[0], &fitted[1]);
    run::write(experiment, "dev-gate.json", &serde_json::to_vec_pretty(&outcome).unwrap()).unwrap();
    println!("{}", serde_json::to_string_pretty(&outcome).unwrap());
    if outcome["dev_gate_passed"] != true {
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_trainer_recipes::eval::SpanScore;

    fn metrics(names: &[&str], f1: f64) -> Metrics {
        let score = SpanScore { gold: 10, f1, ..SpanScore::default() };
        Metrics::Spans {
            all: score,
            per_type: names.iter().map(|&n| (n.to_owned(), score)).collect(),
            coarse: BTreeMap::new(),
        }
    }

    #[test]
    fn a_gain_cannot_hide_a_drop_on_any_guard() {
        let publisher = metrics(&PUBLISHER, 0.6);
        let legacy = metrics(&LEGACY, 0.8);
        let old = [publisher.clone(), legacy.clone(), metrics(&CONTRAST, 0.5)];
        let better = metrics(&CONTRAST, 0.53);
        assert_eq!(gate(&old, &[publisher.clone(), legacy.clone(), better.clone()])["dev_gate_passed"], true);
        assert_eq!(gate(&old, &[metrics(&PUBLISHER, 0.59), legacy.clone(), better.clone()])["dev_gate_passed"], false);
        assert_eq!(gate(&old, &[publisher.clone(), metrics(&LEGACY, 0.79), better])["dev_gate_passed"], false);
        assert_eq!(gate(&old, &[publisher, legacy, metrics(&CONTRAST, 0.51)])["dev_gate_passed"], false);
    }
}
