//! Scores a native language model on converted, pinned FLORES+ gold text.

use std::error::Error;
use std::path::Path;

use turbo_analysis::ModelRef;
use turbo_langdetect::NgramModel;
use turbo_train_recipes::flores::{FLORES_PLUS_REVISION, correct, per_language, read_gold};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    let [model, gold] = args.as_slice() else {
        return Err("usage: turbo-langid-flores MODEL GOLD.tsv".into());
    };
    let model_bytes = std::fs::read(model)?;
    let gold_bytes = std::fs::read(gold)?;
    let model_sha256 = turbo_sha256::sha256(&model_bytes).to_hex();
    let gold_sha256 = turbo_sha256::sha256(&gold_bytes).to_hex();
    let model_ref = ModelRef { id: path_id(Path::new(model)), revision: "1".into(), sha256: model_sha256.clone() };
    let model = NgramModel::from_bytes(&model_bytes, model_ref)?;
    let source = std::str::from_utf8(&gold_bytes)?;
    let examples = read_gold(source)?;
    let started = std::time::Instant::now();
    let correct = correct(&model, &examples);
    let elapsed_seconds = started.elapsed().as_secs_f64();
    let scores = per_language(&model, &examples);
    let supported: Vec<_> = scores.values().filter(|s| s.supported).collect();
    let supported_texts: usize = supported.iter().map(|s| s.texts).sum();
    let supported_correct: usize = supported.iter().map(|s| s.correct).sum();
    let languages: serde_json::Map<_, _> = scores
        .iter()
        .map(|(code, s)| {
            let row = serde_json::json!({ "texts": s.texts, "correct": s.correct, "supported": s.supported });
            (code.clone(), row)
        })
        .collect();
    let receipt = serde_json::json!({
        "dataset": "FLORES+",
        "revision_claim": FLORES_PLUS_REVISION,
        "model_sha256": model_sha256,
        "gold_sha256": gold_sha256,
        "texts": examples.len(),
        "correct": correct,
        "accuracy": correct as f64 / examples.len() as f64,
        "supported_texts": supported_texts,
        "supported_correct": supported_correct,
        "supported_accuracy": supported_correct as f64 / supported_texts.max(1) as f64,
        "languages": languages,
        "elapsed_seconds": elapsed_seconds,
        "texts_per_second": examples.len() as f64 / elapsed_seconds.max(f64::EPSILON),
    });
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(())
}

fn path_id(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "langid".into())
}
