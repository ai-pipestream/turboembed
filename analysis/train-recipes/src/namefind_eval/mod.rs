//! Local-only CoNLL-2003 English scoring and JVM prediction comparison.
//! Records contain digests and token indices, never corpus text or file paths.

mod record;
pub use record::{Mention, Prediction, Record, Sample, Summary};

use turbo_formats::conll::{Conll03Language, Conll03Reader, EntityTypes};
use turbo_maxent::ModelFile;
use turbo_sequence::{
    PatternMode,
    namefind::{MODEL_ENTRY_NAME, NameFinder},
};
use turbo_zoo::Sha256;

/// An explicit gold-label subset. Predictions outside it remain false positives.
/// This deliberately accepts only the listed names, unlike the stream factory's
/// substring matching. `all` includes person, organization, location and misc.
pub fn entity_types(types: &str) -> Result<EntityTypes, String> {
    if types == "all" {
        return Ok(EntityTypes::ALL);
    }
    if !["per", "org", "loc", "misc"].contains(&types) {
        return Err("types must be all, per, org, loc or misc".into());
    }
    Ok(EntityTypes::from_param(types))
}

/// Run a caller-provided model and UTF-8 corpus in document order, resetting
/// adaptive features only where the CoNLL reader requests it. No training,
/// downloads, label remapping, overlap merging or per-sentence reset is added.
pub fn evaluate(model: &[u8], corpus: &str, types: &str) -> Result<Record, String> {
    let flags = entity_types(types)?;
    let file = ModelFile::open(model.to_vec()).map_err(|e| e.to_string())?;
    let classifier = file.maxent(MODEL_ENTRY_NAME).map_err(|e| e.to_string())?;
    let mut finder = NameFinder::new(&file, &classifier, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
    let mut samples = Vec::new();
    for sample in Conll03Reader::new(corpus, Conll03Language::English, flags).map_err(|e| e.to_string())? {
        let sample = sample.map_err(|e| e.to_string())?;
        if sample.clear_adaptive_data() {
            finder.clear_adaptive_data();
        }
        let words: Vec<_> = sample.tokens().iter().map(|f| f.as_str()).collect();
        let predicted = finder
            .find(&words)
            .into_iter()
            .map(|n| Prediction {
                mention: Mention { start: n.span.start as u32, end: n.span.end as u32, label: n.span.kind },
                probability: n.prob,
            })
            .collect();
        let gold = sample
            .names()
            .iter()
            .map(|n| Mention { start: n.start, end: n.end, label: n.kind.map(str::to_owned) })
            .collect();
        samples.push(Sample {
            tokens: words.len() as u32,
            clear_adaptive_data: sample.clear_adaptive_data(),
            gold,
            predicted,
        });
    }
    let record = Record {
        format_version: 1,
        engine: "turbo".into(),
        model_sha256: Sha256::of(model).to_string(),
        corpus_sha256: Sha256::of(corpus.as_bytes()).to_string(),
        types: types.into(),
        samples,
    };
    record.validate()?;
    Ok(record)
}
