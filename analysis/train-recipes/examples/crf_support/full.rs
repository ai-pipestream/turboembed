use std::error::Error;
use std::path::Path;
use turbo_eval::model_gate::Counts;
use turbo_formats::{
    NameSample,
    evaluate::{FMeasure, LabeledSpan},
};
use turbo_maxent::ModelFile;
use turbo_sequence::{PatternMode, crf::CrfSequenceModel, namefind::CrfNameFinder};

pub fn score_crf(
    model: &CrfSequenceModel,
    bytes: &[u8],
    samples: &[NameSample<'_>],
) -> Result<Vec<Counts>, Box<dyn Error>> {
    let file = ModelFile::open(bytes.to_vec())?;
    let mut finder = CrfNameFinder::from_file(model, &file, PatternMode::Utf16Units)?;
    let mut units = Vec::new();
    for sample in samples {
        let words: Vec<_> = sample.tokens().iter().map(|w| w.as_str()).collect();
        let found = finder.find(&words)?;
        let gold: Vec<_> = sample.names().iter().map(|n| LabeledSpan::new(n.start, n.end, n.kind)).collect();
        let predicted: Vec<_> = found
            .iter()
            .map(|n| LabeledSpan::new(n.span.start as u32, n.span.end as u32, n.span.kind.as_deref()))
            .collect();
        let mut measure = FMeasure::default();
        measure.update(&gold, &predicted);
        units.push(turbo_train_recipes::spans::counts(&measure));
    }
    Ok(units)
}

pub fn totals(units: &[Counts]) -> serde_json::Value {
    let (mut tp, mut selected, mut target) = (0, 0, 0);
    for c in units {
        tp += c.true_positives;
        selected += c.selected;
        target += c.target;
    }
    let ratio = |n: u64, d: u64| if d == 0 { 0.0 } else { n as f64 / d as f64 };
    serde_json::json!({"true_positives": tp, "selected": selected, "target": target,
        "precision": ratio(tp, selected), "recall": ratio(tp, target), "f1": ratio(2*tp, selected+target)})
}

pub fn add(total: &mut Vec<Counts>, units: &[Counts]) -> Result<(), Box<dyn Error>> {
    if total.is_empty() {
        total.resize(units.len(), Counts::default());
    }
    if total.len() != units.len() {
        return Err("entity recipes have different sample counts".into());
    }
    for (a, b) in total.iter_mut().zip(units) {
        a.true_positives += b.true_positives;
        a.selected += b.selected;
        a.target += b.target;
    }
    Ok(())
}

pub fn input_bytes(samples: &[NameSample<'_>]) -> Result<Vec<u8>, Box<dyn Error>> {
    let rows: Vec<_> = samples
        .iter()
        .map(|s| {
            serde_json::json!({
                "tokens": s.tokens().iter().map(|w| w.as_str()).collect::<Vec<_>>(),
                "gold": s.names().iter().map(|n| (n.start,n.end,n.kind)).collect::<Vec<_>>()
            })
        })
        .collect();
    Ok(serde_json::to_vec(&rows)?)
}

pub fn write_inputs(path: &Path, samples: &[NameSample<'_>]) -> Result<(), Box<dyn Error>> {
    std::fs::write(path, input_bytes(samples)?)?;
    Ok(())
}
