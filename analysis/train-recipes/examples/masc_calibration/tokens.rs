//! Calibration uses one sentence per document/annotate call, independent of
//! the corpus reader's original document-level adaptive-state flags.
use turbo_analysis::{Annotator, Layer, View};
use turbo_analysis_layers::CalibratedRow;
use turbo_formats::{Field, NameSample};
use turbo_maxent::ModelFile;
use turbo_namefind::calibrated::CalibratedNameAnnotator;
use turbo_sequence::{
    PatternMode,
    namefind::{MODEL_ENTRY_NAME, NameCodec, NameFinder},
};

type Rows = (Vec<Vec<f64>>, Vec<usize>);

pub fn collect(file: &ModelFile, labels: &[String], samples: &[NameSample<'_>]) -> Result<Rows, String> {
    let model = file.maxent(MODEL_ENTRY_NAME).map_err(|e| e.to_string())?;
    let (mut rows, mut gold) = (Vec::new(), Vec::new());
    for sample in samples {
        let mut finder = NameFinder::new(file, &model, PatternMode::Utf16Units).map_err(|e| e.to_string())?;
        let words: Vec<_> = sample.tokens().iter().map(Field::as_str).collect();
        let spans: Vec<_> = sample
            .names()
            .iter()
            .map(|n| turbo_sequence::codec::Span::new(n.start as usize, n.end as usize, n.kind))
            .collect();
        let outcomes = NameCodec::Bio.encode(&spans, words.len());
        let (_, scored) = finder.find_scored(&words);
        if scored.len() != outcomes.len() {
            return Err("token score/gold row count mismatch".into());
        }
        for (row, outcome) in scored.into_iter().zip(outcomes) {
            let g = labels.iter().position(|l| l == &outcome).ok_or("gold token outcome absent from model")?;
            rows.push(row);
            gold.push(g);
        }
    }
    Ok((rows, gold))
}

pub fn visit_sidecars(
    ann: &CalibratedNameAnnotator,
    samples: &[NameSample<'_>],
    mut visit: impl FnMut(usize, CalibratedRow<'_>) -> Result<(), String>,
) -> Result<usize, String> {
    let mut count = 0;
    for sample in samples.iter().filter(|s| !s.tokens().is_empty()) {
        let (doc, sentences, tokens) = super::sample_doc(sample);
        let (_, sidecar) =
            ann.annotate((doc.get::<View>().unwrap(), &sentences, &tokens)).map_err(|e| e.to_string())?;
        if sidecar.len() != sample.tokens().len() {
            return Err("token sidecar row count mismatch".into());
        }
        for row in sidecar.rows() {
            visit(count, row)?;
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(test)]
#[path = "token_tests.rs"]
mod tests;
