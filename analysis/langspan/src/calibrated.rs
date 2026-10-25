//! Calibration of language confidence for text-only runs. Boundaries and
//! labels remain those of the span detector; calibration changes no span.
use crate::{Run, SpanDetector, SpanScratch, annotate::summary};
use std::cell::RefCell;
use turbo_analysis::{Annotator, Error, ModelRef, Refs, Span, Tag, View};
use turbo_analysis_layers::{Language, LanguageSpans};
use turbo_calibration::{Calibration, Identity, Task};
use turbo_langdetect::NO_EVIDENCE;

/// The domain names the labelled population. The inference identity binds
/// all settings and the kept-language inventory; layers are not read.
pub fn calibration_identity(detector: &SpanDetector, domain: &str) -> Identity {
    let model = detector.detector().model();
    let reference = model.model_ref();
    Identity {
        model: reference.id.clone(),
        revision: reference.revision.clone(),
        sha256: reference.sha256.clone(),
        task: Task::TokenTagging,
        domain: format!("langspan-text-only-v1:{domain}"),
        labels: model.supported_languages().map(str::to_owned).collect(),
        inference_hash: detector.config_hash().wrapping_mul(0x0000_0100_0000_01b3) ^ detector.candidate_hash(),
    }
}

#[derive(Debug)]
pub struct CalibratedSpanDetector {
    detector: SpanDetector,
    profile: Calibration,
}
impl CalibratedSpanDetector {
    pub fn new(detector: SpanDetector, profile: Calibration, domain: &str) -> Result<Self, String> {
        if domain.is_empty() {
            return Err("calibration domain is empty".into());
        }
        profile.bind(&calibration_identity(&detector, domain))?;
        Ok(Self { detector, profile })
    }
    pub fn detector(&self) -> &SpanDetector {
        &self.detector
    }
    pub fn profile(&self) -> &Calibration {
        &self.profile
    }
    /// Detect text-only spans and transform the full script-restricted
    /// probability row. An `und` run retains zero confidence.
    pub fn detect(
        &self,
        text: &str,
        scratch: &mut SpanScratch,
        out: &mut Vec<Run>,
        rows: &mut Vec<Vec<f64>>,
    ) -> Result<(), String> {
        self.detector.detect_distributions(text, scratch, out, rows);
        let transform = self.profile.transform().prepare()?;
        let mut calibrated = vec![0.0; self.profile.identity().labels.len()];
        for (run, row) in out.iter_mut().zip(rows) {
            if let Some(outcome) = run.outcome {
                transform.apply(row, &mut calibrated)?;
                run.confidence = calibrated[outcome as usize];
            }
        }
        Ok(())
    }
}

type Buffers = (SpanScratch, Vec<Run>, Vec<Vec<f64>>);
thread_local! {static SCRATCH:RefCell<Buffers>=RefCell::new((SpanScratch::new(),Vec::new(),Vec::new()));}
impl Annotator for CalibratedSpanDetector {
    const ID: &'static str = "turbo-langspan:calibrated-text";
    const VERSION: &'static str = "1";
    type Reads = (View,);
    type Writes = (LanguageSpans, Language);
    fn annotate(&self, (view,): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        SCRATCH.with_borrow_mut(|(scratch, runs, rows)| {
            self.detect(view.text(), scratch, runs, rows)
                .map_err(|detail| Error::Annotator { annotator: Self::ID, detail })?;
            if runs.is_empty() {
                return Err(Error::Skipped(NO_EVIDENCE.into()));
            }
            let mut layer = LanguageSpans::new(view.id(), self.detector.languages().clone());
            for run in runs.iter() {
                let tag = Tag(run.outcome.map_or(self.detector.undetermined(), |o| o as u16));
                layer.push(Span::new(run.span.start, run.span.end), tag, run.script, run.confidence);
            }
            Ok((layer, summary(&self.detector, runs)))
        })
    }
    fn model(&self) -> Option<ModelRef> {
        Some(self.detector.detector().model().model_ref().clone())
    }
    fn config_hash(&self) -> u64 {
        self.profile.to_bytes().iter().fold(self.profile.identity().inference_hash, |h, &b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        })
    }
}
