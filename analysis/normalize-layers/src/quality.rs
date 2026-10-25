//! English readability and language-independent triage over existing pipeline
//! tokens and sentences. See `docs/text-quality.md` for counting policies.

use std::cell::RefCell;

use turbo_analysis::{Annotator, Error, Layer, Refs, Span, Spans, View};
use turbo_analysis_layers::{QualityMetrics, Sentences, SpanQuality, Structure, TextQuality, Tokens};

mod lexical;
mod scan;
mod stats;

pub use scan::QualityScratch;

/// Whole-view metrics and region rows borrowed from reusable Scratch.
pub type QualityMeasurement<'a> = (QualityMetrics, &'a [(Span, QualityMetrics)]);

thread_local! {
    static SCRATCH: RefCell<QualityScratch> = RefCell::default();
}

/// Stateless annotator. Each worker thread reuses its own Scratch; callers
/// needing explicit memory control can use [`Self::measure`] with their own.
#[derive(Debug, Default, Clone, Copy)]
pub struct TextQualityAnnotator;

impl TextQualityAnnotator {
    /// Scan token spellings once and all view characters once. Numeric
    /// summaries operate on cached token facts, not on rescanned text.
    /// Returned rows borrow Scratch and are overwritten on the next call.
    /// After capacity warmup this path allocates no internal hash tables.
    pub fn measure<'s>(
        &self,
        view: &View,
        tokens: &Tokens,
        sentences: &Sentences,
        structure: Option<&Structure>,
        scratch: &'s mut QualityScratch,
    ) -> Result<QualityMeasurement<'s>, Error> {
        validate(view, &tokens.spans)?;
        validate(view, &sentences.spans)?;
        if let Some(blocks) = structure {
            validate(view, &blocks.spans)?;
            blocks.check().map_err(failed)?;
        }
        scratch.scan(view, tokens, sentences, structure)
    }
}

fn failed(detail: impl Into<String>) -> Error {
    Error::Annotator { annotator: TextQualityAnnotator::ID, detail: detail.into() }
}

fn validate<L: Layer>(view: &View, spans: &Spans<L>) -> Result<(), Error> {
    if spans.view() != view.id() {
        return Err(failed("quality inputs must use the same text view"));
    }
    spans.check(L::OVERLAP).map_err(failed)?;
    for (i, _) in spans.iter() {
        for s in spans.fragments(i) {
            if view.text().get(s.start as usize..s.end as usize).is_none() {
                return Err(failed("quality input spans must be UTF-8 slices of the view"));
            }
        }
    }
    Ok(())
}

impl Annotator for TextQualityAnnotator {
    const ID: &'static str = "turbo-normalize:text-quality";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens, Sentences, Option<Structure>);
    type Writes = (TextQuality, SpanQuality);

    fn annotate(&self, (view, tokens, sentences, structure): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let (metrics, rows) = self.measure(view, tokens, sentences, structure, &mut scratch)?;
            let mut spans = SpanQuality::new(view.id());
            for &(span, metrics) in rows {
                spans.push(span, metrics);
            }
            Ok((TextQuality::new(view.id(), view.text().len() as u32, metrics), spans))
        })
    }
}
