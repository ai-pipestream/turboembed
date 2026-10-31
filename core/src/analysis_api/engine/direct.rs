//! Direct sentence and token results over the source view. The rule
//! splitter and tokenizer report no probability.

use turbo_analysis::{Layer, OffsetTable, Span};
use turbo_analysis_layers::{Sentences, Tokens};

use super::{AnalyzerEngine, SOURCE_VIEW, index_u32};
use crate::analysis_api::{
    AnalysisResultInner, CoordRange, LayerRef, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_SENTENCES,
    TURBO_ANALYSIS_LAYER_TOKENS, coordinate_range,
};
use crate::status::Result;

impl AnalyzerEngine {
    fn source_result(
        &self,
        text: &str,
        layer_kind: u32,
        layer: &str,
        spans: Vec<Span>,
        with_token_range: bool,
    ) -> Result<AnalysisResultInner> {
        let offsets = OffsetTable::build(text);
        let mut builder = SpanPoolBuilder::with_capacity(spans.len(), layer.len());
        let layer_ref: LayerRef = builder.layer_ref(layer_kind, layer, "", SOURCE_VIEW)?;
        for (idx, span) in spans.into_iter().enumerate() {
            let item = index_u32(idx, "analysis item count")?;
            let coords: CoordRange = coordinate_range(text, &offsets, (span.start, span.end))?;
            builder.push(
                layer_ref,
                SpanDraft {
                    item_index: item,
                    view: Some(coords),
                    source: Some(coords),
                    tokens: with_token_range.then_some((item, item + 1)),
                    ..SpanDraft::default()
                },
            )?;
        }
        Ok(builder.finish(Vec::new()))
    }

    pub fn detect_sentences(&self, text: &str) -> Result<AnalysisResultInner> {
        let spans = self.sentence_spans(text);
        self.source_result(text, TURBO_ANALYSIS_LAYER_SENTENCES, Sentences::ID, spans, false)
    }

    pub fn tokenize(&self, text: &str) -> Result<AnalysisResultInner> {
        let spans = self.token_spans(text);
        self.source_result(text, TURBO_ANALYSIS_LAYER_TOKENS, Tokens::ID, spans, true)
    }
}
