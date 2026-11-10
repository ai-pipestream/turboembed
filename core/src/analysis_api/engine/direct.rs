//! Direct sentence and token results over the source view. The rule
//! splitter and tokenizer report no probability, and neither does a
//! sentence boundary head. With a token split head, tokenizing also
//! writes each token's syntactic words.

use turbo_analysis::{Layer, OffsetTable, Span};
use turbo_analysis_layers::{Sentences, Tokens};

use super::layers::{tokens_of, word_layer};
use super::{AnalyzerEngine, SOURCE_VIEW, forms, index_u32, sequences};
use crate::analysis_api::graph::words::source_words;
use crate::analysis_api::heads::Want;
use crate::analysis_api::{
    AnalysisResultInner, CoordRange, LayerRef, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_SENTENCES,
    TURBO_ANALYSIS_LAYER_TOKENS, coordinate_range,
};
use crate::status::Result;

/// One row a span of `spans` into `builder`, as layer `layer` of the
/// source view.
fn push_source_rows(
    builder: &mut SpanPoolBuilder,
    text: &str,
    offsets: &OffsetTable,
    (layer_kind, layer): (u32, &str),
    spans: &[Span],
    with_token_range: bool,
) -> Result<()> {
    let layer_ref: LayerRef = builder.layer_ref(layer_kind, layer, "", SOURCE_VIEW)?;
    for (idx, span) in spans.iter().enumerate() {
        let item = index_u32(idx, "analysis item count")?;
        let coords: CoordRange = coordinate_range(text, offsets, (span.start, span.end))?;
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
    Ok(())
}

impl AnalyzerEngine {
    pub fn detect_sentences(&self, text: &str) -> Result<AnalysisResultInner> {
        let spans = self.sentence_spans(text)?;
        let mut builder = SpanPoolBuilder::with_capacity(spans.len(), Sentences::ID.len());
        let layer = (TURBO_ANALYSIS_LAYER_SENTENCES, Sentences::ID);
        push_source_rows(&mut builder, text, &OffsetTable::build(text), layer, &spans, false)?;
        Ok(builder.finish(Vec::new()))
    }

    pub fn tokenize(&self, text: &str) -> Result<AnalysisResultInner> {
        let spans = self.token_spans(text);
        let mut builder = SpanPoolBuilder::with_capacity(spans.len(), Tokens::ID.len());
        let layer = (TURBO_ANALYSIS_LAYER_TOKENS, Tokens::ID);
        push_source_rows(&mut builder, text, &OffsetTable::build(text), layer, &spans, true)?;
        if let Some(heads) = self.heads.as_ref().filter(|h| h.has().words) {
            let tok_strs = forms(text, &spans)?;
            let seqs = sequences(&spans, &self.sentences_over(text, &spans)?);
            let words: Vec<Vec<&str>> = seqs.iter().map(|s| s.iter().map(|&i| tok_strs[i]).collect()).collect();
            let split = heads.run(&words, &Want { words: true, ..Want::default() })?.words.expect("asked for");
            let layer = word_layer(text, &spans, &seqs, split)?;
            source_words(&mut builder, text, &layer, &tokens_of(&spans))?;
        }
        Ok(builder.finish(Vec::new()))
    }
}
