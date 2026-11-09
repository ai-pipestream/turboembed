//! Syntactic word rows (`TURBO_ANALYSIS_LAYER_WORDS`): one row a word,
//! its text range the bytes of its token it tiles (`"ca"` of `"can't"`)
//! or else its whole token, its token range the tokens it belongs to,
//! and a label only when its form is not the text of that range (a
//! fusional `"del"` read as `"de"` and `"el"`), so a word whose form is
//! in the text puts no copy of it in the pool.

use turbo_analysis::{Document, Ix, KeyId, Span, ViewId};
use turbo_analysis_layers::{Tokens, Words};

use super::{Mapper, get_by_id, index_u32, layer_ref};
use crate::analysis_api::{
    CoordRange, LayerRef, SpanDraft, SpanPoolBuilder, TURBO_ANALYSIS_LAYER_WORDS, coordinate_range,
};
use crate::status::{CAPACITY, Error, Result};

/// The rows of `words` over `tokens`, each word's text range given its
/// view and source coordinates by `coords`. A word outside its tokens is
/// refused by `outside`.
pub(crate) fn push_words(
    builder: &mut SpanPoolBuilder,
    layer: LayerRef,
    words: &Words,
    tokens: &Tokens,
    coords: &dyn Fn(Span) -> Result<(CoordRange, CoordRange)>,
    outside: &dyn Fn(String) -> Error,
) -> Result<()> {
    let n = tokens.spans.len();
    for (i, r) in words.tokens.iter().enumerate() {
        let (start, end) = (r.start.get(), r.end.get());
        if start >= end || end > n {
            return Err(outside(format!("word {i} token range {start}..{end} is outside its {n} tokens")));
        }
        let w = Ix::new(index_u32(i, "word count")?);
        let explicit = words.form.as_ref().and_then(|f| f[i].as_deref());
        let span = words.subspan_of(w).unwrap_or_else(|| words.token_span(w, tokens));
        let (view, source) = coords(span)?;
        let label = explicit.map(|f| builder.string_ref(f)).transpose()?;
        builder.push(
            layer,
            SpanDraft {
                item_index: w.get() as u32,
                view: Some(view),
                source: Some(source),
                tokens: Some((index_u32(start, "word token range")?, index_u32(end, "word token range")?)),
                label,
                ..SpanDraft::default()
            },
        )?;
    }
    Ok(())
}

/// The rows of words the analyzer made over `text`'s own tokens.
pub(crate) fn source_words(builder: &mut SpanPoolBuilder, text: &str, words: &Words, tokens: &Tokens) -> Result<()> {
    let offsets = turbo_analysis::OffsetTable::build(text);
    let layer = builder.layer_ref(
        TURBO_ANALYSIS_LAYER_WORDS,
        <Words as turbo_analysis::Layer>::ID,
        "",
        ViewId::SOURCE.0 as u32,
    )?;
    let coords = |span: Span| {
        let c = coordinate_range(text, &offsets, (span.start, span.end))?;
        Ok((c, c))
    };
    push_words(builder, layer, words, tokens, &coords, &|m| Error::new(CAPACITY, m))
}

/// The rows of the words layer `id` of `doc`, over the tokens it indexes.
pub(super) fn extract_words(
    builder: &mut SpanPoolBuilder,
    mapper: &Mapper<'_>,
    doc: &Document,
    id: KeyId,
) -> Result<()> {
    let Some(layer) = get_by_id::<Words>(doc, id) else { return Ok(()) };
    let Some(tokens) = doc.get_key::<Tokens>(&layer.tokens_key) else {
        return Err(mapper.invalid(format!("words layer {id} indexes tokens the graph lacks")));
    };
    let view = tokens.spans.view();
    let layer_ref = layer_ref(builder, TURBO_ANALYSIS_LAYER_WORDS, id, view)?;
    let coords = |span: Span| mapper.coordinates(view, span);
    push_words(builder, layer_ref, layer, tokens, &coords, &|m| mapper.invalid(format!("words layer {id}: {m}")))
}
