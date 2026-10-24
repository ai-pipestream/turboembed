//! `SymbolJoiner`: the symbol map applied to tokens.

use turbo_analysis::{Annotator, Error, Overlap, Refs, View, ViewBuilder};
use turbo_analysis_layers::Tokens;

use super::whole_symbol_word;

/// Replaces every token that is exactly one mapped symbol with its word,
/// in a new view: with tokens `Dungeons`, `&`, `Dragons` the view reads
/// `Dungeons and Dragons`. Tokens that merely contain a symbol (`R&D`)
/// are left alone.
///
/// It reads a `View` and the `Tokens` made on it, and writes a `View`.
/// The tokens of the new view are found by mapping the old ones through
/// the view's alignment, or by tokenizing it again.
///
/// OpenNLP counterpart: `SymbolJoinerCharSequenceNormalizer`, called by
/// the caller once per token. The `symbol-joiner` add-on is the same
/// class; neither joins symbol sequences.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SymbolJoiner;

impl Annotator for SymbolJoiner {
    const ID: &'static str = "turbo:normalize:symbol-joiner";
    const VERSION: &'static str = "2";
    type Reads = (View, Tokens);
    type Writes = (View,);

    fn annotate(&self, (src, tokens): Refs<'_, Self>) -> Result<(View,), Error> {
        if tokens.spans.view() != src.id() {
            return Err(Error::Annotator {
                annotator: Self::ID,
                detail: format!("tokens are in view {}, the text in view {}", tokens.spans.view().0, src.id().0),
            });
        }
        let text = src.text();
        let invalid = |detail| Error::Annotator { annotator: Self::ID, detail };
        tokens.spans.check(Overlap::None).map_err(invalid)?;
        let mut symbols = Vec::new();
        for (ix, span) in tokens.spans.iter() {
            if tokens.spans.is_discontinuous(ix) {
                return Err(invalid(format!("token {ix:?} must be contiguous")));
            }
            let token = text
                .get(span.range())
                .ok_or_else(|| invalid(format!("token {ix:?} must be a UTF-8 span within the view")))?;
            if let Some(word) = whole_symbol_word(token) {
                symbols.push((span.start as usize, span.end as usize, word));
            }
        }
        let mut out = ViewBuilder::new(text);
        for (start, end, word) in symbols {
            out.keep(start);
            out.replace(end, word);
        }
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment),))
    }
}
