//! `SymbolJoiner`: the symbol map applied to tokens.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
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
        // Only one-symbol tokens are replaced, so overlapping or unordered
        // tokens elsewhere cannot disturb the order of the writes.
        let mut symbols: Vec<(usize, usize, &'static str)> = tokens
            .spans
            .iter()
            .filter_map(|(_, s)| whole_symbol_word(&text[s.range()]).map(|w| (s.start as usize, s.end as usize, w)))
            .collect();
        symbols.sort_unstable_by_key(|&(start, _, _)| start);
        symbols.dedup_by_key(|&mut (start, _, _)| start);
        let mut out = ViewBuilder::new(text);
        for (start, end, word) in symbols {
            out.keep(start);
            out.replace(end, word);
        }
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment),))
    }
}
