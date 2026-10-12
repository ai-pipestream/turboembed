//! `SymbolWords`: the symbol map applied to standalone symbols in text.

use turbo_analysis::ViewBuilder;

use super::symbol_word;
use crate::{Normalizer, tables};

/// Replaces each mapped symbol that stands alone, with whitespace (the 25
/// characters of OpenNLP's whitespace table) or the edge of the text on
/// both sides, with its word: `Dungeons & Dragons` becomes
/// `Dungeons and Dragons`, while `R&D`, `TSR®` and `©\u{FE0F}` are left
/// alone.
///
/// OpenNLP applies the same map only to an input that is the symbol and
/// nothing else (`SymbolJoinerCharSequenceNormalizer`); this normalizer
/// finds the symbols in running text instead, so a caller does not have to
/// tokenize first.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SymbolWords;

/// Whether a byte can start a mapped symbol: one of the four ASCII
/// symbols, or the lead byte of the two-byte (U+00A7..U+00B6) and
/// three-byte (U+2122) ones.
fn may_start_symbol(b: u8) -> bool {
    matches!(b, b'&' | b'+' | b'@' | b'%' | 0xC2 | 0xE2)
}

impl Normalizer for SymbolWords {
    const ID: &'static str = "turbo:normalize:symbol-words";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        let bytes = text.as_bytes();
        for (i, &b) in bytes.iter().enumerate() {
            if !may_start_symbol(b) {
                continue;
            }
            let c = text[i..].chars().next().expect("a lead byte starts a character");
            let Some(word) = symbol_word(c) else { continue };
            let end = i + c.len_utf8();
            let alone_before = text[..i].chars().next_back().is_none_or(tables::is_whitespace);
            let alone_after = text[end..].chars().next().is_none_or(tables::is_whitespace);
            if alone_before && alone_after {
                out.keep(i);
                out.replace(end, word);
            }
        }
    }
}
