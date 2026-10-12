//! `EnglishContractionsWithPos`: contraction expansion that resolves `'s`
//! and `'d` from part-of-speech tags.

use turbo_analysis::{Annotator, Error, Refs, View, ViewBuilder};
use turbo_analysis_layers::{PosTags, Tokens};

use super::rules::{Ambiguous, Reading};
use super::scan;

/// [`EnglishContractions`](super::EnglishContractions), and `'s` and `'d`
/// resolved by the Penn Treebank tags of the tokens:
///
/// - `'s` becomes ` has` when the token holding the apostrophe is tagged
///   `VBZ` and the next token `VBN` (`he's gone`), and ` is` when the
///   token is tagged `VBZ` and the next token is not `VBN` (`he's here`).
///   Any other tag (`POS` for a possessive) leaves it alone.
/// - `'d` becomes ` would` when the next token is tagged `VB` (`I'd go`)
///   and ` had` when it is tagged `VBN` (`I'd gone`). Anything else leaves
///   it alone.
///
/// The token holding the apostrophe is `'s` itself under Penn Treebank
/// tokenization, or the whole word under a tokenizer that keeps it whole;
/// either works. The same word-boundary and ASCII-letter rules apply as
/// for the plain normalizer.
///
/// It reads the `View`, and `Tokens` and `PosTags` when the document has
/// them; without both it writes what the plain normalizer writes. The
/// tokens must be in the view it reads.
///
/// OpenNLP has no counterpart: its normalizer leaves every `'s` and `'d`
/// alone.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct EnglishContractionsWithPos;

/// The tokens and their tags, as the resolver reads them.
struct Tagged<'d> {
    tokens: &'d Tokens,
    tags: &'d PosTags,
}

impl Tagged<'_> {
    /// The index of the token whose span holds byte `at`.
    fn token_at(&self, at: usize) -> Option<usize> {
        let (start, end) = self.tokens.spans.columns();
        let at = at as u32;
        let k = end.partition_point(|&e| e <= at);
        (k < start.len() && start[k] <= at).then_some(k)
    }

    fn label(&self, k: usize) -> Option<&str> {
        self.tags.tags.get(k).map(|&t| self.tags.tagset.label(t))
    }

    fn resolve(&self, apostrophe: usize, which: Ambiguous) -> Option<Reading> {
        let k = self.token_at(apostrophe)?;
        let next = self.label(k + 1);
        match which {
            Ambiguous::S if self.label(k) == Some("VBZ") => {
                Some(if next == Some("VBN") { Reading::Has } else { Reading::Is })
            }
            Ambiguous::S => None,
            Ambiguous::D => match next {
                Some("VB") => Some(Reading::Would),
                Some("VBN") => Some(Reading::Had),
                _ => None,
            },
        }
    }
}

impl Annotator for EnglishContractionsWithPos {
    const ID: &'static str = "turbo:normalize:english-contractions-pos";
    type Reads = (View, Option<Tokens>, Option<PosTags>);
    type Writes = (View,);

    fn annotate(&self, (src, tokens, tags): Refs<'_, Self>) -> Result<(View,), Error> {
        let text = src.text();
        let mut out = ViewBuilder::new(text);
        match (tokens, tags) {
            (Some(tokens), Some(tags)) => {
                if tokens.spans.view() != src.id() {
                    return Err(Error::Annotator {
                        annotator: Self::ID,
                        detail: format!(
                            "tokens are in view {}, the text in view {}",
                            tokens.spans.view().0,
                            src.id().0
                        ),
                    });
                }
                let tagged = Tagged { tokens, tags };
                scan::expand(text, &mut out, Some(&mut |at, which| tagged.resolve(at, which)));
            }
            _ => scan::expand(text, &mut out, None),
        }
        let (normalized, alignment) = out.finish();
        Ok((View::derived(src, normalized, alignment),))
    }
}
