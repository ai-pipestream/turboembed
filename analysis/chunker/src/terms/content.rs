//! Term counts that leave boilerplate out.

use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Layer, Refs};
use turbo_analysis_layers::{Boilerplate, Covered, Structure, Tokens};

use super::count::{count_multi, unit_ranges};
use super::layer::{TermStats, Terms, Vocabulary};

/// `TermCounter<L>` with boilerplate left out: reads the tokens, their
/// terms, the vocabulary, the units of `L`, and the items of `B`
/// (structure blocks by default) with their `Boilerplate<B>` flags;
/// writes `TermStats<L>` counting only the tokens that start outside
/// every boilerplate item, so a footer repeated on every page adds to no
/// unit's length or counts. Units are as for `TermCounter`, and a unit
/// that is all boilerplate has length 0. `B` must be in the tokens' view.
/// One binary search per token over the boilerplate intervals, on top of
/// `TermCounter`'s work; `TermCounter` itself is unchanged.
pub struct ContentTermCounter<L, B = Structure> {
    _of: PhantomData<fn() -> (L, B)>,
}

impl<L, B> ContentTermCounter<L, B> {
    pub fn new() -> ContentTermCounter<L, B> {
        ContentTermCounter { _of: PhantomData }
    }
}

impl<L, B> Default for ContentTermCounter<L, B> {
    fn default() -> Self {
        ContentTermCounter::new()
    }
}

impl<L: Layer, B: Layer> Annotator for ContentTermCounter<L, B> {
    const ID: &'static str = "turbo:content-term-counter";
    type Reads = (Tokens, Terms, Vocabulary, L, B, Boilerplate<B>);
    type Writes = (TermStats<L>,);

    fn annotate(
        &self,
        (tokens, terms, vocabulary, units, blocks, flags): Refs<'_, Self>,
    ) -> Result<(TermStats<L>,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let ts = blocks.text_spans().ok_or_else(|| fail(format!("{} has no spans", B::NAME)))?;
        if ts.view != tokens.spans.view() {
            return Err(fail(format!(
                "the boilerplate is in view {}, the tokens in view {}",
                ts.view.0,
                tokens.spans.view().0
            )));
        }
        let ranges = unit_ranges(tokens, units).map_err(fail)?;
        let covered = Covered::new(&ts, flags).map_err(fail)?;
        let (starts, _) = tokens.spans.columns();
        if terms.term.len() != starts.len() {
            return Err(fail(format!("{} terms for {} tokens", terms.term.len(), starts.len())));
        }
        let term: Vec<_> = terms.term.iter().zip(starts).map(|(&t, &s)| t.filter(|_| !covered.contains(s))).collect();
        Ok((count_multi(&term, vocabulary.len(), &ranges).map_err(fail)?,))
    }
}
