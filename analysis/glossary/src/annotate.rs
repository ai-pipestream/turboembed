//! The glossary as an analysis step.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Refs, View};
use turbo_analysis_layers::Tokens;
use turbo_glossary_layers::{Glossary, GlossaryTerms};
use turbo_stem_layers::Stems;

use crate::boundaries::Boundaries;
use crate::matcher::{GlossaryMatcher, MatchInput};

/// Writes the glossary terms a matcher finds in a text view as
/// [`GlossaryTerms`] (`turbo:glossary-terms`): OpenNLP's
/// `GlossaryAnnotator`. It reads the view and its tokens, and
/// `turbo:stems` when a token matcher in it matches by
/// [`TokenValue::Values`](crate::TokenValue::Values), which then stand
/// for each token's value.
///
/// A character match is kept when it starts and ends at a token's start
/// or end or at an end of the text ([`Boundaries::of_tokens`]). With
/// tokens from the UAX #29 word tokenizer these are OpenNLP's word
/// boundaries but for those inside runs of punctuation and symbols,
/// which that tokenizer drops: OpenNLP finds "C+" in "C++", the step
/// does not. With another tokenizer they are that tokenizer's edges.
///
/// To match normalized text (whitespace collapsed, case or dashes folded)
/// make a view with the normalizer, tokens on that view, and bind the
/// step's reads to them (`Bound::read`); the terms must be normalized
/// the same way when the matcher is built. The layer's spans are then in
/// that view, and `Document::to_source` takes them to the source text.
pub struct GlossaryAnnotator {
    matcher: GlossaryMatcher,
    glossary: Arc<Glossary>,
    config: u64,
}

impl GlossaryAnnotator {
    /// The step over `glossary` with `matcher`, which must have been built
    /// from it: the layer's entries index this glossary.
    pub fn new(glossary: Arc<Glossary>, matcher: GlossaryMatcher) -> GlossaryAnnotator {
        let mut h = DefaultHasher::new();
        for (_, e) in glossary.iter() {
            e.id.hash(&mut h);
            e.term.hash(&mut h);
        }
        describe(&matcher, &mut h);
        GlossaryAnnotator { matcher, glossary, config: h.finish() }
    }

    /// The step writing instance `name` of [`GlossaryTerms`].
    pub fn bound(self, name: &'static str) -> Bound<GlossaryAnnotator> {
        Bound::new(self).write(0, name)
    }

    pub fn glossary(&self) -> &Arc<Glossary> {
        &self.glossary
    }
}

/// Feeds the matcher's shape and patterns to the configuration hash, so
/// matchers built with other normalizers, case or analyzers hash
/// differently.
fn describe(m: &GlossaryMatcher, h: &mut DefaultHasher) {
    match m {
        GlossaryMatcher::Text(t) => {
            "text".hash(h);
            t.fingerprint().hash(h);
        }
        GlossaryMatcher::Tokens(t) => {
            "tokens".hash(h);
            t.fingerprint().hash(h);
        }
        GlossaryMatcher::Composite(ms) => {
            "composite".hash(h);
            ms.len().hash(h);
            for m in ms {
                describe(m, h);
            }
        }
    }
}

impl Annotator for GlossaryAnnotator {
    const ID: &'static str = "turbo-glossary:annotator";
    type Reads = (View, Tokens, Option<Stems>);
    type Writes = (GlossaryTerms,);

    fn annotate(&self, reads: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let (view, tokens, stems) = reads;
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if tokens.spans.view() != view.id() {
            return Err(fail("the tokens are in another view".into()));
        }
        let values = if self.matcher.reads_values() {
            Some(&stems.ok_or_else(|| fail("the token matcher reads turbo:stems, which is missing".into()))?.stem)
        } else {
            None
        };
        let text = view.text();
        let boundaries = Boundaries::of_tokens(text.len() as u32, tokens);
        let input = MatchInput { text, tokens, boundaries: &boundaries, values };
        let mut found = Vec::new();
        self.matcher.find(&input, &mut found).map_err(|e| fail(e.to_string()))?;
        let mut layer = GlossaryTerms::new(view.id(), self.glossary.clone());
        for m in found {
            layer.push(m.span, tokens.spans.within(m.span), m.entry);
        }
        Ok((layer,))
    }

    fn config_hash(&self) -> u64 {
        self.config
    }
}
