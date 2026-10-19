//! The glossary as an analysis step.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

use turbo_analysis::{Annotator, Bound, Error, Refs, View};
use turbo_analysis_layers::Tokens;
use turbo_glossary_layers::{Glossary, GlossaryTerms};
use turbo_stem_layers::Stems;

use crate::boundaries::Boundaries;
use crate::matcher::{GlossaryMatcher, MatchInput};
use crate::token::check_tokens;

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
            let v = &stems.ok_or_else(|| fail("the token matcher reads turbo:stems, which is missing".into()))?.stem;
            self.matcher.check_values(tokens, Some(v)).map_err(|e| fail(e.to_string()))?;
            Some(v)
        } else {
            None
        };
        let text = view.text();
        let len = u32::try_from(text.len()).map_err(|_| fail("the text is longer than u32::MAX bytes".into()))?;
        let boundaries = if self.matcher.has_text() {
            Boundaries::of_checked_tokens(text, len, tokens).map_err(|e| fail(e.to_string()))?
        } else {
            check_tokens(text, tokens).map_err(|e| fail(e.to_string()))?;
            Boundaries::none(0)
        };
        let input = MatchInput { text, tokens, boundaries: &boundaries, values };
        let mut found = Vec::new();
        self.matcher.find_checked(&input, &mut found).map_err(|e| fail(e.to_string()))?;
        let mut layer = GlossaryTerms {
            spans: turbo_analysis::Spans::with_capacity(view.id(), found.len()),
            tokens: Vec::with_capacity(found.len()),
            entry: Vec::with_capacity(found.len()),
            glossary: self.glossary.clone(),
        };
        let (starts, ends) = tokens.spans.columns();
        let mut cursor = 0usize;
        for m in found {
            if m.entry.get() >= self.glossary.len() {
                return Err(fail(format!(
                    "matched entry {} is outside the glossary of {}",
                    m.entry.get(),
                    self.glossary.len()
                )));
            }
            let a = cursor + starts[cursor..].partition_point(|&s| s < m.span.start);
            let b = a + ends[a..].partition_point(|&e| e <= m.span.end);
            cursor = a;
            layer.push(m.span, turbo_analysis::IxRange::new(a as u32, b as u32), m.entry);
        }
        Ok((layer,))
    }

    fn config_hash(&self) -> u64 {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text::{Case, TextMatcher};
    use crate::token::{TokenMatcher, TokenValue};
    use turbo_analysis::{Document, Span, Spans};
    use turbo_analysis_layers::Strs;

    #[test]
    fn text_matcher_rejects_unordered_or_out_of_bounds_or_mid_char_token_spans() {
        let mut g = Glossary::new();
        g.push("c1", "dogs").unwrap();
        let g = Arc::new(g);
        let matcher = GlossaryMatcher::Text(TextMatcher::new(&g, Case::Sensitive).unwrap());
        let annotator = GlossaryAnnotator::new(g, matcher);
        let doc = Document::new("hot dogs");
        let view = doc.get::<View>().unwrap();

        // Overlapping / unordered token spans: [0..8, 2..4] would otherwise panic in within(4..8).
        let mut unordered = Spans::new(view.id());
        unordered.push(Span::new(0, 8));
        unordered.push(Span::new(2, 4));
        let err = annotator
            .annotate((view, &Tokens { spans: unordered, sentence: None }, None))
            .err()
            .expect("unordered spans must fail");
        assert!(err.to_string().contains("starts before the token before it ends"), "unexpected error: {err}");

        // Out-of-bounds token span.
        let mut oob = Spans::new(view.id());
        oob.push(Span::new(0, 99));
        let err = annotator
            .annotate((view, &Tokens { spans: oob, sentence: None }, None))
            .err()
            .expect("out-of-bounds span must fail");
        assert!(err.to_string().contains("token 0"), "unexpected error: {err}");

        // Mid-character UTF-8 token span: 'é' in "café" is bytes 3..5.
        let doc_utf8 = Document::new("café");
        let view_utf8 = doc_utf8.get::<View>().unwrap();
        let mut mid_char = Spans::new(view_utf8.id());
        mid_char.push(Span::new(0, 4));
        let err = annotator
            .annotate((view_utf8, &Tokens { spans: mid_char, sentence: None }, None))
            .err()
            .expect("mid-char span must fail");
        assert!(err.to_string().contains("token 0 (0..4)"), "unexpected error: {err}");
    }

    #[test]
    fn token_matcher_reports_misaligned_stems_before_bad_token_spans() {
        let mut g = Glossary::new();
        g.push("c1", "dog").unwrap();
        let g = Arc::new(g);
        let matcher = GlossaryMatcher::Tokens(
            TokenMatcher::new(&g, TokenValue::Values, |t, out| out.push(t.to_string())).unwrap(),
        );
        let annotator = GlossaryAnnotator::new(g, matcher);
        let doc = Document::new("dog");
        let view = doc.get::<View>().unwrap();
        let mut bad_spans = Spans::new(view.id());
        bad_spans.push(Span::new(0, 99));
        let stems = Stems { stem: Strs::default() };
        let err = annotator
            .annotate((view, &Tokens { spans: bad_spans, sentence: None }, Some(&stems)))
            .err()
            .expect("misaligned stems must fail");
        assert!(err.to_string().contains("0 token values for 1 tokens"), "unexpected error: {err}");
    }

    #[test]
    fn rejects_matcher_built_for_larger_glossary() {
        let mut larger = Glossary::new();
        larger.push("c1", "hot").unwrap();
        larger.push("c2", "dogs").unwrap();
        let matcher = GlossaryMatcher::Text(TextMatcher::new(&larger, Case::Sensitive).unwrap());

        let mut smaller = Glossary::new();
        smaller.push("c1", "hot").unwrap();
        let annotator = GlossaryAnnotator::new(Arc::new(smaller), matcher);

        let doc = Document::new("dogs");
        let view = doc.get::<View>().unwrap();
        let mut spans = Spans::new(view.id());
        spans.push(Span::new(0, 4));
        let err = annotator
            .annotate((view, &Tokens { spans, sentence: None }, None))
            .err()
            .expect("entry outside glossary must fail");
        assert!(err.to_string().contains("outside the glossary"), "unexpected error: {err}");
    }
}
