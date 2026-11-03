//! Terms from tokens retokenized on a normalized view, written back
//! as source tokens.
//!
//! `TermIndexer` reads terms from a normalized view for tokens that
//! already exist on its parent. It cannot use tokens made on the
//! normalized view itself: joining a hyphenated word changes the token
//! boundaries, so the source tokens never contained the joined word.
//! This indexer is for that pipeline shape. Bound to the normalized
//! view and the tokens retokenized on it, it projects every token span
//! to the source view through the view alignment, writes those source
//! tokens as its own first output, and indexes the normalized token
//! text as the vocabulary. The written `Tokens`, `Vocabulary`, and
//! `Terms` therefore line up item for item, and downstream counters
//! bound to the written tokens count terms against source units.
//!
//! The projection must be exact: a normalized token whose span does
//! not map to source bytes and back to the same span (for example two
//! tokens splitting one expansion edit) is refused rather than widened
//! to the whole edit. Only a view made directly from the source is
//! accepted, mirroring `TermIndexer`'s one-hop limit, and tokens with
//! sentence indexes are refused because those indexes name sentences
//! of the normalized view, not of the source.

use turbo_analysis::{Annotator, Error, Refs, Spans, View, ViewId};
use turbo_analysis_layers::Tokens;

use super::index::intern;
use super::layer::{Terms, Vocabulary};

/// Projects retokenized normalized tokens to source tokens and indexes
/// their normalized text. See the module docs for the exactness rules.
pub struct RetokenizingTermIndexer;

impl Annotator for RetokenizingTermIndexer {
    const ID: &'static str = "turbo:retokenizing-term-indexer";
    type Reads = (View, Tokens);
    type Writes = (Tokens, Vocabulary, Terms);

    fn annotate(&self, (view, tokens): Refs<'_, Self>) -> Result<(Tokens, Vocabulary, Terms), Error> {
        let refuse = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if tokens.spans.view() != view.id() {
            return Err(refuse(format!(
                "the tokens are in view {}, not the normalized view {} they were retokenized on",
                tokens.spans.view().0,
                view.id().0
            )));
        }
        if view.parent() != Some(ViewId::SOURCE) {
            return Err(refuse(format!(
                "the normalized view {} was not made directly from the source view",
                view.id().0
            )));
        }
        if tokens.sentence.is_some() {
            return Err(refuse(
                "the retokenized tokens carry sentence indexes of the normalized view; \
                 sentence them on the source view instead"
                    .into(),
            ));
        }
        let alignment = view.alignment();
        let text = view.text();
        let mut source_spans = Spans::new(ViewId::SOURCE);
        let mut words: Vec<Option<&str>> = Vec::with_capacity(tokens.spans.len());
        for (i, _) in tokens.spans.iter() {
            let fragments: Vec<_> = tokens.spans.fragments(i).collect();
            let [span] = fragments.as_slice() else {
                return Err(refuse(format!("token {} is discontinuous", i.get())));
            };
            let projected = alignment.span_to_parent(*span);
            if alignment.span_to_child(projected) != *span {
                return Err(refuse(format!(
                    "token {} at {}..{} does not project exactly to the source view",
                    i.get(),
                    span.start,
                    span.end
                )));
            }
            source_spans.push(projected);
            words.push(Some(&text[span.range()]).filter(|w| w.chars().any(char::is_alphanumeric)));
        }
        let (vocabulary, term) = intern(&words);
        Ok((Tokens { spans: source_spans, sentence: None }, vocabulary, Terms { term }))
    }
}
