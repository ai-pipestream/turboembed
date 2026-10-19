use crate::{DependencyGraph, DependencyParserME};
use std::marker::PhantomData;
use turbo_analysis::{Annotator, Error, Ix, Layer, ModelRef, Refs, View};
use turbo_analysis_layers::{Dependencies, MorphPosTags, Sentences, Tokens, WordOrTokenAnchor};

/// Parses each sentence independently, converting local heads to document
/// indexes. Uncovered or crossing words are errors, never silently omitted.
pub struct DependencyAnnotator<A: WordOrTokenAnchor = Tokens> {
    parser: DependencyParserME,
    anchor: PhantomData<fn() -> A>,
}
impl<A: WordOrTokenAnchor> DependencyAnnotator<A> {
    pub fn new(parser: DependencyParserME) -> Self {
        Self { parser, anchor: PhantomData }
    }
}
impl<A: WordOrTokenAnchor> Annotator for DependencyAnnotator<A> {
    const ID: &'static str = "opennlp:dependency-parser";
    const VERSION: &'static str = "1";
    // Tokens is the surface layer needed to resolve expanded Words. For A =
    // Tokens, slots 2 and 3 must bind to the same token instance.
    type Reads = (View, Sentences, Tokens, A, A::Pos);
    type Writes = (Dependencies<A>,);
    fn annotate(&self, (view, sentences, tokens, anchor, pos): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        sentences.check().map_err(fail)?;
        pos.check().map_err(fail)?;
        if sentences.spans.view() != view.id() {
            return Err(fail("sentence view mismatch".into()));
        }
        let words = anchor.resolved(view, tokens).map_err(fail)?;
        if pos.len() != words.len() {
            return Err(fail("POS length mismatch".into()));
        }
        let mut heads = Vec::with_capacity(words.len());
        let mut relations = Vec::with_capacity(words.len());
        let mut cursor = 0;
        for (sentence, span) in sentences.spans.iter() {
            if sentences.spans.is_discontinuous(sentence) || view.text().get(span.range()).is_none() {
                return Err(fail("invalid sentence span".into()));
            }
            let start = cursor;
            while cursor < words.len() && words[cursor].token_span.start < span.end {
                let word = &words[cursor];
                if word.token_span.start < span.start || word.token_span.end > span.end {
                    return Err(fail("word crosses or is outside sentence".into()));
                }
                if let Some(ids) = &tokens.sentence {
                    for &id in &ids[word.tokens.start.get()..word.tokens.end.get()] {
                        if id != sentence {
                            return Err(fail("token sentence index disagrees with span".into()));
                        }
                    }
                }
                cursor += 1;
            }
            if cursor == start {
                continue;
            }
            let forms: Vec<_> = words[start..cursor].iter().map(|w| w.form).collect();
            let tags: Vec<_> = (start..cursor)
                .map(|i| pos.label(i).ok_or_else(|| fail("invalid POS label".into())))
                .collect::<Result<_, _>>()?;
            let graph: DependencyGraph<A> = self.parser.parse(&forms, &tags).map_err(|e| fail(e.to_string()))?;
            for i in 0..graph.len() {
                let arc = graph.arc(Ix::new(i as u32)).map_err(|e| fail(e.to_string()))?;
                heads.push(arc.head.map(|h| Ix::new((start + h.get()) as u32)));
                relations.push(arc.relation.to_owned());
            }
        }
        if cursor != words.len() {
            return Err(fail("words outside sentences".into()));
        }
        // Construct a forest without imposing the single-root sentence check.
        let mut labels: Vec<&str> = Vec::new();
        let mut ids = std::collections::HashMap::new();
        let mut tags = Vec::new();
        for r in &relations {
            let id = match ids.get(r.as_str()) {
                Some(&id) => id,
                None => {
                    let id = u16::try_from(labels.len()).map_err(|_| fail("too many relations".into()))?;
                    labels.push(r);
                    ids.insert(r.as_str(), id);
                    id
                }
            };
            tags.push(turbo_analysis::Tag(id));
        }
        let layer =
            Dependencies { head: heads, relation: tags, relations: turbo_analysis::TagSet::new("dependency", &labels) };
        layer.check().map_err(fail)?;
        layer.check_forest().map_err(fail)?;
        Ok((layer,))
    }
    fn model(&self) -> Option<ModelRef> {
        Some(self.parser.model.identity.clone())
    }
}
