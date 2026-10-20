//! Shared input preparation for the matching and discovery
//! annotators: every input validated before anything is indexed,
//! then the dependency graph, entity heads and per-word
//! sentences both annotators walk.

use std::sync::Arc;

use turbo_analysis::{Error, Ix, Layer, View};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, Sentences, Tokens, Word, WordOrTokenAnchor};
use turbo_depparse::DependencyGraph;

pub(crate) struct Inputs<'a, A: WordOrTokenAnchor> {
    pub view: &'a View,
    pub sentences: &'a Sentences,
    pub anchor: &'a A,
    pub tokens: &'a Tokens,
    pub dependencies: &'a Dependencies<A>,
    pub entities: &'a Entities,
    pub pos_present: bool,
    pub lemmas: Option<&'a Lemmas>,
    pub needs_pos: bool,
    pub needs_lemma: bool,
}

pub(crate) struct Prepared<'a, A: WordOrTokenAnchor> {
    pub words: Vec<Word<'a>>,
    pub graph: DependencyGraph<A>,
    pub entity_heads: Vec<Option<Ix<A>>>,
    pub word_sentence: Vec<Option<usize>>,
}

pub(crate) fn prepare<'a, A: WordOrTokenAnchor>(
    id: &'static str,
    input: Inputs<'a, A>,
) -> Result<Prepared<'a, A>, Error> {
    let Inputs { view, sentences, anchor, tokens, dependencies, entities, pos_present, lemmas, needs_pos, needs_lemma } =
        input;
    let fail = |detail: String| Error::Annotator { annotator: id, detail };
    sentences.check().map_err(fail)?;
    entities.check().map_err(fail)?;
    dependencies.check().map_err(fail)?;
    dependencies.check_forest().map_err(fail)?;
    if let Some(lemmas) = lemmas {
        lemmas.check().map_err(fail)?;
    }
    if sentences.spans.view() != view.id() || entities.spans.view() != view.id() {
        return Err(fail("its inputs are in different views".into()));
    }
    let words = anchor.resolved(view, tokens).map_err(fail)?;
    if dependencies.head.len() != words.len() {
        return Err(fail(format!(
            "dependency layer is not aligned with the token layer: {} heads for {} words",
            dependencies.head.len(),
            words.len()
        )));
    }
    if needs_pos && !pos_present {
        return Err(fail("a pattern restricts the pivot part of speech but no pos layer was given".into()));
    }
    if needs_lemma {
        let Some(lemmas) = lemmas else {
            return Err(fail("a pattern restricts the pivot lemma but no lemmas layer was given".into()));
        };
        if let Some(w) = words.iter().find(|w| w.tokens.start.get() >= lemmas.lemma.len()) {
            return Err(fail(format!("no lemma for token {}", w.tokens.start.get())));
        }
    }
    let graph = DependencyGraph::from_layer(Dependencies {
        head: dependencies.head.clone(),
        relation: dependencies.relation.clone(),
        relations: Arc::clone(&dependencies.relations),
    })
    .map_err(|e| fail(e.to_string()))?;

    let overlaps = |span: turbo_analysis::Span, e: Ix<Entities>| {
        entities.spans.fragments(e).any(|y| span.start.max(y.start) < span.end.min(y.end))
    };
    // Words follow text order, so the words overlapping one entity
    // fragment form one contiguous range; the first outward word
    // overall is the minimum over the fragments.
    let (word_starts, word_ends): (Vec<u32>, Vec<u32>) =
        words.iter().map(|w| (w.token_span.start, w.token_span.end)).unzip();
    let entity_heads: Vec<Option<Ix<A>>> = (0..entities.spans.len() as u32)
        .map(|e| {
            let e = Ix::new(e);
            entities
                .spans
                .fragments(e)
                .filter_map(|y| {
                    let lo = word_ends.partition_point(|&end| end <= y.start);
                    let hi = word_starts.partition_point(|&start| start < y.end);
                    (lo..hi.max(lo)).find(|&i| {
                        overlaps(words[i].token_span, e)
                            && match dependencies.head[i] {
                                None => true,
                                Some(h) => !overlaps(words[h.get()].token_span, e),
                            }
                    })
                })
                .min()
                .map(|i| Ix::new(i as u32))
        })
        .collect();
    let (sent_starts, sent_ends) = sentences.spans.columns();
    let word_sentence: Vec<Option<usize>> = words
        .iter()
        .map(|w| {
            let at = sent_starts.partition_point(|&s| s <= w.token_span.start);
            at.checked_sub(1).filter(|&s| w.token_span.start < sent_ends[s])
        })
        .collect();
    Ok(Prepared { words, graph, entity_heads, word_sentence })
}

/// Entities grouped by the sentence of their head word, each group
/// in layer order, so a pair loop over one group visits exactly the
/// same same-sentence pairs in the same order as a full cross
/// product would, in linear time in the entity count.
pub fn by_sentence<A: WordOrTokenAnchor>(
    entity_heads: &[Option<Ix<A>>],
    word_sentence: &[Option<usize>],
    sentences: usize,
) -> Vec<Vec<u32>> {
    let mut out: Vec<Vec<u32>> = vec![Vec::new(); sentences];
    for (e, h) in entity_heads.iter().enumerate() {
        if let Some(sent) = h.and_then(|h| word_sentence[h.get()]) {
            out[sent].push(e as u32);
        }
    }
    out
}
