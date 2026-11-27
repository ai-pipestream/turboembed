//! The annotator: registered patterns matched against the
//! shortest dependency path between entity head words, as the
//! reference `RelationAnnotator` matches them.
//!
//! An entity head is the first word in layer order that overlaps
//! the entity and whose dependency head is a root arc or a word
//! that does not overlap the entity; an overlap needs at least
//! one shared character, so empty spans select no head. Pairs are
//! visited in entity layer order and patterns in registration
//! order, so the output order is stable. A pair whose entities
//! share a head word, whose heads lie in different sentences, or
//! whose heads are not connected in the dependency forest
//! contributes nothing.
//!
//! Beyond the reference, patterns may restrict the subject and
//! object entity kinds, the pivot's part of speech or lemma, and
//! carry a confidence; the annotator may cap the path length it
//! will match. The annotator writes every mention (duplicates
//! included) to [`Relations`] and the same triples, each once at
//! its best confidence, to `Edges<Entities, Entities>`.

use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Annotator, Edge, Edges, Error, Ix, Layer, Refs, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphPosTags, Sentences, Tokens, Word, WordOrTokenAnchor};
use turbo_depparse::{ArcDirection, PathStep};

use crate::layer::Relations;
use crate::pattern::RelationPattern;
use crate::work::RelationWork;

/// Extracts typed relations by matching [`RelationPattern`] rules
/// against dependency paths between entity heads.
pub struct RelationAnnotator<A: WordOrTokenAnchor = Tokens> {
    patterns: Vec<RelationPattern>,
    kinds: Arc<TagSet>,
    tags: Vec<Tag>,
    /// Longest shortest path, in arcs, a pair may have and still
    /// be matched; `None` is unbounded.
    pub max_path_len: Option<usize>,
    _anchor: PhantomData<fn() -> A>,
}

struct MatchContext<'a, A: WordOrTokenAnchor> {
    words: &'a [Word<'a>],
    entities: &'a Entities,
    pos: Option<&'a A::Pos>,
    lemmas: Option<&'a Lemmas>,
}

impl<A: WordOrTokenAnchor> MatchContext<'_, A> {
    fn entity_label(&self, e: Ix<Entities>) -> &str {
        self.entities.kinds.label(self.entities.kind[e.get()])
    }

    fn matches(&self, pattern: &RelationPattern, pair: (u32, u32), pivot: Ix<A>, path: &[PathStep<'_, A>]) -> bool {
        if pattern.steps().len() != path.len()
            || !pattern
                .steps()
                .iter()
                .zip(path)
                .all(|(step, arc)| step.up == (arc.direction == ArcDirection::ToHead) && step.label == arc.relation)
        {
            return false;
        }
        let (s, o) = pair;
        if pattern.subject_kind.as_deref().is_some_and(|k| k != self.entity_label(Ix::new(s)))
            || pattern.object_kind.as_deref().is_some_and(|k| k != self.entity_label(Ix::new(o)))
        {
            return false;
        }
        if let Some(trigger) = &pattern.trigger
            && !turbo_unicode::java::eq_lowercase_code_points(self.words[pivot.get()].form, trigger)
        {
            return false;
        }
        if let Some(want) = &pattern.pivot_pos
            && self.pos.and_then(|p| p.label(pivot.get())) != Some(want.as_str())
        {
            return false;
        }
        if let (Some(want), Some(lemmas)) = (&pattern.pivot_lemma, self.lemmas)
            && lemmas.lemma.get(self.words[pivot.get()].tokens.start.get()) != want
        {
            return false;
        }
        true
    }
}

/// Deduplicates matched `(from, to, kind)` edges emitted for a single
/// `(subject, object)` pair in place, sorting by `kind` and retaining the
/// highest score for each kind. Because pairs are visited in strictly
/// increasing `(subject, object)` order, `edges` stays globally sorted by
/// `(from, to, kind)` without a hash table.
fn coalesce_pair_edges(edges: &mut Vec<Edge<Entities, Entities>>, start: usize, edge_ops: &mut usize) {
    if edges.len() - start <= 1 {
        return;
    }
    edges[start..].sort_by_key(|e| e.kind);
    let mut write = start;
    for read in (start + 1)..edges.len() {
        *edge_ops += 1;
        if edges[write].kind == edges[read].kind {
            edges[write].score = edges[write].score.max(edges[read].score);
        } else {
            write += 1;
            edges[write] = edges[read];
        }
    }
    edges.truncate(write + 1);
}

impl<A: WordOrTokenAnchor> RelationAnnotator<A> {
    /// An annotator matching `patterns` in registration order.
    /// `Err` when there are none.
    pub fn new(patterns: Vec<RelationPattern>) -> Result<Self, String> {
        if patterns.is_empty() {
            return Err("patterns must not be null or empty".into());
        }
        let mut types: Vec<&str> = Vec::new();
        for p in &patterns {
            if !types.contains(&p.relation_type.as_str()) {
                types.push(&p.relation_type);
            }
        }
        if types.len() > u16::MAX as usize + 1 {
            return Err(format!("{} relation types, more than a tag set holds", types.len()));
        }
        let kinds = TagSet::new("relation", &types);
        let tags = patterns
            .iter()
            .map(|p| Tag(types.iter().position(|t| *t == p.relation_type).expect("registered above") as u16))
            .collect();
        Ok(RelationAnnotator { patterns, kinds, tags, max_path_len: None, _anchor: PhantomData })
    }

    /// The registered patterns, in order.
    pub fn patterns(&self) -> &[RelationPattern] {
        &self.patterns
    }

    fn fail(&self, detail: String) -> Error {
        Error::Annotator { annotator: <Self as Annotator>::ID, detail }
    }

    /// Runs pattern matching on `inputs` and returns deterministic
    /// [`RelationWork`] counters alongside the extracted layers.
    pub fn annotate_with_work(
        &self,
        (view, sentences, anchor, tokens, dependencies, entities, pos, lemmas): Refs<'_, Self>,
    ) -> Result<(<Self as Annotator>::Writes, RelationWork), Error> {
        if let Some(p) = pos {
            p.check().map_err(|d| self.fail(d))?;
        }
        let crate::prepare::Prepared { words, graph, entity_heads, word_sentence, head_probes } =
            crate::prepare::prepare::<A>(
                <Self as Annotator>::ID,
                crate::prepare::Inputs {
                    view,
                    sentences,
                    anchor,
                    tokens,
                    dependencies,
                    entities,
                    pos_present: pos.is_some(),
                    lemmas,
                    needs_pos: self.patterns.iter().any(|p| p.pivot_pos.is_some()),
                    needs_lemma: self.patterns.iter().any(|p| p.pivot_lemma.is_some()),
                },
            )?;
        let fail = |detail: String| self.fail(detail);
        let mut work = RelationWork { head_probes, ..RelationWork::default() };

        let max_pattern_steps = self.patterns.iter().map(|p| p.steps().len()).max().unwrap_or(0);
        let path_cap = Some(self.max_path_len.map_or(max_pattern_steps, |cap| cap.min(max_pattern_steps)));
        let cx = MatchContext { words: &words, entities, pos, lemmas };

        let mut spans = Spans::new(view.id());
        let (mut subject, mut object, mut kind, mut score) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut edges: Vec<Edge<Entities, Entities>> = Vec::new();
        let mut path = Vec::with_capacity(max_pattern_steps);
        // Built on the first pair, so a document with none pays nothing.
        let mut paths = None;
        let by_sentence = crate::prepare::by_sentence(&entity_heads, &word_sentence, sentences.spans.len());

        for s in 0..entities.spans.len() as u32 {
            let Some(sh) = entity_heads[s as usize] else { continue };
            let Some(sent) = word_sentence[sh.get()] else { continue };
            for &o in &by_sentence[sent] {
                work.pair_visits += 1;
                if s == o {
                    continue;
                }
                let Some(oh) = entity_heads[o as usize] else { continue };
                if sh == oh {
                    continue;
                }
                let paths = paths.get_or_insert_with(|| graph.path_index());
                let search = paths.shortest_path_into(sh, oh, path_cap, &mut path).map_err(|e| fail(e.to_string()))?;
                work.arc_reads += search.arc_reads;
                if !search.found {
                    continue;
                }
                let pivot = path.iter().take_while(|p| p.direction == ArcDirection::ToHead).last().map_or(sh, |p| p.to);
                let pair_edges_start = edges.len();
                for (p, pattern) in self.patterns.iter().enumerate() {
                    if !cx.matches(pattern, (s, o), pivot, &path) {
                        continue;
                    }
                    let (se, oe) = (entities.spans.get(Ix::new(s)), entities.spans.get(Ix::new(o)));
                    spans.push(Span::new(se.start.min(oe.start), se.end.max(oe.end)));
                    subject.push(Ix::new(s));
                    object.push(Ix::new(o));
                    kind.push(self.tags[p]);
                    score.push(pattern.confidence);
                    edges.push(Edge::new(Ix::new(s), Ix::new(o), self.tags[p], pattern.confidence));
                    work.edge_ops += 1;
                }
                coalesce_pair_edges(&mut edges, pair_edges_start, &mut work.edge_ops);
            }
        }
        let relations = Relations { spans, subject, object, kind, kinds: Arc::clone(&self.kinds), score };
        relations.check().map_err(fail)?;
        Ok(((relations, Edges::new(Arc::clone(&self.kinds), edges)), work))
    }
}

impl<A: WordOrTokenAnchor> Annotator for RelationAnnotator<A> {
    const ID: &'static str = "turbo:relation-annotator";
    type Reads = (View, Sentences, A, Tokens, Dependencies<A>, Entities, Option<A::Pos>, Option<Lemmas>);
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(&self, inputs: Refs<'_, Self>) -> Result<Self::Writes, Error> {
        self.annotate_with_work(inputs).map(|(writes, _)| writes)
    }

    fn config_hash(&self) -> u64 {
        let mut h = turbo_analysis::Fnv1a::new();
        h.str("turbo:relation-annotator");
        h.u8(u8::from(self.max_path_len.is_some()));
        if let Some(cap) = self.max_path_len {
            h.u64(cap as u64);
        }
        for p in &self.patterns {
            for field in [
                Some(p.relation_type.as_str()),
                Some(p.path.as_str()),
                p.trigger.as_deref(),
                p.subject_kind.as_deref(),
                p.object_kind.as_deref(),
                p.pivot_pos.as_deref(),
                p.pivot_lemma.as_deref(),
            ] {
                match field {
                    Some(s) => {
                        h.u8(1).str(s);
                    }
                    None => {
                        h.u8(0);
                    }
                }
            }
            h.f64(p.confidence);
        }
        h.finish()
    }
}
