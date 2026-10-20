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
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, MorphPosTags, Sentences, Tokens, WordOrTokenAnchor};
use turbo_depparse::ArcDirection;

use crate::layer::Relations;
use crate::pattern::RelationPattern;

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
}

fn fnv(state: &mut u64, bytes: &[u8]) {
    for &b in bytes {
        *state ^= u64::from(b);
        *state = state.wrapping_mul(0x100000001b3);
    }
}

impl<A: WordOrTokenAnchor> Annotator for RelationAnnotator<A> {
    const ID: &'static str = "turbo:relation-annotator";
    type Reads = (View, Sentences, A, Tokens, Dependencies<A>, Entities, Option<A::Pos>, Option<Lemmas>);
    type Writes = (Relations, Edges<Entities, Entities>);

    fn annotate(
        &self,
        (view, sentences, anchor, tokens, dependencies, entities, pos, lemmas): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        if let Some(p) = pos {
            p.check().map_err(|d| self.fail(d))?;
        }
        let crate::prepare::Prepared { words, graph, entity_heads, word_sentence } = crate::prepare::prepare::<A>(
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

        let mut spans = Spans::new(view.id());
        let (mut subject, mut object, mut kind, mut score) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let entity_label = |e: Ix<Entities>| entities.kinds.label(entities.kind[e.get()]);
        for s in 0..entities.spans.len() as u32 {
            for o in 0..entities.spans.len() as u32 {
                if s == o {
                    continue;
                }
                let (Some(sh), Some(oh)) = (entity_heads[s as usize], entity_heads[o as usize]) else { continue };
                if sh == oh || word_sentence[sh.get()] != word_sentence[oh.get()] || word_sentence[sh.get()].is_none() {
                    continue;
                }
                let Some(path) = graph.shortest_path(sh, oh).map_err(|e| fail(e.to_string()))? else { continue };
                if self.max_path_len.is_some_and(|cap| path.len() > cap) {
                    continue;
                }
                let pivot = path.iter().take_while(|p| p.direction == ArcDirection::ToHead).last().map_or(sh, |p| p.to);
                for (p, pattern) in self.patterns.iter().enumerate() {
                    if pattern.steps().len() != path.len()
                        || !pattern.steps().iter().zip(&path).all(|(step, arc)| {
                            step.up == (arc.direction == ArcDirection::ToHead) && step.label == arc.relation
                        })
                    {
                        continue;
                    }
                    if pattern.subject_kind.as_deref().is_some_and(|k| k != entity_label(Ix::new(s)))
                        || pattern.object_kind.as_deref().is_some_and(|k| k != entity_label(Ix::new(o)))
                    {
                        continue;
                    }
                    if let Some(trigger) = &pattern.trigger {
                        let form: String =
                            words[pivot.get()].form.chars().map(turbo_unicode::java::to_lower_char).collect();
                        if form != *trigger {
                            continue;
                        }
                    }
                    if let Some(want) = &pattern.pivot_pos
                        && pos.and_then(|p| p.label(pivot.get())) != Some(want.as_str())
                    {
                        continue;
                    }
                    if let (Some(want), Some(lemmas)) = (&pattern.pivot_lemma, lemmas)
                        && lemmas.lemma.get(words[pivot.get()].tokens.start.get()) != want
                    {
                        continue;
                    }
                    let (se, oe) = (entities.spans.get(Ix::new(s)), entities.spans.get(Ix::new(o)));
                    spans.push(Span::new(se.start.min(oe.start), se.end.max(oe.end)));
                    subject.push(Ix::new(s));
                    object.push(Ix::new(o));
                    kind.push(self.tags[p]);
                    score.push(pattern.confidence);
                }
            }
        }
        let relations = Relations { spans, subject, object, kind, kinds: Arc::clone(&self.kinds), score };
        relations.check().map_err(fail)?;
        // The edge layer holds each (subject, object, type) once,
        // at the best confidence any mention of it carries.
        let mut edges: Vec<Edge<Entities, Entities>> = Vec::new();
        for (i, _) in relations.spans.iter() {
            let (s, o, _, sc) = relations.mention(i);
            let edge = Edge::new(s, o, relations.kind[i.get()], sc);
            match edges.iter_mut().find(|e| (e.from, e.to, e.kind) == (edge.from, edge.to, edge.kind)) {
                Some(e) => e.score = e.score.max(edge.score),
                None => edges.push(edge),
            }
        }
        Ok((relations, Edges::new(Arc::clone(&self.kinds), edges)))
    }

    fn config_hash(&self) -> u64 {
        let mut h = 0xcbf29ce484222325u64;
        fnv(&mut h, b"turbo:relation-annotator");
        if let Some(cap) = self.max_path_len {
            fnv(&mut h, &(cap as u64).to_le_bytes());
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
                        fnv(&mut h, &[1]);
                        fnv(&mut h, s.as_bytes());
                    }
                    None => fnv(&mut h, &[0]),
                }
            }
            fnv(&mut h, &p.confidence.to_bits().to_le_bytes());
        }
        h
    }
}
