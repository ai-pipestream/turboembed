//! The discovery annotator: every connected ordered entity
//! pair's dependency path, untyped, for rule authoring. It is
//! the pattern matcher minus the pattern loop (the same prepared
//! inputs, entity heads and sentence scoping), so every emitted
//! signature is a legal pattern path by construction.

use std::collections::BTreeMap;
use std::marker::PhantomData;

use turbo_analysis::{Annotator, Error, Ix, Layer, Refs, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, Sentences, Tokens, WordOrTokenAnchor};
use turbo_depparse::ArcDirection;

use crate::paths::RelationPaths;
use crate::prepare::{Inputs, prepare};

/// Writes [`RelationPaths`]: the shortest dependency path of
/// every ordered entity pair whose heads lie in one sentence and
/// connect, up to `max_path_len` arcs when set.
pub struct RelationPathAnnotator<A: WordOrTokenAnchor = Tokens> {
    /// Longest path, in arcs, to emit; `None` is unbounded.
    pub max_path_len: Option<usize>,
    _anchor: PhantomData<fn() -> A>,
}

impl<A: WordOrTokenAnchor> Default for RelationPathAnnotator<A> {
    fn default() -> Self {
        RelationPathAnnotator { max_path_len: None, _anchor: PhantomData }
    }
}

impl<A: WordOrTokenAnchor> Annotator for RelationPathAnnotator<A> {
    const ID: &'static str = "turbo:relation-path-annotator";
    type Reads = (View, Sentences, A, Tokens, Dependencies<A>, Entities, Option<A::Pos>, Option<Lemmas>);
    type Writes = (RelationPaths,);

    fn annotate(
        &self,
        (view, sentences, anchor, tokens, dependencies, entities, pos, lemmas): Refs<'_, Self>,
    ) -> Result<Self::Writes, Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        if let Some(p) = pos {
            p.check().map_err(fail)?;
        }
        let p = prepare::<A>(
            Self::ID,
            Inputs {
                view,
                sentences,
                anchor,
                tokens,
                dependencies,
                entities,
                pos_present: pos.is_some(),
                lemmas,
                needs_pos: false,
                needs_lemma: false,
            },
        )?;
        // Labels actually walked, in first-seen order, as the
        // step tag set.
        let mut label_ids: BTreeMap<String, Tag> = BTreeMap::new();
        let mut spans = Spans::new(view.id());
        let (mut subject, mut object, mut subject_head, mut object_head, mut pivot) =
            (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut step_row, mut step_token, mut step_up, mut step_relation) =
            (vec![0u32], Vec::new(), Vec::new(), Vec::new());
        let by_sentence = crate::prepare::by_sentence(&p.entity_heads, &p.word_sentence, sentences.spans.len());
        for s in 0..entities.spans.len() as u32 {
            let Some(sent) = p.entity_heads[s as usize].and_then(|h| p.word_sentence[h.get()]) else { continue };
            for &o in &by_sentence[sent] {
                if s == o {
                    continue;
                }
                let (Some(sh), Some(oh)) = (p.entity_heads[s as usize], p.entity_heads[o as usize]) else {
                    continue;
                };
                if sh == oh
                    || p.word_sentence[sh.get()] != p.word_sentence[oh.get()]
                    || p.word_sentence[sh.get()].is_none()
                {
                    continue;
                }
                let Some(path) = p.graph.shortest_path(sh, oh).map_err(|e| fail(e.to_string()))? else { continue };
                if self.max_path_len.is_some_and(|cap| path.len() > cap) || path.is_empty() {
                    continue;
                }
                let pivot_ix =
                    path.iter().take_while(|st| st.direction == ArcDirection::ToHead).last().map_or(sh, |st| st.to);
                let (se, oe) = (entities.spans.get(Ix::new(s)), entities.spans.get(Ix::new(o)));
                spans.push(Span::new(se.start.min(oe.start), se.end.max(oe.end)));
                subject.push(Ix::new(s));
                object.push(Ix::new(o));
                subject_head.push(sh.get() as u32);
                object_head.push(oh.get() as u32);
                pivot.push(pivot_ix.get() as u32);
                for step in &path {
                    let next = Tag(label_ids.len() as u16);
                    let tag = *label_ids.entry(step.relation.to_owned()).or_insert(next);
                    step_token.push(step.to.get() as u32);
                    step_up.push(step.direction == ArcDirection::ToHead);
                    step_relation.push(tag);
                }
                step_row.push(u32::try_from(step_token.len()).map_err(|_| fail("too many path steps".into()))?);
            }
        }
        let mut labels: Vec<&str> = vec![""; label_ids.len()];
        for (label, tag) in &label_ids {
            labels[tag.0 as usize] = label;
        }
        let paths = RelationPaths {
            spans,
            subject,
            object,
            subject_head,
            object_head,
            pivot,
            step_row,
            step_token,
            step_up,
            step_relation,
            relations: TagSet::new("relation", &labels),
        };
        paths.check().map_err(fail)?;
        Ok((paths,))
    }

    fn config_hash(&self) -> u64 {
        // The only output-affecting setting: hash the Option
        // discriminant and the cap separately, so no cap value
        // (not even usize::MAX) collides with None or overflows.
        let (discriminant, cap) = match self.max_path_len {
            None => (0u8, 0u64),
            Some(cap) => (1u8, cap as u64),
        };
        turbo_analysis::Fnv1a::new().u8(discriminant).u64(cap).finish()
    }
}
