//! A sentence's vocabulary rows, resolved once before parsing, so each
//! transition step reads the network's feature ids from integer arrays
//! without allocating strings or looking up words. The ids equal
//! `FeedforwardDependencyModel::feature_ids` of `FeedforwardContext::extract`
//! for every state.

use super::normalize::normalize;
use super::{FeedforwardContext, FeedforwardDependencyModel};
use crate::{ArcStandardState, Position};
use std::collections::BTreeMap;
use turbo_analysis_layers::WordOrTokenAnchor;

fn row(map: &BTreeMap<String, usize>, s: &str) -> usize {
    *map.get(s).unwrap_or(&map["*UNK*"])
}

pub(super) struct SentenceIds {
    /// Per token, its normalized word's row.
    words: Vec<usize>,
    /// Per token, its tag's row.
    tags: Vec<usize>,
    word_root: usize,
    word_null: usize,
    tag_root: usize,
    tag_null: usize,
    label_null: usize,
}

impl SentenceIds {
    pub(super) fn new(model: &FeedforwardDependencyModel, words: &[&str], tags: &[&str]) -> SentenceIds {
        let v = &model.vocabulary;
        SentenceIds {
            words: words.iter().map(|w| row(&v.words, &normalize(w))).collect(),
            tags: tags.iter().map(|t| row(&v.pos, t)).collect(),
            word_root: row(&v.words, "*ROOT*"),
            word_null: row(&v.words, "*NULL*"),
            tag_root: row(&v.pos, "*ROOT*"),
            tag_null: row(&v.pos, "*NULL*"),
            label_null: row(&v.labels, "*NULL*"),
        }
    }

    /// The feature ids of `state` into `ids` (cleared first), with
    /// `positions` as scratch.
    pub(super) fn fill<A: WordOrTokenAnchor>(
        &self,
        model: &FeedforwardDependencyModel,
        state: &ArcStandardState<A>,
        positions: &mut Vec<Position<A>>,
        ids: &mut Vec<usize>,
    ) {
        FeedforwardContext::positions(state, model.layout, positions);
        ids.clear();
        ids.extend(positions.iter().map(|p| match p {
            Position::Root => self.word_root,
            Position::Missing => self.word_null,
            Position::Node(i) => self.words[i.get()],
        }));
        ids.extend(positions.iter().map(|p| match p {
            Position::Root => self.tag_root,
            Position::Missing => self.tag_null,
            Position::Node(i) => self.tags[i.get()],
        }));
        let labels = &model.vocabulary.labels;
        ids.extend(positions[6..].iter().map(|p| match p {
            Position::Node(i) => state.relations[i.get()].as_deref().map_or(self.label_null, |s| row(labels, s)),
            _ => self.label_null,
        }));
    }
}

#[cfg(test)]
#[path = "sentence_tests.rs"]
mod tests;
