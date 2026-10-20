//! Categories, as a document categorizer scores them: for the whole
//! document, or for each sentence, structure block or chunk.

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Anchor, Layer, LayerName, Tag, TagSet, View};

use crate::{Aspects, Chunks, Opinions, Sentences, Structure};

/// What a [`Categories`] layer can describe. `View` is the whole
/// document: the layer is `turbo:categories`, anchored to the document,
/// with one item. Any other scope is a layer whose items are categorized
/// one by one: `Categories<Sentences>` is
/// `turbo:categories<opennlp:sentences>`, aligned to the sentences.
///
/// A crate with its own layer of units to categorize implements this for
/// that layer, as this crate does for `Sentences`, `Structure` and
/// `Chunks`.
pub trait CategoryScope: Layer {
    /// The `OF` of `Categories<Self>`: `None` for the whole document,
    /// otherwise this layer's name.
    const CATEGORIES_OF: Option<&'static LayerName>;

    /// The anchor of `Categories<Self>`.
    fn categories_anchor() -> Anchor;

    /// Filters `Self` as the aligned anchor of `Categories<Self>` when row
    /// removal changes the item count.
    fn retain_categories_anchor<C: Layer>(
        _doc: &mut turbo_analysis::Document,
        _source_anchor: turbo_analysis::KeyId,
        _target_name: &'static str,
        _kept: &[turbo_analysis::Ix<C>],
        _provenance: turbo_analysis::Provenance,
    ) -> Result<Option<turbo_analysis::KeyId>, turbo_analysis::Error> {
        Ok(None)
    }
}

impl CategoryScope for View {
    const CATEGORIES_OF: Option<&'static LayerName> = None;
    fn categories_anchor() -> Anchor {
        Anchor::Document
    }
}

impl CategoryScope for Sentences {
    const CATEGORIES_OF: Option<&'static LayerName> = Some(Sentences::NAME);
    fn categories_anchor() -> Anchor {
        Anchor::aligned::<Sentences>()
    }
    fn retain_categories_anchor<C: Layer>(
        doc: &mut turbo_analysis::Document,
        source_anchor: turbo_analysis::KeyId,
        target_name: &'static str,
        kept: &[turbo_analysis::Ix<C>],
        provenance: turbo_analysis::Provenance,
    ) -> Result<Option<turbo_analysis::KeyId>, turbo_analysis::Error> {
        turbo_analysis::retain_aligned_anchor::<C, Sentences>(doc, source_anchor, target_name, kept, provenance)
    }
}

impl CategoryScope for Structure {
    const CATEGORIES_OF: Option<&'static LayerName> = Some(Structure::NAME);
    fn categories_anchor() -> Anchor {
        Anchor::aligned::<Structure>()
    }
}

impl CategoryScope for Chunks {
    const CATEGORIES_OF: Option<&'static LayerName> = Some(Chunks::NAME);
    fn categories_anchor() -> Anchor {
        Anchor::aligned::<Chunks>()
    }
}

impl CategoryScope for Opinions {
    const CATEGORIES_OF: Option<&'static LayerName> = Some(Opinions::NAME);
    fn categories_anchor() -> Anchor {
        Anchor::aligned::<Opinions>()
    }
}

impl CategoryScope for Aspects {
    const CATEGORIES_OF: Option<&'static LayerName> = Some(Aspects::NAME);
    fn categories_anchor() -> Anchor {
        Anchor::aligned::<Aspects>()
    }
}

/// The categories of the items of `L`: for each item, the best category
/// and the score of every category the categorizer knows, in the
/// categorizer's category order. `Categories` (`L = View`) is the
/// document's, `turbo:categories`, with one item; `Categories<Sentences>`,
/// `Categories<Structure>` and `Categories<Chunks>` have one item per
/// sentence, block or chunk.
///
/// The best category is the categorizer's own (OpenNLP's `getBestCategory`,
/// the first highest score). [`CategoryRow::ranked`] lists every category
/// best first, ties in category order, as the document categorizers'
/// goldens list them.
pub struct Categories<L: CategoryScope = View> {
    /// The categories, in the categorizer's order.
    pub categories: Arc<TagSet>,
    /// The best category of each item.
    pub category: Vec<Tag>,
    /// The scores, item by item: `categories.len()` per item, in category
    /// order.
    pub scores: Vec<f64>,
    _of: PhantomData<fn() -> L>,
}

impl<L: CategoryScope> Clone for Categories<L> {
    fn clone(&self) -> Self {
        Self {
            categories: Arc::clone(&self.categories),
            category: self.category.clone(),
            scores: self.scores.clone(),
            _of: PhantomData,
        }
    }
}

impl<L: CategoryScope> fmt::Debug for Categories<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Categories")
            .field("categories", &self.categories.name)
            .field("category", &self.category)
            .field("scores", &self.scores)
            .finish()
    }
}

/// Equal when the category names, the best categories and the scores are.
impl<L: CategoryScope> PartialEq for Categories<L> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&self.categories, &other.categories);
        a.len() == b.len()
            && (0..a.len()).all(|i| a.label(Tag(i as u16)) == b.label(Tag(i as u16)))
            && self.category == other.category
            && self.scores == other.scores
    }
}

/// One item of a [`Categories`] layer.
#[derive(Clone, Copy, Debug)]
pub struct CategoryRow<'a> {
    /// The best category.
    pub best_tag: Tag,
    /// The best category's name.
    pub best: &'a str,
    /// The score of every category, in category order.
    pub scores: &'a [f64],
    /// The categories.
    pub categories: &'a TagSet,
}

impl PartialEq for CategoryRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.best_tag == other.best_tag && self.best == other.best && self.scores == other.scores
    }
}

impl<'a> CategoryRow<'a> {
    /// The best category's score.
    pub fn best_score(&self) -> f64 {
        self.scores[self.best_tag.0 as usize]
    }

    /// The score of `category`, or `None` when it is not a category.
    pub fn score_of(&self, category: &str) -> Option<f64> {
        let tag = self.categories.tag(category)?;
        self.scores.get(tag.0 as usize).copied()
    }

    /// Every category with its score, in category order.
    pub fn iter_scores(&self) -> impl Iterator<Item = (&'a str, f64)> + 'a {
        let cats = self.categories;
        self.scores.iter().copied().enumerate().map(move |(i, p)| (cats.label(Tag(i as u16)), p))
    }

    /// The categories best first: by score, highest first, ties in
    /// category order. For the scores a valid layer holds (no NaN) this is
    /// the order `Double.compare` gives.
    pub fn ranked_tags(&self) -> Vec<Tag> {
        let mut order: Vec<Tag> = (0..self.scores.len()).map(|i| Tag(i as u16)).collect();
        order.sort_by(|a, b| self.scores[b.0 as usize].total_cmp(&self.scores[a.0 as usize]));
        order
    }

    /// Every category with its score, best first ([`CategoryRow::ranked_tags`]).
    pub fn ranked(&self) -> Vec<(&'a str, f64)> {
        let (cats, scores) = (self.categories, self.scores);
        self.ranked_tags().into_iter().map(|t| (cats.label(t), scores[t.0 as usize])).collect()
    }
}

impl<L: CategoryScope> Categories<L> {
    /// An empty layer over `categories`.
    pub fn new(categories: Arc<TagSet>) -> Self {
        Self::with_capacity(categories, 0)
    }

    /// An empty layer over `categories` with room for `cap` items.
    pub fn with_capacity(categories: Arc<TagSet>, cap: usize) -> Self {
        let n = categories.len();
        Self {
            categories,
            category: Vec::with_capacity(cap),
            scores: Vec::with_capacity(cap.saturating_mul(n)),
            _of: PhantomData,
        }
    }

    /// Adds an item: its best category and the score of every category,
    /// in category order (`scores.len() == self.categories.len()`).
    pub fn push(&mut self, best: Tag, scores: &[f64]) {
        debug_assert_eq!(scores.len(), self.categories.len());
        self.category.push(best);
        self.scores.extend_from_slice(scores);
    }

    /// The name of item `i`'s best category.
    pub fn best_category(&self, i: usize) -> &str {
        self.categories.label(self.category[i])
    }

    /// The scores of item `i`, in category order.
    pub fn row_scores(&self, i: usize) -> &[f64] {
        let n = self.categories.len();
        &self.scores[i * n..(i + 1) * n]
    }

    /// The score of `category` for item `i`, or `None` when it is not a
    /// category.
    pub fn score_of(&self, i: usize, category: &str) -> Option<f64> {
        let tag = self.categories.tag(category)?;
        Some(self.row_scores(i)[tag.0 as usize])
    }

    /// Item `i`, or `None` past the end.
    pub fn get(&self, i: usize) -> Option<CategoryRow<'_>> {
        (i < self.category.len()).then(|| self.row(i))
    }

    /// The items in order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = CategoryRow<'_>> + '_ {
        (0..self.category.len()).map(move |i| self.row(i))
    }

    fn row(&self, i: usize) -> CategoryRow<'_> {
        let best_tag = self.category[i];
        CategoryRow {
            best_tag,
            best: self.categories.label(best_tag),
            scores: self.row_scores(i),
            categories: &self.categories,
        }
    }
}

impl Categories {
    /// The document's categories: one item, `best` with every category's
    /// score in category order.
    pub fn document(categories: Arc<TagSet>, best: Tag, scores: &[f64]) -> Categories {
        let mut c = Categories::with_capacity(categories, 1);
        c.push(best, scores);
        c
    }

    /// The document's best category and its score, or `None` when the
    /// layer has no item.
    pub fn best(&self) -> Option<(&str, f64)> {
        self.get(0).map(|r| (r.best, r.best_score()))
    }

    /// The document's categories with their scores, best first
    /// ([`CategoryRow::ranked`]); empty when the layer has no item.
    pub fn ranked(&self) -> Vec<(&str, f64)> {
        self.get(0).map(|r| r.ranked()).unwrap_or_default()
    }
}

impl<L: CategoryScope> Layer for Categories<L> {
    const ID: &'static str = "turbo:categories";
    const OF: Option<&'static LayerName> = L::CATEGORIES_OF;

    fn anchor() -> Anchor {
        L::categories_anchor()
    }

    fn len(&self) -> usize {
        self.category.len()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.category.len();
        let k = self.categories.len();
        if Self::anchor() == Anchor::Document && n > 1 {
            return Err(format!("{n} items in the document's categories, at most 1"));
        }
        if k == 0 && n > 0 {
            return Err("no categories to score".into());
        }
        if self.scores.len() != n.saturating_mul(k) {
            return Err(format!("{n} items of {k} categories and {} scores", self.scores.len()));
        }
        if self.category.iter().any(|t| t.0 as usize >= k) {
            return Err(format!("a best category outside {}", self.categories.name));
        }
        if self.scores.iter().any(|s| s.is_nan() || *s < 0.0) {
            return Err("a score that is negative or NaN".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first highest score, as OpenNLP's `getBestCategory`.
    fn first_best(scores: &[f64]) -> Tag {
        let mut best = 0;
        for i in 1..scores.len() {
            if scores[i] > scores[best] {
                best = i;
            }
        }
        Tag(best as u16)
    }

    fn document(names: &[&str], scores: &[f64]) -> Categories {
        Categories::document(TagSet::new("test", names), first_best(scores), scores)
    }

    #[test]
    fn the_document_has_one_item_and_ranks_its_categories() {
        let c = document(&["weather", "sports", "cooking"], &[0.25, 0.5, 0.25]);
        assert_eq!(c.len(), 1);
        assert_eq!(c.best(), Some(("sports", 0.5)));
        assert_eq!(c.ranked(), [("sports", 0.5), ("weather", 0.25), ("cooking", 0.25)]);
        assert_eq!(c.get(0).unwrap().score_of("cooking"), Some(0.25));
        assert_eq!(c.get(0).unwrap().score_of("finance"), None);
        assert_eq!(c.check(), Ok(()));

        let empty: Categories = Categories::new(TagSet::new("test", &["a"]));
        assert_eq!(empty.best(), None);
        assert!(empty.ranked().is_empty());
        assert_eq!(empty.check(), Ok(()));
    }

    #[test]
    fn names_and_anchors() {
        assert_eq!(<Categories>::NAME.to_string(), "turbo:categories");
        assert_eq!(<Categories>::anchor(), Anchor::Document);
        assert_eq!(Categories::<Sentences>::NAME.to_string(), "turbo:categories<opennlp:sentences>");
        assert_eq!(Categories::<Sentences>::anchor(), Anchor::aligned::<Sentences>());
        assert_eq!(Categories::<Structure>::NAME.to_string(), "turbo:categories<turbo:structure>");
        assert_eq!(Categories::<Chunks>::NAME.to_string(), "turbo:categories<turbo:chunks>");
        assert_eq!(Categories::<Chunks>::anchor(), Anchor::aligned::<Chunks>());
    }

    #[test]
    fn an_item_per_unit() {
        let mut c = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
        c.push(Tag(1), &[0.25, 0.75]);
        c.push(Tag(0), &[0.5, 0.5]);
        assert_eq!(c.len(), 2);
        assert_eq!(c.best_category(0), "b");
        assert_eq!(c.row_scores(1), [0.5, 0.5]);
        assert_eq!(c.score_of(0, "a"), Some(0.25));
        let rows: Vec<_> = c.iter().map(|r| (r.best, r.ranked())).collect();
        assert_eq!(rows, [("b", vec![("b", 0.75), ("a", 0.25)]), ("a", vec![("a", 0.5), ("b", 0.5)])]);
        assert!(c.get(2).is_none());
        assert_eq!(c.check(), Ok(()));
    }

    #[test]
    fn check_refuses_inconsistent_layers() {
        let mut c = document(&["a", "b"], &[0.5, 0.5]);
        c.push(Tag(0), &[0.5, 0.5]);
        assert_eq!(c.check(), Err("2 items in the document's categories, at most 1".into()));

        let mut c = Categories::<Sentences>::new(TagSet::new("test", &["a", "b"]));
        c.category.push(Tag(0));
        c.scores.push(1.0);
        assert_eq!(c.check(), Err("1 items of 2 categories and 1 scores".into()));
        c.scores.push(0.0);
        c.category[0] = Tag(2);
        assert_eq!(c.check(), Err("a best category outside test".into()));
        c.category[0] = Tag(0);
        c.scores[1] = f64::NAN;
        assert_eq!(c.check(), Err("a score that is negative or NaN".into()));

        let mut c = Categories::<Sentences>::new(TagSet::new("test", &[]));
        c.category.push(Tag(0));
        assert_eq!(c.check(), Err("no categories to score".into()));
    }
}
