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

    /// Co-written anchor keys when `Categories<Self>` is filtered onto `target_name`.
    fn categories_co_written_anchors(_target_name: &'static str) -> Vec<(turbo_analysis::KeyId, Anchor)> {
        Vec::new()
    }

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
    fn categories_co_written_anchors(target_name: &'static str) -> Vec<(turbo_analysis::KeyId, Anchor)> {
        turbo_analysis::co_written_aligned_anchors::<Sentences>(target_name)
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
///
/// Each item also records the label set the categorizer selected, in
/// category order: every label a multi-label categorizer emitted, `[best]`
/// for a single-label categorizer, and empty where the item abstained
/// (`best` [`Tag::NONE`]). Two items with the same best category and
/// scores but different selected sets (thresholds [0.5, 0.5] emitting
/// {a, b} against [0.5, 0.9] emitting {a}) stay distinguishable. The sets
/// are one CSR column pair, `selected` holding every item's tags
/// contiguous and `selected_end` one prefix offset per item, so pushing an
/// item appends and never allocates once the layer is reserved.
pub struct Categories<L: CategoryScope = View> {
    /// The categories, in the categorizer's order.
    pub categories: Arc<TagSet>,
    /// The best category of each item: the categorizer's winning [`Tag`],
    /// or [`Tag::NONE`] when the item was scored but nothing was predicted
    /// (a multi-label categorizer whose no label cleared its threshold,
    /// abstaining instead of forcing a best category).
    pub category: Vec<Tag>,
    /// Every item's selected labels contiguous, in category order: item
    /// `i` holds `selected[selected_end[i - 1]..selected_end[i]]` (from
    /// the start when `i == 0`). Empty on abstention. `[category[i]]`
    /// unless a categorizer selected a different set
    /// ([`Categories::push_selected`]).
    pub selected: Vec<Tag>,
    /// One prefix offset per item into [`Categories::selected`], the end
    /// of item `i`'s run; the last entry is `selected.len()`.
    pub selected_end: Vec<u32>,
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
            selected: self.selected.clone(),
            selected_end: self.selected_end.clone(),
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
            .field("selected", &self.selected)
            .field("selected_end", &self.selected_end)
            .field("scores", &self.scores)
            .finish()
    }
}

/// Equal when the category names, the best categories, the selected sets,
/// and the scores are.
impl<L: CategoryScope> PartialEq for Categories<L> {
    fn eq(&self, other: &Self) -> bool {
        let (a, b) = (&self.categories, &other.categories);
        a.len() == b.len()
            && (0..a.len()).all(|i| a.label(Tag(i as u16)) == b.label(Tag(i as u16)))
            && self.category == other.category
            && self.selected == other.selected
            && self.selected_end == other.selected_end
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
    /// The label set the item was given, in category order; empty where
    /// the item abstained ([`Tag::NONE`]).
    pub selected: &'a [Tag],
    /// The score of every category, in category order.
    pub scores: &'a [f64],
    /// The categories.
    pub categories: &'a TagSet,
}

impl PartialEq for CategoryRow<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.best_tag == other.best_tag
            && self.best == other.best
            && self.selected == other.selected
            && self.scores == other.scores
    }
}

impl<'a> CategoryRow<'a> {
    /// The best category's score, or 0.0 when the item abstained
    /// ([`Tag::NONE`]: nothing was predicted, so there is no best score).
    pub fn best_score(&self) -> f64 {
        if self.best_tag == Tag::NONE { 0.0 } else { self.scores[self.best_tag.0 as usize] }
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
            selected: Vec::with_capacity(cap),
            selected_end: Vec::with_capacity(cap),
            scores: Vec::with_capacity(cap.saturating_mul(n)),
            _of: PhantomData,
        }
    }

    /// Adds an item: its best category and the score of every category,
    /// in category order (`scores.len() == self.categories.len()`). The
    /// best is [`Tag::NONE`] when the item abstained: scored, but nothing
    /// predicted, as a multi-label categorizer emits when no label clears
    /// its threshold. The selected set is `[best]`, or empty on
    /// abstention; categorizers selecting several labels use
    /// [`Categories::push_selected`]. Appends to the reserved CSR columns:
    /// no heap allocation once `with_capacity` reserved the items.
    pub fn push(&mut self, best: Tag, scores: &[f64]) {
        debug_assert_eq!(scores.len(), self.categories.len());
        if best != Tag::NONE {
            self.selected.push(best);
        }
        self.selected_end.push(self.selected.len() as u32);
        self.category.push(best);
        self.scores.extend_from_slice(scores);
    }

    /// Adds an item like [`Categories::push`], with the full label set the
    /// item is given: every label a multi-label categorizer emitted, in
    /// category order (strictly ascending), including the `best` (its
    /// highest-scoring member) and empty only with `best` [`Tag::NONE`],
    /// the abstention.
    pub fn push_selected(&mut self, best: Tag, selected: &[Tag], scores: &[f64]) {
        debug_assert_eq!(scores.len(), self.categories.len());
        self.selected.extend_from_slice(selected);
        self.selected_end.push(self.selected.len() as u32);
        self.category.push(best);
        self.scores.extend_from_slice(scores);
    }

    /// The label set item `i` was given, in category order; empty where
    /// the item abstained ([`Tag::NONE`]).
    pub fn selected(&self, i: usize) -> &[Tag] {
        let start = if i == 0 { 0 } else { self.selected_end[i - 1] };
        &self.selected[start as usize..self.selected_end[i] as usize]
    }

    /// The name of item `i`'s best category, or `""` when the item
    /// abstained ([`Tag::NONE`]: nothing was predicted).
    pub fn best_category(&self, i: usize) -> &str {
        let tag = self.category[i];
        if tag == Tag::NONE { "" } else { self.categories.label(tag) }
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
            best: if best_tag == Tag::NONE { "" } else { self.categories.label(best_tag) },
            selected: self.selected(i),
            scores: self.row_scores(i),
            categories: &self.categories,
        }
    }
}

impl Categories {
    /// The document's categories: one item, `best` with every category's
    /// score in category order, the selected set `[best]` (empty on
    /// abstention).
    pub fn document(categories: Arc<TagSet>, best: Tag, scores: &[f64]) -> Categories {
        let mut c = Categories::with_capacity(categories, 1);
        c.push(best, scores);
        c
    }

    /// The document's categories with the full selected label set, as
    /// [`Categories::push_selected`] describes.
    pub fn document_selected(categories: Arc<TagSet>, best: Tag, selected: &[Tag], scores: &[f64]) -> Categories {
        let mut c = Categories::with_capacity(categories, 1);
        c.push_selected(best, selected, scores);
        c
    }

    /// The document's best category and its score, or `None` when the layer
    /// has no item or the document abstained ([`Tag::NONE`]).
    pub fn best(&self) -> Option<(&str, f64)> {
        self.get(0).and_then(|r| (r.best_tag != Tag::NONE).then(|| (r.best, r.best_score())))
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
        if self.selected_end.len() != n {
            return Err(format!("{n} items and {} selected row ends", self.selected_end.len()));
        }
        if self.category.iter().any(|t| *t != Tag::NONE && t.0 as usize >= k) {
            return Err(format!("a best category outside {}", self.categories.name));
        }
        let mut start = 0usize;
        for (i, &end) in self.selected_end.iter().enumerate() {
            let end = end as usize;
            if end < start || end > self.selected.len() {
                return Err(format!("selected row {i} ends past the selected labels"));
            }
            let set = &self.selected[start..end];
            if set.iter().any(|t| t.0 as usize >= k) {
                return Err(format!("a selected label of item {i} outside {}", self.categories.name));
            }
            if set.windows(2).any(|w| w[0] >= w[1]) {
                return Err(format!("the selected labels of item {i} not in category order"));
            }
            if self.category[i] == Tag::NONE && !set.is_empty() {
                return Err(format!("item {i} abstained but selected labels"));
            }
            if self.category[i] != Tag::NONE && !set.contains(&self.category[i]) {
                return Err(format!("item {i}'s best category not among its selected labels"));
            }
            start = end;
        }
        if start != self.selected.len() {
            return Err(format!("{} selected labels past the last row end", self.selected.len() - start));
        }
        if self.scores.iter().any(|s| s.is_nan() || *s < 0.0) {
            return Err("a score that is negative or NaN".into());
        }
        Ok(())
    }
}
