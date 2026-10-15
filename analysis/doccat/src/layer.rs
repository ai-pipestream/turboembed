//! The `Categories` layer (`turbo:categories<L>`), aligned to `View` (whole
//! document), `Sentences` (sentence-level), `Structure` (paragraph/section),
//! or `Chunks` (chunk-level).

use std::fmt;
use std::marker::PhantomData;
use std::sync::Arc;

use turbo_analysis::{Anchor, Ix, Layer, LayerName, Span, Spans, Tag, TagSet, View};
use turbo_analysis_layers::{Chunks, Sentences, Structure};

/// Categorization results aligned one-for-one with items of layer `L`.
///
/// When `L = View` (the default), the layer has 1 row describing the whole
/// view (`turbo:categories<turbo:view>`). When `L = Sentences`,
/// `Structure`, or `Chunks`, item `i` holds the best category and full
/// per-category probability distribution for sentence, block, or chunk `i`.
pub struct Categories<L: Layer = View> {
    /// Closed set of category outcome names in model outcome order.
    pub categories: Arc<TagSet>,
    /// Best category tag for each item of `L` (`len == L::len`).
    pub category: Vec<Tag>,
    /// Flattened row-major probability matrix of shape `[len, categories.len()]`.
    pub scores: Vec<f64>,
    _of: PhantomData<fn() -> L>,
}

impl<L: Layer> Clone for Categories<L> {
    fn clone(&self) -> Self {
        Self {
            categories: Arc::clone(&self.categories),
            category: self.category.clone(),
            scores: self.scores.clone(),
            _of: PhantomData,
        }
    }
}

impl<L: Layer> fmt::Debug for Categories<L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Categories")
            .field("categories", &self.categories.name)
            .field("category", &self.category)
            .field("scores", &self.scores)
            .finish()
    }
}

impl<L: Layer> PartialEq for Categories<L> {
    fn eq(&self, other: &Self) -> bool {
        self.categories.len() == other.categories.len()
            && (0..self.categories.len())
                .all(|i| self.categories.label(Tag(i as u16)) == other.categories.label(Tag(i as u16)))
            && self.category == other.category
            && self.scores == other.scores
    }
}

/// Read-only view of one categorized item in [`Categories`].
#[derive(Clone, Copy, Debug)]
pub struct CategoryRow<'a> {
    /// Interned tag of the winning category.
    pub best_tag: Tag,
    /// Name of the winning category.
    pub best: &'a str,
    /// Per-category probability distribution in model outcome order.
    pub scores: &'a [f64],
    /// The model's category tag set.
    pub categories: &'a TagSet,
}

impl<'a> PartialEq for CategoryRow<'a> {
    fn eq(&self, other: &Self) -> bool {
        self.best_tag == other.best_tag && self.best == other.best && self.scores == other.scores
    }
}

impl<'a> CategoryRow<'a> {
    /// Returns the probability assigned to `category`, if it is in the model's
    /// outcome set.
    pub fn score_of(&self, category: &str) -> Option<f64> {
        let tag = self.categories.tag(category)?;
        self.scores.get(tag.0 as usize).copied()
    }

    /// Iterates `(category_name, probability)` pairs in model outcome order.
    pub fn iter_scores(&self) -> impl Iterator<Item = (&'a str, f64)> + 'a {
        let cats = self.categories;
        self.scores.iter().copied().enumerate().map(move |(i, p)| (cats.label(Tag(i as u16)), p))
    }
}

impl<L: Layer> Categories<L> {
    /// Creates an empty [`Categories`] layer for the given `categories` tag set.
    pub fn new(categories: Arc<TagSet>) -> Self {
        Self::with_capacity(categories, 0)
    }

    /// Creates an empty [`Categories`] layer with pre-allocated capacity for
    /// `cap` items.
    pub fn with_capacity(categories: Arc<TagSet>, cap: usize) -> Self {
        let num_cats = categories.len();
        Self {
            categories,
            category: Vec::with_capacity(cap),
            scores: Vec::with_capacity(cap.saturating_mul(num_cats)),
            _of: PhantomData,
        }
    }

    /// Appends one categorized item with its winning `best` tag and per-category
    /// `row_scores` slice (`row_scores.len() == self.categories.len()`).
    pub fn push(&mut self, best: Tag, row_scores: &[f64]) {
        debug_assert_eq!(row_scores.len(), self.categories.len());
        self.category.push(best);
        self.scores.extend_from_slice(row_scores);
    }

    /// Returns the winning category name for item `i`.
    pub fn best_category(&self, i: usize) -> &str {
        self.categories.label(self.category[i])
    }

    /// Returns the probability slice for item `i` in model outcome order.
    pub fn row_scores(&self, i: usize) -> &[f64] {
        let n = self.categories.len();
        &self.scores[i * n..(i + 1) * n]
    }

    /// Returns the probability of `category` for item `i`, if `category` is
    /// in the model's tag set.
    pub fn score_of(&self, i: usize, category: &str) -> Option<f64> {
        let tag = self.categories.tag(category)?;
        Some(self.row_scores(i)[tag.0 as usize])
    }

    /// Returns a [`CategoryRow`] view of item `i`, or `None` if `i >= self.len()`.
    pub fn get(&self, i: usize) -> Option<CategoryRow<'_>> {
        if i >= self.category.len() {
            return None;
        }
        let best_tag = self.category[i];
        Some(CategoryRow {
            best_tag,
            best: self.categories.label(best_tag),
            scores: self.row_scores(i),
            categories: &self.categories,
        })
    }

    /// Iterates over all categorized rows in item order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = CategoryRow<'_>> + '_ {
        (0..self.category.len()).map(move |i| {
            let best_tag = self.category[i];
            CategoryRow {
                best_tag,
                best: self.categories.label(best_tag),
                scores: self.row_scores(i),
                categories: &self.categories,
            }
        })
    }
}

impl<L: Layer> Layer for Categories<L> {
    const ID: &'static str = "turbo:categories";
    const OF: Option<&'static LayerName> = Some(L::NAME);

    fn anchor() -> Anchor {
        Anchor::aligned::<L>()
    }

    fn len(&self) -> usize {
        self.category.len()
    }

    fn check(&self) -> Result<(), String> {
        let n = self.category.len();
        let num_cats = self.categories.len();
        if num_cats == 0 && n > 0 {
            return Err("categories tagset cannot be empty when items are present".into());
        }
        if self.scores.len() != n.saturating_mul(num_cats) {
            return Err(format!(
                "expected {} category scores ({} items * {} categories), got {}",
                n.saturating_mul(num_cats),
                n,
                num_cats,
                self.scores.len()
            ));
        }
        if self.category.iter().any(|t| t.0 as usize >= num_cats) {
            return Err(format!("category tag out of bounds for {}", self.categories.name));
        }
        if self.scores.iter().any(|s| s.is_nan() || *s < 0.0) {
            return Err("category scores must be non-negative and not NaN".into());
        }
        Ok(())
    }
}

/// Trait implemented by span-carrying layers (`Sentences`, `Structure`, `Chunks`)
/// that [`crate::SpanCategorizer`] can categorize row-by-row.
pub trait SpanCategorizable: Layer {
    /// Returns the spans of this layer.
    fn layer_spans(&self) -> &Spans<Self>;

    /// Returns the `i`-th span of this layer.
    fn span_at(&self, i: usize) -> Span {
        self.layer_spans().get(Ix::new(i as u32))
    }
}

impl SpanCategorizable for Sentences {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}

impl SpanCategorizable for Structure {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}

impl SpanCategorizable for Chunks {
    fn layer_spans(&self) -> &Spans<Self> {
        &self.spans
    }
}
