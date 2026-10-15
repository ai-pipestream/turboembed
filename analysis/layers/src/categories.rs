//! The document's categories, as a document categorizer scores them.

use turbo_analysis::{Anchor, Layer};

use crate::Strs;

/// The document's categories: a document-scoped layer holding every
/// category the categorizer knows, best first, each with its score. Item 0
/// is the best category. OpenNLP's SVM document categorizer scores each
/// with its probability estimate; categories of equal score keep the
/// model's category order.
pub struct Categories {
    pub category: Strs,
    pub score: Vec<f64>,
}

impl Categories {
    /// The best category and its score: item 0, or `None` when the layer
    /// is empty.
    pub fn best(&self) -> Option<(&str, f64)> {
        self.score.first().map(|&s| (self.category.get(0), s))
    }

    /// The categories with their scores, best first.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.category.iter().zip(self.score.iter().copied())
    }
}

impl Layer for Categories {
    const ID: &'static str = "turbo:categories";
    fn anchor() -> Anchor {
        Anchor::Document
    }
    fn len(&self) -> usize {
        self.category.len()
    }
    fn check(&self) -> Result<(), String> {
        if self.category.len() != self.score.len() {
            return Err(format!("{} categories and {} scores", self.category.len(), self.score.len()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn categories(rows: &[(&str, f64)]) -> Categories {
        let mut category = Strs::default();
        for (c, _) in rows {
            category.push(c);
        }
        Categories { category, score: rows.iter().map(|r| r.1).collect() }
    }

    #[test]
    fn the_best_category_is_the_first_row() {
        let c = categories(&[("sports", 0.75), ("weather", 0.25)]);
        assert_eq!(c.best(), Some(("sports", 0.75)));
        assert_eq!(c.iter().collect::<Vec<_>>(), [("sports", 0.75), ("weather", 0.25)]);
        assert_eq!(categories(&[]).best(), None);
    }

    #[test]
    fn a_score_for_every_category() {
        let mut c = categories(&[("a", 0.5)]);
        assert_eq!(c.check(), Ok(()));
        c.score.push(0.5);
        assert_eq!(c.check(), Err("1 categories and 2 scores".into()));
        assert_eq!(Categories::anchor(), Anchor::Document);
        assert_eq!(Categories::NAME.to_string(), "turbo:categories");
    }
}
