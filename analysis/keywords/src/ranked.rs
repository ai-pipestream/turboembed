//! The extractor output both algorithms share, before it becomes layer
//! items.

use turbo_analysis::Span;

/// One ranked keyphrase, every occurrence span attached. `salience` is
/// what the `turbo:keywords` layer stores: a score in `[0, 1]` where
/// higher is more salient (YAKE normalizes its lower-is-better raw
/// score across the survivors; TextRank's centrality already lies in
/// `[0, 1]`). `rank` is 0-based, best first.
#[derive(Clone, Debug, PartialEq)]
pub struct RankedKeyword {
    /// The normalized (folded) keyphrase, terms joined by one space.
    pub phrase: String,
    /// How many terms the keyphrase holds.
    pub terms: u16,
    /// The salience in `[0, 1]`, higher better.
    pub salience: f64,
    /// Every occurrence span, in document order.
    pub occurrences: Vec<Span>,
    /// The 0-based position in the ranking, best first.
    pub rank: usize,
}
