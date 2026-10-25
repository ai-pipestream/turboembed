//! Offsets. Every offset in the library is a UTF-8 byte offset into one
//! text view, half-open, as u32: the unit turboembed's tokenizer spans
//! already use. UTF-16 and code point offsets are conversions at the edge
//! (Java, the wire), never a second coordinate system inside.

use std::fmt;
use std::ops::Range;

mod index;
mod join;
mod spans;
mod text;

pub use index::{Ix, IxRange};
pub use join::{ContainedJoin, OverlappingJoin};
pub use spans::{Fragments, Spans, filter_spans};
pub use text::TextSpans;

#[cfg(test)]
mod tests;

/// Bytes `[start, end)` of one text view.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    pub const fn new(start: u32, end: u32) -> Span {
        assert!(start <= end, "a span ends at or after its start");
        Span { start, end }
    }

    pub const fn len(self) -> u32 {
        self.end - self.start
    }

    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }

    pub const fn contains(self, other: Span) -> bool {
        self.start <= other.start && other.end <= self.end
    }

    /// Whether `self` covers byte `offset` (for an empty span at `offset`,
    /// `self.start == offset`; otherwise `self.start <= offset < self.end`).
    pub const fn covers(self, offset: u32) -> bool {
        if self.is_empty() { self.start == offset } else { self.start <= offset && offset < self.end }
    }

    /// Whether `self` and `other` share at least one byte, or an empty
    /// span lies at the start of or inside the other.
    pub const fn overlaps(self, other: Span) -> bool {
        if self.is_empty() && other.is_empty() {
            self.start == other.start
        } else if self.is_empty() {
            other.start <= self.start && self.start < other.end
        } else if other.is_empty() {
            self.start <= other.start && other.start < self.end
        } else {
            self.start < other.end && other.start < self.end
        }
    }

    pub fn range(self) -> Range<usize> {
        self.start as usize..self.end as usize
    }
}

/// What overlap a positional layer's spans may have.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Overlap {
    /// Continuous spans in text order (`start` non-decreasing) that do not
    /// overlap (`end[i - 1] <= start[i]`).
    #[default]
    None,
    /// Spans and individual fragments never partially cross (`a.start <
    /// b.start && b.start < a.end && a.end < b.end`), and whenever two
    /// items share covered bytes, every fragment of one item lies inside a
    /// fragment of the other (whilst a separate item in the uncovered gap
    /// of a discontinuous item may overlap its bounding envelope without
    /// sharing covered bytes).
    Nesting,
    /// Any valid spans in the view, including partially crossing spans.
    Any,
}

/// How overlapping or duplicate spans are resolved into a subset whose
/// bounding envelopes do not overlap (`!a.overlaps(b)`), returned in text
/// order.
///
/// Conflicts and span lengths (`end - start`) are evaluated on each item's
/// bounding envelope (`first.start..last.end`), so for a discontinuous item
/// the gap between fragments is counted in its length and treated as covered
/// when checking conflicts. When every kept item is continuous, the result
/// satisfies [`Overlap::None`].
///
/// Scores are compared with [`f64::total_cmp`] (layers such as `Entities`
/// refuse `NaN` scores in [`crate::Layer::check`]).
///
/// Compared with OpenNLP's `NameFinderME.dropOverlappingSpans`:
/// - [`FilterRule::LeftmostLongest`] matches `dropOverlappingSpans` on all
///   non-empty, distinct spans.
/// - On empty spans, [`Span::overlaps`] treats an empty span at the end of a
///   non-empty span (`[3, 3)` after `[1, 3)`) as non-overlapping, whereas
///   OpenNLP's `Span.intersects` counts containment.
/// - On identical spans of different entity types, `dropOverlappingSpans`
///   keeps the UTF-16-first type (`String.compareTo`), whereas `filter_spans`
///   breaks ties by higher score and then earlier item index.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FilterRule {
    /// Longer envelope (`end - start`) wins; ties broken by higher score,
    /// then earlier start, then earlier item index. Matches spaCy's
    /// `util.filter_spans` when scores are equal.
    LongestFirst,
    /// Higher score wins; ties broken by longer envelope (`end - start`),
    /// then earlier start, then earlier item index.
    HighestScore,
    /// Earlier start (`start`) wins; at the same start, longer envelope
    /// (`end - start`) wins, then higher score, then earlier item index.
    LeftmostLongest,
}

/// Why [`Spans::try_push_fragments`] refused a slice of fragments.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FragmentError {
    /// `fragments` is empty.
    Empty,
    /// A fragment's `start` is greater than its `end`.
    EndsBeforeStart { start: u32, end: u32 },
    /// A discontinuous item (`fragments.len() > 1`) has an empty fragment.
    EmptyFragment { start: u32 },
    /// Two adjacent fragments are not strictly separated (`first.end >= second.start`).
    NotSeparated { first: Span, second: Span },
}

impl fmt::Display for FragmentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            FragmentError::Empty => f.write_str("a span needs at least one fragment"),
            FragmentError::EndsBeforeStart { start, end } => {
                write!(f, "fragment {start}..{end} ends before it starts")
            }
            FragmentError::EmptyFragment { start } => {
                write!(f, "discontinuous span has an empty fragment at {start}..{start}")
            }
            FragmentError::NotSeparated { first, second } => write!(
                f,
                "fragments {}..{} and {}..{} are not separated in text order",
                first.start, first.end, second.start, second.end
            ),
        }
    }
}

impl std::error::Error for FragmentError {}
