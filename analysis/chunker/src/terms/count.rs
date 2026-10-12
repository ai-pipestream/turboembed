//! Term counts per unit of one level.

use std::marker::PhantomData;
use std::ops::Range;

use turbo_analysis::{Annotator, Error, Ix, Layer, Refs};
use turbo_analysis_layers::Tokens;

use super::layer::{TermStats, Terms, Vocabulary};

/// Reads the tokens, their terms and the vocabulary, and the units of one
/// level (`L`); writes the term counts of each unit (`TermStats<L>`).
///
/// - `TermCounter<Chunks>`: a row per chunk, at every level of the chunk
///   hierarchy.
/// - `TermCounter<Structure>`: a row per structure block. The sections are
///   the blocks of kind `Block::Section`; a nested section's terms count in
///   it and in every section around it.
/// - `TermCounter<View>`: one row, the whole document.
///
/// A token is in a unit when its start is: units at sentence boundaries
/// never cut a token, and a token a unit's end does cut is counted once,
/// in the unit it starts in. A layer with spans must be in the tokens'
/// view; one without (a `View`) is one unit holding every token. Two
/// binary searches per unit over the token starts, then a sort of the
/// unit's term ids.
pub struct TermCounter<L> {
    _of: PhantomData<fn() -> L>,
}

impl<L> TermCounter<L> {
    pub fn new() -> TermCounter<L> {
        TermCounter { _of: PhantomData }
    }
}

impl<L> Default for TermCounter<L> {
    fn default() -> TermCounter<L> {
        TermCounter::new()
    }
}

impl<L: Layer> Annotator for TermCounter<L> {
    const ID: &'static str = "turbo:term-counter";
    type Reads = (Tokens, Terms, Vocabulary, L);
    type Writes = (TermStats<L>,);

    fn annotate(&self, (tokens, terms, vocabulary, units): Refs<'_, Self>) -> Result<(TermStats<L>,), Error> {
        let fail = |detail: String| Error::Annotator { annotator: Self::ID, detail };
        let (starts, _) = tokens.spans.columns();
        if !starts.windows(2).all(|w| w[0] <= w[1]) {
            return Err(fail("the tokens are not in text order".into()));
        }
        let ranges: Vec<Range<usize>> = match units.text_spans() {
            Some((view, _, _)) if view != tokens.spans.view() => {
                return Err(fail(format!(
                    "the units are in view {}, the tokens in view {}",
                    view.0,
                    tokens.spans.view().0
                )));
            }
            Some((_, a, b)) => a
                .iter()
                .zip(b)
                .map(|(&a, &b)| starts.partition_point(|&s| s < a)..starts.partition_point(|&s| s < b))
                .collect(),
            None if units.len() == 1 => std::iter::once(0..starts.len()).collect(),
            None => return Err(fail(format!("{} units with no spans; only one can hold every token", units.len()))),
        };
        Ok((count(&terms.term, vocabulary.len(), &ranges).map_err(fail)?,))
    }
}

/// The term counts of each run of tokens. Refused when a unit holds, or
/// all units together hold, more terms than a u32 counts.
pub fn count<L>(
    term: &[Option<Ix<Vocabulary>>],
    vocabulary: usize,
    units: &[Range<usize>],
) -> Result<TermStats<L>, String> {
    let mut s = TermStats {
        row: Vec::with_capacity(units.len() + 1),
        term: Vec::new(),
        tf: Vec::new(),
        length: Vec::with_capacity(units.len()),
        df: vec![0; vocabulary],
        _of: PhantomData,
    };
    s.row.push(0);
    let mut ids = Vec::new();
    for r in units {
        ids.clear();
        ids.extend(term[r.clone()].iter().flatten().copied());
        ids.sort_unstable();
        for &t in &ids {
            if s.term.len() > *s.row.last().unwrap() as usize && s.term.last() == Some(&t) {
                *s.tf.last_mut().unwrap() += 1;
            } else {
                s.term.push(t);
                s.tf.push(1);
                s.df[t.get()] += 1;
            }
        }
        let too_many = |what: &str, n: usize| format!("{what} of {n} terms, more than a u32 counts");
        s.length.push(u32::try_from(ids.len()).map_err(|_| too_many("a unit", ids.len()))?);
        s.row.push(u32::try_from(s.term.len()).map_err(|_| too_many("rows", s.term.len()))?);
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use turbo_analysis_layers::Chunks;

    fn ids(xs: &[Option<u32>]) -> Vec<Option<Ix<Vocabulary>>> {
        xs.iter().map(|x| x.map(Ix::new)).collect()
    }

    #[test]
    fn counts_are_sparse_sorted_rows_with_lengths_and_df() {
        // Terms 0..3; None is a token that is no term.
        let term = ids(&[Some(2), Some(0), None, Some(2), Some(1), Some(2)]);
        let s: TermStats<Chunks> = count(&term, 4, &[0..4, 4..6, 0..6, 3..3]).unwrap();
        assert_eq!(s.row, [0, 2, 4, 7, 7]);
        let (t, tf) = s.unit(Ix::new(0));
        assert_eq!((t.iter().map(|t| t.get()).collect::<Vec<_>>(), tf), (vec![0, 2], &[1, 2][..]));
        assert_eq!(s.length, [3, 2, 5, 0]);
        assert_eq!(s.df, [2, 2, 3, 0]);
        assert_eq!(s.tf(Ix::new(2), Ix::new(2)), 3);
        assert_eq!(s.tf(Ix::new(1), Ix::new(0)), 0);
        assert_eq!(s.check(), Ok(()));
    }

    #[test]
    fn a_repeated_term_after_a_unit_ends_starts_a_new_entry() {
        // Unit 0 ends with term 1, unit 1 starts with it.
        let s: TermStats<Chunks> = count(&ids(&[Some(1), Some(1)]), 2, &[0..1, 1..2]).unwrap();
        assert_eq!((s.row.as_slice(), s.tf.as_slice(), s.df.as_slice()), (&[0, 1, 2][..], &[1, 1][..], &[0, 2][..]));
        assert_eq!(s.check(), Ok(()));
    }

    #[test]
    fn the_check_refuses_inconsistent_columns() {
        let good = || count::<Chunks>(&ids(&[Some(0), Some(1), Some(1)]), 2, std::slice::from_ref(&(0..3))).unwrap();
        let mut s = good();
        s.length[0] = 2;
        assert!(s.check().unwrap_err().contains("sum of its counts"));
        let mut s = good();
        s.term.swap(0, 1);
        assert!(s.check().unwrap_err().contains("ascending"));
        let mut s = good();
        s.df[0] = 2;
        assert!(s.check().unwrap_err().contains("df"));
        let mut s = good();
        s.df.pop();
        assert!(s.check().unwrap_err().contains("past the vocabulary"));
        let mut s = good();
        s.row.pop();
        assert!(s.check().unwrap_err().contains("row start"));
        let mut s = good();
        s.tf[0] = 0;
        s.length[0] = 2;
        assert!(s.check().unwrap_err().contains("zero"));
    }
}
