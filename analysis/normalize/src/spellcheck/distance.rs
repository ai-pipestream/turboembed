//! Edit distances with an upper bound, over code points.
//!
//! Both metrics count insertions, deletions and substitutions at cost one.
//! The optimal string alignment distance (restricted Damerau-Levenshtein)
//! also counts a transposition of two adjacent symbols as one edit, with
//! the restriction that no substring is edited twice; plain Levenshtein
//! counts a transposition as two edits. A symbol is a Unicode code point,
//! so a supplementary character is one symbol, as in OpenNLP's
//! `DamerauOSADistance` and `LevenshteinDistance`.
//!
//! The computation keeps three rows of the dynamic programming table (two
//! for Levenshtein), trims the common prefix and suffix first, and stops as
//! soon as every cell of a row exceeds the bound.

/// An edit distance with an early exit.
pub trait EditDistance {
    /// The distance between `a` and `b`, or `None` when it is greater than
    /// `max`. A `max` of 0 asks only whether the two are equal.
    fn distance(&self, a: &str, b: &str, max: usize) -> Option<usize>;
}

/// The two metrics the engine and the model format know, by the ids the
/// model format stores.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum Metric {
    /// Optimal string alignment: a transposition is one edit.
    #[default]
    DamerauOsa,
    /// Levenshtein: a transposition is two edits.
    Levenshtein,
}

impl Metric {
    /// The id the model format stores: `damerau-osa` or `levenshtein`.
    pub fn id(self) -> &'static str {
        match self {
            Metric::DamerauOsa => "damerau-osa",
            Metric::Levenshtein => "levenshtein",
        }
    }

    /// The metric an id names.
    pub fn from_id(id: &str) -> Option<Metric> {
        [Metric::DamerauOsa, Metric::Levenshtein].into_iter().find(|m| m.id() == id)
    }

    /// The bounded distance between two code point sequences, with the rows
    /// of the table in `rows` so that repeated calls do not allocate.
    pub(crate) fn chars(self, a: &[char], b: &[char], max: usize, rows: &mut Rows) -> Option<usize> {
        bounded(a, b, max, self == Metric::DamerauOsa, rows)
    }
}

impl EditDistance for Metric {
    fn distance(&self, a: &str, b: &str, max: usize) -> Option<usize> {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        self.chars(&a, &b, max, &mut Rows::default())
    }
}

/// The optimal string alignment distance, OpenNLP's default metric.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DamerauOsa;

/// The Levenshtein distance.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Levenshtein;

impl EditDistance for DamerauOsa {
    fn distance(&self, a: &str, b: &str, max: usize) -> Option<usize> {
        Metric::DamerauOsa.distance(a, b, max)
    }
}

impl EditDistance for Levenshtein {
    fn distance(&self, a: &str, b: &str, max: usize) -> Option<usize> {
        Metric::Levenshtein.distance(a, b, max)
    }
}

/// The rows of the table: the row being filled, the one before it, and
/// the one before that (read only for transpositions).
#[derive(Debug, Default)]
pub(crate) struct Rows {
    cur: Vec<usize>,
    prev: Vec<usize>,
    before: Vec<usize>,
}

fn bounded(a: &[char], b: &[char], max: usize, transpositions: bool, rows: &mut Rows) -> Option<usize> {
    // The shorter sequence indexes the columns, which keeps the rows short.
    let (s1, s2) = if a.len() > b.len() { (b, a) } else { (a, b) };
    // A common suffix and a common prefix add nothing to the distance.
    let mut len1 = s1.len();
    let mut len2 = s2.len();
    while len1 > 0 && s1[len1 - 1] == s2[len2 - 1] {
        len1 -= 1;
        len2 -= 1;
    }
    let mut offset = 0;
    while offset < len1 && s1[offset] == s2[offset] {
        offset += 1;
    }
    let s1 = &s1[offset..len1];
    let s2 = &s2[offset..len2];
    if s1.is_empty() {
        return (s2.len() <= max).then_some(s2.len());
    }
    // The distance is at least the difference of the lengths.
    if s2.len() - s1.len() > max {
        return None;
    }

    let n = s1.len();
    let Rows { cur, prev, before } = rows;
    for row in [&mut *cur, &mut *prev, &mut *before] {
        row.clear();
        row.resize(n + 1, 0);
    }
    for (j, cell) in prev.iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=s2.len() {
        let c2 = s2[i - 1];
        let prev_c2 = if i >= 2 { Some(s2[i - 2]) } else { None };
        cur[0] = i;
        let mut row_min = i;
        for j in 1..=n {
            let c1 = s1[j - 1];
            let cost = usize::from(c1 != c2);
            let mut value = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
            if transpositions && j >= 2 && prev_c2 == Some(c1) && s1[j - 2] == c2 {
                value = value.min(before[j - 2] + 1);
            }
            cur[j] = value;
            row_min = row_min.min(value);
        }
        // Every later cell is at least the smallest of this row (a
        // transposition reads two rows back, but its cell is at least the
        // diagonal cell of the row before, which is in this bound too).
        if row_min > max {
            return None;
        }
        // The row before becomes the one before that, this row the row
        // before; the oldest row is overwritten by the next row.
        std::mem::swap(before, prev);
        std::mem::swap(prev, cur);
    }
    let result = prev[n];
    (result <= max).then_some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The full table without trimming or early exit, as the definition
    /// states it.
    fn reference(a: &[char], b: &[char], transpositions: bool) -> usize {
        let mut d = vec![vec![0usize; b.len() + 1]; a.len() + 1];
        for (i, row) in d.iter_mut().enumerate() {
            row[0] = i;
        }
        for (j, cell) in d[0].iter_mut().enumerate() {
            *cell = j;
        }
        for i in 1..=a.len() {
            for j in 1..=b.len() {
                let cost = usize::from(a[i - 1] != b[j - 1]);
                let mut v = (d[i - 1][j] + 1).min(d[i][j - 1] + 1).min(d[i - 1][j - 1] + cost);
                if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                    v = v.min(d[i - 2][j - 2] + 1);
                }
                d[i][j] = v;
            }
        }
        d[a.len()][b.len()]
    }

    #[test]
    fn bounded_distance_equals_the_full_table_under_every_bound() {
        // A small alphabet makes repeated symbols, shared prefixes and
        // suffixes and transpositions frequent.
        let alphabet = ['a', 'b', 'c', '\u{1F600}'];
        let mut seed = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut rows = Rows::default();
        for _ in 0..20_000 {
            let mut word = |n: u64| -> Vec<char> {
                (0..next() % n).map(|_| alphabet[(next() % alphabet.len() as u64) as usize]).collect()
            };
            let (a, b) = (word(8), word(8));
            for metric in [Metric::DamerauOsa, Metric::Levenshtein] {
                let full = reference(&a, &b, metric == Metric::DamerauOsa);
                for max in 0..9 {
                    let expected = (full <= max).then_some(full);
                    assert_eq!(metric.chars(&a, &b, max, &mut rows), expected, "{metric:?} {a:?} {b:?} max {max}");
                }
            }
        }
    }
}
