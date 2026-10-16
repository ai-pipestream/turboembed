//! The training data as liblinear-java holds it (`Problem`, `Feature[][]`)
//! in one sparse table, the sparse vector operations its solvers use
//! (`SparseOperator`), and the class grouping of `Linear.train`.
//!
//! Feature indexes are kept zero based here, as every solver subtracts one
//! before it reads a weight; the one based index of the Java side is the
//! index here plus one.

/// Sparse rows in one table: row `i` is `index[start[i]..start[i + 1]]`
/// with its `value`s. A row of the instance table is an instance (one
/// unique event); a row of the transposed table is a feature, with the
/// instances that have it.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Rows {
    start: Vec<u32>,
    index: Vec<u32>,
    value: Vec<f64>,
}

impl Rows {
    pub(crate) fn with_capacity(rows: usize, entries: usize) -> Rows {
        let mut start = Vec::with_capacity(rows + 1);
        start.push(0);
        Rows { start, index: Vec::with_capacity(entries), value: Vec::with_capacity(entries) }
    }

    pub(crate) fn len(&self) -> usize {
        self.start.len() - 1
    }

    pub(crate) fn push(&mut self, index: u32, value: f64) {
        self.index.push(index);
        self.value.push(value);
    }

    pub(crate) fn end_row(&mut self) {
        self.start.push(self.index.len() as u32);
    }

    /// Row `i`'s indexes and values.
    #[inline]
    pub(crate) fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let (a, b) = (self.start[i] as usize, self.start[i + 1] as usize);
        (&self.index[a..b], &self.value[a..b])
    }

    /// `Linear.transpose`: the table by feature, each feature's entries in
    /// instance order. `columns` is the number of features.
    pub(crate) fn transpose(&self, columns: usize) -> Rows {
        let mut count = vec![0u32; columns + 1];
        for &j in &self.index {
            count[j as usize + 1] += 1;
        }
        for j in 0..columns {
            count[j + 1] += count[j];
        }
        let mut next = count.clone();
        let mut index = vec![0u32; self.index.len()];
        let mut value = vec![0.0f64; self.value.len()];
        for i in 0..self.len() {
            let (idx, val) = self.row(i);
            for (&j, &v) in idx.iter().zip(val) {
                let at = &mut next[j as usize];
                index[*at as usize] = i as u32;
                value[*at as usize] = v;
                *at += 1;
            }
        }
        Rows { start: count, index, value }
    }
}

/// `SparseOperator.nrm2_sq`.
#[inline]
pub(crate) fn nrm2_sq(value: &[f64]) -> f64 {
    let mut ret = 0.0;
    for &v in value {
        ret += v * v;
    }
    ret
}

/// `SparseOperator.dot`: `s` at the row's indexes times its values, summed
/// in row order.
#[inline]
pub(crate) fn dot(s: &[f64], (index, value): (&[u32], &[f64])) -> f64 {
    let mut ret = 0.0;
    for (&j, &v) in index.iter().zip(value) {
        ret += s[j as usize] * v;
    }
    ret
}

/// `SparseOperator.axpy`: `y += a * x` at the row's indexes.
#[inline]
pub(crate) fn axpy(a: f64, (index, value): (&[u32], &[f64]), y: &mut [f64]) {
    for (&j, &v) in index.iter().zip(value) {
        y[j as usize] += a * v;
    }
}

/// `Math.max` on doubles: NaN when either is NaN, and `0.0` above `-0.0`.
#[inline]
pub(crate) fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { b } else { a };
    }
    if a >= b { a } else { b }
}

/// `Math.min` on doubles: NaN when either is NaN, and `-0.0` below `0.0`.
#[inline]
pub(crate) fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() { a } else { b };
    }
    if a <= b { a } else { b }
}

/// The classes of the instances, as `Linear.groupClasses` finds them.
pub(crate) struct Classes {
    /// The class labels in order of first occurrence.
    pub(crate) label: Vec<i32>,
    /// Where each class's instances start in `perm`.
    pub(crate) start: Vec<usize>,
    pub(crate) count: Vec<usize>,
    /// The instances grouped by class, each class in instance order.
    pub(crate) perm: Vec<u32>,
}

/// `Linear.groupClasses`. Labels are numbered by first occurrence; liblinear
/// swaps the two labels of a `-1`/`+1` problem whose first instance is
/// `-1`, which cannot happen here, as the labels are outcome ids and never
/// negative.
pub(crate) fn group_classes(y: &[u32]) -> Classes {
    let mut label: Vec<i32> = Vec::new();
    let mut count: Vec<usize> = Vec::new();
    let mut data_label = Vec::with_capacity(y.len());
    for &this in y {
        let this = this as i32;
        let j = match label.iter().position(|&l| l == this) {
            Some(j) => {
                count[j] += 1;
                j
            }
            None => {
                label.push(this);
                count.push(1);
                label.len() - 1
            }
        };
        data_label.push(j);
    }
    let mut start = vec![0usize; label.len()];
    for i in 1..label.len() {
        start[i] = start[i - 1] + count[i - 1];
    }
    let mut next = start.clone();
    let mut perm = vec![0u32; y.len()];
    for (i, &c) in data_label.iter().enumerate() {
        perm[next[c]] = i as u32;
        next[c] += 1;
    }
    Classes { label, start, count, perm }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(rows: &[&[(u32, f64)]]) -> Rows {
        let mut t = Rows::with_capacity(rows.len(), 0);
        for r in rows {
            for &(j, v) in *r {
                t.push(j, v);
            }
            t.end_row();
        }
        t
    }

    #[test]
    fn transpose_lists_each_feature_in_instance_order() {
        let t = table(&[&[(0, 1.0), (2, 2.0)], &[(1, 3.0)], &[(0, 4.0), (1, 5.0)]]);
        let c = t.transpose(4);
        assert_eq!(c.len(), 4);
        assert_eq!(c.row(0), (&[0u32, 2][..], &[1.0, 4.0][..]));
        assert_eq!(c.row(1), (&[1u32, 2][..], &[3.0, 5.0][..]));
        assert_eq!(c.row(2), (&[0u32][..], &[2.0][..]));
        assert_eq!(c.row(3), (&[][..], &[][..]));
    }

    #[test]
    fn classes_are_numbered_by_first_occurrence() {
        let c = group_classes(&[2, 0, 2, 1, 0]);
        assert_eq!(c.label, [2, 0, 1]);
        assert_eq!(c.count, [2, 2, 1]);
        assert_eq!(c.start, [0, 2, 4]);
        assert_eq!(c.perm, [0, 2, 1, 4, 3]);
    }

    #[test]
    fn java_max_and_min_keep_nan_and_signed_zero() {
        assert!(java_max(f64::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f64::NAN).is_nan());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert_eq!(java_max(1.0, 2.0), 2.0);
        assert_eq!(java_min(1.0, 2.0), 1.0);
    }

    #[test]
    fn sparse_operations_follow_row_order() {
        let t = table(&[&[(0, 2.0), (3, 0.5)]]);
        let mut y = vec![1.0; 4];
        axpy(2.0, t.row(0), &mut y);
        assert_eq!(y, [5.0, 1.0, 1.0, 2.0]);
        assert_eq!(dot(&y, t.row(0)), 11.0);
        assert_eq!(nrm2_sq(t.row(0).1), 4.25);
    }
}
