//! Levenshtein distance in characters, bounded: it stops as soon as the
//! distance is known to exceed the bound.

/// Whether `a` and `b` are within `max` insertions, deletions and
/// substitutions of each other, counted in characters.
pub(crate) fn within(a: &[char], b: &[char], max: u32, row: &mut Vec<u32>) -> bool {
    let max = max as usize;
    if a.len().abs_diff(b.len()) > max {
        return false;
    }
    if a.is_empty() || b.is_empty() {
        return a.len().max(b.len()) <= max;
    }
    // One row of the distance table over `b`, updated per character of `a`.
    row.clear();
    row.extend(0..=b.len() as u32);
    for (i, &ca) in a.iter().enumerate() {
        let mut diagonal = row[0];
        row[0] = i as u32 + 1;
        let mut best = row[0];
        for (j, &cb) in b.iter().enumerate() {
            let substitute = diagonal + u32::from(ca != cb);
            diagonal = row[j + 1];
            let v = substitute.min(row[j] + 1).min(diagonal + 1);
            row[j + 1] = v;
            best = best.min(v);
        }
        if best as usize > max {
            return false;
        }
    }
    row[b.len()] as usize <= max
}

/// spaCy's default bound for `FUZZY`: 30 percent of the pattern's length,
/// rounded half to even as Python's `round` does, and at least two.
pub(crate) fn default_max(pattern_len: usize) -> u32 {
    let r = (0.3 * pattern_len as f64).round_ties_even() as u32;
    r.max(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(a: &str, b: &str, max: u32) -> bool {
        let a: Vec<char> = a.chars().collect();
        let b: Vec<char> = b.chars().collect();
        within(&a, &b, max, &mut Vec::new())
    }

    #[test]
    fn distances() {
        assert!(d("kitten", "sitting", 3));
        assert!(!d("kitten", "sitting", 2));
        assert!(d("", "ab", 2));
        assert!(!d("", "abc", 2));
        assert!(d("Straße", "Strasse", 2));
        assert!(!d("Straße", "Strasse", 1));
        assert!(d("same", "same", 0));
    }

    #[test]
    fn default_bound_rounds_half_to_even() {
        assert_eq!(default_max(3), 2);
        assert_eq!(default_max(10), 3);
        assert_eq!(default_max(15), 4);
        assert_eq!(default_max(25), 8);
    }
}
