//! String predicates outside the interned equality path.

/// A comparison of individual morphology features, such as `Number=Sing`.
#[derive(Clone, Copy, Debug)]
pub enum SetRelation {
    Subset,
    Superset,
    Intersects,
}

/// Tests used by imported spaCy patterns and available to Rust callers.
#[derive(Clone, Debug)]
pub enum ValueTest {
    /// Unicode-scalar Levenshtein distance. `None` uses spaCy's default
    /// of max(2, round(0.3 * pattern length)); `Some(n)` allows n edits.
    /// With `exclude`, succeeds only when no value matches.
    Fuzzy { values: Box<[Box<str>]>, distance: Option<u8>, exclude: bool },
    /// Compare a token's individual morphological features with a set.
    Morph { values: Box<[Box<str>]>, relation: SetRelation },
}

impl ValueTest {
    pub(crate) fn holds(&self, text: &str) -> bool {
        match self {
            Self::Fuzzy { values, distance, exclude } => {
                values.iter().any(|p| {
                    let max = distance.map_or_else(
                        || ((p.chars().count() as f64 * 0.3).round_ties_even() as usize).max(2),
                        usize::from,
                    );
                    within(text, p, max)
                }) != *exclude
            }
            Self::Morph { values, relation } => {
                let fields = features(text);
                match relation {
                    SetRelation::Subset => fields.iter().all(|v| values.iter().any(|x| **x == *v)),
                    SetRelation::Superset => values.iter().all(|v| fields.iter().any(|x| *x == **v)),
                    SetRelation::Intersects => fields.iter().any(|v| values.iter().any(|x| **x == *v)),
                }
            }
        }
    }
}

fn features(text: &str) -> Vec<String> {
    if text.is_empty() || text == "_" {
        return Vec::new();
    }
    let mut result = Vec::new();
    for field in text.split('|') {
        if let Some((key, values)) = field.split_once('=') {
            for value in values.split(',') {
                result.push(format!("{key}={value}"));
            }
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

/// spaCy's canonical string sorts feature keys and values.
pub(crate) fn morph(text: &str) -> String {
    let fields = features(text);
    let mut out = String::new();
    let mut last = "";
    for field in &fields {
        let (key, value) = field.split_once('=').expect("constructed above");
        if key == last {
            out.push(',');
            out.push_str(value);
        } else {
            if !out.is_empty() {
                out.push('|');
            }
            out.push_str(field);
        }
        last = key;
    }
    out
}

/// Banded Wagner-Fischer, with no transposition operation. Short-circuit
/// when the length difference or an entire row exceeds the edit limit.
fn within(text: &str, pattern: &str, max: usize) -> bool {
    if text == pattern {
        return true;
    }
    let n = text.chars().count();
    let p: Vec<char> = pattern.chars().collect();
    if n.abs_diff(p.len()) > max {
        return false;
    }
    let inf = max.saturating_add(1);
    let mut prev: Vec<usize> = (0..=p.len()).map(|i| i.min(inf)).collect();
    let mut row = vec![inf; p.len() + 1];
    for (i, c) in text.chars().enumerate() {
        row.fill(inf);
        row[0] = (i + 1).min(inf);
        let lo = (i + 1).saturating_sub(max).max(1);
        let hi = (i + 1).saturating_add(max).min(p.len());
        for j in lo..=hi {
            row[j] = (prev[j] + 1).min(row[j - 1] + 1).min(prev[j - 1] + usize::from(c != p[j - 1]));
        }
        if row.iter().copied().min().unwrap_or(inf) > max {
            return false;
        }
        std::mem::swap(&mut row, &mut prev);
    }
    prev[p.len()] <= max
}
