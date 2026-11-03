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
                let has = |key: &str, value: &str| values.iter().any(|x| is_field(x, key, value));
                match relation {
                    SetRelation::Subset => fields(text).all(|(k, v)| has(k, v)),
                    SetRelation::Superset => values.iter().all(|x| fields(text).any(|(k, v)| is_field(x, k, v))),
                    SetRelation::Intersects => fields(text).any(|(k, v)| has(k, v)),
                }
            }
        }
    }
}

/// The individual features of a FEATS string, as key and value, in
/// written order and with any repeats: `Number=Sing,Plur|Case=Nom` gives
/// `Number=Sing`, `Number=Plur` and `Case=Nom`. A field without `=` is
/// skipped, and `_` has none. The set is the same for a string and its
/// canonical form ([`morph`]), so a set test can read either without
/// allocating.
fn fields(text: &str) -> impl Iterator<Item = (&str, &str)> {
    let text = if text == "_" { "" } else { text };
    text.split('|')
        .filter_map(|field| field.split_once('='))
        .flat_map(|(key, values)| values.split(',').map(move |value| (key, value)))
}

/// Whether `field` is `key=value`.
fn is_field(field: &str, key: &str, value: &str) -> bool {
    field.len() == key.len() + 1 + value.len()
        && field.starts_with(key)
        && field.as_bytes()[key.len()] == b'='
        && field.ends_with(value)
}

fn features(text: &str) -> Vec<String> {
    let mut result: Vec<String> = fields(text).map(|(key, value)| format!("{key}={value}")).collect();
    result.sort_unstable();
    result.dedup();
    result
}

/// spaCy's canonical string sorts feature keys and values.
pub(crate) fn morph(text: &str) -> String {
    let fields = features(text);
    let mut out = String::new();
    // `None` before the first field, so an empty key starts a field too.
    let mut last = None;
    for field in &fields {
        let (key, value) = field.split_once('=').expect("constructed above");
        if last == Some(key) {
            out.push(',');
            out.push_str(value);
        } else {
            if !out.is_empty() {
                out.push('|');
            }
            out.push_str(field);
        }
        last = Some(key);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The set test as it was: on the features of the canonical string.
    fn on_canonical(values: &[Box<str>], relation: SetRelation, text: &str) -> bool {
        let fields = features(&morph(text));
        match relation {
            SetRelation::Subset => fields.iter().all(|v| values.iter().any(|x| **x == *v)),
            SetRelation::Superset => values.iter().all(|v| fields.iter().any(|x| *x == **v)),
            SetRelation::Intersects => fields.iter().any(|v| values.iter().any(|x| **x == *v)),
        }
    }

    #[test]
    fn set_tests_on_written_feats_match_the_canonical_form() {
        // Every string of up to six symbols from an alphabet with the
        // separators, `_`, empty values and repeated keys.
        let alphabet = ["N=S", "N=P", "C=a", "|", ",", "=", "_", "S", "C", "a"];
        let sets: [&[&str]; 5] = [&[], &["N=S"], &["N=S", "C=a"], &["N=P", "C="], &["C=a=S", "_"]];
        let mut texts = vec![String::new()];
        let mut frontier = vec![String::new()];
        for _ in 0..4 {
            let mut next = Vec::new();
            for t in &frontier {
                for a in alphabet {
                    next.push(format!("{t}{a}"));
                }
            }
            texts.extend(next.iter().cloned());
            frontier = next;
        }
        // An empty key starts its own field in the canonical form.
        assert_eq!(morph("=a|N=S"), "=a|N=S");
        for set in sets {
            let values: Box<[Box<str>]> = set.iter().map(|&v| v.into()).collect();
            for relation in [SetRelation::Subset, SetRelation::Superset, SetRelation::Intersects] {
                let test = ValueTest::Morph { values: values.clone(), relation };
                for text in &texts {
                    assert_eq!(
                        test.holds(text),
                        on_canonical(&values, relation, text),
                        "{set:?} {relation:?} {text:?}"
                    );
                }
            }
        }
    }
}
