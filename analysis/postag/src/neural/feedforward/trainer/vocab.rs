//! Building the trainer's word, suffix, shape, and tag vocabularies and checking the pretrained
//! vectors a lookup returns.

use std::collections::{HashMap, HashSet};

use crate::neural::NeuralError;
use crate::neural::feedforward::context::{ABSENT, UNKNOWN};

pub(super) fn check_vector(word: &str, found: &[f32], pretrained_size: &mut usize) -> Result<Vec<f32>, NeuralError> {
    if found.is_empty() {
        return Err(NeuralError::InvalidArgument(format!("empty vector returned for '{word}'")));
    }
    if *pretrained_size == 0 {
        *pretrained_size = found.len();
    } else if found.len() != *pretrained_size {
        return Err(NeuralError::InvalidArgument(format!(
            "vector length mismatch for '{word}': expected {}, got {}",
            *pretrained_size,
            found.len()
        )));
    }
    for (i, &v) in found.iter().enumerate() {
        if !v.is_finite() {
            return Err(NeuralError::InvalidArgument(format!(
                "non-finite vector component for '{word}' at index {i}: {v}"
            )));
        }
    }
    Ok(found.to_vec())
}

pub(super) fn register_vocab(
    entries: &mut Vec<(String, usize)>,
    order: &[String],
    counts: &HashMap<String, usize>,
    cutoff: usize,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in order {
        if counts[s] >= cutoff && seen.insert(s.clone()) {
            entries.push((s.clone(), next_row));
            next_row += 1;
        }
    }
    next_row
}

pub(super) fn register_all<'a>(
    entries: &mut Vec<(String, usize)>,
    symbols: impl IntoIterator<Item = &'a str>,
    mut next_row: usize,
) -> usize {
    let mut seen = HashSet::new();
    entries.push((UNKNOWN.to_owned(), next_row));
    seen.insert(UNKNOWN.to_owned());
    next_row += 1;
    entries.push((ABSENT.to_owned(), next_row));
    seen.insert(ABSENT.to_owned());
    next_row += 1;
    for s in symbols {
        if seen.insert(s.to_owned()) {
            entries.push((s.to_owned(), next_row));
            next_row += 1;
        }
    }
    next_row
}
