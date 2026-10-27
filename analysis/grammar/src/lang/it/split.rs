//! Split only complete Italian cardinal compounds; source cuts remain byte offsets.
use super::WORD;
use super::numbers::ENTRIES;

// Zero means one vocabulary word; another value is the hundred/remainder cut.
fn small(word: &str) -> Option<u32> {
    if ENTRIES.iter().any(|e| e.dim == WORD && e.value < 100 && e.word == word) {
        return Some(0);
    }
    for e in ENTRIES.iter().filter(|e| e.dim == WORD && e.value >= 100) {
        let Some(rest) = word.strip_prefix(e.word) else { continue };
        if rest.is_empty() {
            return e.word.ends_with('o').then_some(0);
        }
        if !e.word.ends_with('o') && !rest.starts_with("otto") && !rest.starts_with("ottant") {
            continue;
        }
        if ENTRIES.iter().any(|r| r.dim == WORD && r.value > 0 && r.value < 100 && r.word == rest) {
            return Some(e.word.len() as u32);
        }
    }
    None
}

pub(super) fn split(word: &str, cuts: &mut Vec<u32>) {
    if word.len() > 96 || !word.bytes().all(|b| b.is_ascii_lowercase()) {
        return;
    }
    if ENTRIES.iter().any(|e| e.word == word) {
        return;
    }
    let mut parts = [0; 4];
    let mut count = 0;
    let mut push = |i| {
        parts[count] = i;
        count += 1;
    };
    let thousand = if let Some(rest) = word.strip_prefix("mille") {
        Some((5, rest))
    } else if let Some(at) = word.find("mila") {
        let Some(cut) = small(&word[..at]) else { return };
        if cut > 0 {
            push(cut);
        }
        push(at as u32);
        Some((at + 4, &word[at + 4..]))
    } else {
        None
    };
    if let Some((at, rest)) = thousand {
        if !rest.is_empty() {
            let Some(cut) = small(rest) else { return };
            push(at as u32);
            if cut > 0 {
                push(cut + at as u32);
            }
        }
    } else {
        let Some(cut) = small(word) else { return };
        if cut > 0 {
            push(cut);
        }
    }
    cuts.extend_from_slice(&parts[..count]);
}
