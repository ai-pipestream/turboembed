//! A small token- and BIO-count-matched edit to the historical synthetic
//! training mix. Only eligible legacy rows are replaced, at their original
//! indices. No dev fixtures are used to choose rows or surface forms.

use std::collections::BTreeMap;

use crate::Result;
use crate::corpus::Sentence;

use super::context_contrasts::{KINDS, TRAIN_NAMES};
use super::spans::to_sentence;

pub const REPLACEMENTS: usize = 600;

const FRAMES: &[[(&str, &str); 4]] = &[
    [("", " reopened the clinic."), ("", " lost a coat."), ("Counsel ", " signed twice."), ("", " issued a dividend.")],
    [("From ", " came a bus."), ("From ", " came a postcard."), ("From ", " came a note."), ("From ", " came a bill.")],
    [
        ("We sent maps to ", "."),
        ("We sent mittens to ", "."),
        ("We sent the draft to ", "."),
        ("We sent equipment orders to ", "."),
    ],
    [
        ("", " closed the public road."),
        ("", " packed a snack."),
        ("", " read the docket."),
        ("", " hired more mechanics."),
    ],
];
// The two novel-dev particles del/von remain held out. Mix particles and
// non-particle compounds. The same lexical token fills the four roles.
const SHORT_SURNAME: &[&str] = &["de", "Van", "Le", "Leigh", "Mills", "al", "da", "di"];
const LONG_SURNAME: &[&str] = &["de la", "van der", "dos Reis", "da Costa"];
const COMPANY: &[&str] = &["Labs", "Works", "Supply", "Studio"];
// Neutral notes also include ordinary lower-case particle-like words.
const PAD4: &[&str] = &["A van passed.", "The bin rattled.", "More paper arrived.", "Nothing else changed."];

fn replacement(i: usize, words: usize, positive: usize) -> Result<Sentence> {
    let tag = TRAIN_NAMES[(i.wrapping_mul(17) + i / 7) % TRAIN_NAMES.len()];
    let frame = &FRAMES[(i / 3) % FRAMES.len()];
    let long_surname = positive == 6 && (i / 7).is_multiple_of(2);
    let last = if long_surname { LONG_SURNAME[i % LONG_SURNAME.len()] } else { SHORT_SURNAME[i % SHORT_SURNAME.len()] };
    let mut text = String::new();
    let mut labels = Vec::new();
    for n in 0..4 {
        let role = (n + i / 11) % 4;
        let (before, after) = frame[role];
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(before);
        let start = text.len();
        if role == 2 {
            text.push_str(last);
            text.push(' ');
        }
        text.push_str(tag);
        if role == 3 && positive == 6 && !long_surname {
            text.push(' ');
            text.push_str(COMPANY[i % COMPANY.len()]);
        }
        let end = text.len();
        labels.push(format!("{{'start': {start}, 'end': {end}, 'text': 'x', 'label': '{}'}}", KINDS[role]));
        text.push_str(after);
    }
    let repr = format!("[{}]", labels.join(", "));
    assert!(text.is_ascii());
    let first = to_sentence("context-overlay", i, &text, &repr)?.expect("four role clauses");
    assert!(first.words.len() <= words);
    let extra = words - first.words.len();
    for k in 0..extra / 4 {
        text.push(' ');
        text.push_str(PAD4[(i + k) % PAD4.len()]);
    }
    text.push_str(["", " Notes", " Done.", " It worked."][extra % 4]);
    let row = to_sentence("context-overlay", i, &text, &repr)?.expect("all four role clauses");
    assert_eq!(row.words.len(), words);
    assert_eq!(row.labels.iter().filter(|l| *l != "O").count(), positive);
    Ok(row)
}

type Shape = (usize, usize, usize, usize);

/// Stable largest-remainder quotas across eligible legacy city/person
/// counts, row length bins, and BIO-labelled word counts.
fn selected(legacy: &[Sentence]) -> Vec<(usize, usize)> {
    let mut strata: BTreeMap<Shape, Vec<usize>> = BTreeMap::new();
    for (i, r) in legacy.iter().enumerate() {
        let words = r.words.len();
        let city = r.labels.iter().filter(|&l| l == "B-city").count();
        let first = r.labels.iter().filter(|&l| l == "B-first_name").count();
        let positive = r.labels.iter().filter(|&l| l != "O").count();
        if (30..=60).contains(&words) && city > 0 && first > 0 && (5..=6).contains(&positive) {
            strata.entry((city, first, words / 8, positive)).or_default().push(i);
        }
    }
    let total: usize = strata.values().map(Vec::len).sum();
    assert!(total >= REPLACEMENTS);
    let mut quotas: Vec<_> = strata
        .into_iter()
        .map(|(shape, ids)| {
            let weighted = REPLACEMENTS * ids.len();
            (shape, ids, weighted / total, weighted % total)
        })
        .collect();
    quotas.sort_by_key(|(shape, _, _, rest)| (std::cmp::Reverse(*rest), *shape));
    let leftover = REPLACEMENTS - quotas.iter().map(|(_, _, n, _)| *n).sum::<usize>();
    let mut selected = Vec::with_capacity(REPLACEMENTS);
    for (i, (shape, ids, floor, _)) in quotas.into_iter().enumerate() {
        let n = floor + usize::from(i < leftover);
        selected.extend((0..n).map(|k| (ids[(2 * k + 1) * ids.len() / (2 * n)], shape.3)));
    }
    selected.sort_unstable();
    selected
}

/// Materializes only the edited rows as (original row index, replacement).
pub fn edits(legacy: &[Sentence]) -> Result<Vec<(usize, Sentence)>> {
    selected(legacy)
        .into_iter()
        .map(|(i, positive)| replacement(i, legacy[i].words.len(), positive).map(|s| (i, s)))
        .collect()
}

/// Neutral target tokens for a raw-decoder false-positive diagnostic.
pub fn particle_negatives() -> Result<Vec<(Sentence, usize)>> {
    [
        ("A green van stopped outside.", "van"),
        ("The parcel fell from the van.", "van"),
        ("The spare bolts are in the bin.", "bin"),
        ("Take the empty bin to the shed.", "bin"),
        ("The caption uses de as a short label.", "de"),
        ("The damaged sheet reads del on the side.", "del"),
        ("Type von before pressing Enter.", "von"),
        ("The sample is marked al in the legend.", "al"),
        ("Print el beside the empty box.", "el"),
        ("The string begins with da here.", "da"),
        ("The letter di appears in the note.", "di"),
        ("Draw a line from the bin to the table.", "bin"),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (text, token))| {
        let row = to_sentence("context-overlay-neutral", i, text, "[]")?.expect("neutral text");
        let at = row.words.iter().position(|w| w == token).expect("target in neutral text");
        assert!(row.labels.iter().all(|l| l == "O"));
        Ok((row, at))
    })
    .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn replacement_matches_each_original_row_and_has_all_four_roles() {
        let (old, _) = super::super::shared_names::documents().unwrap();
        let edits = edits(&old).unwrap();
        assert_eq!(edits.len(), REPLACEMENTS);
        assert!(edits.windows(2).all(|w| w[0].0 < w[1].0));
        for (i, row) in edits {
            assert_eq!(row.words.len(), old[i].words.len());
            assert_eq!(
                row.labels.iter().filter(|l| *l != "O").count(),
                old[i].labels.iter().filter(|l| *l != "O").count()
            );
            let spans = crate::eval::spans(&row.labels);
            assert_eq!(spans.len(), 4);
            for kind in KINDS {
                let (a, b, _) = spans.iter().find(|(_, _, k)| k == kind).unwrap();
                if kind == "last_name" {
                    assert!(b - a >= 2);
                    assert!(row.words[*a..*b].iter().all(|w| !["del", "von"].contains(&w.as_str())));
                }
            }
        }
    }
}
