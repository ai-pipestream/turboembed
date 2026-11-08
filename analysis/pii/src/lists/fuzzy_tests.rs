//! The indexed fuzzy scan against comparing every term with every window:
//! both must leave selection with the same matches.

use super::*;

/// The scan before #1935: every fuzzy term against every word window.
fn scan_every_term(
    index: &FuzzyIndex,
    text: &str,
    case: Case,
    max_edits: u8,
    capitalised_only: bool,
) -> Vec<Candidate> {
    let words = word_spans(text);
    let folded: Vec<Vec<char>> = words.iter().map(|&(s, e)| fold_chars(&text[s..e], case)).collect();
    let mut band = Band::default();
    let mut out = Vec::new();
    for ft in &index.terms {
        for i in 0..words.len() {
            if let Some(c) = window_match(text, &words, &folded, i, ft, max_edits, capitalised_only, &mut band) {
                out.push(c);
            }
        }
    }
    out
}

fn indexed(index: &FuzzyIndex, text: &str, case: Case, max_edits: u8, capitalised_only: bool) -> Vec<Candidate> {
    let mut out = Vec::new();
    index.scan(text, case, max_edits, capitalised_only, &mut out);
    out
}

/// What selection keeps of `c`, as `(start, end, entry)`.
fn sorted(mut c: Vec<Candidate>) -> Vec<(u32, u32, u32)> {
    let mut kept = Vec::new();
    turbo_glossary::select(&mut c, &mut kept);
    kept.iter().map(|m| (m.span.start, m.span.end, m.entry.0)).collect()
}

fn build(terms: &[String], case: Case) -> FuzzyIndex {
    FuzzyIndex::new(
        terms.iter().enumerate().filter_map(|(i, t)| build_fuzzy_term(EntryIx(i as u32), t, case)).collect(),
    )
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> usize {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize
    }
    fn pick<'a>(&mut self, from: &[&'a str]) -> &'a str {
        from[self.next() % from.len()]
    }
}

/// A word over a small alphabet so that near misses are common.
fn word(rng: &mut Lcg, len: usize) -> String {
    const LETTERS: [char; 8] = ['a', 'b', 'c', 'A', 'B', 'é', 'É', '1'];
    (0..len).map(|_| LETTERS[rng.next() % LETTERS.len()]).collect()
}

/// `w` with up to three random substitutions, insertions or deletions.
fn mutate(rng: &mut Lcg, w: &str) -> String {
    let mut chars: Vec<char> = w.chars().collect();
    for _ in 0..rng.next() % 4 {
        let at = rng.next() % (chars.len() + 1);
        let c = word(rng, 1).chars().next().unwrap();
        match rng.next() % 3 {
            0 if at < chars.len() => chars[at] = c,
            1 if at < chars.len() && chars.len() > 1 => {
                chars.remove(at);
            }
            _ => chars.insert(at, c),
        }
    }
    chars.into_iter().collect()
}

fn random_case(seed: u64) -> (Vec<String>, String) {
    let mut rng = Lcg(seed);
    let seps = [" ", "-", "  ", "'"];
    let mut terms = Vec::new();
    for _ in 0..1 + rng.next() % 40 {
        let words = 1 + rng.next() % 3;
        let mut t = String::new();
        for w in 0..words {
            if w > 0 {
                t.push_str(rng.pick(&seps));
            }
            let len = if rng.next() % 10 == 0 { 25 + rng.next() % 6 } else { 1 + rng.next() % 10 };
            t.push_str(&word(&mut rng, len));
        }
        terms.push(t);
    }
    let mut text = String::new();
    for _ in 0..200 {
        let piece = if rng.next() % 2 == 0 {
            let t = rng.next() % terms.len();
            mutate(&mut rng, &terms[t])
        } else {
            let len = 1 + rng.next() % 12;
            word(&mut rng, len)
        };
        text.push_str(&piece);
        text.push_str(rng.pick(&[" ", " ", ", ", "-", "  ", "\n"]));
    }
    (terms, text)
}

#[test]
fn indexed_scan_equals_every_term_scan_on_random_lists() {
    let mut nonempty = 0;
    for seed in 0..300 {
        let (terms, text) = random_case(seed);
        for case in [Case::Sensitive, Case::Insensitive] {
            let index = build(&terms, case);
            for max_edits in 1..=2 {
                for cap in [false, true] {
                    let want = sorted(scan_every_term(&index, &text, case, max_edits, cap));
                    let got = sorted(indexed(&index, &text, case, max_edits, cap));
                    assert_eq!(got, want, "seed {seed} {case:?} max_edits {max_edits} capitalised_only {cap}");
                    nonempty += usize::from(!want.is_empty());
                }
            }
        }
    }
    assert!(nonempty > 1000, "random cases should find matches, found {nonempty}");
}

#[test]
fn indexed_scan_finds_each_edit_count_and_long_terms() {
    let long = "a".repeat(30) + "b";
    let terms = vec!["Palantir".to_owned(), "Ed Vera".to_owned(), long.clone()];
    let index = build(&terms, Case::Sensitive);
    let text = format!("Palantir Palantyr Palxntyr Pxlxntyr Ed Verb Ed  Vxrx {} {long}", "a".repeat(31));
    for max_edits in 1..=2 {
        let want = sorted(scan_every_term(&index, &text, Case::Sensitive, max_edits, false));
        assert!(want.len() >= 4, "{want:?}");
        assert_eq!(sorted(indexed(&index, &text, Case::Sensitive, max_edits, false)), want);
    }
}

/// The deterministic part of the ticket's measurement: with 2000 terms of
/// one length, the index compares a window with a handful of terms, not
/// all of them. Counts work, never time.
#[test]
fn indexed_scan_compares_far_fewer_windows_than_every_term() {
    let terms: Vec<String> = (0..2000).map(|i| format!("Name{i:04}son")).collect();
    let index = build(&terms, Case::Sensitive);
    let filler = ["information", "government", "development", "Anderson", "Johnson", "the", "of", "Nameson"];
    let mut text = String::new();
    for i in 0..2000 {
        text.push_str(filler[i % filler.len()]);
        text.push(' ');
        if i % 50 == 0 {
            text.push_str(&format!("Name{:04}sen Nane{:04}son ", i % 2000, (i * 7) % 2000));
        }
    }
    let windows = word_spans(&text).len();
    let mut out = Vec::new();
    let stats = index.scan(&text, Case::Sensitive, 2, false, &mut out);
    let every = windows * terms.len();
    assert!(stats.compared * 200 < every, "compared {} of {every} windows", stats.compared);
    assert_eq!(sorted(out), sorted(scan_every_term(&index, &text, Case::Sensitive, 2, false)));
}
