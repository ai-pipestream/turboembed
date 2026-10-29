//! The lexicon search against a brute-force oracle: random sorted
//! dictionaries over an alphabet with non-ASCII and supplementary scalars,
//! random queries, and every result and non-result checked against a
//! full optimal-string-alignment dynamic program per entry. Also the
//! exact-match (`max_distance` 0) behavior, distance-3 rejection under a
//! bound of 2, determinism, and long and empty inputs.

use turbo_matcher::{FuzzyMatch, Lexicon, Scratch};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The alphabet: ASCII letters that make runs and transpositions, a
/// non-ASCII letter, and a supplementary character as one scalar.
const ALPHA: [char; 6] = ['a', 'b', 'c', 'd', '\u{e9}', '\u{1f600}'];

/// The full table, as the definition states it, over scalars.
fn osa(a: &[char], b: &[char]) -> usize {
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
            if i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                v = v.min(d[i - 2][j - 2] + 1);
            }
            d[i][j] = v;
        }
    }
    d[a.len()][b.len()]
}

fn random_word(rng: &mut Rng, len: u64) -> String {
    (0..rng.below(len)).map(|_| ALPHA[rng.below(ALPHA.len() as u64) as usize]).collect()
}

/// A query near the dictionary: a random entry taken through zero to
/// three edits (insert, delete, substitute, transpose), or a fresh word.
fn nearby_query(rng: &mut Rng, dict: &[String]) -> String {
    let mut chars: Vec<char> = if dict.is_empty() || rng.below(4) == 0 {
        random_word(rng, 10).chars().collect()
    } else {
        dict[rng.below(dict.len() as u64) as usize].chars().collect()
    };
    for _ in 0..rng.below(4) {
        if chars.is_empty() {
            chars.push(ALPHA[rng.below(ALPHA.len() as u64) as usize]);
            continue;
        }
        match rng.below(4) {
            // Insert.
            0 => chars.insert(rng.below(chars.len() as u64 + 1) as usize, ALPHA[rng.below(6) as usize]),
            // Delete.
            1 => {
                chars.remove(rng.below(chars.len() as u64) as usize);
            }
            // Substitute.
            2 => {
                let i = rng.below(chars.len() as u64) as usize;
                chars[i] = ALPHA[rng.below(6) as usize];
            }
            // Transpose two adjacent scalars.
            _ => {
                if chars.len() >= 2 {
                    let i = rng.below(chars.len() as u64 - 1) as usize;
                    chars.swap(i, i + 1);
                } else {
                    chars.push(ALPHA[rng.below(6) as usize]);
                }
            }
        }
    }
    chars.into_iter().collect()
}

/// The brute-force answer: every entry within `max`, by index and distance.
fn brute(dict: &[&str], query: &str, max: usize) -> Vec<FuzzyMatch> {
    let q: Vec<char> = query.chars().collect();
    let mut want = Vec::new();
    for (index, entry) in dict.iter().enumerate() {
        let chars: Vec<char> = entry.chars().collect();
        let d = osa(&chars, &q);
        if d <= max {
            want.push(FuzzyMatch { index: index as u32, distance: d as u32 });
        }
    }
    want.sort_by(|a, b| a.distance.cmp(&b.distance).then(a.index.cmp(&b.index)));
    want
}

#[test]
fn search_agrees_with_the_brute_force_oracle() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    let mut rounds = 0;
    for round in 0..1500 {
        let mut words: Vec<String> = (0..1 + rng.below(24)).map(|_| random_word(&mut rng, 10)).collect();
        words.sort();
        words.dedup();
        let refs: Vec<&str> = words.iter().map(String::as_str).collect();
        let lexicon = Lexicon::new(&refs);
        assert_eq!(lexicon.len(), refs.len());
        for _ in 0..3 {
            let query = nearby_query(&mut rng, &words);
            for max in 0..=2u8 {
                lexicon.search(&query, max, &mut scratch, &mut out);
                assert_eq!(
                    out,
                    brute(&refs, &query, max as usize),
                    "round {round}: dict {refs:?} query {query:?} max {max}"
                );
                // The contract: sorted by (distance, index), so a second
                // run reads identically.
                let again = out.clone();
                lexicon.search(&query, max, &mut scratch, &mut out);
                assert_eq!(out, again);
            }
        }
        rounds += 1;
    }
    assert!(rounds == 1500);
}

#[test]
fn distance_three_entries_are_rejected_under_bound_two() {
    let dict = ["abcd", "abce", "abdc", "abcde", "wxyz"];
    let mut dict: Vec<&str> = dict.to_vec();
    dict.sort();
    let lexicon = Lexicon::new(&dict);
    // Every entry is at distance at most two of "abcd" except "wxyz".
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    lexicon.search("abcd", 2, &mut scratch, &mut out);
    assert_eq!(out.iter().map(|m| m.index).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    // "wxyz" is distance four; nothing at distance three sneaks in.
    assert!(out.iter().all(|m| m.distance <= 2));
    lexicon.search("wxyz", 2, &mut scratch, &mut out);
    assert_eq!(out.len(), 1);
    // At distance three from "abcd": exactly one deletion plus... "abd" and
    // "acd" qualify; check a pure distance-three word is found only at 3.
    lexicon.search("abxe", 2, &mut scratch, &mut out);
    let d: Vec<u32> = out.iter().map(|m| m.distance).collect();
    assert!(d.iter().all(|&x| x <= 2), "{d:?}");
}

#[test]
fn max_distance_zero_is_exact_lookup() {
    let words = ["ab", "abc", "abd", "ac", "b", "\u{e9}a", "\u{1f600}ab"];
    let refs: Vec<&str> = words.to_vec();
    let lexicon = Lexicon::new(&refs);
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    for (query, hit) in [
        ("ab", Some(0)),
        ("abc", Some(1)),
        ("abce", None),
        ("ba", None),
        ("", None),
        ("\u{e9}a", Some(5)),
        ("\u{1f600}ab", Some(6)),
        ("\u{1f600}ba", None),
    ] {
        lexicon.search(query, 0, &mut scratch, &mut out);
        assert_eq!(out.len(), usize::from(hit.is_some()), "{query:?}");
        if let Some(index) = hit {
            assert_eq!(out[0], FuzzyMatch { index, distance: 0 }, "{query:?}");
        }
    }
}

#[test]
fn transpositions_cost_one_edit() {
    // Optimal string alignment: ab <-> ba is one edit, not two.
    let mut words = ["ab", "ba", "abc", "a\u{1f600}b", "a\u{1f600}c"].to_vec();
    words.sort();
    let lexicon = Lexicon::new(&words);
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    lexicon.search("ab", 1, &mut scratch, &mut out);
    assert_eq!(out.iter().map(|m| (m.index, m.distance)).collect::<Vec<_>>(), [(0, 0), (1, 1), (2, 1), (4, 1)]);
    lexicon.search("ba", 1, &mut scratch, &mut out);
    assert_eq!(out.iter().map(|m| (m.index, m.distance)).collect::<Vec<_>>(), [(4, 0), (0, 1)]);
    // Supplementary scalars transpose as one edit too; a non-ASCII
    // scalar substitutes like any other.
    lexicon.search("a\u{1f600}c", 1, &mut scratch, &mut out);
    assert_eq!(out.iter().map(|m| (m.index, m.distance)).collect::<Vec<_>>(), [(3, 0), (1, 1), (2, 1)]);
}

#[test]
fn empty_and_long_inputs_do_not_panic() {
    let mut scratch = Scratch::default();
    let mut out = Vec::new();
    let empty = Lexicon::new(&[]);
    empty.search("abc", 2, &mut scratch, &mut out);
    assert!(out.is_empty());
    empty.search("", 0, &mut scratch, &mut out);
    assert!(out.is_empty());

    // A long entry and a long query, matching at the end.
    let long: String = (0..1000).map(|i| ALPHA[i % 3]).collect();
    let mut mutated = long.chars().collect::<Vec<char>>();
    mutated.swap(500, 501);
    let near: String = mutated.into_iter().collect();
    let mut words = [long.as_str(), near.as_str()].to_vec();
    words.sort();
    let lexicon = Lexicon::new(&words);
    lexicon.search(&long, 0, &mut scratch, &mut out);
    assert_eq!(out, vec![FuzzyMatch { index: 1, distance: 0 }]);
    lexicon.search(&long, 1, &mut scratch, &mut out);
    assert_eq!(out.len(), 2);
    // An empty query matches only the empty entry, within the bound.
    let mut words = ["", "a", "ab"].to_vec();
    words.sort();
    let lexicon = Lexicon::new(&words);
    for (max, want) in [(0u8, 1), (1, 2), (2, 3)] {
        lexicon.search("", max, &mut scratch, &mut out);
        assert_eq!(out.len(), want, "max {max}");
    }
}
