//! Correction accuracy and speed on SymSpell's benchmark queries
//! (`symspell-noisy-query-en-1000`, MIT): 1,000 lines of a query, the
//! word meant and their edit distance, 0 to 4. A query is right when the
//! corrector's answer is the word meant (for distance 0, the query left as
//! it is). The counts are pinned; the timings are printed and recorded in
//! docs/normalizers.md.

use std::time::Instant;

use turbo_normalize::spellcheck::{EN_UNIGRAMS, SpellCorrect, SymSpell, SymSpellConfig, Verbosity};
use turbo_normalize::{ContextSpellCorrector, normalize};

use super::{dataset, engine, lexicon, lm};

/// A query, the word meant, and their edit distance.
type Query = (String, String, usize);

/// The set's lines.
fn queries(set: &str) -> Vec<Query> {
    set.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let mut c = l.split_whitespace();
            let (q, m, d) = (c.next().unwrap(), c.next().unwrap(), c.next().unwrap());
            (q.to_owned(), m.to_owned(), d.parse().unwrap())
        })
        .collect()
}

/// Right answers by the set's distance, 0 to 4.
fn by_distance(qs: &[Query], right: impl Fn(usize, &Query) -> bool) -> [usize; 5] {
    let mut out = [0; 5];
    for (i, q) in qs.iter().enumerate() {
        out[q.2] += usize::from(right(i, q));
    }
    out
}

/// Prints one row and returns the right answers by distance.
fn row(name: &str, qs: &[Query], micros: f64, right: impl Fn(usize, &Query) -> bool) -> [usize; 5] {
    let r = by_distance(qs, right);
    let total: usize = r.iter().sum();
    println!("{name}: {total}/{} right, by distance {r:?}, {micros:.1} us per query", qs.len());
    r
}

/// Microseconds per query of `f` over the set.
fn timed<T>(qs: &[Query], f: impl Fn(&str) -> T) -> (Vec<T>, f64) {
    let t = Instant::now();
    let out: Vec<T> = qs.iter().map(|q| f(&q.0)).collect();
    (out, t.elapsed().as_secs_f64() * 1e6 / qs.len() as f64)
}

/// An engine over the unigram list alone, indexed to `distance`.
fn unigram_engine(distance: usize) -> SymSpell {
    let config = SymSpellConfig::builder().max_dictionary_edit_distance(distance).build().unwrap();
    let uni = lexicon(EN_UNIGRAMS);
    let t = Instant::now();
    let e = SymSpell::from_lexicons(config, &uni, None).unwrap();
    println!("index to distance {distance}: {:.2} s, {} delete keys", t.elapsed().as_secs_f64(), e.entry_count());
    e
}

#[test]
#[ignore = "fetches the SymSpell lexicons and benchmark queries through the zoo"]
fn accuracy_and_speed_on_the_symspell_noisy_queries() {
    let qs = queries(&dataset("symspell-noisy-query-en-1000"));
    assert_eq!(qs.len(), 1000);
    assert_eq!(by_distance(&qs, |_, _| true), [296, 292, 268, 114, 30]);
    let e = engine();
    // A word outside the unigram list cannot be suggested.
    assert_eq!(qs.iter().filter(|q| e.contains(&q.1)).count(), 977);

    // Top: the suggestion is the word meant, or there is none and the
    // query is. All: the word meant is among the suggestions.
    for (distance, want_top, want_found) in
        [(2, [283, 183, 104, 0, 0], [282, 287, 266, 0, 0]), (3, [282, 183, 104, 27, 0], [282, 287, 266, 113, 0])]
    {
        let e = unigram_engine(distance);
        let top = |q: &str| e.lookup(q, Verbosity::Top, distance).unwrap().first().map(|s| s.term.to_owned());
        let (answers, us) = timed(&qs, top);
        let got = row(&format!("lookup Top, distance {distance}"), &qs, us, |i, q| {
            answers[i].as_deref().unwrap_or(&q.0) == q.1
        });
        assert_eq!(got, want_top, "Top at {distance}");
        let all = |q: &str| -> Vec<String> {
            e.lookup(q, Verbosity::All, distance).unwrap().iter().map(|s| s.term.to_owned()).collect()
        };
        let (answers, us) = timed(&qs, all);
        let got = row(&format!("lookup All, distance {distance}, meant among them"), &qs, us, |i, q| {
            answers[i].contains(&q.1)
        });
        assert_eq!(got, want_found, "All at {distance}");
    }

    let n = SpellCorrect::new(e.clone());
    let (answers, us) = timed(&qs, |q| normalize(&n, q).0);
    let got = row("SpellCorrect per token, distance 2", &qs, us, |i, q| answers[i] == q.1);
    assert_eq!(got, [290, 171, 104, 0, 0]);

    let c = ContextSpellCorrector::new(e.clone(), lm());
    let (answers, us) = timed(&qs, |q| normalize(&c, q).0);
    let got = row("ContextSpellCorrector, one word, distance 2", &qs, us, |i, q| answers[i] == q.1);
    assert_eq!(got, [280, 121, 109, 0, 0]);
}
