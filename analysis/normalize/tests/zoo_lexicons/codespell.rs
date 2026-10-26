//! Correction accuracy and speed on human misspellings: codespell's list
//! (`codespell-dictionary-en`), derived from English Wikipedia's lists of
//! common misspellings under CC BY-SA 3.0, so the zoo fetches it only
//! after `turbo-zoo accept CC-BY-SA-3.0`. Each line is `typo->fix` or
//! `typo->fix1, fix2,` when the typo has several readings. The queries are
//! the typos of lowercase letters and apostrophes only; an answer is right
//! when it is one of the fixes listed. The counts are pinned; the timings
//! are printed and recorded in docs/normalizers.md.

use std::time::Instant;

use turbo_normalize::spellcheck::{SpellCorrect, Verbosity};
use turbo_normalize::{ContextSpellCorrector, normalize};

use super::{dataset, engine, lm};

/// A typo and its fixes.
struct Entry {
    typo: String,
    fixes: Vec<String>,
}

/// The lines whose typo is lowercase letters and apostrophes. A line that
/// does not end in a comma and lists several parts gives a reason last,
/// which is not a fix.
fn entries(text: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some((typo, rest)) = line.split_once("->") else { continue };
        if typo.is_empty() || !typo.chars().all(|c| c.is_ascii_lowercase() || c == '\'') {
            continue;
        }
        let mut parts: Vec<&str> = rest.split(',').map(str::trim).collect();
        if !rest.trim_end().ends_with(',') && parts.len() > 1 {
            parts.pop();
        }
        let fixes = parts.into_iter().filter(|p| !p.is_empty()).map(str::to_owned).collect();
        out.push(Entry { typo: typo.to_owned(), fixes });
    }
    out
}

/// Prints one row and returns (right on one-fix lines, right on all).
fn row(name: &str, set: &[Entry], answers: &[String], micros: f64) -> (usize, usize) {
    let right = |e: &Entry, a: &String| e.fixes.contains(a);
    let single = set.iter().zip(answers).filter(|(e, a)| e.fixes.len() == 1 && right(e, a)).count();
    let all = set.iter().zip(answers).filter(|(e, a)| right(e, a)).count();
    println!("{name}: {all}/{} right ({single} on one-fix lines), {micros:.1} us per query", set.len());
    (single, all)
}

/// Each typo's answer, and microseconds per typo.
fn timed(set: &[Entry], f: impl Fn(&str) -> String) -> (Vec<String>, f64) {
    let t = Instant::now();
    let out: Vec<String> = set.iter().map(|e| f(&e.typo)).collect();
    (out, t.elapsed().as_secs_f64() * 1e6 / set.len() as f64)
}

#[test]
#[ignore = "fetches the SymSpell lexicons and, once CC BY-SA 3.0 is accepted, codespell's list through the zoo"]
fn accuracy_and_speed_on_the_codespell_misspellings() {
    let set = entries(&dataset("codespell-dictionary-en"));
    let e = engine();
    let one_fix = set.iter().filter(|x| x.fixes.len() == 1).count();
    let known = set.iter().filter(|x| x.fixes.iter().any(|f| e.contains(f))).count();
    let typo_known = set.iter().filter(|x| e.contains(&x.typo)).count();
    println!(
        "{} typos, {one_fix} with one fix, {known} with a fix in the unigram list, {typo_known} typos in it",
        set.len()
    );
    assert_eq!((set.len(), one_fix, known, typo_known), (0, 0, 0, 0));

    let top =
        |q: &str| e.lookup(q, Verbosity::Top, 2).unwrap().first().map_or_else(|| q.to_owned(), |s| s.term.to_owned());
    let (answers, us) = timed(&set, top);
    assert_eq!(row("lookup Top, distance 2", &set, &answers, us), (0, 0));

    let t = Instant::now();
    let found = set
        .iter()
        .filter(|x| {
            let all = e.lookup(&x.typo, Verbosity::All, 2).unwrap();
            x.fixes.iter().any(|f| all.iter().any(|s| s.term == f))
        })
        .count();
    let us = t.elapsed().as_secs_f64() * 1e6 / set.len() as f64;
    println!("lookup All, distance 2, a fix among them: {found}/{}, {us:.1} us per query", set.len());
    assert_eq!(found, 0);

    let n = SpellCorrect::new(e.clone());
    let (answers, us) = timed(&set, |q| normalize(&n, q).0);
    assert_eq!(row("SpellCorrect per token, distance 2", &set, &answers, us), (0, 0));

    let c = ContextSpellCorrector::new(e.clone(), lm());
    let (answers, us) = timed(&set, |q| normalize(&c, q).0);
    assert_eq!(row("ContextSpellCorrector, one word, distance 2", &set, &answers, us), (0, 0));
}
