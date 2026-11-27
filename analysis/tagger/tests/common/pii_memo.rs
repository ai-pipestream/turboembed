//! Shared scoring of the real-head Places memos.

use std::path::Path;
use turbo_analysis::{Analyzer, Document, Layer};
use turbo_analysis_layers::{Pii, PiiCandidates, PiiKind};

/// A memo's text and its gold city spans.
fn memo(file: &str, labels: &str) -> (String, Vec<(u32, u32)>) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../pii/tests/data/human-eval");
    let text = std::fs::read_to_string(dir.join(file)).unwrap();
    let gold = std::fs::read_to_string(dir.join(labels))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f[3], "city", "{l}");
            let start: u32 = f[1].parse().unwrap();
            let end: u32 = f[2].parse().unwrap();
            assert_eq!(text.get(start as usize..end as usize), Some(f[4]), "{labels}: {l}");
            (start, end)
        })
        .collect();
    (text, gold)
}

/// What the head and the Places pack read on a memo.
pub struct Read {
    pub text: String,
    /// Number of labeled city mentions in the memo.
    pub gold_count: usize,
    /// The start of each false city.
    pub false_cities: Vec<usize>,
    /// The city uses not found.
    pub missed: Vec<String>,
    /// The city uses not found that no `city` candidate of the head
    /// covers either: a reader of the candidates has no sign of them.
    pub unread: Vec<String>,
}

impl Read {
    pub fn names(&self, starts: &[usize]) -> Vec<&str> {
        starts.iter().map(|&s| self.text[s..].split(|c: char| !c.is_alphabetic()).next().unwrap()).collect()
    }
}

/// Runs the head with the Places pack over a memo. Prints every scored
/// candidate, for the record.
pub fn read(analyzer: &Analyzer, file: &str, labels: &str) -> Read {
    let (text, gold) = memo(file, labels);
    let mut docs = vec![Document::new(text.as_str())];
    for r in analyzer.run(&mut docs) {
        r.unwrap();
    }
    let pii = docs[0].get::<Pii>().unwrap();
    pii.check().unwrap();
    let candidates = docs[0].get::<PiiCandidates>().unwrap();
    println!("{file}");
    for c in candidates.candidates() {
        if let Some(score) = c.score {
            let kind = candidates.kind_name(c.kind).unwrap_or_default();
            let covered = &text[c.span.range()];
            println!("  {:>5} {covered:<10} {kind:<12} {:.3} {:?}", c.span.start, score.value, c.disposition);
        }
    }
    let read_as_city: Vec<(u32, u32)> = candidates
        .candidates()
        .filter(|c| c.score.is_some() && candidates.kind_name(c.kind).is_some_and(|k| k == "city"))
        .map(|c| (c.span.start, c.span.end))
        .collect();
    let cities: Vec<(u32, u32)> =
        pii.mentions().filter(|m| m.kind == PiiKind::City).map(|m| (m.span.start, m.span.end)).collect();
    let people: Vec<&str> = pii
        .mentions()
        .filter(|m| pii.kind_name(m.kind).is_some_and(|k| k == "first-name" || k == "last-name"))
        .map(|m| &text[m.span.range()])
        .collect();
    let false_cities: Vec<usize> = cities.iter().filter(|c| !gold.contains(c)).map(|c| c.0 as usize).collect();
    let covered = |&(s, e): &(u32, u32)| text[s as usize..e as usize].to_owned();
    let lost: Vec<&(u32, u32)> = gold.iter().filter(|g| !cities.contains(g)).collect();
    let missed: Vec<String> = lost.iter().copied().map(covered).collect();
    let unread: Vec<String> = lost.iter().copied().filter(|g| !read_as_city.contains(g)).map(covered).collect();
    println!("head person spans: {people:?}");
    println!("city tp {} fp {} fn {}", cities.len() - false_cities.len(), false_cities.len(), missed.len());
    let read = Read { text, gold_count: gold.len(), false_cities, missed, unread };
    println!("false cities ({} gold): {:?}", read.gold_count, read.names(&read.false_cities));
    read
}
