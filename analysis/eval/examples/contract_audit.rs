//! Verify pinned contract inputs and print counts without retrieval scoring.
use std::path::Path;
fn main() -> Result<(), String> {
    let cache = std::env::args().nth(1).ok_or("usage: contract_audit <cache>")?;
    for name in ["cuad", "contract-nli"] {
        let s = turbo_eval::sets::find(name)?.load(Path::new(&cache))?;
        let judgments = s
            .queries
            .iter()
            .flat_map(|q| s.docs.iter().map(move |d| (&q.id, &d.id)))
            .filter(|(q, d)| s.qrels.grade(q, d) > 0)
            .count();
        let spans: usize = s
            .queries
            .iter()
            .flat_map(|q| s.docs.iter().map(move |d| (&q.id, &d.id)))
            .map(|(q, d)| s.spans.as_ref().unwrap().spans(q, d).len())
            .sum();
        println!(
            "{name}: digest={} documents={} templates={} positive_pairs={judgments} spans={spans}",
            s.sha256,
            s.docs.len(),
            s.queries.len()
        );
    }
    Ok(())
}
