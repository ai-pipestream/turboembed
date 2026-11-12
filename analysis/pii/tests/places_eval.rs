//! #1886: precision and recall of the Places pack (`state`, `county`,
//! `city`) on the human-written evaluation set and a project-authored memo
//! of places and near misses (`places-memos.txt`), gold in
//! `labels-places.tsv`. The test prints a table (`--nocapture`) and holds
//! each kind to its measured counts.

mod support;

use support::{HUMAN_DOCS, layer, load_labelled, measure_with, pack, ratio_text};
use turbo_analysis::Layer as _;
use turbo_pii::Pack;

#[test]
fn places_precision_and_recall_on_human_written_text() {
    let mut docs = HUMAN_DOCS.to_vec();
    docs.push((
        "places-memos",
        false,
        "places-memos.txt",
        "5b6d7e9f28069ea39d9fd63fe1f8dcb27e2d7d873fe755211a456a7cd823fb8b",
    ));
    let set = load_labelled(&docs, "labels-places.tsv", false);
    let d = pack(Pack::Places);
    for l in &set {
        layer(&d, &l.text).check().expect("layer is ordered");
    }
    let by_kind = measure_with(&d, &set);
    eprintln!("\nplaces on the human-written set:");
    eprintln!("{:<8} {:>5} {:>5} {:>5} {:>9} {:>7}", "kind", "tp", "fp", "fn", "precision", "recall");
    for (name, c) in &by_kind {
        let (p, r) = (ratio_text(c.tp, c.tp + c.fp), ratio_text(c.tp, c.tp + c.fn_));
        eprintln!("{name:<8} {:>5} {:>5} {:>5} {p:>9} {r:>7}", c.tp, c.fp, c.fn_);
    }
    let got: Vec<(&str, usize, usize, usize)> = by_kind.iter().map(|(k, c)| (*k, c.tp, c.fp, c.fn_)).collect();
    assert_eq!(got, [("city", 9, 0, 27), ("county", 9, 0, 0), ("state", 40, 0, 0)]);
}
