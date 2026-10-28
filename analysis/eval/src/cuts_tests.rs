use turbo_chunker::Costs;
use turbo_structure::Syntax;

use super::*;
use crate::chunkers::tests::analyzed;
use crate::chunkers::{Chunker, Signals, TurboChunker};
use crate::embedder::Embedder;
use crate::embedder::stub::Stub;

const TEXT: &str = "cat purrs loud.\ncat naps long.\n\nrust compiles fast.\nrust ships now.\n";

fn chunker(costs: Costs) -> TurboChunker {
    TurboChunker {
        budget: Budget { max: 9, target: 8, size_weight: 1000, overhead: 2 },
        costs,
        signals: Signals::Structure,
        cohesion: None,
        clauses: false,
    }
}

fn header(set: &str, backend: &str, c: &TurboChunker) -> Header {
    Header {
        set: set.into(),
        sha256: "00".repeat(32),
        backend: backend.into(),
        device: "test".into(),
        precision: "model".into(),
        max: c.budget.max,
        target: c.budget.target,
        size_weight: c.budget.size_weight,
        overhead: c.budget.overhead,
        chunker: c.config(),
    }
}

fn dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("turbo-eval-cuts-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

/// Dumps the two-paragraph text, cut with `c`, as set `set` in `d`.
fn dump(d: &Path, set: &str, backend: &str, c: &TurboChunker) -> PathBuf {
    let mut e = Stub::new(&["cat"]);
    let doc = analyzed(TEXT, &e, Some(Syntax::Plain));
    let (spans, cut) = c.chunk_cut(&doc, &mut e).unwrap();
    assert_eq!(spans, c.chunk(&doc, &mut e).unwrap());
    let mut w = Writer::create(d, &header(set, backend, c)).unwrap();
    w.document("d1", &cut).unwrap();
    let (path, n) = w.finish().unwrap();
    assert_eq!(n, 1);
    assert!(e.dim() > 0);
    path
}

#[test]
fn a_dump_reads_back_and_matches_itself() {
    let (a, b) = (dir("same-a"), dir("same-b"));
    let c = chunker(Costs::default());
    let path = dump(&a, "s", "cpu", &c);
    dump(&b, "s", "levelzero", &c);
    let read = Dump::read(&path).unwrap();
    assert_eq!(read.header, header("s", "cpu", &c));
    assert_eq!(read.docs.len(), 1);
    assert_eq!(read.docs[0].1.source(), TEXT);
    // The chunks and what they were cut from cross.
    assert!(read.docs[0].1.get::<Chunks>().is_some());
    let sets = compare_dirs(&a, &b).unwrap();
    assert_eq!(sets.len(), 1);
    assert!(sets[0].is_identical(), "{}", sets[0]);
    // Every chunk's start is a cut, at every level: the two leaves and
    // the one chunk above them.
    assert_eq!(sets[0].matched, 3);
    assert_eq!(sets[0].b.backend, "levelzero");
    for d in [a, b] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_moved_cut_is_reported_with_its_margins() {
    let (a, b) = (dir("moved-a"), dir("moved-b"));
    dump(&a, "s", "cpu", &chunker(Costs::default()));
    // A paragraph end dearer than a sentence boundary moves the cut off
    // the paragraph: sizes 5 and 11 are over max, so the cut goes after
    // the first sentence or the third, and the earliest of equal costs wins.
    let dear = Costs { paragraph: 5000, ..Costs::default() };
    dump(&b, "s", "cpu", &chunker(dear));
    let sets = compare_dirs(&a, &b).unwrap();
    let s = &sets[0];
    assert!(!s.is_identical());
    assert_eq!(s.differing.len(), 1);
    let d = &s.differing[0].disagreements;
    assert!(!d.is_empty());
    assert!(d.iter().any(|x| x.margin_a.is_some_and(|m| m > 0)), "{s}");
    assert!(s.to_string().contains("d1 level 0"), "{s}");
    for d in [a, b] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn mismatched_dumps_are_refused() {
    let (a, b) = (dir("refused-a"), dir("refused-b"));
    let c = chunker(Costs::default());
    dump(&a, "s", "cpu", &c);
    dump(&b, "t", "cpu", &c);
    assert!(compare_dirs(&a, &b).unwrap_err().contains("holds sets"));
    let wider = TurboChunker { budget: Budget { max: 20, ..c.budget }, ..c };
    let other = dir("refused-c");
    dump(&other, "s", "cpu", &wider);
    assert!(compare_dirs(&a, &other).unwrap_err().contains("different budgets"));
    // A dump is never replaced.
    let e = Writer::create(&a, &header("s", "cpu", &c)).err().unwrap();
    assert!(e.contains("never replaced"), "{e}");
    for d in [a, b, other] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn a_chunker_without_an_optimizer_has_no_cuts() {
    let mut e = Stub::new(&["cat"]);
    let doc = analyzed(TEXT, &e, None);
    let w = crate::chunkers::SentenceWindows::new(2).unwrap();
    assert!(w.cut_budget().is_none());
    assert!(w.chunk_cut(&doc, &mut e).err().unwrap().contains("cannot be dumped"));
}

#[test]
fn sentence_vectors_stay_out_of_the_dump() {
    let d = dir("vectors");
    let c = TurboChunker { cohesion: Some(turbo_chunker::Cohesion { weight: 1000 }), ..chunker(Costs::default()) };
    let path = dump(&d, "s", "cpu", &c);
    let read = Dump::read(&path).unwrap();
    let doc = &read.docs[0].1;
    assert!(doc.get::<turbo_analysis_layers::Vectors<Sentences>>().is_none());
    assert!(doc.get::<Chunks>().is_some());
    fs::remove_dir_all(d).unwrap();
}

#[test]
fn weighed_drift_boundaries_cross_and_compare() {
    // Drift writes the structure's and the drift's costs as their own
    // instances and the weighed costs as the default one, which the
    // optimizer and the comparison read.
    let (a, b) = (dir("drift-a"), dir("drift-b"));
    let w = turbo_chunker::WeighBoundaries::default();
    let c = TurboChunker {
        signals: Signals::StructureDrift {
            drift: turbo_chunker::Drift::default(),
            structure: w.structure,
            weight: w.drift,
        },
        ..chunker(Costs::default())
    };
    dump(&a, "s", "cpu", &c);
    dump(&b, "s", "cpu", &c);
    let sets = compare_dirs(&a, &b).unwrap();
    assert!(sets[0].is_identical(), "{}", sets[0]);
    assert!(sets[0].matched > 0);
    for d in [a, b] {
        fs::remove_dir_all(d).unwrap();
    }
}

#[test]
fn an_unfinished_dump_leaves_no_file() {
    let d = dir("partial");
    let c = chunker(Costs::default());
    drop(Writer::create(&d, &header("s", "cpu", &c)).unwrap());
    assert!(!cuts_path(&d, "s").exists());
    // A later run is not refused by it.
    dump(&d, "s", "cpu", &c);
    fs::remove_dir_all(d).unwrap();
}
