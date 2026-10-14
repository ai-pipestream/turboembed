//! The checked in goldens and dumps under `data_dir()`: every golden of
//! the segmentation runs reads with `segmentation_shapes()`, its input
//! checks, and writing it gives its bytes back; every dump reads, its
//! header is its golden's, and the sentence detector's `acceptable`
//! records are on exactly the points whose best outcome is a split.

use std::path::{Path, PathBuf};

use turbo_goldens::{Dump, Golden, data_dir, segmentation_shapes};

/// Every file under `dir` with this extension, sorted.
fn files(dir: &Path, ext: &str, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            files(&p, ext, out);
        } else if p.extension().is_some_and(|x| x == ext) {
            out.push(p);
        }
    }
}

fn annotator_files(annotator: &str, ext: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    files(&data_dir().join(annotator), ext, &mut out);
    out.sort();
    out
}

const SEGMENTATION: [&str; 4] = ["tokenize", "sentdetect", "detokenize", "unigram"];

#[test]
fn every_segmentation_golden_reads_and_writes_back_its_bytes() {
    let shapes = segmentation_shapes();
    let mut n = 0;
    for annotator in SEGMENTATION {
        for path in annotator_files(annotator, "golden") {
            let g = Golden::read_with(&path, &shapes).unwrap_or_else(|e| panic!("{e}"));
            g.input_text(data_dir()).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
            let bytes = std::fs::read(&path).unwrap();
            assert_eq!(g.to_text().unwrap().as_bytes(), &bytes[..], "{}", path.display());
            n += 1;
        }
    }
    // 779 tokenize, 294 sentdetect, 19 detokenize and 264 unigram goldens.
    assert_eq!(n, 1356);
}

#[test]
fn every_dump_reads_and_has_its_goldens_header() {
    let shapes = segmentation_shapes();
    let mut dumps = 0;
    let mut points = 0;
    for annotator in ["tokenize", "sentdetect", "langdetect"] {
        for path in annotator_files(annotator, "dump") {
            let d = Dump::read(&path).unwrap_or_else(|e| panic!("{e}"));
            let g = Golden::read_with(path.with_extension("golden"), &shapes).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d.header, g.header, "{}", path.display());
            assert!(!d.header.models.is_empty(), "{}", path.display());
            for p in &d.points {
                assert!(p.position <= d.header.input.utf8_len, "{}", path.display());
                let sum: f64 = p.outcomes.iter().map(|o| o.prob).sum();
                assert!((sum - 1.0).abs() < 1e-9, "{}: the probabilities add up to {sum}", path.display());
                let wants_split = annotator == "sentdetect" && p.best().label == "s";
                assert_eq!(p.acceptable.is_some(), wants_split, "{} at {}", path.display(), p.position);
            }
            dumps += 1;
            points += d.points.len();
        }
    }
    // 473 tokenize, 226 sentdetect and 22 langdetect dumps.
    assert_eq!(dumps, 721);
    assert_eq!(points, 24777, "every point of every dump");
}

#[test]
fn the_fixture_sentence_model_refuses_three_splits_in_each_mode() {
    // `en-abbrev` has an abbreviation dictionary; it refuses a split the
    // model wanted three times over the fixture inputs, in each mode.
    for run in ["en-abbrev", "en-abbrev+legacy"] {
        let dir = data_dir().join("sentdetect").join(run);
        let mut paths = Vec::new();
        files(&dir, "dump", &mut paths);
        let refused: usize = paths
            .iter()
            .map(|p| Dump::read(p).unwrap().points.iter().filter(|p| p.acceptable == Some(false)).count())
            .sum();
        assert_eq!(refused, 3, "{run}");
    }
}
