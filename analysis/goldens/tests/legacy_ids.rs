//! A golden file written with the legacy shared-layer ids (on its
//! layer lines, its aligned anchors and its end lines) reads as the
//! same golden as one written with the current ids.

use std::path::PathBuf;

use turbo_analysis::ids::legacy_ids;
use turbo_goldens::Golden;

fn zurich() -> String {
    std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/zurich.golden")).unwrap()
}

/// `text` with each current shared-layer id on a layer or end line
/// written as its legacy id, as files from before the rename have them.
fn with_legacy_ids(text: &str) -> String {
    let mut out = String::new();
    for line in text.split_inclusive('\n') {
        let mut line = line.to_owned();
        if line.starts_with("layer\t") || line.starts_with("end\t") {
            for (old, new) in legacy_ids() {
                for end in ['\t', '\n', '>', ','] {
                    line = line.replace(&format!("{new}{end}"), &format!("{old}{end}"));
                }
            }
        }
        out.push_str(&line);
    }
    out
}

#[test]
fn a_golden_with_legacy_ids_reads_as_the_current_one() {
    let current = zurich();
    let legacy = with_legacy_ids(&current);
    assert_ne!(legacy, current, "the fixture should name shared layers");
    assert!(legacy.contains("<opennlp:sentences>"), "nested ids are rewritten too");
    let (a, b) = (Golden::parse(&current).unwrap(), Golden::parse(&legacy).unwrap());
    assert_eq!(a, b);
    // Written back, it carries the current ids.
    assert_eq!(b.to_text().unwrap(), current);
}
