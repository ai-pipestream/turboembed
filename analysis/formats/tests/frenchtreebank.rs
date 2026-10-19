//! The French Treebank reader against the JVM: every `frenchtreebank`
//! golden of `analysis/goldens` (written by `cargo run -p
//! turbo-goldens-tool -- generate`, see analysis/goldens-tool/README.md and
//! `FrenchTreebankGolden.java`) is what OpenNLP's `ConstitParseSampleStream`
//! read from a data directory bundled in one input of the `frenchtreebank`
//! set. Each test writes the directory out, reads its files with the reader
//! here and rebuilds every layer: the parses as written and node by node,
//! the errors and where they came, and the files in the order they were
//! read.
//!
//! The class of an XML error is compared, not its message, which is
//! Xerces's own. Beside the comparison, every token is checked against
//! the bytes it was read from.

use std::path::Path;

use turbo_formats::frenchtreebank::{ConstitStream, FtbError, decode, french_treebank_files};
use turbo_goldens::Golden;

mod bundle;
mod treebank;

use treebank::Rebuilt;

fn rebuild(run: &str, root: &Path) -> Rebuilt {
    assert_eq!(run, "parse", "unknown frenchtreebank run");
    let mut out = Rebuilt::default();
    let files = french_treebank_files(root).unwrap();
    let bytes: Vec<Vec<u8>> = files.iter().map(|f| std::fs::read(f).unwrap()).collect();
    let mut parses = Vec::new();
    for item in ConstitStream::new(bytes.iter().map(Vec::as_slice)) {
        match item {
            Ok((doc, p)) => {
                let xml = decode(&bytes[doc]).unwrap();
                assert!(xml[p.source().range()].starts_with("<SENT"));
                let tokens = p.parse().token_nodes();
                assert_eq!(tokens.len(), p.tokens().len());
                for (node, span) in tokens.iter().zip(p.tokens()) {
                    let source = &xml[span.range()];
                    // A word written as it reads points at itself; one with a
                    // reference, a CDATA section, a comment or a carriage return
                    // points at the text it was read from.
                    if !source.contains(['&', '<', '\r']) {
                        assert_eq!(source, node.covered_text());
                    }
                }
                parses.push(p);
            }
            Err(e) => {
                let message = (!matches!(e, FtbError::Xml(_))).then(|| e.to_string());
                out.error(parses.len(), e.java_class(), message);
            }
        }
    }
    let trees: Vec<_> = parses.iter().map(|p| p.parse()).collect();
    out.parses(&trees);
    out.listing(root, &files);
    out
}

#[test]
fn every_frenchtreebank_golden_matches() {
    let Some(all) = treebank::goldens("frenchtreebank") else { return };
    let data = turbo_goldens::data_dir();
    let shapes = treebank::shapes();
    assert!(all.len() >= 4, "only {} frenchtreebank goldens", all.len());
    for (run, path) in all {
        let golden = Golden::read_with(&path, &shapes).unwrap();
        let input = golden.input_text(&data).unwrap();
        let root = treebank::scratch("frenchtreebank-goldens", &path);
        bundle::write_bundle(&input, &root);
        treebank::compare(&path, &golden, rebuild(&run, &root));
    }
}
