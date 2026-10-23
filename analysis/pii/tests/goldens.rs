//! Parity of the packs with OpenNLP's `PiiPacks` on the JVM.
//!
//! For every text in `analysis/goldens/data/inputs/pii/`,
//! `pii/<pack>/pii/<id>.golden` holds the mentions OpenNLP's
//! `PiiAnnotator` found with the pack's extractor: span, type and
//! normalized form (see `analysis/goldens-tool/README.md`). Each test runs
//! `PiiDetector::pack(pack).keeping_nested()` in an analyzer over every
//! text and compares the `turbo:pii` layer with the golden row for row.
//! No tolerance is applied.
//!
//! The goldens come from `cargo run -p turbo-goldens-tool -- generate`;
//! while the `pii` directory is not on disk the tests say so and pass.

use std::path::PathBuf;

use turbo_analysis::{Analyzer, Document};
use turbo_goldens::{Golden, Tolerance, data_dir, matches_golden};
use turbo_pii::{Pack, Pii, PiiDetector};

/// The ids of the texts in `inputs/pii`, sorted.
fn inputs() -> Vec<String> {
    let mut ids: Vec<String> = std::fs::read_dir(data_dir().join("inputs/pii"))
        .expect("the pii input set")
        .map(|e| {
            let name = e.expect("a directory entry").file_name().into_string().expect("a UTF-8 file name");
            name.strip_suffix(".txt").expect("an input set holds only <id>.txt files").to_string()
        })
        .collect();
    ids.sort();
    ids
}

/// Whether the goldens are on disk; when not, says why the test passes.
fn goldens_present() -> bool {
    let dir = data_dir().join("pii");
    let present = dir.is_dir();
    if !present {
        eprintln!(
            "skipped: {} is absent; make the goldens with `cargo run -p turbo-goldens-tool -- generate`",
            dir.display()
        );
    }
    present
}

fn golden(pack: Pack, id: &str) -> Golden {
    let path: PathBuf = data_dir().join("pii").join(pack.name()).join("pii").join(format!("{id}.golden"));
    Golden::read(path).unwrap_or_else(|e| panic!("{e}"))
}

fn config<'g>(golden: &'g Golden, key: &str) -> &'g str {
    let entry = golden.header.config.iter().find(|c| c.key == key);
    &entry.unwrap_or_else(|| panic!("no {key} config record")).value
}

/// Runs `pack` over every text, compares the layer with its goldens and
/// returns how many mentions were compared.
fn check(pack: Pack) -> usize {
    let ids = inputs();
    assert!(ids.len() > 50, "only {} inputs in the pii set", ids.len());
    let analyzer = Analyzer::builder().add(PiiDetector::pack(pack).keeping_nested()).build().expect("the analyzer");
    let mut mentions = 0;
    let mut failures = Vec::new();
    for id in &ids {
        let g = golden(pack, id);
        assert_eq!(config(&g, "pack"), pack.name(), "{id}: the golden was made with another pack");
        let text = g.input_text(data_dir()).unwrap_or_else(|e| panic!("{e}"));
        let mut docs = vec![Document::new(text.as_str())];
        for result in analyzer.run(&mut docs) {
            result.unwrap_or_else(|e| panic!("{id}: {e}"));
        }
        mentions += docs[0].get::<Pii>().map_or(0, |p| p.spans.len());
        if let Err(m) = matches_golden::<Pii>(&docs[0], &g, Tolerance::exact()) {
            failures.push(format!("{id}: {m}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} texts differ from the JVM in the {} pack:\n{}",
        failures.len(),
        ids.len(),
        pack.name(),
        failures.join("\n")
    );
    mentions
}

macro_rules! pack_tests {
    ($($name:ident: $pack:expr, $at_least:expr;)*) => {$(
        #[test]
        fn $name() {
            if goldens_present() {
                let n = check($pack);
                assert!(n >= $at_least, "only {n} mentions compared");
            }
        }
    )*};
}

pack_tests! {
    payment_matches_the_jvm: Pack::Payment, 15;
    contact_matches_the_jvm: Pack::Contact, 25;
    network_matches_the_jvm: Pack::Network, 12;
    secrets_matches_the_jvm: Pack::Secrets, 5;
    crypto_matches_the_jvm: Pack::Crypto, 5;
    us_identity_matches_the_jvm: Pack::UsIdentity, 3;
    eu_identity_matches_the_jvm: Pack::EuIdentity, 4;
    ca_identity_matches_the_jvm: Pack::CaIdentity, 2;
    device_matches_the_jvm: Pack::Device, 2;
    all_structured_matches_the_jvm: Pack::AllStructured, 80;
}
