//! The Snowball stemmers over the whole reference vocabularies of
//! snowball-data, when `TURBO_SNOWBALL_DATA` names a checkout of it.
//! Without it the test passes without checking anything.
//!
//! At snowball-data commit da07690d20ce69ee16b5fb95796f92090cc9b4a1, the
//! one that matches the Snowball commit OpenNLP's classes were generated
//! from, every word must stem to its line of `output.txt`. At any other
//! commit the test fails where Snowball changed since, and prints how many
//! words differ per language and the first few of them.

use std::fs;
use std::path::Path;
use std::process::Command;

use turbo_stem::{SnowballAlgorithm, SnowballStemmer, Stemmer};

/// A file of the checkout, read through `gzip -dc` when only its `.gz`
/// form is there (arabic).
fn read(dir: &Path, name: &str) -> String {
    let path = dir.join(name);
    if path.exists() {
        return fs::read_to_string(&path).unwrap();
    }
    let gz = dir.join(format!("{name}.gz"));
    let out = Command::new("gzip").arg("-dc").arg(&gz).output().unwrap_or_else(|e| panic!("{}: {e}", gz.display()));
    assert!(out.status.success(), "gzip -dc {}", gz.display());
    String::from_utf8(out.stdout).unwrap()
}

#[test]
fn snowball_matches_the_reference_vocabularies() {
    let Some(data) = std::env::var_os("TURBO_SNOWBALL_DATA") else {
        eprintln!("TURBO_SNOWBALL_DATA is not set; the vocabularies were not checked");
        return;
    };
    let data = Path::new(&data);
    let mut report = Vec::new();
    let mut total = 0;
    for a in SnowballAlgorithm::ALL {
        let dir = data.join(a.java_name().to_lowercase());
        let (voc, output) = (read(&dir, "voc.txt"), read(&dir, "output.txt"));
        let (voc, output): (Vec<&str>, Vec<&str>) = (voc.lines().collect(), output.lines().collect());
        assert_eq!(voc.len(), output.len(), "{}", dir.display());
        let stemmer = SnowballStemmer::new(a);
        let mut out = Vec::new();
        let mut differ = Vec::new();
        for (word, want) in voc.iter().zip(&output) {
            let units: Vec<u16> = word.encode_utf16().collect();
            stemmer.stem_utf16(&units, &mut out);
            let got = String::from_utf16_lossy(&out);
            if got != *want {
                differ.push(format!("{word} -> {got}, Snowball {want}"));
            }
        }
        total += differ.len();
        report.push(format!(
            "{}: {} of {} differ {:?}",
            a.java_name(),
            differ.len(),
            voc.len(),
            &differ[..differ.len().min(5)]
        ));
    }
    eprintln!("{}", report.join("\n"));
    assert_eq!(total, 0, "words differ from the reference; the counts are above");
}
