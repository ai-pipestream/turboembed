//! Precision and recall of the detector on Ab3P's labelled MED1250 set.
//!
//! ```text
//! git clone https://github.com/ncbi-nlp/Ab3P <dir>
//! git -C <dir> checkout 41130cddfcba1449ba612905d4a51274f8f565a8
//! cargo run --release -p turbo-abbrev --example med1250 -- <dir>/MED1250_labeled
//! ```
//!
//! Each line of a record (the title, the abstract) is one sentence, as the
//! Schwartz and Hearst reference implementation reads its input a line at a
//! time. A detected pair is the text of its short and long form spans; the
//! example checks that every span is on character boundaries of the
//! record's text before taking it. Pairs are compared exactly, after
//! whitespace runs are made single spaces.

use std::collections::BTreeSet;
use std::process::ExitCode;

use turbo_abbrev::gold::{Counts, normalize, read_labeled};
use turbo_abbrev::{AbbreviationForm, Abbreviations, detect};
use turbo_analysis::{Ix, Layer, ViewId};

/// SHA-256 of `MED1250_labeled` at the commit above.
const SHA256: &str = "5093fa8f130ee250add0d0fbde7fc736478e18fbcc4447b00b4179db47f4cf53";

fn main() -> ExitCode {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("usage: med1250 <path to MED1250_labeled>");
        return ExitCode::from(2);
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    let digest: String = turbo_goldens::sha256(&bytes).iter().map(|b| format!("{b:02x}")).collect();
    if digest != SHA256 {
        eprintln!("{path}: SHA-256 {digest}, expected {SHA256}");
        return ExitCode::from(2);
    }
    let records = match read_labeled(&bytes) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{path}: {e}");
            return ExitCode::from(2);
        }
    };
    let verbose = std::env::args().any(|a| a == "--verbose");
    let mut counts = Counts::default();
    let mut gold = 0;
    let mut mentions = 0;
    for record in &records {
        let (layer, links) = detect(&record.text, ViewId::SOURCE, record.lines.iter().copied(), None);
        if let Err(e) = layer.check().and_then(|()| links.check()) {
            eprintln!("{}: invalid layer: {e}", record.pmid);
            return ExitCode::from(1);
        }
        let mut detected = BTreeSet::new();
        for (i, span) in layer.spans.iter() {
            assert!(
                record.text.is_char_boundary(span.start as usize) && record.text.is_char_boundary(span.end as usize)
            );
            match layer.form(i) {
                AbbreviationForm::Short => {
                    let long = layer.long_form(&links, i).expect("a defining short form has a long form");
                    detected.insert((normalize(text(record, &layer, i)), normalize(text(record, &layer, long))));
                }
                AbbreviationForm::Mention => mentions += 1,
                AbbreviationForm::Long => {}
            }
        }
        if verbose {
            for p in detected.iter().filter(|p| !record.pairs.contains(*p) && !record.synonyms.contains(*p)) {
                println!("{} FP {}|{}", record.pmid, p.0, p.1);
            }
            for p in record.pairs.iter().filter(|p| !detected.contains(*p)) {
                println!("{} FN {}|{}", record.pmid, p.0, p.1);
            }
        }
        gold += record.pairs.len();
        counts.add(record, &detected);
    }
    println!("records {}, gold pairs {gold}, mentions linked {mentions}", records.len());
    println!(
        "true positives {}, false positives {}, false negatives {}, synonyms {}",
        counts.true_positives, counts.false_positives, counts.false_negatives, counts.synonyms
    );
    println!("precision {:.4}, recall {:.4}, F1 {:.4}", counts.precision(), counts.recall(), counts.f1());
    ExitCode::SUCCESS
}

fn text<'a>(record: &'a turbo_abbrev::gold::Record, layer: &Abbreviations, i: Ix<Abbreviations>) -> &'a str {
    &record.text[layer.spans.get(i).range()]
}
