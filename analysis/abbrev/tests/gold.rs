//! The MED1250 reader and the pair scoring, on a small sample written for
//! these tests in the file's format, with the file's quirks: carriage
//! returns, a record split in two, synonyms, comments in Windows-1252.

use std::collections::BTreeSet;

use turbo_abbrev::gold::{Counts, Record, normalize, read_labeled};
use turbo_abbrev::{AbbreviationForm, detect};
use turbo_analysis::ViewId;

/// The sample's lines, without their line feeds.
const SAMPLE_LINES: &[&[u8]] = &[
    b"111",
    b"Effects of tumour necrosis factor (TNF) on cells.\r ",
    b"We measured TNF in   the World Health Organization (WHO) cohort.\r Tumour size rose.\r ",
    b"  TNF|tumour necrosis factor\r",
    b"  WHO|World  Health Organization \r",
    b"//!nch\r",
    b"//\x93quoted\x94 comment\r",
    b"//*  WHO cohort|World Health Organization cohort\r",
    b"",
    b"222",
    b"",
    b"A title without definitions.",
    b"",
    b"222",
    b"",
    b"The rest of it, with magnetic resonance (MR) imaging.",
    b"  MR|magnetic resonance",
    b"  MRI|magnetic resonance imaging",
];

fn sample() -> Vec<u8> {
    SAMPLE_LINES.join(&b'\n')
}

#[test]
fn records_lines_and_pairs_are_read() {
    let records = read_labeled(&sample()).unwrap();
    assert_eq!(records.len(), 2, "the split record is joined");
    let first = &records[0];
    assert_eq!(first.pmid, "111");
    assert_eq!(first.lines.len(), 2);
    assert_eq!(&first.text[first.lines[0].range()], "Effects of tumour necrosis factor (TNF) on cells.\r ");
    assert!(first.text[first.lines[1].range()].contains("cohort.\r Tumour"));
    let pairs: Vec<(&str, &str)> = first.pairs.iter().map(|(s, l)| (s.as_str(), l.as_str())).collect();
    assert_eq!(pairs, [("TNF", "tumour necrosis factor"), ("WHO", "World Health Organization")]);
    assert_eq!(first.synonyms.len(), 1);
    let second = &records[1];
    assert_eq!(second.pmid, "222");
    assert_eq!(second.lines.len(), 2);
    assert_eq!(second.text, "A title without definitions.\nThe rest of it, with magnetic resonance (MR) imaging.");
    assert_eq!(second.pairs.len(), 2);
}

#[test]
fn malformed_lines_are_refused_with_their_number() {
    assert_eq!(read_labeled(b"Some text first\n").unwrap_err(), "line 1: text before the first PubMed id");
    assert_eq!(read_labeled(b"1\nTitle\n  no bar\n").unwrap_err(), "line 3: no '|' in \"no bar\"");
    assert_eq!(read_labeled(b"1\nTitle\n  |x\n").unwrap_err(), "line 3: an empty side in \"|x\"");
    assert!(read_labeled(b"1\nTitle \xff\n").unwrap_err().starts_with("line 2: not UTF-8"));
}

#[test]
fn whitespace_is_normalized_for_comparison() {
    assert_eq!(normalize("  World \t Health\r Organization "), "World Health Organization");
}

fn detected(record: &Record) -> BTreeSet<(String, String)> {
    let (layer, links) = detect(&record.text, ViewId::SOURCE, record.lines.iter().copied(), None);
    let text = |i| normalize(&record.text[layer.spans.get(i).range()]);
    layer
        .spans
        .iter()
        .filter(|&(i, _)| layer.form(i) == AbbreviationForm::Short)
        .map(|(i, _)| (text(i), text(layer.long_form(&links, i).unwrap())))
        .collect()
}

#[test]
fn counts_give_precision_and_recall() {
    let records = read_labeled(&sample()).unwrap();
    let mut counts = Counts::default();
    for r in &records {
        counts.add(r, &detected(r));
    }
    // TNF, WHO and MR are found; MRI is not.
    assert_eq!(counts, Counts { true_positives: 3, false_positives: 0, false_negatives: 1, synonyms: 0 });
    assert_eq!(counts.precision(), 1.0);
    assert_eq!(counts.recall(), 0.75);
    assert!((counts.f1() - 6.0 / 7.0).abs() < 1e-12);

    // A synonym is counted neither way; anything else wrong is a false
    // positive.
    let mut counts = Counts::default();
    let guess: BTreeSet<(String, String)> = [
        ("WHO cohort".to_string(), "World Health Organization cohort".to_string()),
        ("TNF".to_string(), "necrosis factor".to_string()),
    ]
    .into();
    counts.add(&records[0], &guess);
    assert_eq!(counts, Counts { true_positives: 0, false_positives: 1, false_negatives: 2, synonyms: 1 });
    assert_eq!(Counts::default().precision(), 0.0);
    assert_eq!(Counts::default().f1(), 0.0);
}
