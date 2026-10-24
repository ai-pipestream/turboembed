//! The finder on the CoNLL-2003 English test set (`eng.testb`), scored
//! as OpenNLP scores a name finder: the samples `Conll03NameSampleStream`
//! reads (`turbo_formats::conll::Conll03Reader`) and the counts of
//! `FMeasure`, a name right when its tokens and type both are.
//!
//! Each sentence is one text, its tokens joined by single spaces, as
//! `NameFinderDL.find` joins them, and read as one sentence. A name the
//! finder gives covers the tokens its bytes overlap, and names with the
//! same tokens and type count once.
//!
//! The finder runs with its defaults, each word labelled by its first
//! piece; `TURBO_NER_DECODING=piece` labels every piece, as `NameFinderDL`
//! does.
//!
//! The set's Reuters text is for research only: the test reads a copy the
//! caller has and keeps nothing of it.

use std::path::PathBuf;

use turbo_formats::conll::{Conll03Language, Conll03Reader, EntityTypes};
use turbo_formats::evaluate::{FMeasure, LabeledSpan};
use turbo_namefind_dl::{Decoding, NameFinderDl, NameFinderDlOptions};

/// The model's type for each of the reader's.
fn reader_kind(model: &str) -> &'static str {
    match model {
        "PER" => "person",
        "ORG" => "organization",
        "LOC" => "location",
        "MISC" => "misc",
        other => panic!("a type the CoNLL-2003 labels do not name: {other}"),
    }
}

fn of_kind<'a>(names: &[LabeledSpan<'a>], kind: &str) -> Vec<LabeledSpan<'a>> {
    names.iter().copied().filter(|s| s.kind == Some(kind)).collect()
}

#[test]
#[ignore = "needs TURBO_NER_BUNDLE (bundle/recipes/dslim-bert-base-ner.json) and TURBO_CONLL2003_TEST naming eng.testb"]
fn conll2003_english_test_set() {
    let bundle = PathBuf::from(std::env::var("TURBO_NER_BUNDLE").expect("TURBO_NER_BUNDLE"));
    let source = std::fs::read_to_string(std::env::var("TURBO_CONLL2003_TEST").expect("TURBO_CONLL2003_TEST")).unwrap();
    let samples: Vec<_> = Conll03Reader::new(&source, Conll03Language::English, EntityTypes::ALL)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    // Each sentence's text and its tokens' bytes.
    let texts: Vec<(String, Vec<(usize, usize)>)> = samples
        .iter()
        .map(|s| {
            let (mut text, mut spans) = (String::new(), Vec::new());
            for t in s.tokens() {
                if !text.is_empty() {
                    text.push(' ');
                }
                spans.push((text.len(), text.len() + t.text.len()));
                text.push_str(&t.text);
            }
            (text, spans)
        })
        .collect();
    let piece = std::env::var("TURBO_NER_DECODING").is_ok_and(|d| d == "piece");
    let decoding = if piece { Decoding::Piece } else { Decoding::Word };
    let finder =
        NameFinderDl::open(&bundle, &NameFinderDlOptions { decoding, ..NameFinderDlOptions::default() }).unwrap();
    let inputs: Vec<(&str, Vec<(usize, usize)>)> =
        texts.iter().map(|(t, _)| (t.as_str(), vec![(0, t.len())])).collect();
    let found = finder.find(&inputs).unwrap();
    let kinds = ["person", "organization", "location", "misc"];
    let mut all = FMeasure::default();
    let mut by_kind = [FMeasure::default(); 4];
    for ((sample, (_, spans)), names) in samples.iter().zip(&texts).zip(&found) {
        let gold: Vec<LabeledSpan<'_>> =
            sample.names().iter().map(|n| LabeledSpan::new(n.start, n.end, n.kind)).collect();
        let mut predicted: Vec<LabeledSpan<'_>> = names
            .iter()
            .map(|c| {
                let first = spans.iter().position(|&(_, e)| e > c.start).unwrap();
                let last = spans.iter().rposition(|&(s, _)| s < c.end).unwrap();
                let kind = reader_kind(finder.kinds().label(turbo_analysis_layers::Tag(c.kind as u16)));
                LabeledSpan::new(first as u32, last as u32 + 1, Some(kind))
            })
            .collect();
        // Two names of one word's pieces (`Ko` `##fi`, each a `B-`) cover
        // the same token: one name there, as `FMeasure` would count a
        // repeated one twice.
        let mut seen = Vec::new();
        predicted.retain(|p| {
            let new = !seen.contains(p);
            seen.push(*p);
            new
        });
        all.update(&gold, &predicted);
        for (k, m) in kinds.iter().zip(by_kind.iter_mut()) {
            m.update(&of_kind(&gold, k), &of_kind(&predicted, k));
        }
    }
    eprintln!(
        "{} sentences: {} names, {} found, {} right; precision {:.4}, recall {:.4}, F1 {:.4}",
        samples.len(),
        all.target(),
        all.selected(),
        all.true_positives(),
        all.precision(),
        all.recall(),
        all.f_measure()
    );
    for (k, m) in kinds.iter().zip(&by_kind) {
        eprintln!("  {k}: precision {:.4}, recall {:.4}, F1 {:.4}", m.precision(), m.recall(), m.f_measure());
    }
    // 0.8952 by the tokenizer's words, 0.8641 by pieces.
    let floor = if piece { 0.86 } else { 0.89 };
    assert!(all.f_measure() > floor, "F1 {}", all.f_measure());
}
