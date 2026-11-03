use std::ops::Range;

use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval;
use turbo_trainer_recipes::ner_documents::{self, Report, Request};

fn labels(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

fn sentence(text: &str) -> Sentence {
    let labels = labels(text);
    Sentence { words: (0..labels.len()).map(|i| format!("word{i}")).collect(), labels }
}

fn one_document(range: Range<usize>) -> Vec<Range<usize>> {
    std::iter::once(range).collect()
}

fn compare(
    gold: &[Sentence],
    candidate: &[Vec<String>],
    peer: &[Vec<String>],
    groups: &[Range<usize>],
) -> turbo_trainer_recipes::Result<Report> {
    ner_documents::compare(Request {
        gold,
        candidate,
        peer,
        documents: groups,
        types: &labels("person location organization date"),
        resamples: 512,
        seed: 7,
    })
}

#[test]
fn correlated_sentences_stay_together_in_each_document_draw() {
    // Twenty successes belong to one document, one failure to the other.
    // The four equally likely document draws yield deltas -1, 19/21, 19/21, 1.
    let gold = vec![sentence("B-person"); 21];
    let candidate: Vec<_> = (0..21).map(|i| labels(if i < 20 { "B-person" } else { "B-organization" })).collect();
    let peer: Vec<_> = (0..21).map(|i| labels(if i < 20 { "B-organization" } else { "B-person" })).collect();
    let report = compare(&gold, &candidate, &peer, &[0..20, 20..21]).unwrap();
    assert_eq!((report.all.low, report.all.high), (-1.0, 1.0));
    assert!((report.all.delta - 19.0 / 21.0).abs() < 1e-12);
    assert_eq!(report.statistical_unit, "document");
    assert_eq!(report.document_ranges, [[0, 20], [20, 21]]);
    assert_eq!((report.per_type["person"].low, report.per_type["person"].high), (-1.0, 1.0));
    // Treating sentences as independent would incorrectly exclude zero here.
    assert!(eval::bootstrap(Task::Bio, &gold, &candidate, &peer, 512, 7).low > 0.0);
}

#[test]
fn spans_require_exact_boundaries_and_type_and_do_not_cross_sentences() {
    let gold = [sentence("B-person I-person O B-location"), sentence("B-person O")];
    let candidate = [labels("B-person O O B-location"), labels("B-location O")];
    let peer = [labels("B-person I-person O B-location"), labels("B-person O")];
    let report = compare(&gold, &candidate, &peer, &[0..1, 1..2]).unwrap();
    let score = report.candidate.all;
    assert_eq!((score.gold, score.found, score.correct), (3, 3, 1));
    assert!((score.f1 - 1.0 / 3.0).abs() < 1e-12);
    assert_eq!(report.peer.all.f1, 1.0);
    let per = &report.candidate.per_type;
    assert_eq!((per["person"].gold, per["person"].found, per["person"].correct), (2, 1, 0));
    assert_eq!((per["location"].gold, per["location"].found, per["location"].correct), (1, 2, 1));
    assert_eq!((per["date"].gold, per["date"].found, per["date"].f1), (0, 0, 0.0));
    assert_eq!((report.per_type["date"].low, report.per_type["date"].high), (0.0, 0.0));
}

#[test]
fn grouped_bootstrap_is_seeded_paired_and_serializes_its_units() {
    let gold = [sentence("B-person O"), sentence("O B-date"), sentence("B-location O")];
    let predicted: Vec<_> = gold.iter().map(|s| s.labels.clone()).collect();
    let groups = [0..2, 2..3];
    let first = compare(&gold, &predicted, &predicted, &groups).unwrap();
    assert_eq!(first, compare(&gold, &predicted, &predicted, &groups).unwrap());
    assert_eq!((first.all.delta, first.all.low, first.all.high), (0.0, 0.0, 0.0));
    assert!(first.per_type.values().all(|c| c.delta == 0.0 && c.low == 0.0 && c.high == 0.0));
    let serialized = serde_json::to_string(&first).unwrap();
    assert_eq!(first, serde_json::from_str(&serialized).unwrap());
}

#[test]
fn refuses_incomplete_overlapping_or_reordered_document_membership() {
    let gold = vec![sentence("O"); 3];
    let predicted = vec![labels("O"); 3];
    for groups in [
        vec![],
        one_document(1..3),
        one_document(0..2),
        vec![0..0, 0..3],
        vec![0..1, 2..3],
        vec![0..2, 1..3],
        one_document(0..4),
        vec![2..3, 0..2],
    ] {
        assert!(compare(&gold, &predicted, &predicted, &groups).is_err(), "{groups:?}");
    }
}

#[test]
fn refuses_malformed_sequences_unmapped_types_and_token_misalignment() {
    let gold = [sentence("B-person I-person")];
    let good = [labels("B-person I-person")];
    for bad in [
        vec![],
        vec![labels("O")],
        vec![labels("I-person I-person")],
        vec![labels("B-person I-location")],
        vec![labels("B-unknown O")],
        vec![labels("B-person X")],
        vec![labels("B-person I-")],
    ] {
        assert!(compare(&gold, &bad, &good, &one_document(0..1)).is_err());
        assert!(compare(&gold, &good, &bad, &one_document(0..1)).is_err());
    }
    assert!(compare(&[sentence("I-person O")], &good, &good, &one_document(0..1)).is_err());
    assert!(compare(&[sentence("")], &[vec![]], &[vec![]], &one_document(0..1)).is_err());
    let mut wrong = gold.to_vec();
    wrong[0].words.pop();
    assert!(compare(&wrong, &good, &good, &one_document(0..1)).is_err());
}

#[test]
fn refuses_empty_schema_duplicate_types_and_zero_resamples() {
    let gold = [sentence("O")];
    let prediction = [labels("O")];
    for types in [vec![], labels("person person"), vec!["".into()], vec!["O".into()], vec!["two words".into()]] {
        assert!(
            ner_documents::compare(Request {
                gold: &gold,
                candidate: &prediction,
                peer: &prediction,
                documents: &one_document(0..1),
                types: &types,
                resamples: 5,
                seed: 1,
            })
            .is_err()
        );
    }
    assert!(
        ner_documents::compare(Request {
            gold: &gold,
            candidate: &prediction,
            peer: &prediction,
            documents: &one_document(0..1),
            types: &labels("person"),
            resamples: 0,
            seed: 1,
        })
        .is_err()
    );
    assert!(compare(&[], &[], &[], &[]).is_err());
}
