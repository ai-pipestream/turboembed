//! Scoring, the majority baseline and the paired bootstrap.

use turbo_trainer_recipes::corpus::{Sentence, Task};
use turbo_trainer_recipes::eval::{Majority, Metrics, Seen, bootstrap, evaluate, spans};

fn s(words: &str, labels: &str) -> Sentence {
    Sentence {
        words: words.split(' ').map(str::to_owned).collect(),
        labels: labels.split(' ').map(str::to_owned).collect(),
    }
}

fn v(labels: &str) -> Vec<String> {
    labels.split(' ').map(str::to_owned).collect()
}

#[test]
fn spans_are_read_the_conlleval_way() {
    let got = spans(&v("B-PER I-PER O I-LOC I-LOC B-LOC I-PER B-ORG"));
    let want = [(0, 2, "PER"), (3, 5, "LOC"), (5, 6, "LOC"), (6, 7, "PER"), (7, 8, "ORG")];
    assert_eq!(got, want.map(|(a, b, k)| (a, b, k.to_owned())));
    assert!(spans(&v("O O")).is_empty());
}

#[test]
fn tags_score_by_accuracy_with_unseen_words_and_labels() {
    let train = [s("the dog runs", "DT NN VBZ")];
    let gold = [s("the cat runs .", "DT NN VBZ .")];
    let m = evaluate(Task::Tags, &gold, &[v("DT NN NNS .")], &Seen::of(&train), None);
    let Metrics::Tags { words, correct, accuracy, unseen_words, unseen_correct, unseen_labels } = m else { panic!() };
    assert_eq!((words, correct, unseen_words, unseen_correct, unseen_labels), (4, 3, 2, 2, 1));
    assert!((accuracy - 0.75).abs() < 1e-12);
}

#[test]
fn spans_score_exactly_per_type_and_per_group() {
    let gold = [s("a b c d", "B-Athlete I-Athlete O B-Station")];
    let pred = [v("B-Athlete I-Athlete O B-Facility")];
    let group = |t: &str| -> &str { if t == "Athlete" { "Person" } else { "Location" } };
    let Metrics::Spans { all, per_type, coarse } = evaluate(Task::Bio, &gold, &pred, &Seen::default(), Some(group))
    else {
        panic!()
    };
    assert_eq!((all.gold, all.found, all.correct), (2, 2, 1));
    assert!((all.f1 - 0.5).abs() < 1e-12);
    assert_eq!(per_type["Station"].correct, 0);
    assert_eq!(per_type["Facility"].found, 1);
    assert!((coarse["Location"].f1 - 1.0).abs() < 1e-12, "a wrong fine type in the right group counts there");
}

#[test]
fn the_majority_baseline_takes_each_words_commonest_label() {
    let train = [s("run run run the", "VB NN NN DT"), s("a a", "DT DT"), s("x", "NN")];
    let m = Majority::train(&train);
    assert_eq!(m.predict(&v("run the unseen")), v("NN DT DT"), "DT and NN tie overall: the lesser wins");
    let tie = Majority::train(&[s("w w", "B A")]);
    assert_eq!(tie.predict(&v("w")), v("A"));
}

#[test]
fn the_bootstrap_is_seeded_and_paired() {
    let gold: Vec<Sentence> = (0..40).map(|i| s("a b", if i % 2 == 0 { "X Y" } else { "Y Y" })).collect();
    let model: Vec<Vec<String>> = gold.iter().map(|g| g.labels.clone()).collect();
    let base: Vec<Vec<String>> = gold.iter().map(|_| v("Y Y")).collect();
    let a = bootstrap(Task::Tags, &gold, &model, &base, 200, 7);
    assert_eq!(a, bootstrap(Task::Tags, &gold, &model, &base, 200, 7));
    assert!((a.model - 1.0).abs() < 1e-12 && (a.baseline - 0.75).abs() < 1e-12);
    assert!(a.low > 0.0 && a.low <= a.delta && a.delta <= a.high);
    let same = bootstrap(Task::Tags, &gold, &model, &model, 50, 1);
    assert_eq!((same.low, same.high), (0.0, 0.0));
}
