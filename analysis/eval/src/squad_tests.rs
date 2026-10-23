use super::*;

#[test]
fn normalize_matches_the_official_script() {
    // Values from evaluate-v2.0.py's normalize_answer on the same input.
    assert_eq!(normalize("The Cat!"), "cat");
    assert_eq!(normalize("  a  dog,  AN apple and the  end. "), "dog apple and end");
    assert_eq!(normalize("theater"), "theater");
    assert_eq!(normalize("Ann's"), "anns");
    assert_eq!(normalize("the\u{2019}s"), "\u{2019}s");
    assert_eq!(normalize("Ünïcode\u{00A0}Text"), "ünïcode text");
    assert_eq!(normalize("."), "");
}

#[test]
fn exact_and_f1_on_known_pairs() {
    assert_eq!(exact("the Eiffel Tower", "Eiffel tower."), 1.0);
    assert_eq!(exact("Eiffel", "Eiffel Tower"), 0.0);
    // One common token of 1 and 2: precision 1, recall 0.5.
    assert!((f1("Eiffel", "Eiffel Tower") - 2.0 / 3.0).abs() < 1e-12);
    // Repeated tokens count once each: 2 common of 3 and 2.
    assert!((f1("tower tower paris", "tower tower") - 0.8).abs() < 1e-12);
    assert_eq!(f1("london", "paris"), 0.0);
    assert_eq!(f1("", ""), 1.0);
    assert_eq!(f1("the", "paris"), 0.0);
}

fn ex(answers: &[&str]) -> Example {
    Example {
        id: "q".into(),
        question: "q?".into(),
        context: "c".into(),
        answers: answers.iter().map(|a| a.to_string()).collect(),
    }
}

#[test]
fn a_question_takes_its_best_gold_answer_and_no_answer_wants_empty() {
    assert_eq!(question_scores(&ex(&["Paris", "the city of Paris"]), "city of Paris"), (1.0, 1.0));
    assert_eq!(question_scores(&ex(&[]), ""), (1.0, 1.0));
    assert_eq!(question_scores(&ex(&[]), "Paris"), (0.0, 0.0));
}

#[test]
fn scores_and_the_best_threshold() {
    let examples = [ex(&["Paris"]), ex(&["Rome"]), ex(&[]), ex(&[])];
    let p = |text: &str, margin: f64| Prediction { text: text.into(), margin };
    let preds = [p("Paris", 3.0), p("Rome", 1.0), p("Oslo", 2.0), p("Bern", -1.0)];
    // At 0: the first three answer; the third is wrong, the fourth abstains.
    let s = score(&examples, &preds, 0.0);
    assert_eq!((s.total, s.has_ans_total, s.no_ans_total), (4, 2, 2));
    assert_eq!((s.exact, s.has_ans_exact, s.no_ans_exact), (75.0, 100.0, 50.0));
    // Above -1 and above 0 give 3 of 4; above 1 loses Rome (2 of 4); above
    // 2 loses Rome and gains Oslo's abstention (3 of 4). The first best,
    // the lowest threshold, stays.
    let (t, best) = best_threshold(&examples, &preds);
    assert_eq!(best.exact, 75.0);
    assert_eq!(t, -1.0);
    // Abstaining everywhere: only the unanswerable questions are right.
    assert_eq!(score(&examples, &preds, 3.0).exact, 50.0);
}

#[test]
fn read_takes_squad_v2_layout() {
    let text = r#"{"version":"v2.0","data":[{"title":"T","paragraphs":[{"context":"Paris is in France.",
        "qas":[{"id":"a","question":"Where?","answers":[{"text":"France","answer_start":12}],"is_impossible":false},
               {"id":"b","question":"Who?","answers":[],"is_impossible":true}]}]}]}"#;
    let e = read(text).unwrap();
    assert_eq!(e.len(), 2);
    assert_eq!(e[0].answers, ["France"]);
    assert!(e[1].answers.is_empty());
    assert!(read(r#"{"data":[{"paragraphs":[{"context":"c","qas":[{"question":"q","answers":[]}]}]}]}"#).is_err());
}
