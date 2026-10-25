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
    // Python splits at U+0085 and U+001F as well.
    assert_eq!(normalize("New\u{0085}York\u{001F}City"), "new york city");
}

#[test]
fn python_whitespace_preserves_squad_answer_scores() {
    let whitespace = [
        '\u{09}', '\u{0a}', '\u{0b}', '\u{0c}', '\u{0d}', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\u{85}',
        '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}', '\u{2006}',
        '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}', '\u{205f}', '\u{3000}',
    ];
    for c in whitespace {
        let answer = format!("{c}The{c}{c}New{c}York{c}");
        assert_eq!(normalize(&answer), "new york", "U+{:04X}", c as u32);
        assert_eq!(exact(&answer, "New York"), 1.0);
        assert_eq!(f1(&answer, "New York"), 1.0);
    }
    for c in ['\u{200b}', '\u{feff}', '\u{180e}'] {
        let answer = format!("New{c}York");
        assert_eq!(normalize(&answer), format!("new{c}york"));
        assert_eq!(exact(&answer, "New York"), 0.0);
        assert_eq!(f1(&answer, "New York"), 0.0);
    }
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
    // From abstaining everywhere (2 of 4) down: above 2 answers Paris (3
    // of 4), above 1 adds Oslo (2), above -1 adds Rome (3, a tie, so 2
    // stays), answering all adds Bern (2).
    let (t, best) = best_threshold(&examples, &preds, |s| s.f1);
    assert_eq!((t, best.exact), (2.0, 75.0));
    // Answering every question is a candidate: with Bern right it wins.
    let answerable = [ex(&["Paris"]), ex(&["Rome"]), ex(&[]), ex(&["Bern"])];
    let all = [p("Paris", 3.0), p("Rome", 1.0), p("", 2.0), p("Bern", -1.0)];
    let (t, best) = best_threshold(&answerable, &all, |s| s.exact);
    assert_eq!((t, best.exact), (f64::NEG_INFINITY, 100.0));
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
