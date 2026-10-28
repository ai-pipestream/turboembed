use turbo_eval::analysis::Sentencer;
use turbo_eval::analysis_bench::{
    command, parse_conllu, parse_lang_gold, parse_lang_pred, parse_ner_gold, parse_ner_pred, parse_sentence_list,
    score_conllu, score_language, score_ner, score_sentence_list, score_sentencer, script_fallback_predictions,
};
use turbo_eval::sha::{sha256, to_hex};

const GOLD: &str = concat!(
    "# text = Cats purr.\n",
    "1\tCats\tcat\tNOUN\tNN\tNumber=Plur\t2\tnsubj\t_\t_\n",
    "2\tpurr\tpurr\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
    "3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n",
    "\n",
    "# text = Dogs bark.\n",
    "1-2\tDogs\t_\t_\t_\t_\t_\t_\t_\t_\n",
    "1\tDogs\tdog\tNOUN\tNN\tNumber=Plur\t2\tnsubj\t_\t_\n",
    "2\tbark\tbark\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
    "3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n",
);

#[test]
fn parse_skips_multiword_ranges_and_keeps_sentence_text() {
    let gold = parse_conllu(GOLD).unwrap();
    assert_eq!(gold.len(), 2);
    assert_eq!(gold[0].text.as_deref(), Some("Cats purr."));
    assert_eq!(gold[1].tokens.len(), 3);
    assert_eq!(gold[1].tokens[0].form, "Dogs");
}

#[test]
fn malformed_rows_are_refused_with_the_line_number() {
    let err = parse_conllu("1\tCats\tcat\n").unwrap_err();
    assert!(err.contains("line 1"), "{err}");
    assert!(err.contains("expected 10"), "{err}");
}

#[test]
fn a_perfect_prediction_scores_one_hundred_everywhere() {
    let gold = parse_conllu(GOLD).unwrap();
    let scores = score_conllu("self", &gold, &gold, 20);
    assert_eq!(scores.sentences.score, 100.0);
    assert_eq!(scores.tokens.unwrap().score, 100.0);
    assert_eq!(scores.upos.unwrap().score, 100.0);
    assert_eq!(scores.las.unwrap().score, 100.0);
    assert_eq!(scores.token_mismatch_sentences, 0);
    assert!(scores.entities.is_none());
}

#[test]
fn field_errors_and_token_mismatches_are_counted_not_aligned_away() {
    let gold = parse_conllu(GOLD).unwrap();
    let mut pred = gold.clone();
    pred[0].tokens[0].upos = "PROPN".into();
    pred[0].tokens[0].head = Some(0);
    pred[1].tokens.remove(1);
    let scores = score_conllu("altered", &gold, &pred, 0);
    // The second gold sentence holds a token ("bark") with no aligned
    // partner, so it is counted; its other tokens still score.
    assert_eq!(scores.token_mismatch_sentences, 1);
    // UPOS: all but "bark" (unaligned) and the altered "Cats": 4 of 6.
    assert_eq!(scores.upos.unwrap().correct, 4);
    // UAS: sentence one scores the root and the punctuation; the
    // shortened second sentence's heads name the wrong tokens.
    assert_eq!(scores.uas.as_ref().unwrap().correct, 2);
    assert!(scores.uas.unwrap().low.is_none(), "no resamples were requested");
}

#[test]
fn resegmented_predictions_keep_their_field_scores() {
    // The same six tokens as GOLD but one sentence: fields must stay
    // perfect while the sentence boundary is missed, the raw-text peer
    // shape that sentence-index pairing used to score as near zero.
    let pred = parse_conllu(concat!(
        "1\tCats\tcat\tNOUN\tNN\tNumber=Plur\t2\tnsubj\t_\t_\n",
        "2\tpurr\tpurr\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
        "3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n",
        "4\tDogs\tdog\tNOUN\tNN\tNumber=Plur\t5\tnsubj\t_\t_\n",
        "5\tbark\tbark\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
        "6\t.\t.\tPUNCT\t.\t_\t5\tpunct\t_\t_\n",
    ))
    .unwrap();
    let gold = parse_conllu(GOLD).unwrap();
    let scores = score_conllu("merged", &gold, &pred, 0);
    assert_eq!(scores.tokens.unwrap().score, 100.0);
    assert_eq!(scores.upos.unwrap().score, 100.0);
    assert_eq!(scores.las.unwrap().score, 100.0);
    assert_eq!(scores.sentences.correct, 0);
    assert_eq!(scores.sentences.target, 1);
    assert_eq!(scores.token_mismatch_sentences, 0);
}

#[test]
fn split_tokens_resync_instead_of_shifting_the_stream() {
    // "Cats" split in two in the prediction: the walk must resync at
    // "purr" so the rest of the stream still scores.
    let pred = parse_conllu(concat!(
        "1\tCat\tcat\tNOUN\tNN\tNumber=Plur\t3\tnsubj\t_\t_\n",
        "2\ts\ts\tNOUN\tNN\tNumber=Plur\t3\tnsubj\t_\t_\n",
        "3\tpurr\tpurr\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
        "4\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\t_\n",
        "\n",
        "1\tDogs\tdog\tNOUN\tNN\tNumber=Plur\t2\tnsubj\t_\t_\n",
        "2\tbark\tbark\tVERB\tVB\tTense=Pres\t0\troot\t_\t_\n",
        "3\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\t_\n",
    ))
    .unwrap();
    let gold = parse_conllu(GOLD).unwrap();
    let scores = score_conllu("split", &gold, &pred, 0);
    assert_eq!(scores.tokens.as_ref().unwrap().correct, 5);
    assert_eq!(scores.tokens.as_ref().unwrap().selected, 7);
    assert_eq!(scores.upos.unwrap().correct, 5);
    assert_eq!(scores.token_mismatch_sentences, 1);
    assert_eq!(scores.sentences.score, 100.0);
}

#[test]
fn sentence_lists_tile_gold_documents_or_are_refused() {
    let gold = parse_conllu(GOLD).unwrap();
    let perfect = parse_sentence_list("Cats purr.\nDogs bark.\n");
    let scores = score_sentence_list("peer", &gold, &perfect).unwrap();
    assert_eq!(scores.sentences.score, 100.0);
    assert!(scores.tokens.is_none());

    let merged = parse_sentence_list("Cats purr. Dogs bark.\n");
    let scores = score_sentence_list("peer", &gold, &merged).unwrap();
    assert_eq!(scores.sentences.correct, 0);
    assert_eq!(scores.sentences.target, 1);

    let swapped = parse_sentence_list("Dogs bark.\nCats purr.\n");
    let err = score_sentence_list("peer", &gold, &swapped).unwrap_err();
    assert!(err.contains("in order") || err.contains("skips"), "{err}");

    let two_docs = parse_sentence_list("Cats purr.\n\nDogs bark.\n");
    let err = score_sentence_list("peer", &gold, &two_docs).unwrap_err();
    assert!(err.contains("predicted documents"), "{err}");
}

#[test]
fn the_rule_sentencer_scores_the_joined_gold_text() {
    let gold = parse_conllu(GOLD).unwrap();
    let scores = score_sentencer("turbo-rule", &gold, &Sentencer::rule()).unwrap();
    assert_eq!(scores.sentences.score, 100.0);
    assert!(scores.tokens.is_none());
    assert!(scores.throughput_sentences_per_second.is_some_and(|t| t > 0.0));
}

#[test]
fn language_scoring_and_the_script_fallback() {
    let gold = parse_lang_gold(concat!(
        "ell\tΚαλημέρα κόσμε\n",
        "kor\t안녕하세요 세상\n",
        "eng\tGood morning world\n",
        "spa\tBuenos días mundo\n",
        "zho\t早上好世界\n",
    ))
    .unwrap();
    // Greek and Hangul identify their language by script in this label
    // set; Latin is shared (eng/spa) and Han has no unique candidate
    // once zho's table alias shares it with Japanese's group alone...
    // here zho is the only Han-group label, so it is named.
    let fallback = script_fallback_predictions(&gold);
    assert_eq!(fallback, ["ell", "kor", "und", "und", "zho"]);
    let scores = score_language("fallback", &gold, &fallback, 0);
    assert_eq!(scores.language.correct, 3);
    assert_eq!(scores.language.target, 5);
    assert_eq!(scores.per_language.iter().find(|l| l.code == "eng").unwrap().score, 0.0);

    let pred = parse_lang_pred("ell\nkor\neng\t0.9\nspa\nzho\n", 5).unwrap();
    assert_eq!(score_language("peer", &gold, &pred, 0).language.score, 100.0);
    let err = parse_lang_pred("ell\nkor\n", 5).unwrap_err();
    assert!(err.contains("2 predicted codes for 5"), "{err}");
    assert!(parse_lang_gold("no-tab-here").is_err());
}

#[test]
fn the_language_command_writes_a_record() {
    let dir = std::env::temp_dir().join(format!("turbo-eval-lang-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let gold_text = "ell\tΚαλημέρα κόσμε\nkor\t안녕하세요 세상\n";
    let gold = dir.join("gold.tsv");
    std::fs::write(&gold, gold_text).unwrap();
    let digest = to_hex(&sha256(gold_text.as_bytes()));
    let out = dir.join("record.json");
    command(&[
        "--lang-gold".into(),
        gold.to_string_lossy().into_owned(),
        "--sha256".into(),
        digest.clone(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--script-fallback".into(),
        "--resamples".into(),
        "0".into(),
    ])
    .unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(record["gold_sha256"], digest);
    assert_eq!(record["texts"], 2);
    assert_eq!(record["systems"][0]["language"]["score"], 100.0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn entity_spans_score_exactly_and_shapes_are_checked() {
    let gold = parse_ner_gold(concat!(
        "# a header comment\n",
        "#hashtag\tO\n",
        "Ada\tB-person\n",
        "Lovelace\tI-person\n",
        "met\tO\n",
        "Acme\tB-corporation\n",
        "\n",
        "Paris\tB-location\n",
        "sleeps\tO\n",
    ))
    .unwrap();
    assert_eq!(gold.len(), 2);
    let perfect = parse_ner_pred("O\nB-person\nI-person\nO\nB-corporation\n\nB-location\nO\n", &gold).unwrap();
    assert_eq!(score_ner("self", &gold, &perfect, 0).entities.score, 100.0);

    // A split person span and a missed corporation: 1 of 3 spans right.
    let flawed = parse_ner_pred("O\nB-person\nO\nO\nO\n\nB-location\nO\n", &gold).unwrap();
    let scores = score_ner("flawed", &gold, &flawed, 0);
    assert_eq!(scores.entities.correct, 1);
    assert_eq!(scores.entities.selected, 2);
    assert_eq!(scores.entities.target, 3);
    assert_eq!(scores.per_type.iter().find(|t| t.kind == "location").unwrap().score, 100.0);
    assert_eq!(scores.per_type.iter().find(|t| t.kind == "person").unwrap().score, 0.0);

    let err = parse_ner_pred("B-person\n", &gold).unwrap_err();
    assert!(err.contains("predicted sentences"), "{err}");
    assert!(parse_ner_gold("Ada\tX-person\n").is_err());
}

#[test]
fn the_entity_command_writes_a_record() {
    let dir = std::env::temp_dir().join(format!("turbo-eval-ner-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let gold_text = "Ada\tB-person\nruns\tO\n";
    let gold = dir.join("gold.tsv");
    std::fs::write(&gold, gold_text).unwrap();
    let pred = dir.join("pred.tsv");
    std::fs::write(&pred, "B-person\nO\n").unwrap();
    let digest = to_hex(&sha256(gold_text.as_bytes()));
    let out = dir.join("record.json");
    command(&[
        "--ner-gold".into(),
        gold.to_string_lossy().into_owned(),
        "--sha256".into(),
        digest.clone(),
        "--out".into(),
        out.to_string_lossy().into_owned(),
        "--ner-pred".into(),
        format!("peer={}", pred.display()),
        "--resamples".into(),
        "0".into(),
    ])
    .unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(record["gold_sha256"], digest);
    assert_eq!(record["systems"][0]["entities"]["score"], 100.0);
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn the_command_writes_a_record_and_refuses_a_wrong_pin() {
    let dir = std::env::temp_dir().join(format!("turbo-eval-analysis-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let gold = dir.join("gold.conllu");
    std::fs::write(&gold, GOLD).unwrap();
    let digest = to_hex(&sha256(GOLD.as_bytes()));
    let out = dir.join("record.json");
    let args = |sha: &str| {
        vec![
            "--gold".into(),
            gold.to_string_lossy().into_owned(),
            "--sha256".into(),
            sha.into(),
            "--out".into(),
            out.to_string_lossy().into_owned(),
            "--pred".into(),
            format!("gold-as-pred={}", gold.display()),
            "--sentences".into(),
            "rule".into(),
            "--resamples".into(),
            "10".into(),
        ]
    };
    let err = command(&args("00")).unwrap_err();
    assert!(err.contains("sha256"), "{err}");

    command(&args(&digest)).unwrap();
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(record["gold_sha256"], digest);
    assert_eq!(record["sentences"], 2);
    assert_eq!(record["systems"].as_array().unwrap().len(), 2);
    assert_eq!(record["systems"][0]["upos"]["score"], 100.0);
    std::fs::remove_dir_all(&dir).unwrap();
}
