use turbo_eval::analysis::Sentencer;
use turbo_eval::analysis_bench::{
    command, parse_conllu, parse_sentence_list, score_conllu, score_sentence_list, score_sentencer,
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
