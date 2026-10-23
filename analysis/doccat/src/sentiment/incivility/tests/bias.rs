use super::*;

fn close(got: Option<f64>, want: f64) {
    let got = got.expect("an AUC");
    assert!((got - want).abs() < 1e-12, "got {got}, want {want}");
}

#[test]
fn roc_auc_on_fixed_scores() {
    close(roc_auc(&[0.9, 0.8], &[0.1, 0.2]), 1.0);
    close(roc_auc(&[0.1, 0.2], &[0.9, 0.8]), 0.0);
    close(roc_auc(&[0.5, 0.5], &[0.5]), 0.5);
    // 0.8 beats both negatives, 0.4 beats only 0.2: 3 of 4 pairs.
    close(roc_auc(&[0.8, 0.4], &[0.6, 0.2]), 0.75);
    // A tie counts one half: (0.5 + 1) / 2.
    close(roc_auc(&[0.5], &[0.5, 0.3]), 0.75);
    assert_eq!(roc_auc(&[], &[0.1]), None);
    assert_eq!(roc_auc(&[0.1], &[]), None);
    assert_eq!(roc_auc(&[f64::NAN], &[0.5]), None);
    assert_eq!(roc_auc(&[0.5], &[f64::INFINITY]), None);
    assert_eq!(roc_auc(&[0.5, f64::NAN], &[0.5, 0.2]), None);
}

#[test]
fn identity_terms_match_case_punctuation_and_phrases() {
    let terms = IdentityTerms::new([
        ("muslim", "Muslim"),
        ("muslim", "muslims"),
        ("black", "African American"),
        ("black", "black"),
        ("gay", "gay"),
    ])
    .unwrap();
    assert_eq!(terms.subgroups(), ["muslim", "black", "gay"]);
    let find = |text: &str| terms.find(&text.split_whitespace().collect::<Vec<_>>());
    assert_eq!(find("I am MUSLIM."), ["muslim"]);
    assert_eq!(find("\u{201C}Muslims\u{201D} and gay friends"), ["gay", "muslim"]);
    assert_eq!(find("an african\u{2010}american author"), Vec::<&str>::new());
    assert_eq!(find("an African American author"), ["black"]);
    assert_eq!(find("an american author"), Vec::<&str>::new());
    assert_eq!(find("gaiety"), Vec::<&str>::new());
    assert!(IdentityTerms::new([("", "x")]).is_err());
    assert!(IdentityTerms::new([("x", " ")]).is_err());
    assert!(IdentityTerms::new([("x", "...")]).is_err());
    assert!(IdentityTerms::new([("x", "gay \u{2026}")]).is_err());
    assert!(IdentityTerms::new([("x", "\u{1F3F3}\u{FE0F}\u{200D}\u{1F308}")]).is_ok());
}

/// Nine units, insult scores only. `muslim` has a neutral mention scored
/// 0.75, above two real insults elsewhere.
fn fixture() -> (Vec<GoldBiasUnit>, Incivility) {
    let rows: [(&[&str], bool, f64); 9] = [
        (&[], true, 0.9),
        (&[], true, 0.7),
        (&[], false, 0.1),
        (&[], false, 0.2),
        (&["muslim"], true, 0.8),
        (&["muslim"], false, 0.75),
        (&["muslim"], false, 0.3),
        (&["gay"], false, 0.55),
        (&["gay"], true, 0.65),
    ];
    let mut layer = Incivility::new();
    let mut gold = Vec::new();
    for (i, (groups, positive, insult)) in rows.into_iter().enumerate() {
        let scores = IncivilityScores { insult, ..Default::default() };
        layer.push(IncivilityItem { target: IncivilityTarget::Sentence(Ix::new(i as u32)), scores, evidence: &[] });
        gold.push(GoldBiasUnit {
            positive_labels: if positive { vec![IncivilityLabel::Insult] } else { Vec::new() },
            subgroups: groups.iter().map(|g| g.to_string()).collect(),
        });
    }
    (gold, layer)
}

#[test]
fn bias_metrics_on_a_fixed_set() {
    let (gold, layer) = fixture();
    let m = evaluate_incivility_bias(&gold, &layer, 0.5).unwrap();
    assert_eq!(m.per_label.len(), 6);
    let insult = &m.per_label[0];
    assert_eq!(insult.label, IncivilityLabel::Insult);
    // Positives 0.9, 0.7, 0.8, 0.65 beat 5, 4, 5 and 4 of the 5 negatives.
    close(insult.overall_auc, 18.0 / 20.0);
    assert_eq!(insult.subgroups.iter().map(|s| s.subgroup.as_str()).collect::<Vec<_>>(), ["gay", "muslim"]);

    let gay = &insult.subgroups[0];
    assert_eq!((gay.units, gay.positives, gay.neutral_mentions, gay.neutral_false_positives), (2, 1, 1, 1));
    close(gay.subgroup_auc, 1.0);
    close(gay.bpsn_auc, 1.0);
    // 0.65 against the background negatives 0.1, 0.2, 0.75, 0.3.
    close(gay.bnsp_auc, 0.75);
    close(gay.neutral_fpr, 1.0);

    let muslim = &insult.subgroups[1];
    assert_eq!((muslim.units, muslim.positives, muslim.neutral_mentions, muslim.neutral_false_positives), (3, 1, 2, 1));
    close(muslim.subgroup_auc, 1.0);
    // Background positives 0.9, 0.7, 0.65 against 0.75 and 0.3: 4 of 6.
    close(muslim.bpsn_auc, 4.0 / 6.0);
    close(muslim.bnsp_auc, 1.0);
    close(muslim.neutral_fpr, 0.5);

    // No unit is positive for the other labels: no AUC, and all neutral
    // mentions score 0.
    for l in &m.per_label[1..] {
        assert_eq!(l.overall_auc, None);
        for s in &l.subgroups {
            assert_eq!((s.subgroup_auc, s.bpsn_auc, s.bnsp_auc), (None, None, None));
            assert_eq!(s.neutral_fpr, Some(0.0));
        }
    }
}

#[test]
fn bias_metrics_reject_a_count_mismatch_and_a_bad_score() {
    let (mut gold, mut layer) = fixture();
    layer.threat[2] = f64::NAN;
    assert!(evaluate_incivility_bias(&gold, &layer, 0.5).unwrap_err().contains("item 2"));
    layer.threat[2] = 0.0;
    for bad in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(evaluate_incivility_bias(&gold, &layer, bad).unwrap_err().contains("threshold"), "{bad}");
    }
    assert!(evaluate_incivility_bias(&gold, &layer, 0.0).is_ok());
    assert!(evaluate_incivility_bias(&gold, &layer, 1.0).is_ok());
    let mut short = layer.clone();
    short.threat.truncate(3);
    assert!(evaluate_incivility_bias(&gold, &short, 0.5).unwrap_err().contains("predicted layer"));
    gold.pop();
    assert!(evaluate_incivility_bias(&gold, &layer, 0.5).is_err());
}
