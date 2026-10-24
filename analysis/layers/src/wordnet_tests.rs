use super::*;

fn sense<'a>(token: u32, synset_id: &'a str, pos: u8, synset_lemmas: &'a [String]) -> SenseRow<'a> {
    SenseRow {
        token: Ix::new(token),
        synset_id,
        sense_id: "",
        pos,
        lemma: "dog",
        synset_lemmas,
        gloss: "a domestic canine",
        sense_rank: 0,
    }
}

fn expansion(token: u32, relation: u8, depth: u8, weight: f64) -> ExpansionRow<'static> {
    ExpansionRow { token: Ix::new(token), term: "canine", relation, depth, sense_rank: 0, weight, synset_id: "s1" }
}

#[test]
fn senses_check_columns_rows_parts_of_speech_and_token_order() {
    let members = vec!["dog".to_owned(), "domestic dog".to_owned()];
    let mut s = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    s.push(Span::new(4, 8), sense(1, "s1", b'n', &members));
    s.push(Span::new(4, 8), sense(1, "s2", b'v', &[]));
    assert_eq!(s.check(), Ok(()));
    assert_eq!(s.members(0).collect::<Vec<_>>(), ["dog", "domestic dog"]);
    assert_eq!(s.members(1).count(), 0);
    assert_eq!(s.indexes()[0].1, 2);

    let mut bad = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), sense(1, "s1", b's', &members));
    assert!(bad.check().unwrap_err().contains("part of speech code"));

    let mut bad = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), sense(1, "s1", b'n', &members));
    bad.gloss = Strs::default();
    assert!(bad.check().unwrap_err().contains("columns of"));

    let mut bad = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), sense(1, "s1", b'n', &members));
    bad.synset_lemma_row[1] = 1;
    assert!(bad.check().unwrap_err().contains("synset lemma row"));

    let mut bad = WordNetSenses::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), sense(1, "s1", b'n', &members));
    bad.push(Span::new(0, 3), sense(0, "s2", b'n', &members));
    assert_eq!(bad.check().unwrap_err(), "senses not in token order");
}

#[test]
fn expansions_check_relation_codes_depths_and_weights() {
    let mut e = LexicalExpansions::new(ViewId::SOURCE, "test:1".into());
    e.push(Span::new(4, 8), expansion(1, SAME_SYNSET, 0, 1.0));
    e.push(Span::new(4, 8), expansion(1, WORDNET_RELATIONS, 2, 0.25));
    assert_eq!(e.check(), Ok(()));

    let mut bad = LexicalExpansions::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), expansion(1, WORDNET_RELATIONS + 1, 1, 0.5));
    assert!(bad.check().unwrap_err().contains("relation code 107"));

    let mut bad = LexicalExpansions::new(ViewId::SOURCE, "test:1".into());
    bad.push(Span::new(4, 8), expansion(1, 2, 0, 0.5));
    assert!(bad.check().unwrap_err().contains("depth 0 if and only if"));

    for w in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
        let mut bad = LexicalExpansions::new(ViewId::SOURCE, "test:1".into());
        bad.push(Span::new(4, 8), expansion(1, 2, 1, w));
        assert!(bad.check().unwrap_err().contains("weight outside"), "accepted {w}");
    }
}
