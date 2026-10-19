use super::*;

#[test]
fn sentiment_numeric_and_vector_layers_round_trip() {
    let mut doc = document();
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Opinions {
            tokens: vec![IxRange::new(0, 1), IxRange::new(3, 4)],
            polarity: vec![0.8, -0.4],
            intensity: vec![1.0, 2.0],
            negated: vec![false, true],
            tokens_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Aspects {
            tokens: vec![Some(IxRange::new(0, 1)), None],
            sentence: vec![Some(Ix::new(0)), Some(Ix::new(1))],
            category: vec![Some(Tag(0)), None],
            categories: tagset("aspect-cats", &["food", "service"]),
            positive: vec![0.5, 0.1],
            negative: vec![0.2, 0.6],
            neutral: vec![0.3, 0.3],
            opinions: vec![Ix::new(0)],
            opinion_end: vec![1, 1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            opinions_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Pragmatics {
            score: vec![0.7, 0.1],
            evidence: vec![Some(Incongruity { positive: IxRange::new(0, 1), negative: IxRange::new(1, 2) }), None],
            tokens_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Incivility {
            target: vec![IncivilityTarget::Sentence(Ix::new(0))],
            insult: vec![0.9],
            profanity: vec![0.0],
            threat: vec![0.1],
            identity_attack: vec![0.0],
            intolerance: vec![0.2],
            severe_toxicity: vec![0.3],
            evidence_label: vec![IncivilityLabel::Insult],
            evidence_tokens: vec![IxRange::new(0, 1)],
            evidence_end: vec![1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            chunks_key: Key::DEFAULT,
            structure_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Numbers {
            spans: spans(&[(0, 5)]),
            numeral_end: vec![5],
            negative: vec![false],
            coefficient: strs(&["42"]),
            scale: vec![0],
            multiplier: vec![Multiplier::One],
            marker: vec![ScaleMarker::None],
            notation: vec![NumberNotation::LatinUs],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Money {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["500"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Ordinals { spans: spans(&[(0, 5)]), base: vec![OrdinalBase::End], offset: vec![-2] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Quantities {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["50"]),
            scale: vec![0],
            unit: strs(&["%"]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Times {
            spans: spans(&[(13, 16)]),
            kind: vec![TimeKind::Date],
            grain: vec![TimeGrain::Day],
            modifier: vec![Modifier { relation: Relation::After, portion: Portion::Start }],
            start: vec![Some(1_800_000_000_000)],
            end: vec![Some(1_800_086_400_000)],
            duration_ms: vec![None],
            ambiguous: vec![true],
            timex: strs(&["2027-01-15"]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Pii {
            spans: spans(&[(0, 5)]),
            kind: vec![PiiKind::Email],
            confidence: vec![PiiConfidence::Checksum],
            normalized: strs(&["a@b.c"]),
        },
        by.clone(),
    )
    .unwrap();
    let mut regions = Regions::new();
    regions.push(turbo_analysis_layers::RegionVote::new(RegionCode::parse("US").unwrap(), 0.75).unwrap());
    regions.push(turbo_analysis_layers::RegionVote::new(RegionCode::parse("IE").unwrap(), 0.25).unwrap());
    doc.add(&Key::DEFAULT, regions, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Matches {
            spans: spans(&[(0, 11)]),
            tokens: vec![IxRange::new(0, 2)],
            label: vec![Tag(0)],
            labels: tagset("match-labels", &["GREET"]),
            rule: vec![7],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Levels { spans: spans(&[(0, 21)]), unit: vec![Unit::Document], parent: vec![None] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Containment {
            spans: spans(&[(0, 5)]),
            ancestor_id: strs(&["ie", "europe"]),
            ancestor_name: strs(&["Ireland", "Europe"]),
            ancestor_kind: strs(&["country", "continent"]),
            chain_end: vec![2],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Phrases {
            spans: spans(&[(0, 11), (0, 5)]),
            tokens: vec![IxRange::new(0, 2), IxRange::new(0, 1)],
            kind: vec![Tag(0), Tag(0)],
            kinds: tagset("phrase-kinds", &["NP"]),
            head: vec![Ix::new(1), Ix::new(0)],
            parent: vec![None, Some(Ix::new(0))],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            3,
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            vec![Ix::new(1)],
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Confidence::<Tokens>::new(vec![0.5; 6]), by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Utf16Spans::<Tokens>::new(
            vec![0, 6, 11, 13, 17, 20],
            vec![5, 11, 12, 16, 20, 21],
            vec![false, true, false, false, false, false],
        ),
        by,
    )
    .unwrap();

    let mut opinion_cats = Categories::<Opinions>::new(tagset("cats", &["a", "b"]));
    opinion_cats.push(Tag(1), &[0.1, 0.9]);
    opinion_cats.push(Tag(0), &[0.8, 0.2]);
    doc.add(&Key::DEFAULT, opinion_cats, Provenance::by("test")).unwrap();
    let mut aspect_cats = Categories::<Aspects>::new(tagset("cats", &["a", "b"]));
    aspect_cats.push(Tag(0), &[0.6, 0.4]);
    aspect_cats.push(Tag(1), &[0.2, 0.8]);
    doc.add(&Key::DEFAULT, aspect_cats, Provenance::by("test")).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.get::<Categories<Opinions>>().unwrap().best_category(0), "b");
    assert_eq!(back.get::<Categories<Aspects>>().unwrap().best_category(1), "b");
    assert_eq!(back.get::<Opinions>().unwrap().get(Ix::new(1)).effective_polarity(), 0.8);
    assert_eq!(back.get::<Aspects>().unwrap().get(Ix::new(0)).opinions, [Ix::new(0)]);
    assert_eq!(back.get::<Pragmatics>().unwrap().score, [0.7, 0.1]);
    assert_eq!(back.get::<Incivility>().unwrap().target[0], IncivilityTarget::Sentence(Ix::new(0)));
    assert_eq!(back.get::<Numbers>().unwrap().coefficient.get(0), "42");
    assert_eq!(back.get::<Money>().unwrap().currency[0].as_str(), "USD");
    assert_eq!(back.get::<Ordinals>().unwrap().offset, [-2]);
    assert_eq!(back.get::<Quantities>().unwrap().unit.get(0), "%");
    assert_eq!(back.get::<Times>().unwrap().timex.get(0), "2027-01-15");
    assert_eq!(back.get::<Pii>().unwrap().kind, [PiiKind::Email]);
    assert_eq!(back.get::<Regions>().unwrap().share, [0.75, 0.25]);
    assert_eq!(back.get::<Matches>().unwrap().rule, [7]);
    assert_eq!(back.get::<Levels>().unwrap().unit, [Unit::Document]);
    assert_eq!(back.get::<Containment>().unwrap().ancestor_name.get(1), "Europe");
    assert_eq!(back.get::<Phrases>().unwrap().parent[1], Some(Ix::new(0)));
    let vectors = back.get::<Vectors<Sentences>>().unwrap();
    assert_eq!(vectors.row(Ix::new(0)), [1.0, 2.0, 3.0]);
    assert!(vectors.is_empty(Ix::new(1)));
    assert_eq!(back.get::<Confidence<Tokens>>().unwrap().value, [0.5; 6]);
    assert_eq!(back.get::<Utf16Spans<Tokens>>().unwrap().cut, [false, true, false, false, false, false]);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}
