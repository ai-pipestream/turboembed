//! Import checks and an optional local oracle over original EWT text.
mod spacy_oracle;
use turbo_analysis::{Span, Spans, TagSet, ViewId};
use turbo_analysis_layers::{MorphFeatures, PosTags, Strs, Tokens};
use turbo_matcher::{
    Input, Matcher,
    spacy::{ImportError, RulePattern, read_pattern, read_rules},
};

fn strs(values: &[&str]) -> Strs {
    let mut s = Strs::default();
    for value in values {
        s.push(value);
    }
    s
}

#[test]
fn published_rules_and_operators_agree_with_spacy_on_ewt() {
    let Some(cases) = spacy_oracle::load() else { return };
    let mut matcher = Matcher::new();
    for file in [include_str!("data/drugs_patterns.jsonl"), include_str!("data/operators.jsonl")] {
        for rule in read_rules(file).unwrap() {
            rule.add_to(&mut matcher).unwrap();
        }
    }
    assert_eq!(read_rules(include_str!("data/drugs_patterns.jsonl")).unwrap().len(), 118);
    assert_eq!(cases.len(), 31);
    assert_eq!(cases.iter().map(|c| c.spans.len()).sum::<usize>(), 485);
    assert_eq!(cases.iter().map(|c| c.hits.len()).sum::<usize>(), 2580);
    for case in &cases {
        let mut spans = Spans::new(ViewId::SOURCE);
        for &(a, b) in &case.spans {
            spans.push(Span::new(a, b));
        }
        let tokens = Tokens { spans, sentence: None };
        let pos_names: Vec<&str> = case.pos.iter().map(String::as_str).collect();
        let tagset = TagSet::new("upos", &pos_names);
        let pos = PosTags { tags: case.pos.iter().map(|s| tagset.tag(s).unwrap()).collect(), tagset, prob: None };
        let column = |v: &[String]| strs(&v.iter().map(String::as_str).collect::<Vec<_>>());
        let tags = column(&case.tags);
        let lemmas = column(&case.lemmas);
        let morph = MorphFeatures::new(column(&case.morph));
        let input = Input::new(&case.text, &tokens)
            .unwrap()
            .with_pos(&pos)
            .unwrap()
            .with_tags(&tags)
            .unwrap()
            .with_lemmas(&lemmas)
            .unwrap()
            .with_morph(&morph)
            .unwrap();
        let mut hits: Vec<_> = matcher
            .find(&input)
            .unwrap()
            .iter()
            .map(|h| {
                let span = input.span(h.tokens);
                assert_eq!(span.start, case.spans[h.tokens.start.get()].0);
                assert_eq!(span.end, case.spans[h.tokens.end.get() - 1].1);
                (matcher.label_name(h.label).to_owned(), h.tokens.start.get(), h.tokens.end.get())
            })
            .collect();
        hits.sort_unstable();
        assert_eq!(hits, case.hits, "{}", case.text);
    }
}

#[test]
fn file_shapes_metadata_escapes_and_rejections() {
    let rules = read_rules(
        r#"[{"label":"ORG","id":"x","pattern":"New York"},{"label":"face","pattern":[{"TEXT":"\ud83d\ude00"}]}]"#,
    )
    .unwrap();
    assert_eq!(rules[0].id.as_deref(), Some("x"));
    assert!(matches!(&rules[0].pattern, RulePattern::Phrase(s) if s == "New York"));
    assert!(rules[0].add_to(&mut Matcher::new()).is_err());
    rules[1].add_to(&mut Matcher::new()).unwrap();
    for source in [r#"[{"TEXT":{"REGEX":"cat"}}]"#, r#"[{"TEXT":{"FUZZY1":{"REGEX":["cat"]}}}]"#] {
        assert_eq!(read_pattern(source).unwrap_err(), ImportError::Unsupported { operator: "REGEX".into() });
    }
    for source in [
        "[",
        "[{}] garbage",
        "[{},]",
        r#"[{"LENGTH":01}]"#,
        r#"[{"TEXT":"\ud800"}]"#,
        r#"[{"TEXT":"cat","TEXT":"dog"}]"#,
        r#"[{"OP":"+","op":"?"}]"#,
        r#"[{"LENGTH":-1}]"#,
        r#"[{"LENGTH":1e2}]"#,
        r#"[{"IS_ALPHA":"true"}]"#,
        r#"[{"TEXT":{"IN":[1]}}]"#,
        r#"[{"OP":"{3,2}"}]"#,
        r#"[{"OP":"{,}"}]"#,
        r#"[{"OP":"{999999999999999999999999}"}]"#,
        r#"[{"OP":"{1000000}"}]"#,
        "[]",
    ] {
        assert!(read_pattern(source).is_err(), "{source}");
    }
    let deep = format!("{}{}", "[".repeat(100), "]".repeat(100));
    assert!(read_pattern(&deep).is_err());
    assert_eq!(
        read_rules("{\"label\":\"x\",\"pattern\":[{}]}\n{}").unwrap_err(),
        ImportError::Schema { field: "label".into() }
    );
}

#[test]
fn fuzzy_unicode_distance_missing_layers_and_morph_order() {
    let text = "résumé resume resumé résumé";
    let mut spans = Spans::new(ViewId::SOURCE);
    let mut at = 0;
    for word in text.split(' ') {
        spans.push(Span::new(at, at + word.len() as u32));
        at += word.len() as u32 + 1;
    }
    let tokens = Tokens { spans, sentence: None };
    let input = Input::new(text, &tokens).unwrap();
    let mut m = Matcher::new();
    m.add("accent", &read_pattern(r#"[{"TEXT":{"FUZZY1":"résumé"}}]"#).unwrap()).unwrap();
    assert_eq!(m.find(&input).unwrap().iter().map(|h| h.tokens.start.get()).collect::<Vec<_>>(), [0, 2, 3]);
    let mut m = Matcher::new();
    m.add("noun", &read_pattern(r#"[{"MORPH":{"IS_SUPERSET":["Case=Acc"]}}]"#).unwrap()).unwrap();
    assert!(m.find(&input).is_err());
    let features = MorphFeatures::new(strs(&["Number=Sing|Case=Nom,Acc", "_", "Case=Nom", "Case=Acc|Number=Sing"]));
    let input = input.with_morph(&features).unwrap();
    assert_eq!(m.find(&input).unwrap().iter().map(|h| h.tokens.start.get()).collect::<Vec<_>>(), [0, 3]);
    let bad = MorphFeatures::new(strs(&["_"]));
    assert!(input.with_morph(&bad).is_err());
}

#[test]
fn morphology_runs_as_a_source_aligned_pipeline_step() {
    use turbo_analysis::{Analyzer, Annotator, Document, Key, Provenance};
    use turbo_matcher::{Attr, Field, Matches, Policy, SpanRuler};
    let text = "The engineers repaired systems.";
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 13), (14, 22), (23, 30), (30, 31)] {
        spans.push(Span::new(a, b));
    }
    let tokens = Tokens { spans, sentence: None };
    let feats = MorphFeatures::<Tokens>::new(strs(&[
        "Definite=Def|PronType=Art",
        "Number=Plur",
        "Tense=Past",
        "Number=Plur",
        "_",
    ]));
    let mut doc = Document::new(text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("annotation")).unwrap();
    doc.add(&Key::DEFAULT, feats, Provenance::by("annotation")).unwrap();
    let mut ruler = SpanRuler::new(Field::Attr(Attr::Text), Policy::Keep);
    ruler.add_pattern("plural", &read_pattern(r#"[{"MORPH":{"IS_SUPERSET":["Number=Plur"]}}]"#).unwrap()).unwrap();
    let hash = ruler.config_hash();
    let mut other = SpanRuler::new(Field::Attr(Attr::Text), Policy::Keep);
    other.add_pattern("plural", &read_pattern(r#"[{"MORPH":{"IS_SUPERSET":["Number=Sing"]}}]"#).unwrap()).unwrap();
    assert_ne!(hash, other.config_hash());
    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<MorphFeatures<Tokens>>::DEFAULT)
        .step(ruler.bound("rules"))
        .build()
        .unwrap();
    let results = analyzer.run(std::slice::from_mut(&mut doc));
    assert!(results.iter().all(Result::is_ok), "{results:?}");
    let matches = doc.get_key(&Key::<Matches>::named("rules")).unwrap();
    assert_eq!(matches.spans.len(), 2);
    assert_eq!(matches.spans.get(turbo_analysis::Ix::new(0)), Span::new(4, 13));
    assert_eq!(matches.spans.get(turbo_analysis::Ix::new(1)), Span::new(23, 30));
}
