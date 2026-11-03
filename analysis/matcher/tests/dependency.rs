//! Strict schema and bounded graph checks. The external oracle uses original UD text.
use turbo_analysis::{Ix, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::{Dependencies, Tokens};
use turbo_matcher::{
    DependencyError, DependencyMatcher, Input, MatchError, MatchLimits, spacy::read_dependency_pattern,
};

fn pattern(op: &str) -> String {
    format!(
        r#"[{{"RIGHT_ID":"a","RIGHT_ATTRS":{{}}}},{{"LEFT_ID":"a","REL_OP":"{op}","RIGHT_ID":"b","RIGHT_ATTRS":{{}}}}]"#
    )
}

#[test]
fn schema_and_add_are_atomic() {
    let mut matcher = DependencyMatcher::new();
    let nodes = read_dependency_pattern(&pattern(">")).unwrap();
    let id = matcher.add("valid", &nodes).unwrap();
    let mut broken = nodes.clone();
    broken[1].relation.as_mut().unwrap().0 = 1;
    assert!(matcher.add("bad", &broken).is_err());
    assert!(matcher.label("bad").is_none());
    assert_eq!(matcher.add("valid", &nodes).unwrap().0, id.0 + 1);
    assert_eq!(matcher.len(), 1);
    broken[1].token = turbo_matcher::Tok::any().star();
    assert!(matcher.add("bad", &broken).is_err());
    assert!(matcher.add("bad", &[]).is_err());
    assert!(matcher.add("bad", &vec![nodes[0].clone(); 65]).is_err());
    for raw in [
        "[]",
        "[{}]",
        r#"[{"RIGHT_ID":"a","RIGHT_ATTRS":{},"REL_OP":">"}]"#,
        r#"[{"RIGHT_ID":"a","RIGHT_ATTRS":{"TEXT":{"REGEX":"x"}}}]"#,
        r#"[{"RIGHT_ID":"a","RIGHT_ATTRS":{"OP":"+"}}]"#,
    ] {
        assert!(read_dependency_pattern(raw).is_err(), "{raw}");
    }
    for raw in [
        pattern("unknown"),
        pattern(">").replace("\"b\"", "\"a\""),
        pattern(">").replace("\"LEFT_ID\":\"a\"", "\"LEFT_ID\":\"missing\""),
    ] {
        assert!(read_dependency_pattern(&raw).is_err(), "{raw}");
    }
}

// Malformed-input tests use actual text with deliberately invalid annotations.
fn document() -> (&'static str, Tokens, Dependencies) {
    let text = "The price was £5.";
    let mut spans = Spans::new(ViewId::SOURCE);
    for (a, b) in [(0, 3), (4, 9), (10, 13), (14, 17), (17, 18)] {
        spans.push(Span::new(a, b));
    }
    let relations = TagSet::new("UD", &["det", "nsubj", "root", "obj", "punct"]);
    let deps = Dependencies {
        head: vec![Some(Ix::new(1)), Some(Ix::new(2)), None, Some(Ix::new(2)), Some(Ix::new(2))],
        relation: (0..5).map(Tag).collect(),
        relations,
    };
    (text, Tokens { spans, sentence: None }, deps)
}

#[test]
fn invalid_forests_and_alignment_refuse_without_panics() {
    let (text, mut tokens, deps) = document();
    let input = Input::new(text, &tokens).unwrap();
    for kind in 0..5 {
        let mut bad = deps.clone();
        match kind {
            0 => {
                bad.head.pop();
            }
            1 => bad.head[0] = Some(Ix::new(99)),
            2 => bad.head[0] = Some(Ix::new(0)),
            3 => bad.head[1] = Some(Ix::new(0)),
            _ => bad.relation[0] = Tag(99),
        }
        assert!(input.with_dependencies(&bad).is_err());
    }
    tokens.sentence = Some(vec![Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1), Ix::new(1)]);
    assert!(Input::new(text, &tokens).unwrap().with_dependencies(&deps).is_err());
}

#[test]
fn caps_missing_layer_removal_and_dependency_attributes() {
    let (text, tokens, deps) = document();
    let input = Input::new(text, &tokens).unwrap();
    let mut matcher = DependencyMatcher::new();
    matcher.add("pairs", &read_dependency_pattern(&pattern(">>")).unwrap()).unwrap();
    assert!(matches!(matcher.find(&input), Err(DependencyError::Input(MatchError::Missing(_)))));
    let input = input.with_dependencies(&deps).unwrap();
    assert!(matches!(
        matcher.find_with_limits(&input, MatchLimits { visits: 1, matches: 100 }),
        Err(DependencyError::Limit("visits"))
    ));
    assert!(matches!(
        matcher.find_with_limits(&input, MatchLimits { visits: 1000, matches: 0 }),
        Err(DependencyError::Limit("matches"))
    ));
    assert!(!matcher.find(&input).unwrap().is_empty());
    matcher.remove("pairs").unwrap();
    assert!(matcher.find(&input).unwrap().is_empty());
    let mut token_matcher = turbo_matcher::Matcher::new();
    token_matcher.add("root", &turbo_matcher::spacy::read_pattern(r#"[{"DEP":"root"}]"#).unwrap()).unwrap();
    let hits = token_matcher.find(&input).unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].tokens.start.get(), 2);
}
