//! Real spaCy 3.8.7 on original UD EWT gold annotations, including every operator.
use turbo_analysis::{Ix, IxRange, OffsetTable, Span, Spans, TagSet, ViewBuilder, ViewId};
use turbo_analysis_layers::{Dependencies, MorphFeatures, PosTags, Strs, Tokens};
use turbo_matcher::{
    DependencyMatcher, Input,
    spacy::{ImportError, read_dependency_pattern},
};
#[allow(dead_code)]
#[path = "../src/spacy/json.rs"]
mod json;
use json::Value;
fn array(v: &Value) -> &[Value] {
    let Value::Array(a) = v else { panic!("array") };
    a
}
fn field<'a>(v: &'a Value, k: &str) -> &'a Value {
    let Value::Object(o) = v else { panic!("object") };
    &o.iter().find(|(name, _)| name == k).unwrap().1
}
fn string(v: &Value) -> &str {
    let Value::String(s) = v else { panic!("string") };
    s
}
fn number(v: &Value) -> u32 {
    let Value::Number(n) = v else { panic!("number") };
    n.parse().unwrap()
}
fn strings(v: &Value) -> Vec<&str> {
    array(v).iter().map(string).collect()
}

#[test]
fn every_dependency_operator_matches_pinned_spacy_on_ewt() {
    let Some(path) = std::env::var_os("TURBO_SPACY_DEPENDENCY_ORACLE") else {
        assert!(
            !std::env::var("TURBO_TEST_REQUIRE_SPACY_ORACLE").is_ok_and(|v| v == "1"),
            "TURBO_SPACY_DEPENDENCY_ORACLE required; see tests/data/README.md"
        );
        eprintln!("skipping external dependency oracle: TURBO_SPACY_DEPENDENCY_ORACLE unset");
        return;
    };
    let raw = std::fs::read_to_string(path).unwrap();
    let root = json::parse(&raw).unwrap();
    assert_eq!(string(field(&root, "spacy")), "3.8.7");
    assert_eq!(
        string(field(&root, "corpus_sha256")),
        "3fb78e2b55482c8ee00caa653af436c88c9e5bc9a73ce26e4ae3ddc79d2b7e7b"
    );
    let mut matcher = DependencyMatcher::new();
    for rule in array(field(&root, "rules")) {
        matcher
            .add(string(field(rule, "label")), &read_dependency_pattern(string(field(rule, "pattern"))).unwrap())
            .unwrap();
    }
    assert_eq!(matcher.len(), 24);
    let cases = array(field(&root, "cases"));
    assert_eq!(cases.len(), 220);
    let (mut count, mut nodes, mut shifted) = (0, 0, 0);
    for case in cases {
        let text = string(field(case, "text"));
        let mut spans = Spans::new(ViewId::SOURCE);
        for pair in array(field(case, "spans")) {
            let pair = array(pair);
            spans.push(Span::new(number(&pair[0]), number(&pair[1])));
        }
        let tokens = Tokens { spans, sentence: None };
        let relation_names = strings(field(case, "deps"));
        let relations = TagSet::new("UD", &relation_names);
        let dependencies = Dependencies {
            head: array(field(case, "heads"))
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let head = number(v);
                    (head as usize != i).then_some(Ix::new(head))
                })
                .collect(),
            relation: relation_names.iter().map(|r| relations.tag(r).unwrap()).collect(),
            relations,
        };
        let pos_names = strings(field(case, "pos"));
        let tagset = TagSet::new("upos", &pos_names);
        let pos = PosTags { tags: pos_names.iter().map(|p| tagset.tag(p).unwrap()).collect(), tagset, prob: None };
        let mut morph = Strs::default();
        for v in strings(field(case, "morph")) {
            morph.push(v);
        }
        let morph = MorphFeatures::new(morph);
        let input = Input::new(text, &tokens)
            .unwrap()
            .with_dependencies(&dependencies)
            .unwrap()
            .with_pos(&pos)
            .unwrap()
            .with_morph(&morph)
            .unwrap();
        let matches = matcher.find(&input).unwrap();
        let mut actual: Vec<_> = matches.iter().map(|m| (matcher.label_name(m.label), m.tokens.clone())).collect();
        actual.sort();
        let expected: Vec<_> = array(field(case, "hits"))
            .iter()
            .map(|v| {
                let a = array(v);
                (string(&a[0]), array(&a[1]).iter().map(number).collect::<Vec<_>>())
            })
            .collect();
        assert_eq!(actual, expected, "{text}");
        count += matches.len();
        // Expand original spaces without changing token text; prove each matched
        // node retains exact source boundaries in both directions and offset units.
        let mut builder = ViewBuilder::new(text);
        for (at, c) in text.char_indices() {
            if c == ' ' {
                builder.keep(at);
                builder.replace(at + 1, "  ");
            }
        }
        let (child_text, alignment) = builder.finish();
        let mut child_spans = Spans::new(ViewId(1));
        for (_, span) in tokens.spans.iter() {
            child_spans.push(alignment.span_to_child(span));
        }
        let child_tokens = Tokens { spans: child_spans, sentence: None };
        let child = Input::new(&child_text, &child_tokens)
            .unwrap()
            .with_dependencies(&dependencies)
            .unwrap()
            .with_pos(&pos)
            .unwrap()
            .with_morph(&morph)
            .unwrap();
        assert_eq!(matcher.find(&child).unwrap(), matches);
        let offsets = OffsetTable::build(text);
        for m in &matches {
            for &i in &m.tokens {
                let span = input.span(IxRange::new(i, i + 1));
                let projected = child.span(IxRange::new(i, i + 1));
                assert_eq!(alignment.span_to_parent(projected), span);
                assert_eq!(text[span.range()], child_text[projected.range()]);
                assert_eq!(offsets.span_from_utf16(text, offsets.span_to_utf16_typed(text, span)).unwrap(), span);
                assert_eq!(
                    offsets.span_from_code_point(text, offsets.span_to_code_point_typed(text, span)).unwrap(),
                    span
                );
                nodes += 1;
                shifted += usize::from(projected != span);
            }
        }
    }
    assert_eq!(count, 136_932);
    assert!(shifted > 1000);
    println!(
        "dependency oracle: {} sentences, {count} matches, {nodes} node projections, {shifted} shifted",
        cases.len()
    );
}
