//! Relation matching over hand-built layers, mirroring the
//! reference `RelationAnnotatorTest` and
//! `RelationEntityOverlapTest` fixtures, plus this crate's
//! extension constraints (entity kinds, pivot POS and lemma, path
//! cap, sentence containment, confidence).

use std::sync::Arc;

use turbo_analysis::{
    Analyzer, Annotator, Document, Edges, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Dependencies, Entities, Lemmas, PosTags, Sentences, Strs, Tokens};
use turbo_relation::{RelationAnnotator, RelationPattern, Relations};

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}

/// The acquisition fixture: "Acme Corp acquired Bolt in 2024."
struct Fixture {
    doc: Document,
    sentences: Sentences,
    tokens: Tokens,
    deps: Dependencies,
    entities: Entities,
    pos: PosTags,
    lemmas: Lemmas,
}

fn fixture() -> Fixture {
    let doc = Document::new("Acme Corp acquired Bolt in 2024.");
    let sentences = Sentences { spans: spans(&[(0, 32)]) };
    let tokens = Tokens {
        spans: spans(&[(0, 4), (5, 9), (10, 18), (19, 23), (24, 26), (27, 31), (31, 32)]),
        sentence: Some(vec![Ix::new(0); 7]),
    };
    let labels = ["compound", "nsubj", "root", "obj", "case", "obl", "punct"];
    let relations = TagSet::new("dependency", &labels);
    let tag = |l: &str| Tag(labels.iter().position(|x| *x == l).unwrap() as u16);
    let deps = Dependencies {
        head: vec![
            Some(Ix::new(1)),
            Some(Ix::new(2)),
            None,
            Some(Ix::new(2)),
            Some(Ix::new(5)),
            Some(Ix::new(2)),
            Some(Ix::new(2)),
        ],
        relation: vec![tag("compound"), tag("nsubj"), tag("root"), tag("obj"), tag("case"), tag("obl"), tag("punct")],
        relations,
    };
    let kinds = TagSet::new("entity", &["organization", "date", "person"]);
    let entities = Entities {
        spans: spans(&[(0, 9), (19, 23), (27, 31)]),
        tokens: vec![IxRange::new(0, 2), IxRange::new(3, 4), IxRange::new(5, 6)],
        kind: vec![Tag(0), Tag(0), Tag(1)],
        kinds,
        score: vec![1.0; 3],
    };
    let pos = PosTags {
        tags: ["PROPN", "PROPN", "VERB", "PROPN", "ADP", "NUM", "PUNCT"]
            .iter()
            .map(|t| Tag(["PROPN", "VERB", "ADP", "NUM", "PUNCT"].iter().position(|x| x == t).unwrap() as u16))
            .collect(),
        tagset: TagSet::new("upos", &["PROPN", "VERB", "ADP", "NUM", "PUNCT"]),
        prob: None,
    };
    let mut lemma = Strs::default();
    for l in ["Acme", "Corp", "acquire", "Bolt", "in", "2024", "."] {
        lemma.push(l);
    }
    Fixture { doc, sentences, tokens, deps, entities, pos, lemmas: Lemmas { lemma } }
}

fn extract(f: &Fixture, annotator: &RelationAnnotator) -> (Relations, Edges<Entities, Entities>) {
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    annotator
        .annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &f.entities, Some(&f.pos), Some(&f.lemmas)))
        .unwrap()
}

fn pattern(trigger: Option<&str>) -> RelationPattern {
    match trigger {
        Some(t) => RelationPattern::with_trigger("acquisition", "<nsubj >obj", t).unwrap(),
        None => RelationPattern::new("acquisition", "<nsubj >obj").unwrap(),
    }
}

#[test]
fn matches_subject_verb_object_path() {
    let f = fixture();
    let annotator = RelationAnnotator::new(vec![pattern(Some("acquired"))]).unwrap();
    let (relations, edges) = extract(&f, &annotator);
    assert_eq!(relations.spans.len(), 1);
    let (s, o, kind, score) = relations.mention(Ix::new(0));
    assert_eq!((s.get(), o.get(), kind, score), (0, 1, "acquisition", 1.0));
    let span = relations.spans.get(Ix::new(0));
    assert_eq!(&f.doc.source()[span.range()], "Acme Corp acquired Bolt");
    assert_eq!(edges.len(), 1);
    let edge = edges.get(Ix::new(0));
    assert_eq!((edge.from, edge.to), (s, o));
}

#[test]
fn trigger_direction_and_path_length_matter() {
    let f = fixture();
    let miss = RelationAnnotator::new(vec![pattern(Some("bought"))]).unwrap();
    assert_eq!(extract(&f, &miss).0.spans.len(), 0);
    // The reversed shape binds the swapped pair.
    let reversed = RelationAnnotator::new(vec![RelationPattern::new("acquired-by", "<obj >nsubj").unwrap()]).unwrap();
    let (relations, _) = extract(&f, &reversed);
    assert_eq!(relations.spans.len(), 1);
    let (s, o, _, _) = relations.mention(Ix::new(0));
    assert_eq!((s.get(), o.get()), (1, 0));
    // A longer pattern over the same pair does not match.
    let longer = RelationAnnotator::new(vec![RelationPattern::new("x", "<nsubj >obj >case").unwrap()]).unwrap();
    assert_eq!(extract(&f, &longer).0.spans.len(), 0);
    // Nor does a path cap below the actual path length.
    let mut capped = RelationAnnotator::new(vec![pattern(None)]).unwrap();
    capped.max_path_len = Some(1);
    assert_eq!(extract(&f, &capped).0.spans.len(), 0);
    capped.max_path_len = Some(2);
    assert_eq!(extract(&f, &capped).0.spans.len(), 1);
}

#[test]
fn several_and_duplicate_patterns() {
    let f = fixture();
    let annotator = RelationAnnotator::new(vec![pattern(None), pattern(None)]).unwrap();
    let (relations, edges) = extract(&f, &annotator);
    // Every registration emits a mention, as in the reference;
    // the edge layer holds the triple once.
    assert_eq!(relations.spans.len(), 2);
    assert_eq!(edges.len(), 1);
    // Distinct types over one pair are distinct edges.
    let two =
        RelationAnnotator::new(vec![pattern(None), RelationPattern::new("deal", "<nsubj >obj").unwrap()]).unwrap();
    let (relations, edges) = extract(&f, &two);
    assert_eq!((relations.spans.len(), edges.len()), (2, 2));
}

#[test]
fn extension_constraints_filter_matches() {
    let f = fixture();
    let by_kind = RelationAnnotator::new(vec![pattern(None).subject_kind("organization").unwrap()]).unwrap();
    assert_eq!(extract(&f, &by_kind).0.spans.len(), 1);
    let wrong_kind = RelationAnnotator::new(vec![pattern(None).subject_kind("person").unwrap()]).unwrap();
    assert_eq!(extract(&f, &wrong_kind).0.spans.len(), 0);
    let by_pos = RelationAnnotator::new(vec![pattern(None).pivot_pos("VERB").unwrap()]).unwrap();
    assert_eq!(extract(&f, &by_pos).0.spans.len(), 1);
    let wrong_pos = RelationAnnotator::new(vec![pattern(None).pivot_pos("NOUN").unwrap()]).unwrap();
    assert_eq!(extract(&f, &wrong_pos).0.spans.len(), 0);
    let by_lemma = RelationAnnotator::new(vec![pattern(None).pivot_lemma("acquire").unwrap()]).unwrap();
    assert_eq!(extract(&f, &by_lemma).0.spans.len(), 1);
    let scored = RelationAnnotator::new(vec![pattern(None).confidence(0.4).unwrap()]).unwrap();
    assert_eq!(extract(&f, &scored).0.score, vec![0.4]);
}

#[test]
fn empty_and_invalid_inputs() {
    let f = fixture();
    assert!(RelationAnnotator::<Tokens>::new(vec![]).is_err());
    let annotator = RelationAnnotator::new(vec![pattern(None)]).unwrap();
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    // No entities: empty outputs, not an error.
    let empty = Entities {
        spans: spans(&[]),
        tokens: vec![],
        kind: vec![],
        kinds: Arc::clone(&f.entities.kinds),
        score: vec![],
    };
    let (relations, edges) =
        annotator.annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &empty, None, None)).unwrap();
    assert!(relations.spans.is_empty() && edges.len() == 0);
    // A cyclic dependency layer is refused.
    let mut cyclic = f.deps.clone();
    cyclic.head[2] = Some(Ix::new(3));
    assert!(annotator.annotate((view, &f.sentences, &f.tokens, &f.tokens, &cyclic, &f.entities, None, None)).is_err());
    // Misaligned heads are refused.
    let mut short = f.deps.clone();
    short.head.pop();
    short.relation.pop();
    assert!(annotator.annotate((view, &f.sentences, &f.tokens, &f.tokens, &short, &f.entities, None, None)).is_err());
    // A pivot lemma pattern without lemmas is refused, not ignored.
    let needs_lemma = RelationAnnotator::new(vec![pattern(None).pivot_lemma("acquire").unwrap()]).unwrap();
    assert!(
        needs_lemma.annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &f.entities, None, None)).is_err()
    );
}

#[test]
fn pairs_do_not_cross_sentences() {
    // "Bolt fell. Acme acquired it." with an artificial arc from
    // the second sentence's verb into the first: connected heads
    // in different sentences still extract nothing.
    let text = "Bolt fell. Acme acquired it.";
    let doc = Document::new(text);
    let sentences = Sentences { spans: spans(&[(0, 10), (11, 29)]) };
    let tokens = Tokens {
        spans: spans(&[(0, 4), (5, 9), (11, 15), (16, 24), (25, 27)]),
        sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1)]),
    };
    let relations = TagSet::new("dependency", &["nsubj", "root", "obj"]);
    let deps: Dependencies = Dependencies {
        // Bolt(0) object of acquired(3) across the boundary;
        // Acme(2) its subject; fell(1) attaches to acquired too.
        head: vec![Some(Ix::new(3)), Some(Ix::new(3)), Some(Ix::new(3)), None, Some(Ix::new(3))],
        relation: vec![Tag(2), Tag(0), Tag(0), Tag(1), Tag(2)],
        relations,
    };
    let kinds = TagSet::new("entity", &["organization"]);
    let entities = Entities {
        spans: spans(&[(0, 4), (11, 15)]),
        tokens: vec![IxRange::new(0, 1), IxRange::new(2, 3)],
        kind: vec![Tag(0), Tag(0)],
        kinds,
        score: vec![1.0; 2],
    };
    let annotator = RelationAnnotator::new(vec![RelationPattern::new("acquisition", "<nsubj >obj").unwrap()]).unwrap();
    let view = doc.get::<turbo_analysis::View>().unwrap();
    let (out, _) = annotator.annotate((view, &sentences, &tokens, &tokens, &deps, &entities, None, None)).unwrap();
    assert_eq!(out.spans.len(), 0, "a cross-sentence pair matched");
}

#[test]
fn pipeline_writes_both_layers() {
    let f = fixture();
    let mut doc = Document::new(f.doc.source());
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, f.sentences, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.tokens, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.deps, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.entities, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.pos, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.lemmas, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<turbo_analysis_layers::PosTags>::DEFAULT)
        .given(Key::<Lemmas>::DEFAULT)
        .add(RelationAnnotator::<Tokens>::new(vec![pattern(Some("acquired"))]).unwrap())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    assert_eq!(docs[0].get::<Relations>().unwrap().spans.len(), 1);
    assert_eq!(docs[0].get::<Edges<Entities, Entities>>().unwrap().len(), 1);
    let first = analyzer.run(&mut docs)[0].clone();
    assert!(first.is_err(), "re-running over written relations should refuse or skip, got {first:?}");
}

#[test]
fn pivot_form_uses_the_project_case_mapping() {
    // U+0130 lowers to a plain i under the project's simple
    // per-character mapping, so trigger "istanbul" matches the
    // dotted-capital pivot and the JDK string-lowercased spelling
    // (i + combining dot) does not.
    let text = "\u{130}stanbul, home of Bolt.";
    let doc = Document::new(text);
    let at = |s: &str| {
        let a = text.find(s).unwrap();
        Span::new(a as u32, (a + s.len()) as u32)
    };
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    for s in ["\u{130}stanbul", ",", "home", "of", "Bolt", "."] {
        tok_spans.push(at(s));
    }
    let sentences = Sentences { spans: spans(&[(0, text.len() as u32)]) };
    let tokens = Tokens { spans: tok_spans, sentence: Some(vec![Ix::new(0); 6]) };
    let labels = ["root", "punct", "appos", "case", "nmod"];
    let tag = |l: &str| Tag(labels.iter().position(|x| *x == l).unwrap() as u16);
    let deps: Dependencies = Dependencies {
        head: vec![None, Some(Ix::new(2)), Some(Ix::new(0)), Some(Ix::new(4)), Some(Ix::new(2)), Some(Ix::new(0))],
        relation: vec![tag("root"), tag("punct"), tag("appos"), tag("case"), tag("nmod"), tag("punct")],
        relations: TagSet::new("dependency", &labels),
    };
    let entities = Entities {
        spans: {
            let mut s = Spans::new(ViewId::SOURCE);
            s.push(at("\u{130}stanbul"));
            s.push(at("Bolt"));
            s
        },
        tokens: vec![IxRange::new(0, 1), IxRange::new(4, 5)],
        kind: vec![Tag(0), Tag(1)],
        kinds: TagSet::new("entity", &["location", "organization"]),
        score: vec![1.0; 2],
    };
    let view = doc.get::<turbo_analysis::View>().unwrap();
    let run = |trigger: &str| {
        let annotator =
            RelationAnnotator::new(vec![RelationPattern::with_trigger("located_in", ">appos >nmod", trigger).unwrap()])
                .unwrap();
        annotator.annotate((view, &sentences, &tokens, &tokens, &deps, &entities, None, None)).unwrap().0.spans.len()
    };
    assert_eq!(run("istanbul"), 1);
    assert_eq!(run("i\u{307}stanbul"), 0);
}

#[test]
fn entity_overlap_edge_cases() {
    let f = fixture();
    let annotator = RelationAnnotator::new(vec![pattern(None)]).unwrap();
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    let run_with = |entities: &Entities| {
        annotator
            .annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, entities, None, None))
            .unwrap()
            .0
            .spans
            .len()
    };
    let copy = |spans: Spans<Entities>| Entities {
        spans,
        tokens: f.entities.tokens.clone(),
        kind: f.entities.kind.clone(),
        kinds: std::sync::Arc::clone(&f.entities.kinds),
        score: f.entities.score.clone(),
    };
    // An empty entity span shares no character with any token
    // and selects no head.
    let mut empty = copy(f.entities.spans.clone());
    empty.spans = spans(&[(5, 5), (19, 23), (27, 31)]);
    assert_eq!(run_with(&empty), 0);
    // A partially overlapping subject span ("me Cor" over the
    // first two tokens) still finds the outward head token.
    let mut partial = copy(f.entities.spans.clone());
    partial.spans = spans(&[(2, 8), (19, 23), (27, 31)]);
    assert_eq!(run_with(&partial), 1);
    // A discontinuous subject whose fragments are the compound
    // tokens behaves like its envelope for head selection.
    let mut discontinuous = copy(f.entities.spans.clone());
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.try_push_fragments(&[Span::new(0, 4), Span::new(5, 9)]).unwrap();
    spans.push(Span::new(19, 23));
    spans.push(Span::new(27, 31));
    discontinuous.spans = spans;
    assert_eq!(run_with(&discontinuous), 1);
}
