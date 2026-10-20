//! Discovery mode, the default pattern libraries, and the two
//! corpus miners, over the worked example: "Acme Corp acquired
//! Bolt in 2024." with entities Acme Corp, Bolt and 2024.

use turbo_analysis::export::{export, import};
use turbo_analysis::{
    Analyzer, Annotator, Document, Ix, IxRange, Key, Layer, Provenance, Span, Spans, Tag, TagSet, ViewId,
};
use turbo_analysis_layers::{Dependencies, Entities, PosTags, Sentences, Tokens};
use turbo_relation::defaults::{english_defaults, english_semantic_defaults};
use turbo_relation::export::codecs;
use turbo_relation::{
    DistantSupervisionMiner, RelationAnnotator, RelationPathAnnotator, RelationPathExplorer, RelationPaths,
    RelationPattern, Relations, Seed,
};

fn spans<L>(ranges: &[(u32, u32)]) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for &(a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}

struct Fixture {
    doc: Document,
    sentences: Sentences,
    tokens: Tokens,
    deps: Dependencies,
    entities: Entities,
    pos: PosTags,
}

fn fixture() -> Fixture {
    let doc = Document::new("Acme Corp acquired Bolt in 2024.");
    let sentences = Sentences { spans: spans(&[(0, 32)]) };
    let tokens = Tokens {
        spans: spans(&[(0, 4), (5, 9), (10, 18), (19, 23), (24, 26), (27, 31), (31, 32)]),
        sentence: Some(vec![Ix::new(0); 7]),
    };
    let labels = ["compound", "nsubj", "root", "obj", "case", "obl", "punct"];
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
        relations: TagSet::new("dependency", &labels),
    };
    let entities = Entities {
        spans: spans(&[(0, 9), (19, 23), (27, 31)]),
        tokens: vec![IxRange::new(0, 2), IxRange::new(3, 4), IxRange::new(5, 6)],
        kind: vec![Tag(0), Tag(0), Tag(1)],
        kinds: TagSet::new("entity", &["organization", "date"]),
        score: vec![1.0; 3],
    };
    let pos_tags = ["PROPN", "PROPN", "VERB", "PROPN", "ADP", "NUM", "PUNCT"];
    let pos = PosTags {
        tags: pos_tags
            .iter()
            .map(|t| Tag(["PROPN", "VERB", "ADP", "NUM", "PUNCT"].iter().position(|x| x == t).unwrap() as u16))
            .collect(),
        tagset: TagSet::new("upos", &["PROPN", "VERB", "ADP", "NUM", "PUNCT"]),
        prob: None,
    };
    Fixture { doc, sentences, tokens, deps, entities, pos }
}

fn discover(f: &Fixture) -> RelationPaths {
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    RelationPathAnnotator::default()
        .annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &f.entities, Some(&f.pos), None))
        .unwrap()
        .0
}

#[test]
fn discovery_emits_the_worked_example_paths() {
    let f = fixture();
    let paths = discover(&f);
    let rows: Vec<(usize, usize, String, u32)> = (0..paths.len() as u32)
        .map(|i| {
            let p = paths.path(Ix::new(i));
            (p.subject.get(), p.object.get(), p.signature(), p.pivot)
        })
        .collect();
    assert_eq!(
        rows,
        [
            (0, 1, "<nsubj >obj".into(), 2),
            (0, 2, "<nsubj >obl".into(), 2),
            (1, 0, "<obj >nsubj".into(), 2),
            (1, 2, "<obj >obl".into(), 2),
            (2, 0, "<obl >nsubj".into(), 2),
            (2, 1, "<obl >obj".into(), 2),
        ]
    );
    // Every emitted signature parses as a pattern unchanged, and
    // the token walk reads as language.
    for i in 0..paths.len() as u32 {
        assert!(RelationPattern::new("found", paths.signature(Ix::new(i))).is_ok());
    }
    let first = paths.path(Ix::new(0));
    assert_eq!(first.tokens, [1, 2, 3]);
    let forms: Vec<&str> =
        first.tokens.iter().map(|&t| ["Acme", "Corp", "acquired", "Bolt", "in", "2024", "."][t as usize]).collect();
    assert_eq!(forms, ["Corp", "acquired", "Bolt"]);
    // A path cap below the walk length emits nothing here.
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    let mut capped = RelationPathAnnotator::default();
    capped.max_path_len = Some(1);
    let capped =
        capped.annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &f.entities, None, None)).unwrap().0;
    assert_eq!(capped.len(), 0);
}

#[test]
fn english_defaults_extract_subject_object() {
    let f = fixture();
    let view = f.doc.get::<turbo_analysis::View>().unwrap();
    let (relations, _) = RelationAnnotator::new(english_defaults())
        .unwrap()
        .annotate((view, &f.sentences, &f.tokens, &f.tokens, &f.deps, &f.entities, None, None))
        .unwrap();
    assert_eq!(relations.spans.len(), 1);
    let (s, o, kind, _) = relations.mention(Ix::new(0));
    assert_eq!((s.get(), o.get(), kind), (0, 1, "subject-object"));
    assert_eq!(english_defaults().len(), 6);
    assert_eq!(english_semantic_defaults().len(), 8);
    for p in english_semantic_defaults() {
        assert!(p.pivot_lemma_constraint().is_some() && p.subject_kind_constraint().is_some());
    }
}

#[test]
fn explorer_ranks_signatures_and_exports_patterns() {
    let f = fixture();
    let paths = discover(&f);
    let forms = ["Acme", "Corp", "acquired", "Bolt", "in", "2024", "."];
    let pos: Vec<Option<&str>> =
        ["PROPN", "PROPN", "VERB", "PROPN", "ADP", "NUM", "PUNCT"].iter().map(|s| Some(*s)).collect();
    let mut explorer = RelationPathExplorer::new(2);
    explorer.observe(&paths, &f.entities, &forms, Some(&pos), None);
    assert!(explorer.rows().is_empty(), "below the support floor at one sighting");
    explorer.observe(&paths, &f.entities, &forms, Some(&pos), None);
    let rows = explorer.rows();
    let row = rows.iter().find(|r| r.key.signature == "<nsubj >obj" && r.key.subject_kind == "organization").unwrap();
    assert_eq!(row.count, 2);
    assert_eq!(row.key.pivot_pos.as_deref(), Some("VERB"));
    assert_eq!(row.examples, vec!["Corp acquired Bolt".to_string()], "examples deduplicated, capped");
    let pattern = row.to_pattern("acquisition").unwrap();
    assert_eq!(pattern.path(), "<nsubj >obj");
    assert_eq!(pattern.pivot_pos_constraint(), Some("VERB"));
}

#[test]
fn distant_supervision_ranks_seed_signatures() {
    let f = fixture();
    let paths = discover(&f);
    let mut miner = DistantSupervisionMiner::new(
        vec![Seed { subject: "acme corp".into(), relation_type: "acquisition".into(), object: "Bolt".into() }],
        1,
        0.5,
    )
    .unwrap();
    miner.observe(&paths, &f.entities, &["Acme Corp", "Bolt", "2024"]);
    let rules = miner.rules();
    assert_eq!(rules.len(), 1);
    assert_eq!((rules[0].support, rules[0].matches), (1, 1));
    assert_eq!(rules[0].precision, 1.0);
    assert_eq!(rules[0].signature, "<nsubj >obj");
    let pattern = rules[0].pattern().unwrap();
    assert_eq!(pattern.confidence_value(), 1.0);
    // No seeds, blank seeds and bad precision floors are refused.
    assert!(DistantSupervisionMiner::new(vec![], 1, 0.5).is_err());
    assert!(
        DistantSupervisionMiner::new(
            vec![Seed { subject: " ".into(), relation_type: "x".into(), object: "y".into() }],
            1,
            0.5
        )
        .is_err()
    );
}

#[test]
fn relation_paths_cross_the_graph_boundary() {
    let f = fixture();
    let mut doc = Document::new(f.doc.source());
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, f.sentences, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.tokens, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.deps, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, f.entities, by).unwrap();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(RelationPathAnnotator::<Tokens>::default())
        .add(RelationAnnotator::<Tokens>::new(english_defaults()).unwrap())
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs)[0].as_ref().unwrap();
    let registry = codecs();
    let graph = export(&docs[0], &registry).unwrap();
    let back = import(&graph, &registry).unwrap();
    assert_eq!(back.get::<RelationPaths>().unwrap().len(), 6);
    assert_eq!(back.get::<Relations>().unwrap().spans.len(), 1);
    assert_eq!(export(&back, &registry).unwrap().to_bytes(), graph.to_bytes());
}
