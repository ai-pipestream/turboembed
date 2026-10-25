//! `EntityConsolidator`: coreference chains, knowledge-base links, surface
//! and alias merges, head roles from dependencies, aspect sentiment,
//! resolved places and relation edges lifted to document entities, each
//! input optional but `Entities`; the layer through the analyzer and the
//! document graph bytes.

mod common;

use std::sync::Arc;

use common::{doc_with_names, spans_of};
use turbo_analysis::{
    Analyzer, Annotator, Document, DocumentGraph, Edge, Edges, Groups, Ix, IxRange, Key, Layer, Provenance, TagSet,
    ViewId, export, import,
};
use turbo_analysis_layers::{
    Aspect, AspectTarget, Aspects, Dependencies, DocumentEntities, Entities, PolarityScores, Sentences, SyntacticRole,
    Tokens,
};
use turbo_coref::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person};
use turbo_entity_linker::{
    CatalogProvider, ConsolidatorConfig, EntityConsolidator, EntityLinker, KbEntity, KnowledgeBase, codecs,
};
use turbo_geo::{EntryParts, GazetteerEntry, GeoPoint, GeoResolution, Locations};

const TEXT: &str = "Abraham Lincoln lived in Springfield . Lincoln was praised . He loved it .";
//                  0       1       2     3  4           5 6       7   8       9 10 11    12 13

fn names() -> Document {
    doc_with_names(TEXT, &[("Abraham Lincoln", "PERSON"), ("Springfield", "GPE"), ("Lincoln", "PERSON")])
}

/// Mentions `Abraham Lincoln`, `Lincoln` (lifted from names 0 and 2) and
/// `He`, one chain.
fn add_coref(doc: &mut Document) {
    let e = doc.require(&Key::<Entities>::DEFAULT).unwrap();
    let (spans, tokens) = ([0, 2].map(|r| e.spans.get(Ix::new(r))), [e.tokens[0], e.tokens[2]]);
    let he = turbo_analysis::Span::new(61, 63);
    assert_eq!(&TEXT[61..63], "He");
    let unk = Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown);
    let mentions = Mentions {
        spans: spans_of(ViewId(0), [spans[0], spans[1], he]),
        sentence: vec![Ix::new(0); 3],
        tokens: vec![tokens[0], tokens[1], IxRange::new(10, 11)],
        head: vec![Ix::new(1), Ix::new(6), Ix::new(10)],
        entity: Some(vec![Some(Ix::new(0)), Some(Ix::new(2)), None]),
        kind: vec![MentionKind::Proper, MentionKind::Proper, MentionKind::Pronominal],
        origin: vec![Origin::Entity, Origin::Entity, Origin::Pronoun],
        agreement: vec![unk; 3],
    };
    doc.add(&Key::<Mentions>::DEFAULT, mentions, Provenance::by("test:mentions")).unwrap();
    doc.add(
        &Key::<Groups<Mentions>>::DEFAULT,
        Groups::new([vec![Ix::new(0), Ix::new(1), Ix::new(2)]]),
        Provenance::by("test:coref"),
    )
    .unwrap();
}

fn add_links(doc: &mut Document) {
    let kb = KnowledgeBase::from_entities(
        "wikidata",
        vec![
            KbEntity::new("Q91", "Abraham Lincoln").with_alias("Lincoln", 3.5).with_types(["PERSON"]).with_prior(0.95),
            KbEntity::new("Q28515", "Springfield, Illinois")
                .with_alias("Springfield", 3.0)
                .with_types(["GPE"])
                .with_prior(0.9),
        ],
    )
    .unwrap();
    EntityLinker::new().add_provider(CatalogProvider::new(kb)).annotate_document(doc).unwrap();
}

fn add_dependencies(doc: &mut Document) {
    let relations = TagSet::new("ud", &["aux", "case", "flat", "nsubj", "nsubj:pass", "obj", "obl", "punct", "root"]);
    let rows: [(Option<u32>, &str); 14] = [
        (Some(1), "flat"),
        (Some(2), "nsubj"),
        (None, "root"),
        (Some(4), "case"),
        (Some(2), "obl"),
        (Some(2), "punct"),
        (Some(8), "nsubj:pass"),
        (Some(8), "aux"),
        (None, "root"),
        (Some(8), "punct"),
        (Some(11), "nsubj"),
        (None, "root"),
        (Some(11), "obj"),
        (Some(11), "punct"),
    ];
    let deps = Dependencies {
        head: rows.iter().map(|r| r.0.map(Ix::new)).collect(),
        relation: rows.iter().map(|r| relations.tag(r.1).unwrap()).collect(),
        relations,
    };
    doc.add(&Key::<Dependencies>::DEFAULT, deps, Provenance::by("test:deps")).unwrap();
}

fn add_aspects(doc: &mut Document) {
    let tokens = doc.require(&Key::<Tokens>::DEFAULT).unwrap();
    let term = |a, b| AspectTarget::Term { tokens: IxRange::new(a, b), sentence: None };
    let items = [
        Aspect { target: term(6, 7), category: None, polarity: PolarityScores::new(0.9, 0.05, 0.05), opinions: &[] },
        Aspect { target: term(10, 11), category: None, polarity: PolarityScores::new(0.7, 0.1, 0.2), opinions: &[] },
        Aspect { target: term(12, 13), category: None, polarity: PolarityScores::new(0.0, 0.8, 0.2), opinions: &[] },
    ];
    let aspects = Aspects::from_items(tokens, None, None, Arc::new(TagSet::default()), items).unwrap();
    doc.add(&Key::<Aspects>::DEFAULT, aspects, Provenance::by("test:aspects")).unwrap();
}

fn add_place(doc: &mut Document) {
    let e = doc.require(&Key::<Entities>::DEFAULT).unwrap();
    let span = e.spans.get(Ix::new(1));
    let entry = GazetteerEntry::new(EntryParts {
        source: "test".into(),
        record_id: "4250542".into(),
        name: "Springfield".into(),
        alternate_names: vec![],
        location: GeoPoint::new(39.80, -89.64).unwrap(),
        bounding_box: None,
        country_code: Some("US".into()),
        containment: vec![],
        population: 114_000,
        feature_class: None,
        attributes: vec![],
    })
    .unwrap();
    let mut locations = Locations::new(ViewId(0));
    locations.push(GeoResolution::new(span, entry, 0.9).unwrap());
    doc.add(&Key::<Locations>::DEFAULT, locations, Provenance::by("test:geo")).unwrap();
}

/// `lived_in` found on `Abraham Lincoln` and again on `Lincoln`.
fn add_relations(doc: &mut Document) {
    let kinds = TagSet::new("rel", &["lived_in"]);
    let lived = kinds.tag("lived_in").unwrap();
    let edges: Edges<Entities, Entities> = Edges::new(
        kinds,
        [Edge::new(Ix::new(0), Ix::new(1), lived, 0.6), Edge::new(Ix::new(2), Ix::new(1), lived, 0.8)],
    );
    doc.add(&Key::<Edges<Entities, Entities>>::DEFAULT, edges, Provenance::by("test:relations")).unwrap();
}

fn full_doc() -> Document {
    let mut doc = names();
    add_coref(&mut doc);
    add_links(&mut doc);
    add_dependencies(&mut doc);
    add_aspects(&mut doc);
    add_relations(&mut doc);
    doc
}

#[test]
fn coref_links_roles_sentiment_and_relations_make_one_lincoln() {
    let mut doc = full_doc();
    add_place(&mut doc);
    let graph = EntityConsolidator::new().consolidate(&doc).unwrap();
    graph.check().unwrap();
    assert_eq!(graph.entities.len(), 2);

    let lincoln = &graph.entities[0];
    assert_eq!(&*lincoln.canonical_name, "Abraham Lincoln");
    assert_eq!(&*lincoln.kind, "PERSON");
    assert_eq!(lincoln.kb_id.as_deref(), Some("Q91"));
    let surfaces: Vec<&str> =
        lincoln.mentions.iter().map(|m| &TEXT[m.span.start as usize..m.span.end as usize]).collect();
    assert_eq!(surfaces, ["Abraham Lincoln", "Lincoln", "He"]);
    assert_eq!(lincoln.mentions.iter().filter(|m| m.is_representative).count(), 1);
    assert_eq!(lincoln.mentions[2].entity, None);
    assert_eq!(lincoln.role_counts.subject_count, 3);
    assert_eq!(lincoln.mentions[1].role, Some(SyntacticRole::Subject));
    assert_eq!(lincoln.sentiment_mentions, 2);
    assert!((lincoln.mean_sentiment - 0.725).abs() < 1e-6, "{}", lincoln.mean_sentiment);
    assert_eq!(lincoln.geo_coord, None);

    let springfield = &graph.entities[1];
    assert_eq!(&*springfield.canonical_name, "Springfield, Illinois");
    assert_eq!(springfield.kb_id.as_deref(), Some("Q28515"));
    assert_eq!(springfield.role_counts.other_count, 1);
    assert_eq!(springfield.country_code, Some(*b"US"));
    let (lat, lon) = springfield.geo_coord.unwrap();
    assert!((lat - 39.80).abs() < 1e-4 && (lon + 89.64).abs() < 1e-4);

    assert_eq!(graph.relations.len(), 1);
    let r = &graph.relations[0];
    assert_eq!((r.subject, r.object, &*r.kind, r.support), (0, 1, "lived_in", 2));
    assert!((r.score - 0.8).abs() < 1e-6);
}

#[test]
fn optional_layers_absent_still_merge_names_by_surface_and_alias() {
    let text = "Barack Obama met Merkel . Obama smiled . Merkel spoke .";
    let doc = doc_with_names(
        text,
        &[("Barack Obama", "PERSON"), ("Merkel", "PERSON"), ("Obama", "PERSON"), ("Merkel", "PERSON")],
    );
    let graph = EntityConsolidator::new().consolidate(&doc).unwrap();
    graph.check().unwrap();
    let names: Vec<(&str, usize)> = graph.entities.iter().map(|e| (&*e.canonical_name, e.mentions.len())).collect();
    assert_eq!(names, [("Barack Obama", 2), ("Merkel", 2)]);
    assert!(graph.entities.iter().all(|e| e.kb_id.is_none() && e.sentiment_mentions == 0 && e.geo_coord.is_none()));
    assert!(graph.relations.is_empty());

    let off = ConsolidatorConfig { surface_merge: false, alias_merge: false };
    let apart = EntityConsolidator::new().with_config(off).consolidate(&doc).unwrap();
    assert_eq!(apart.entities.len(), 4);
}

#[test]
fn an_ambiguous_alias_and_a_different_kind_stay_apart() {
    let text = "John Smith and Mary Smith met . Smith left Washington ; Washington stayed .";
    let doc = doc_with_names(
        text,
        &[
            ("John Smith", "PERSON"),
            ("Mary Smith", "PERSON"),
            ("Smith", "PERSON"),
            ("Washington", "GPE"),
            ("Washington", "PERSON"),
        ],
    );
    let graph = EntityConsolidator::new().consolidate(&doc).unwrap();
    let names: Vec<(&str, &str)> = graph.entities.iter().map(|e| (&*e.canonical_name, &*e.kind)).collect();
    assert_eq!(
        names,
        [
            ("John Smith", "PERSON"),
            ("Mary Smith", "PERSON"),
            ("Smith", "PERSON"),
            ("Washington", "GPE"),
            ("Washington", "PERSON")
        ]
    );
}

#[test]
fn hash_compat_config_hash_is_deterministic_and_tracks_the_switches() {
    let a = EntityConsolidator::new();
    assert_eq!(a.config_hash(), EntityConsolidator::new().config_hash());
    let b = EntityConsolidator::new().with_config(ConsolidatorConfig { surface_merge: false, alias_merge: true });
    let c = EntityConsolidator::new().with_config(ConsolidatorConfig { surface_merge: true, alias_merge: false });
    assert_ne!(a.config_hash(), b.config_hash());
    assert_ne!(b.config_hash(), c.config_hash());
    let d = EntityConsolidator::new().with_config(ConsolidatorConfig { surface_merge: false, alias_merge: false });
    let hashes = [a.config_hash(), b.config_hash(), c.config_hash(), d.config_hash()];
    // Fixed values from the previous implementation, before migration.
    assert_eq!(hashes, [5048337373682345938, 5047382997589248015, 5048338473193974149, 5047381898077619804]);
}

#[test]
fn analyzer_records_inputs_and_the_graph_round_trips_through_bytes() {
    let doc = full_doc();
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<Mentions>::DEFAULT)
        .given(Key::<Groups<Mentions>>::DEFAULT)
        .given(Key::<turbo_entity_linker::LinkedEntities>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Aspects>::DEFAULT)
        .given(Key::<Edges<Entities, Entities>>::DEFAULT)
        .add(EntityConsolidator::new())
        .build()
        .unwrap();
    let mut batch = vec![doc];
    for r in analyzer.run(&mut batch) {
        r.unwrap();
    }
    let doc = batch.pop().unwrap();
    let graph = doc.require(&Key::<DocumentEntities>::DEFAULT).unwrap().clone();
    assert_eq!(graph.entities.len(), 2);
    let prov = doc.provenance(Key::<DocumentEntities>::DEFAULT.id()).unwrap();
    assert_eq!(prov.annotator, EntityConsolidator::ID);
    assert_eq!(prov.config, EntityConsolidator::new().config_hash());
    for read in [
        Key::<Entities>::DEFAULT.id(),
        Key::<Mentions>::DEFAULT.id(),
        Key::<Groups<Mentions>>::DEFAULT.id(),
        Key::<turbo_entity_linker::LinkedEntities>::DEFAULT.id(),
        Key::<Dependencies>::DEFAULT.id(),
        Key::<Aspects>::DEFAULT.id(),
        Key::<Edges<Entities, Entities>>::DEFAULT.id(),
    ] {
        assert!(prov.reads.contains(&read), "{read:?} not in {:?}", prov.reads);
    }
    assert!(!prov.reads.contains(&Key::<Locations>::DEFAULT.id()));

    let mut codecs = codecs();
    codecs.register::<Edges<Entities, Entities>>();
    let bytes = export(&doc, &codecs).unwrap().to_bytes().unwrap();
    let back = import(&DocumentGraph::from_bytes(&bytes).unwrap(), &codecs).unwrap();
    assert_eq!(back.require(&Key::<DocumentEntities>::DEFAULT).unwrap(), &graph);
}
