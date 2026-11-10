//! The relation stages of the full document graph, together (#1952): the
//! dependency-path rules and the GLiREL scorer run over one shared
//! `Entities` layer in one analyzer, each writing `Relations` and
//! deduplicated `Edges<Entities, Entities>` with its own provenance, and
//! the graph crosses export and import with every entity link and offset
//! intact. The text holds multi-byte characters, so an offset that drifted
//! by a byte would slice another string or fail to slice at all.
//!
//! The GLiREL weights are the seeded reduced fixture: its scores show the
//! wiring, not quality. Quality on a labelled corpus is measured on the
//! release corpus, not here.

use std::collections::BTreeSet;

use turbo::deberta::{Config, Encoder};
use turbo::safetensors::File;
use turbo::sentencepiece::SentencePiece;
use turbo_analysis::export::{export, import};
use turbo_analysis::{
    Analyzer, Bound, Document, Edges, Ix, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, View,
    ViewId,
};
use turbo_analysis_layers::{Dependencies, Entities, Sentences, Tokens};
use turbo_relation::export::codecs;
use turbo_relation::{GlirelRelations, RelationAnnotator, RelationPattern, Relations};

const TEXT: &str = "Ørsted acquired Bølt in 2024.";
/// Each token's byte range; `Ø` and `ø` are two bytes each.
const TOKENS: [(u32, u32); 6] = [(0, 7), (8, 16), (17, 22), (23, 25), (26, 30), (30, 31)];
/// The entities, as token ranges, kinds and the text each covers.
const ENTITIES: [(u32, u32, u16, &str); 3] = [(0, 1, 0, "Ørsted"), (2, 3, 0, "Bølt"), (4, 5, 1, "2024")];
const NEURAL: &str = "glirel";

fn spans<L>(ranges: impl IntoIterator<Item = (u32, u32)>) -> Spans<L> {
    let mut out = Spans::new(ViewId::SOURCE);
    for (a, b) in ranges {
        out.push(Span::new(a, b));
    }
    out
}

/// Each span's byte range, in layer order.
fn ranges<L>(s: &Spans<L>) -> Vec<(u32, u32)> {
    s.iter().map(|(_, span)| (span.start, span.end)).collect()
}

fn model_ref() -> ModelRef {
    ModelRef { id: "reduced".into(), revision: "seeded".into(), sha256: "fixture".into() }
}

fn glirel() -> GlirelRelations {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glirel");
    let bytes = std::fs::read(format!("{dir}/reduced.safetensors")).unwrap();
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 800,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let canonical = turbo_bundle::deberta::convert(&config, &bytes, "token_rep_layer.bert_layer.model.").unwrap();
    let encoder = Encoder::from_safetensors(config, &canonical).unwrap();
    let heads = turbo_relation::glirel::heads::Heads::load(&File::parse("reduced", &bytes).unwrap(), 32).unwrap();
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../testdata/sentencepiece/unigram-identity.model");
    let tokenizer = SentencePiece::parse(&std::fs::read(path).unwrap()).unwrap();
    let specials = turbo_relation::glirel::prompt::Specials { cls: 1, sep: 2, rel: 3 };
    let mut model = GlirelRelations::new(encoder, heads, tokenizer, specials, model_ref(), 512).unwrap();
    model.configure(vec!["acquired".into()], 0.0).unwrap();
    model
}

/// The document with its parse and entities given, after both relation
/// stages ran: the rules into the default instances, GLiREL into
/// [`NEURAL`].
fn analyzed() -> Document {
    let mut doc = Document::new(TEXT);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans([(0, 31)]) }, by.clone()).unwrap();
    let tokens = Tokens { spans: spans(TOKENS), sentence: Some(vec![Ix::new(0); TOKENS.len()]) };
    doc.add(&Key::DEFAULT, tokens, by.clone()).unwrap();
    let labels = ["nsubj", "root", "obj", "case", "obl", "punct"];
    let tag = |l: &str| Tag(labels.iter().position(|x| *x == l).unwrap() as u16);
    let deps = Dependencies::<Tokens> {
        head: [Some(1), None, Some(1), Some(4), Some(1), Some(1)].map(|h| h.map(Ix::new)).to_vec(),
        relation: ["nsubj", "root", "obj", "case", "obl", "punct"].map(tag).to_vec(),
        relations: TagSet::new("dependency", &labels),
    };
    doc.add(&Key::DEFAULT, deps, by.clone()).unwrap();
    let entities = Entities {
        spans: spans(ENTITIES.map(|(a, b, _, _)| (TOKENS[a as usize].0, TOKENS[b as usize - 1].1))),
        tokens: ENTITIES.map(|(a, b, _, _)| IxRange::new(a, b)).to_vec(),
        kind: ENTITIES.map(|(_, _, k, _)| Tag(k)).to_vec(),
        kinds: TagSet::new("entity", &["organization", "date"]),
        score: vec![1.0; ENTITIES.len()],
    };
    doc.add(&Key::DEFAULT, entities, by).unwrap();
    let rules = RelationAnnotator::<Tokens>::new(vec![
        RelationPattern::with_trigger("acquisition", "<nsubj >obj", "acquired").unwrap().confidence(0.75).unwrap(),
        // A second rule for the same triple: two mentions, one edge.
        RelationPattern::with_trigger("acquisition", "<nsubj >obj", "acquired").unwrap().confidence(0.9).unwrap(),
        // A negative: no pair's path pivots on this trigger.
        RelationPattern::with_trigger("sale", "<nsubj >obj", "sold").unwrap(),
    ])
    .unwrap();
    let neural = Bound::new(glirel()).write(0, NEURAL).write(1, NEURAL).write(2, NEURAL);
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Dependencies>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(rules)
        .step(neural)
        .build()
        .unwrap();
    let mut docs = vec![doc];
    analyzer.run(&mut docs).into_iter().for_each(Result::unwrap);
    docs.pop().unwrap()
}

/// Each mention of `name`'s relations as (subject text, object text,
/// type, score), read through the entities `name`'s stage wrote or kept.
fn mentions(doc: &Document, name: &'static str) -> Vec<(String, String, String, f64)> {
    let relations = doc.get_key(&Key::<Relations>::named(name)).expect("relations");
    let entities = doc.get_key(&Key::<Entities>::named(name)).or_else(|| doc.get::<Entities>()).unwrap();
    let text = |e: Ix<Entities>| {
        let s = entities.spans.get(e);
        TEXT.get(s.start as usize..s.end as usize).expect("an entity on a char boundary").to_owned()
    };
    (0..relations.len() as u32)
        .map(|i| {
            let (s, o, kind, score) = relations.mention(Ix::new(i));
            (text(s), text(o), kind.to_owned(), score)
        })
        .collect()
}

/// Each edge of `name` as (from, to, type, score).
fn edges(doc: &Document, name: &'static str) -> Vec<(usize, usize, String, f64)> {
    let e = doc.get_key(&Key::<Edges<Entities, Entities>>::named(name)).expect("edges");
    (0..e.len()).map(|i| (e.from[i].get(), e.to[i].get(), e.kinds.label(e.kind[i]).to_owned(), e.score[i])).collect()
}

#[test]
fn rules_and_a_neural_scorer_share_the_entities_and_write_their_own_edges() {
    let doc = analyzed();
    // The rules: two mentions of one acquisition, one edge at the best
    // confidence, no mention of the reverse pair or the date, and none of
    // the rule whose trigger the text does not hold.
    let rules = mentions(&doc, "");
    let acquisition = |score| ("Ørsted".to_owned(), "Bølt".to_owned(), "acquisition".to_owned(), score);
    assert_eq!(rules, vec![acquisition(0.75), acquisition(0.9)]);
    assert_eq!(edges(&doc, ""), vec![(0, 1, "acquisition".to_owned(), 0.9)]);
    let relations = doc.get::<Relations>().unwrap();
    let covered = relations.spans.get(Ix::new(0));
    assert_eq!(&TEXT[covered.start as usize..covered.end as usize], "Ørsted acquired Bølt");

    // GLiREL over the same entities: every ordered pair of distinct
    // entities scored, its entities the given spans, and one edge per
    // distinct triple.
    let given = doc.get::<Entities>().unwrap();
    let kept = doc.get_key(&Key::<Entities>::named(NEURAL)).unwrap();
    assert_eq!(ranges(&kept.spans), ranges(&given.spans), "the scorer keeps the shared entity spans");
    let neural = mentions(&doc, NEURAL);
    assert_eq!(neural.len(), ENTITIES.len() * (ENTITIES.len() - 1));
    let names: BTreeSet<&str> = ENTITIES.iter().map(|e| e.3).collect();
    for (s, o, kind, score) in &neural {
        assert!(s != o && names.contains(s.as_str()) && names.contains(o.as_str()), "{s} -> {o}");
        assert_eq!(kind, "acquired");
        assert!((0.0..=1.0).contains(score), "{score}");
    }
    let neural_edges = edges(&doc, NEURAL);
    let triples: BTreeSet<(usize, usize, &str)> = neural_edges.iter().map(|e| (e.0, e.1, e.2.as_str())).collect();
    assert_eq!(triples.len(), neural_edges.len(), "edges are deduplicated");
    assert_eq!(neural_edges.len(), neural.len(), "each pair once");

    // Each stage is its own provenance; the scorer names its model.
    let rules_by = doc.provenance(Key::<Relations>::DEFAULT.id()).unwrap();
    assert_eq!((rules_by.annotator, rules_by.model.as_ref()), ("turbo:relation-annotator", None));
    let neural_by = doc.provenance(Key::<Relations>::named(NEURAL).id()).unwrap();
    assert_eq!((neural_by.annotator, neural_by.model.as_ref()), ("turbo:glirel-relations", Some(&model_ref())));
}

#[test]
fn the_relation_graph_survives_export_and_import_with_links_and_offsets() {
    let doc = analyzed();
    let registry = codecs();
    let graph = export(&doc, &registry).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let back = import(&turbo_analysis::export::DocumentGraph::from_bytes(&bytes).unwrap(), &registry).unwrap();
    assert_eq!(back.get::<View>().unwrap().text(), TEXT);
    for name in ["", NEURAL] {
        assert_eq!(mentions(&back, name), mentions(&doc, name), "{name:?}: every link reads the same text");
        assert_eq!(edges(&back, name), edges(&doc, name), "{name:?}");
        let key = Key::<Relations>::named(name).id();
        let (a, b) = (back.provenance(key).unwrap(), doc.provenance(key).unwrap());
        assert_eq!((a.annotator, a.config, &a.model), (b.annotator, b.config, &b.model), "{name:?}");
        let (r, s) = (
            back.get_key(&Key::<Relations>::named(name)).unwrap(),
            doc.get_key(&Key::<Relations>::named(name)).unwrap(),
        );
        assert_eq!(ranges(&r.spans), ranges(&s.spans), "{name:?}: relation spans keep their offsets");
        let e = back.get_key(&Key::<Edges<Entities, Entities>>::named(name)).unwrap();
        e.check().unwrap();
    }
    assert_eq!(ranges(&back.get::<Entities>().unwrap().spans), ranges(&doc.get::<Entities>().unwrap().spans));
    let again = export(&back, &registry).unwrap();
    assert_eq!(again.to_bytes().unwrap(), bytes, "re-export is byte-identical");
}
