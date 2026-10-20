//! Entity continuity costs and chunk entity headers (#178) through a
//! pipeline: the `entity` instance keeps a run of sentences about one
//! entity in one chunk where the same pipeline without it cuts the run,
//! the headers name entities by index, and both survive the document
//! graph boundary. Configs that leave the signal out keep their outputs
//! and config hashes.

use turbo_analysis::export::{Codecs, DocumentGraph, Writer, export, import};
use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{
    Analyzer, Annotator, Bound, Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, View,
    ViewId,
};
use turbo_analysis_layers::{Boundaries, Chunks, Entities, Sentences, Subwords, Tokens};
use turbo_chunker::{
    Budget, ChunkEntities, ChunkEntityHeaders, Chunker, Costs, DistilledBoundaryScorer, DistilledWeights, ENTITY,
    EntityBoundaries, EntityBoundariesProvider, STRUCTURE, StructureBoundaries, WeighProvider, Weight,
};

const SENTENCES: [&str; 6] = [
    "Marie Curie studied physics and chemistry in her youth.",
    "Later marie  curie moved abroad to continue her studies.",
    "The prize made MARIE CURIE famous across the world.",
    "Paris was then a city of science and art.",
    "Many laboratories in Paris welcomed eager young foreign researchers.",
    "The museums of Paris still show that history today.",
];

fn text() -> String {
    SENTENCES.join(" ")
}

/// A document of six ten-token sentences: three about Marie Curie
/// (written three ways), three about Paris.
fn document() -> Document {
    let text = text();
    let mut doc = Document::new(text.clone());
    let by = || Provenance::by("t");
    let (mut sents, mut toks, mut subs) =
        (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let mut at = 0u32;
    for s in SENTENCES {
        sents.push(Span::new(at, at + s.len() as u32));
        at += s.len() as u32 + 1;
    }
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b' ' {
            i += 1;
            continue;
        }
        let start = i;
        i += 1;
        if bytes[start] != b'.' {
            while i < bytes.len() && bytes[i] != b' ' && bytes[i] != b'.' {
                i += 1;
            }
        }
        toks.push(Span::new(start as u32, i as u32));
        subs.push(Span::new(start as u32, i as u32));
    }
    let n = toks.len();
    let model = ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() };
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, by()).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: toks, sentence: None }, by()).unwrap();
    doc.add(&Key::DEFAULT, Subwords { spans: subs, ids: vec![1; n], model, token: None }, by()).unwrap();
    let mut found: Vec<(usize, usize, u16)> = Vec::new();
    for (needle, kind) in [("Marie Curie", 0), ("marie  curie", 0), ("MARIE CURIE", 0), ("Paris", 1)] {
        found.extend(text.match_indices(needle).map(|(a, _)| (a, a + needle.len(), kind)));
    }
    found.sort_unstable();
    let mut spans = Spans::new(ViewId::SOURCE);
    for &(a, b, _) in &found {
        spans.push(Span::new(a as u32, b as u32));
    }
    let m = found.len();
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1); m],
        kind: found.iter().map(|f| Tag(f.2)).collect(),
        kinds: TagSet::new("ner", &["PER", "LOC"]),
        score: vec![1.0; m],
    };
    doc.add(&Key::DEFAULT, entities, by()).unwrap();
    doc
}

/// Structure at full weight and every other distilled signal at zero,
/// so the entity instance is the one signal added.
fn weights(entity_runs: Weight) -> DistilledWeights {
    DistilledWeights {
        structure: Weight::ONE,
        drift: Weight(0),
        sentiment: Weight(0),
        coref: Weight(0),
        lexical_overlap: 0,
        entity_continuity: 0,
        bias: 0,
        entity_runs,
    }
}

/// Chunks the document; `with_entity` adds the `entity` instance.
fn run(with_entity: bool, entity_runs: Weight) -> Document {
    let mut doc = document();
    let mut b = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE));
    if with_entity {
        b = b.step(EntityBoundaries::default().bound());
    }
    let analyzer = b
        .step(DistilledBoundaryScorer { weights: weights(entity_runs) }.bound())
        .add(Chunker { budget: Budget { max: 40, target: 20, size_weight: 3000, overhead: 0 } })
        .add(ChunkEntityHeaders::default())
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();
    doc
}

fn ranges(doc: &Document) -> Vec<(usize, usize)> {
    doc.get::<Chunks>().unwrap().sentences.iter().map(|r| (r.start.get(), r.end.get())).collect()
}

fn entries(doc: &Document, c: u32) -> Vec<(u32, u32)> {
    doc.get::<ChunkEntities>().unwrap().entries(Ix::new(c)).collect()
}

#[test]
fn entity_runs_keep_sentences_about_one_entity_together() {
    let without = run(false, Weight::ONE);
    // Without the signal every gap costs one sentence boundary and pairs
    // hit the target: both runs are cut.
    assert_eq!(ranges(&without), [(0, 2), (2, 4), (4, 6)]);
    let with = run(true, Weight::ONE);
    let costs = with.get_key(&Key::<Boundaries>::named(ENTITY)).unwrap();
    assert_eq!(costs.cost, [500, 500, 0, 500, 500, 0]);
    // With it the one cut falls where Marie Curie gives way to Paris.
    assert_eq!(ranges(&with), [(0, 3), (3, 6)]);
    // Weighed at zero the instance changes nothing.
    assert_eq!(ranges(&run(true, Weight(0))), ranges(&without));
    assert_eq!(without.get::<Boundaries>().unwrap().cost, run(true, Weight(0)).get::<Boundaries>().unwrap().cost);
}

#[test]
fn headers_name_entities_by_index_and_carry_the_cut_run() {
    let text = text();
    let with = run(true, Weight::ONE);
    let ents = with.get::<Entities>().unwrap();
    let h = with.get::<ChunkEntities>().unwrap();
    // Key 0 is Marie Curie (all three spellings), key 1 Paris.
    assert_eq!(h.first, [Ix::new(0), Ix::new(3)]);
    assert_eq!(entries(&with, 0), [(0, 3)]);
    assert_eq!(entries(&with, 1), [(1, 3), (0, 0)]);
    assert_eq!(h.header(Ix::new(1), ents, &text).unwrap(), "[Entities: Paris | in scope: Marie Curie]");
    // The run cut without the signal: the middle chunk names both, the
    // last carries Marie Curie from the chunk before.
    let without = run(false, Weight::ONE);
    assert_eq!(entries(&without, 1), [(0, 1), (1, 1)]);
    assert_eq!(entries(&without, 2), [(1, 2), (0, 0)]);
}

fn codecs() -> Codecs {
    let mut codecs = turbo_analysis_layers::export::codecs();
    turbo_chunker::export::register_codecs(&mut codecs);
    codecs
}

#[test]
fn chunk_entities_round_trip_through_the_graph() {
    let text = text();
    let mut doc = document();
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 40, target: 20, size_weight: 3000, overhead: 0 } })
        .add(ChunkEntityHeaders::default())
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let (a, b) = (doc.get::<ChunkEntities>().unwrap(), back.get::<ChunkEntities>().unwrap());
    assert_eq!(a, b);
    let ents = back.get::<Entities>().unwrap();
    for c in 0..back.get::<Chunks>().unwrap().spans.len() as u32 {
        let c = Ix::new(c);
        assert_eq!(a.header(c, doc.get::<Entities>().unwrap(), &text), b.header(c, ents, &text));
    }
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // Frames that do not hold together are refused, never clamped.
    let body = |first: &[u32], row: &[u32], key: &[u32], mentions: &[u32]| {
        let mut w = Writer::new();
        for col in [first, row, key, mentions] {
            w.usize(col.len()).unwrap();
            for &v in col {
                w.u32(v);
            }
        }
        w.finish()
    };
    let at = graph.layers.iter().position(|l| l.layer.starts_with("turbo:chunk-entities")).unwrap();
    for bad in [
        body(&[0, 99], &[0, 1, 1, 1], &[0], &[1]),
        body(&[0], &[0, 1, 1, 1], &[1], &[1]),
        body(&[0], &[0, 1], &[0], &[1]),
        body(&[0, 3], &[0, 2, 2, 2], &[0, 0], &[1, 1]),
    ] {
        let mut g = DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap();
        g.layers[at].body = Some(bad);
        assert!(import(&g, &codecs()).is_err());
    }
}

#[test]
fn legacy_distilled_hash_is_unchanged_and_providers_build_the_steps() {
    // The fold every distilled config hashed to before `entity_runs`.
    let legacy = |w: &DistilledWeights| {
        [w.structure.0, w.drift.0, w.sentiment.0, w.coref.0, w.lexical_overlap, w.entity_continuity, w.bias]
            .iter()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, &x| (h ^ x as u32 as u64).wrapping_mul(0x0100_0000_01b3))
    };
    let d = DistilledWeights::default();
    assert_eq!(d.entity_runs, Weight::ONE);
    assert_eq!(DistilledBoundaryScorer { weights: d }.config_hash(), legacy(&d));
    let w = weights(Weight::ONE);
    assert_eq!(DistilledBoundaryScorer { weights: w }.config_hash(), legacy(&w));
    let off = weights(Weight(0));
    assert_ne!(DistilledBoundaryScorer { weights: off }.config_hash(), legacy(&off));

    let p = EntityBoundariesProvider;
    let spec = Spec::default().with("per_entity", "400").with("max_gap", "1").with("kind", "true");
    assert!(p.supports(&spec));
    assert!(p.create(&spec).is_ok());
    assert!(!p.supports(&Spec::default().with("unknown", "1")));
    assert!(p.create(&Spec::default().with("max_gap", "0")).is_err());
    assert!(p.create(&Spec::default().with("chains", "maybe")).is_err());
    let weigh = WeighProvider;
    assert!(weigh.supports(&Spec::default().with("entity_runs", "500")));
    assert!(weigh.create(&Spec::default().with("entity_runs", "500")).is_ok());
}
