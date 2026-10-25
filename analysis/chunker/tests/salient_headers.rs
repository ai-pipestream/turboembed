//! Salience in chunk entity headers (#1201): with `SalienceHeaders` on
//! and the document's consolidated entities and salience present, a
//! chunk's own entries follow document salience and the most salient
//! entities join every header after its carried entries; off (the
//! default) or with either layer absent, headers and the config hash
//! are unchanged.

use turbo_analysis::{
    Analyzer, Annotator, Document, Ix, IxRange, Key, Layer, ModelRef, Provenance, Span, Spans, Tag, TagSet, View,
    ViewId,
};
use turbo_analysis_layers::{
    Chunks, ConsolidatedMention, DocumentEntities, DocumentEntity, Entities, EntitySalience, EntitySalienceScore,
    Sentences, Subwords, SyntacticRoleCounts, Tokens,
};
use turbo_chunker::{Budget, ChunkEntities, ChunkEntityHeaders, Chunker, Costs, SalienceHeaders, StructureBoundaries};

const SENTENCES: [&str; 6] = [
    "Marie Curie studied physics and chemistry.",
    "Later Marie Curie moved abroad to study.",
    "The prize made Marie Curie famous everywhere.",
    "Paris was then a city of science.",
    "Many laboratories in Paris welcomed researchers.",
    "The museums of Paris still show history.",
];

/// Six sentences, Marie Curie in the first three and Paris in the last
/// three, with both consolidated and Paris ranked the more salient.
fn document(with_graph: bool) -> Document {
    let text = SENTENCES.join(" ");
    let mut doc = Document::new(text.clone());
    let by = || Provenance::by("t");
    let (mut sents, mut toks, mut subs) =
        (Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE), Spans::new(ViewId::SOURCE));
    let mut at = 0u32;
    for s in SENTENCES {
        sents.push(Span::new(at, at + s.len() as u32));
        for (i, w) in s.split(' ').enumerate() {
            let off = s.split(' ').take(i).map(|p| p.len() as u32 + 1).sum::<u32>();
            toks.push(Span::new(at + off, at + off + w.len() as u32));
            subs.push(Span::new(at + off, at + off + w.len() as u32));
        }
        at += s.len() as u32 + 1;
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sents }, by()).unwrap();
    let (n, model) = (toks.len(), ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() });
    doc.add(&Key::DEFAULT, Subwords { spans: subs, ids: vec![1; n], model, token: None }, by()).unwrap();
    doc.add(&Key::DEFAULT, Tokens { spans: toks, sentence: None }, by()).unwrap();
    let mut found: Vec<(usize, usize, u16)> = Vec::new();
    for (needle, kind) in [("Marie Curie", 0), ("Paris", 1)] {
        found.extend(text.match_indices(needle).map(|(a, _)| (a, a + needle.len(), kind)));
    }
    found.sort_unstable();
    let mut spans = Spans::new(ViewId::SOURCE);
    found.iter().for_each(|&(a, b, _)| {
        spans.push(Span::new(a as u32, b as u32));
    });
    let m = found.len();
    let entities = Entities {
        spans,
        tokens: vec![IxRange::new(0, 1); m],
        kind: found.iter().map(|f| Tag(f.2)).collect(),
        kinds: TagSet::new("ner", &["PER", "LOC"]),
        score: vec![1.0; m],
    };
    doc.add(&Key::DEFAULT, entities, by()).unwrap();
    if with_graph {
        let mut graph = DocumentEntities::new(ViewId::SOURCE);
        for (name, kind) in [("Marie Curie", 0u16), ("Paris", 1)] {
            let mentions: Vec<ConsolidatedMention> = (0..m)
                .filter(|&r| found[r].2 == kind)
                .map(|r| ConsolidatedMention {
                    span: Span::new(found[r].0 as u32, found[r].1 as u32),
                    entity: Some(Ix::new(r as u32)),
                    is_representative: false,
                    role: None,
                })
                .collect();
            graph.entities.push(DocumentEntity {
                canonical_name: name.into(),
                kind: "X".into(),
                kb_id: None,
                geo_coord: None,
                country_code: None,
                mentions: mentions.into(),
                role_counts: SyntacticRoleCounts::default(),
                mean_sentiment: 0.0,
                sentiment_mentions: 0,
                confidence: 1.0,
            });
        }
        doc.add(&Key::DEFAULT, graph, by()).unwrap();
        let row = |entity_ix, rank, salience| EntitySalienceScore {
            entity_ix,
            salience,
            rank,
            position_score: salience,
            syntax_score: salience,
            frequency_score: salience,
            centrality_score: salience,
        };
        doc.add(&Key::DEFAULT, EntitySalience { scores: vec![row(0, 1, 0.4), row(1, 0, 0.9)] }, by()).unwrap();
    }
    doc
}

fn run(with_graph: bool, headers: ChunkEntityHeaders) -> Document {
    let mut doc = document(with_graph);
    let mut b = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Entities>::DEFAULT);
    if with_graph {
        b = b.given(Key::<DocumentEntities>::DEFAULT).given(Key::<EntitySalience>::DEFAULT);
    }
    let analyzer = b
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 20, target: 12, size_weight: 3000, overhead: 0 } })
        .add(headers)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();
    doc
}

fn all_entries(doc: &Document) -> Vec<Vec<(u32, u32)>> {
    let h = doc.get::<ChunkEntities>().unwrap();
    h.check().unwrap();
    (0..doc.get::<Chunks>().unwrap().spans.len() as u32).map(|c| h.entries(Ix::new(c)).collect()).collect()
}

fn salient(top: u32, order: bool) -> ChunkEntityHeaders {
    ChunkEntityHeaders {
        salience: SalienceHeaders { top_document_entities: top, order_by_salience: order },
        ..Default::default()
    }
}

#[test]
fn off_by_default_and_without_the_layers_nothing_changes() {
    let plain = all_entries(&run(false, ChunkEntityHeaders::default()));
    assert_eq!(all_entries(&run(true, ChunkEntityHeaders::default())), plain);
    assert_eq!(all_entries(&run(false, salient(2, true))), plain, "layers absent");
    assert_eq!(ChunkEntityHeaders::default().config_hash(), salient(0, false).config_hash());
    assert_ne!(ChunkEntityHeaders::default().config_hash(), salient(1, false).config_hash());
    assert_ne!(salient(1, false).config_hash(), salient(1, true).config_hash());
}

#[test]
fn salience_orders_own_entries_and_appends_the_top_entities() {
    // Key 0 is Marie Curie, key 1 Paris; Paris is the more salient.
    let plain = all_entries(&run(true, ChunkEntityHeaders::default()));
    assert_eq!(plain, [vec![(0, 2)], vec![(0, 1), (1, 1)], vec![(1, 2), (0, 0)]]);
    let ordered = all_entries(&run(true, salient(0, true)));
    assert_eq!(ordered, [vec![(0, 2)], vec![(1, 1), (0, 1)], vec![(1, 2), (0, 0)]]);
    let topped = all_entries(&run(true, salient(1, false)));
    assert_eq!(topped, [vec![(0, 2), (1, 0)], vec![(0, 1), (1, 1)], vec![(1, 2), (0, 0)]]);
    // Both on, top two: nothing is listed twice, carried entries stay.
    let both = all_entries(&run(true, salient(2, true)));
    assert_eq!(both, [vec![(0, 2), (1, 0)], vec![(1, 1), (0, 1)], vec![(1, 2), (0, 0)]]);

    let doc = run(true, salient(1, true));
    let (h, ents, text) = (doc.get::<ChunkEntities>().unwrap(), doc.get::<Entities>().unwrap(), SENTENCES.join(" "));
    let headers: Vec<String> = (0..3).map(|c| h.header(Ix::new(c), ents, &text).unwrap()).collect();
    assert_eq!(
        headers,
        [
            "[Entities: Marie Curie | in scope: Paris]",
            "[Entities: Paris; Marie Curie]",
            "[Entities: Paris | in scope: Marie Curie]"
        ]
    );
    let prov = doc.provenance(Key::<ChunkEntities>::DEFAULT.id()).unwrap();
    assert_eq!(prov.config, salient(1, true).config_hash());
}
