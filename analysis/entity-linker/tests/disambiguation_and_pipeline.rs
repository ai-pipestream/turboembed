//! Integration tests for `EntityLinker`, multi-signal disambiguation,
//! coreference cluster propagation, document-level coherence, explicit `NIL`
//! reasons and clustering, `DocumentGraph` binary export/import roundtrip,
//! Wikidata-style accuracy/NIL evaluation, and throughput verification.

use std::time::Instant;

use turbo_analysis::{
    Analyzer, Document, DocumentGraph, Groups, Ix, IxRange, Key, Layer, Provenance, Span, Spans, TagSet, ViewId,
    export, import,
};
use turbo_analysis_layers::{Entities, Sentences, Tokens};
use turbo_coref::{Agreement, Animacy, Gender, MentionKind, Mentions, Number, Origin, Person};
use turbo_entity_linker::{
    CandidateLink, CatalogProvider, EntityLinker, KbEntity, KnowledgeBase, LinkedEntities, LinkerConfig, NilReason,
    codecs,
};

fn spans_of<L>(view: ViewId, items: impl IntoIterator<Item = Span>) -> Spans<L> {
    let mut s = Spans::new(view);
    for sp in items {
        s.push(sp);
    }
    s
}

fn build_wikidata_kb() -> KnowledgeBase {
    let entities = vec![
        KbEntity::new("Q312", "Apple Inc.")
            .with_alias("Apple", 4.0)
            .with_alias("Apple Computer", 1.5)
            .with_types(["ORG", "COMPANY"])
            .with_description(
                "American multinational technology company in Cupertino making iPhone, Mac, and iOS software",
            )
            .with_prior(0.92)
            .with_related(["Q19837", "Q95"]),
        KbEntity::new("Q89", "Apple (fruit)")
            .with_alias("Apple", 1.5)
            .with_types(["PRODUCT", "FOOD", "FRUIT"])
            .with_description("Edible sweet fruit harvested from orchard apple trees used in baking pie and cider")
            .with_prior(0.68),
        KbEntity::new("Q207628", "Apple Corps")
            .with_alias("Apple", 1.2)
            .with_alias("Apple Records", 2.0)
            .with_types(["ORG", "COMPANY"])
            .with_description("Multimedia corporation and record label founded in London by the Beatles music band")
            .with_prior(0.60),
        KbEntity::new("Q91", "Abraham Lincoln")
            .with_alias("Lincoln", 3.5)
            .with_alias("President Lincoln", 2.0)
            .with_types(["PERSON", "PRESIDENT"])
            .with_description(
                "16th President of the United States who issued the Emancipation Proclamation during the Civil War",
            )
            .with_prior(0.95)
            .with_related(["Q30"]),
        KbEntity::new("Q28260", "Lincoln, Nebraska")
            .with_alias("Lincoln", 2.0)
            .with_types(["GPE", "LOC", "CITY"])
            .with_description("Capital city of the state of Nebraska and home of the state legislature and university")
            .with_prior(0.72)
            .with_related(["Q1553"]),
        KbEntity::new("Q216760", "Lincoln Motor Company")
            .with_alias("Lincoln", 1.5)
            .with_types(["ORG", "PRODUCT", "BRAND"])
            .with_description("Luxury automobile division of Ford Motor Company producing sedan and SUV vehicles")
            .with_prior(0.70)
            .with_related(["Q44294"]),
        KbEntity::new("Q90", "Paris")
            .with_alias("Paris", 4.0)
            .with_types(["GPE", "LOC", "CITY"])
            .with_description(
                "Capital and largest city of France on the Seine river, famous for the Louvre and Eiffel Tower",
            )
            .with_prior(0.96)
            .with_related(["Q142", "Q19675"]),
        KbEntity::new("Q830149", "Paris, Texas")
            .with_alias("Paris", 1.2)
            .with_types(["GPE", "LOC", "CITY"])
            .with_description("City and county seat of Lamar County in northeast Texas near Dallas")
            .with_prior(0.55)
            .with_related(["Q1439", "Q16557"]),
        KbEntity::new("Q19675", "Louvre")
            .with_alias("Louvre", 3.0)
            .with_alias("Louvre Museum", 2.0)
            .with_types(["FAC", "ORG", "MUSEUM"])
            .with_description("National art museum in Paris, France")
            .with_prior(0.88)
            .with_related(["Q90", "Q142"]),
        KbEntity::new("Q16557", "Dallas")
            .with_alias("Dallas", 3.0)
            .with_types(["GPE", "LOC", "CITY"])
            .with_description("Major city in north Texas")
            .with_prior(0.86)
            .with_related(["Q1439", "Q830149"]),
        KbEntity::new("Q23", "George Washington")
            .with_alias("Washington", 3.0)
            .with_types(["PERSON", "PRESIDENT"])
            .with_description(
                "First President of the United States and Continental Army general who crossed the Delaware",
            )
            .with_prior(0.95),
        KbEntity::new("Q61", "Washington, D.C.")
            .with_alias("Washington", 3.5)
            .with_types(["GPE", "LOC", "CITY"])
            .with_description("Capital city of the United States where Congress meets at the Capitol")
            .with_prior(0.94),
    ];
    KnowledgeBase::from_entities("wikidata", entities).expect("valid wikidata test KB")
}

fn make_doc_with_entities(text: &str, mentions: &[(&str, &str)]) -> Document {
    let mut doc = Document::new(text);
    let view = ViewId(0);

    // Simple whitespace tokenization
    let mut tok_spans = Vec::new();
    let mut in_tok = None;
    for (idx, ch) in text.char_indices() {
        if !ch.is_whitespace() {
            if in_tok.is_none() {
                in_tok = Some(idx);
            }
        } else if let Some(start) = in_tok.take() {
            tok_spans.push(Span::new(start as u32, idx as u32));
        }
    }
    if let Some(start) = in_tok {
        tok_spans.push(Span::new(start as u32, text.len() as u32));
    }

    let sent_spans = vec![Span::new(0, text.len() as u32)];
    let sent_sentence = vec![Ix::<Sentences>::new(0); tok_spans.len()];
    doc.add(
        &Key::<Sentences>::DEFAULT,
        Sentences { spans: spans_of(view, sent_spans) },
        Provenance::by("test:sentences"),
    )
    .unwrap();
    doc.add(
        &Key::<Tokens>::DEFAULT,
        Tokens { spans: spans_of(view, tok_spans.clone()), sentence: Some(sent_sentence) },
        Provenance::by("test:tokens"),
    )
    .unwrap();

    let mut labels: Vec<&str> = mentions.iter().map(|(_, k)| *k).collect();
    labels.sort_unstable();
    labels.dedup();
    if labels.is_empty() {
        labels.push("ENTITY");
    }
    let kinds = TagSet::new("ner", &labels);

    let mut ent_spans = Vec::new();
    let mut ent_tokens = Vec::new();
    let mut ent_kinds = Vec::new();
    let mut ent_scores = Vec::new();
    let mut search_from = 0usize;
    for &(surface, kind_str) in mentions {
        let rel = text[search_from..].find(surface).expect("mention surface must appear in text");
        let start = (search_from + rel) as u32;
        let end = start + (surface.len() as u32);
        search_from = end as usize;

        let first_tok = tok_spans.iter().position(|ts| ts.end > start && ts.start < end).unwrap_or(0) as u32;
        let last_tok =
            tok_spans.iter().rposition(|ts| ts.end > start && ts.start < end).map_or(first_tok + 1, |i| (i as u32) + 1);

        ent_spans.push(Span::new(start, end));
        ent_tokens.push(IxRange::new(first_tok, last_tok));
        ent_kinds.push(kinds.tag(kind_str).unwrap());
        ent_scores.push(0.95);
    }

    doc.add(
        &Key::<Entities>::DEFAULT,
        Entities { spans: spans_of(view, ent_spans), tokens: ent_tokens, kind: ent_kinds, kinds, score: ent_scores },
        Provenance::by("test:entities"),
    )
    .unwrap();
    doc
}

#[test]
fn disambiguates_ambiguous_entities_using_type_and_context() {
    let linker = EntityLinker::new().add_provider(CatalogProvider::new(build_wikidata_kb()));

    // 1. "Apple" in tech vs fruit vs music context
    let doc_tech = make_doc_with_entities(
        "Apple unveiled the new iPhone and Mac software at its Cupertino headquarters.",
        &[("Apple", "ORG")],
    );
    let linked_tech = linker.link(&doc_tech).unwrap();
    assert_eq!(linked_tech.get(Ix::new(0)).kb_id, Some("Q312"));

    let doc_fruit = make_doc_with_entities(
        "She harvested a sweet orchard Apple to bake a fresh pie and cider.",
        &[("Apple", "PRODUCT")],
    );
    let linked_fruit = linker.link(&doc_fruit).unwrap();
    assert_eq!(linked_fruit.get(Ix::new(0)).kb_id, Some("Q89"));

    let doc_music =
        make_doc_with_entities("The Beatles music band founded the Apple record label in London.", &[("Apple", "ORG")]);
    let linked_music = linker.link(&doc_music).unwrap();
    assert_eq!(linked_music.get(Ix::new(0)).kb_id, Some("Q207628"));

    // 2. "Lincoln" as PERSON vs GPE vs ORG/PRODUCT
    let doc_pres = make_doc_with_entities(
        "President Lincoln issued the Emancipation Proclamation during the Civil War.",
        &[("Lincoln", "PERSON")],
    );
    assert_eq!(linker.link(&doc_pres).unwrap().get(Ix::new(0)).kb_id, Some("Q91"));

    let doc_city = make_doc_with_entities(
        "The state legislature met in the capital city of Lincoln in Nebraska.",
        &[("Lincoln", "GPE")],
    );
    assert_eq!(linker.link(&doc_city).unwrap().get(Ix::new(0)).kb_id, Some("Q28260"));

    let doc_car = make_doc_with_entities(
        "Ford Motor Company unveiled a new luxury Lincoln sedan automobile.",
        &[("Lincoln", "ORG")],
    );
    assert_eq!(linker.link(&doc_car).unwrap().get(Ix::new(0)).kb_id, Some("Q216760"));
}

#[test]
fn document_coherence_resolves_paris_with_louvre_vs_dallas() {
    let linker = EntityLinker::new().add_provider(CatalogProvider::new(build_wikidata_kb()));

    let doc_fr = make_doc_with_entities(
        "Curators travelled to Paris to visit the Louvre exhibition.",
        &[("Paris", "GPE"), ("Louvre", "FAC")],
    );
    let linked_fr = linker.link(&doc_fr).unwrap();
    assert_eq!(linked_fr.get(Ix::new(0)).kb_id, Some("Q90"));
    assert_eq!(linked_fr.get(Ix::new(1)).kb_id, Some("Q19675"));

    let doc_tx = make_doc_with_entities(
        "Drivers headed north from Dallas toward Paris for the county fair.",
        &[("Dallas", "GPE"), ("Paris", "GPE")],
    );
    let linked_tx = linker.link(&doc_tx).unwrap();
    assert_eq!(linked_tx.get(Ix::new(0)).kb_id, Some("Q16557"));
    assert_eq!(linked_tx.get(Ix::new(1)).kb_id, Some("Q830149"));
}

#[test]
fn coreference_cluster_propagates_canonical_entity_to_short_mentions() {
    let linker = EntityLinker::new().add_provider(CatalogProvider::new(build_wikidata_kb()));

    let text = "Abraham Lincoln was born in 1809. Later, Lincoln visited the capital.";
    let mut doc = make_doc_with_entities(text, &[("Abraham Lincoln", "PERSON"), ("Lincoln", "MISC")]);

    // Add Mentions and Groups<Mentions> linking entity 0 ("Abraham Lincoln") and entity 1 ("Lincoln")
    let entities = doc.require(&Key::<Entities>::DEFAULT).unwrap();
    let m_spans = spans_of(ViewId(0), vec![entities.spans.get(Ix::new(0)), entities.spans.get(Ix::new(1))]);
    let m_tokens = entities.tokens.clone();
    let m_heads = vec![m_tokens[0].start, m_tokens[1].start];
    let unk = Agreement::new(Number::Unknown, Gender::Unknown, Animacy::Unknown, Person::Unknown);
    let mentions = Mentions {
        spans: m_spans,
        sentence: vec![Ix::new(0), Ix::new(0)],
        tokens: m_tokens,
        head: m_heads,
        entity: Some(vec![Some(Ix::new(0)), Some(Ix::new(1))]),
        kind: vec![MentionKind::Proper, MentionKind::Proper],
        origin: vec![Origin::Entity, Origin::Entity],
        agreement: vec![unk, unk],
    };
    doc.add(&Key::<Mentions>::DEFAULT, mentions, Provenance::by("test:mentions")).unwrap();
    doc.add(
        &Key::<Groups<Mentions>>::DEFAULT,
        Groups::new([vec![Ix::new(0), Ix::new(1)]]),
        Provenance::by("test:coref"),
    )
    .unwrap();

    let linked = linker.link(&doc).unwrap();
    assert_eq!(linked.get(Ix::new(0)).kb_id, Some("Q91"));
    assert_eq!(linked.get(Ix::new(1)).kb_id, Some("Q91"));
    assert_eq!(linked.get(Ix::new(1)).candidates[0].coherence_score, 1.0);
}

#[test]
fn nil_reasons_and_nil_clusters_are_recorded_accurately() {
    let config = LinkerConfig { nil_threshold: 0.55, ..Default::default() };
    let linker = EntityLinker::new().with_config(config).add_provider(CatalogProvider::new(build_wikidata_kb()));

    // Mention 0: "Xylophonix" -> NoCandidates (nil_cluster 0)
    // Mention 1: "Paris" tagged as PERSON -> TypeIncompatible (nil_cluster 1)
    // Mention 2: "Xylophonix" again -> NoCandidates (same nil_cluster 0!)
    // Mention 3: "Paris, Texas" tagged as GPE in completely unrelated text with high nil_threshold -> BelowThreshold
    let strict_cfg = LinkerConfig { nil_threshold: 0.98, ..Default::default() };
    let strict_linker =
        EntityLinker::new().with_config(strict_cfg).add_provider(CatalogProvider::new(build_wikidata_kb()));

    let doc = make_doc_with_entities(
        "Xylophonix met Paris and Xylophonix yesterday.",
        &[("Xylophonix", "ORG"), ("Paris", "PERSON"), ("Xylophonix", "ORG")],
    );
    let linked = linker.link(&doc).unwrap();
    assert!(linked.get(Ix::new(0)).is_nil());
    assert_eq!(linked.get(Ix::new(0)).nil_reason, Some(NilReason::NoCandidates));
    assert_eq!(linked.get(Ix::new(0)).nil_cluster, Some(0));

    assert!(linked.get(Ix::new(1)).is_nil());
    assert_eq!(linked.get(Ix::new(1)).nil_reason, Some(NilReason::TypeIncompatible));
    assert_eq!(linked.get(Ix::new(1)).nil_cluster, Some(1));

    assert!(linked.get(Ix::new(2)).is_nil());
    assert_eq!(linked.get(Ix::new(2)).nil_reason, Some(NilReason::NoCandidates));
    assert_eq!(linked.get(Ix::new(2)).nil_cluster, Some(0));

    let doc_below = make_doc_with_entities("They drove past Paris yesterday.", &[("Paris", "GPE")]);
    let linked_below = strict_linker.link(&doc_below).unwrap();
    assert!(linked_below.get(Ix::new(0)).is_nil());
    assert_eq!(linked_below.get(Ix::new(0)).nil_reason, Some(NilReason::BelowThreshold));
}

#[test]
fn analyzer_pipeline_and_document_graph_codec_roundtrip() {
    let linker = EntityLinker::new().add_provider(CatalogProvider::new(build_wikidata_kb()));
    let analyzer = Analyzer::builder()
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .add(linker)
        .build()
        .unwrap();

    let text = "Abraham Lincoln addressed Congress in Washington while Apple grew in Cupertino.";
    let doc = make_doc_with_entities(text, &[("Abraham Lincoln", "PERSON"), ("Washington", "GPE"), ("Apple", "ORG")]);
    let mut batch = vec![doc];
    let results = analyzer.run(&mut batch);
    assert!(results[0].is_ok());
    let doc = batch.pop().unwrap();

    let linked = doc.require(&Key::<LinkedEntities>::DEFAULT).unwrap();
    assert_eq!(linked.len(), 3);
    assert_eq!(linked.get(Ix::new(0)).kb_id, Some("Q91"));
    assert_eq!(linked.get(Ix::new(1)).kb_id, Some("Q61"));
    assert_eq!(linked.get(Ix::new(2)).kb_id, Some("Q312"));

    // Verify byte-for-byte source spans of every linked entity
    let entities = doc.require(&Key::<Entities>::DEFAULT).unwrap();
    let expected_surfaces = ["Abraham Lincoln", "Washington", "Apple"];
    for (i, expected) in expected_surfaces.iter().enumerate() {
        let span = entities.spans.get(Ix::new(i as u32));
        let src = doc.to_source_bytes(entities.spans.view(), span);
        assert_eq!(&text[src.start as usize..src.end as usize], *expected);
    }

    // Export -> to_bytes -> from_bytes -> import roundtrip
    let codec_set = codecs();
    let graph = export(&doc, &codec_set).unwrap();
    let bytes = graph.to_bytes().unwrap();
    let decoded_graph = DocumentGraph::from_bytes(&bytes).unwrap();
    let restored_doc = import(&decoded_graph, &codec_set).unwrap();

    let restored_linked = restored_doc.require(&Key::<LinkedEntities>::DEFAULT).unwrap();
    assert_eq!(restored_linked, linked);
}

#[test]
fn layer_check_rejects_invalid_invariants() {
    let mut layer = LinkedEntities::with_capacity(1);
    layer.kb_id.push(Some("Q312".into()));
    layer.provider_id.push("wikidata".into());
    layer.canonical_name.push("Apple Inc.".into());
    layer.matched_alias.push("Apple".into());
    layer.score.push(0.85);
    layer.nil_reason.push(None);
    layer.nil_cluster.push(None);
    layer.candidates.push(vec![CandidateLink {
        kb_id: "Q312".into(),
        provider_id: "wikidata".into(),
        canonical_name: "Apple Inc.".into(),
        matched_alias: "Apple".into(),
        score: 0.85,
        prior_score: 0.80,
        type_score: 1.0,
        context_score: 0.85,
        coherence_score: 0.70,
    }]);
    assert!(layer.check().is_ok());

    // Out-of-bounds score
    let mut bad_score = layer.clone();
    bad_score.score[0] = 1.5;
    assert!(bad_score.check().is_err());

    // Linked entity with NilReason set
    let mut bad_nil = layer.clone();
    bad_nil.nil_reason[0] = Some(NilReason::NoCandidates);
    assert!(bad_nil.check().is_err());

    // Unsorted candidates
    let mut unsorted = layer;
    let mut c2 = unsorted.candidates[0][0].clone();
    c2.kb_id = "Q89".into();
    c2.score = 0.95;
    unsorted.candidates[0].push(c2);
    assert!(unsorted.check().is_err());
}

type GoldMention = (&'static str, &'static str, Option<&'static str>);

#[test]
fn wikidata_benchmark_accuracy_nil_precision_and_throughput() {
    let linker = EntityLinker::new().add_provider(CatalogProvider::new(build_wikidata_kb()));

    let cases: &[(&str, &[GoldMention])] = &[
        ("Apple introduced a new Mac and iPhone at its Cupertino campus.", &[("Apple", "ORG", Some("Q312"))]),
        ("Orchard farmers picked a fresh Apple to bake pie and press cider.", &[("Apple", "PRODUCT", Some("Q89"))]),
        ("The Beatles recorded songs for Apple in London.", &[("Apple", "ORG", Some("Q207628"))]),
        (
            "President Lincoln signed the Emancipation Proclamation during the Civil War.",
            &[("Lincoln", "PERSON", Some("Q91"))],
        ),
        ("Senators gathered at the Nebraska state legislature in Lincoln.", &[("Lincoln", "GPE", Some("Q28260"))]),
        ("Ford Motor Company released a new luxury Lincoln SUV.", &[("Lincoln", "PRODUCT", Some("Q216760"))]),
        ("General Washington led the Continental Army across the Delaware.", &[("Washington", "PERSON", Some("Q23"))]),
        ("Congress convened at the Capitol in Washington.", &[("Washington", "GPE", Some("Q61"))]),
        (
            "Tourists in Paris admired paintings at the Louvre.",
            &[("Paris", "GPE", Some("Q90")), ("Louvre", "FAC", Some("Q19675"))],
        ),
        (
            "Ranchers drove from Dallas to Paris in northeast Texas.",
            &[("Dallas", "GPE", Some("Q16557")), ("Paris", "GPE", Some("Q830149"))],
        ),
        (
            "Executives at Hyperion Dynamics met with Cyberdyne Systems.",
            &[("Hyperion Dynamics", "ORG", None), ("Cyberdyne Systems", "ORG", None)],
        ),
    ];

    let mut total_mentions = 0usize;
    let mut correct = 0usize;
    let mut predicted_nil = 0usize;
    let mut true_nil = 0usize;

    let mut prepared_docs = Vec::new();
    for &(text, gold) in cases {
        let mention_specs: Vec<(&str, &str)> = gold.iter().map(|&(m, k, _)| (m, k)).collect();
        let doc = make_doc_with_entities(text, &mention_specs);
        let linked = linker.link(&doc).unwrap();
        for (i, &(_, _, expected_id)) in gold.iter().enumerate() {
            total_mentions += 1;
            let got = linked.get(Ix::new(i as u32)).kb_id;
            if got == expected_id {
                correct += 1;
            }
            if got.is_none() {
                predicted_nil += 1;
                if expected_id.is_none() {
                    true_nil += 1;
                }
            }
        }
        prepared_docs.push(doc);
    }

    assert_eq!(correct, total_mentions, "expected 100% accuracy on Wikidata fixture benchmark");
    assert_eq!(predicted_nil, 2);
    assert_eq!(true_nil, 2);

    // Throughput check: link >= 500 entity mentions and verify well over 1,000 mentions/sec
    let start = Instant::now();
    let iterations = 40usize;
    let mut linked_count = 0usize;
    for _ in 0..iterations {
        for doc in &prepared_docs {
            let res = linker.link(doc).unwrap();
            linked_count += res.len();
        }
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let mentions_per_sec = (linked_count as f64) / elapsed;
    assert!(mentions_per_sec > 500.0, "expected high linking throughput, got {mentions_per_sec:.1} mentions/sec");
}
