//! Integration tests for Issue #692:
//! - Syntactic clause splitting (`Cut::Clause`) and token-window fallback for
//!   over-budget sentences (`ClauseCuts`, `ClauseBoundaries`, `Chunker`).
//! - Coreference chain boundary costs (`CorefCohesion`, `CorefBoundaries`) and
//!   per-chunk antecedent context headers (`ChunkAntecedents`, `ChunkAntecedentBuilder`).
//! - Distilled linear boundary scorer (`DistilledBoundaryScorer`, `distilled_boundaries`)
//!   and pipeline registry providers (`ClauseCutProvider`, `CorefBoundariesProvider`,
//!   `DistilledBoundariesProvider`, `WeighProvider`).

use turbo_analysis::registry::{Provider, Spec};
use turbo_analysis::{
    Analyzer, Bound, Document, Edge, Edges, Groups, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet,
    View, ViewId,
};
use turbo_analysis_layers::{
    Boundaries, Chunks, Cut, Dependencies, Entities, PhraseChunks, Phrases, Sentences, Subwords, Tokens,
};
use turbo_chunker::{
    Budget, ChunkAntecedentBuilder, ChunkAntecedents, Chunker, ClauseBoundaries, ClauseCosts, ClauseCutProvider,
    ClauseCuts, CorefBoundaries, CorefBoundariesProvider, CorefCohesion, Costs, DistilledBoundariesProvider,
    DistilledBoundaryScorer, DistilledWeights, MentionKind, Mentions, STRUCTURE, StructureBoundaries,
    StructureBoundariesProvider, WeighProvider, Weight, weigh_with_coref,
};

fn tokenize_words_and_punct(text: &str) -> (Tokens, Subwords) {
    let mut tok_spans = Spans::new(ViewId::SOURCE);
    let mut sw_spans = Spans::new(ViewId::SOURCE);
    let mut sw_ids = Vec::new();
    let bytes = text.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_whitespace() {
            i += 1;
            continue;
        }
        let start = i as u32;
        if bytes[i].is_ascii_punctuation() {
            i += 1;
        } else {
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && !bytes[i].is_ascii_punctuation() {
                i += 1;
            }
        }
        let end = i as u32;
        tok_spans.push(Span::new(start, end));
        sw_spans.push(Span::new(start, end));
        sw_ids.push(1);
    }
    (
        Tokens { spans: tok_spans, sentence: None },
        Subwords {
            spans: sw_spans,
            ids: sw_ids,
            model: ModelRef { id: "test".into(), revision: "0".into(), sha256: "0".into() },
            token: None,
        },
    )
}

fn single_sentence(text: &str) -> Sentences {
    let mut spans = Spans::new(ViewId::SOURCE);
    spans.push(Span::new(0, text.len() as u32));
    Sentences { spans }
}

#[test]
fn over_budget_sentence_splits_at_dependency_clause_boundaries() {
    // 11 tokens (including comma and period). Budget max = 7 tokens forces the
    // single over-budget sentence to split at the coordinate clause boundary
    // before "and" ("Alpha beta gamma delta, | and epsilon zeta eta theta.").
    let text = "Alpha beta gamma delta, and epsilon zeta eta theta.";
    let mut doc = Document::new(text);
    let (tokens, subwords) = tokenize_words_and_punct(text);
    assert_eq!(tokens.spans.len(), 11);

    let rels = TagSet::new("ud-rel", &["root", "compound", "punct", "cc", "conj"]);
    let h = |i: u32| Some(Ix::new(i));
    let deps = Dependencies::<Tokens> {
        head: vec![None, h(0), h(0), h(0), h(0), h(6), h(0), h(6), h(6), h(6), h(0)],
        relation: vec![Tag(0), Tag(1), Tag(1), Tag(1), Tag(2), Tag(3), Tag(4), Tag(1), Tag(1), Tag(1), Tag(2)],
        relations: rels,
    };

    doc.add(&Key::DEFAULT, single_sentence(text), Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, subwords, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, deps, Provenance::by("t")).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Dependencies<Tokens>>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(ClauseBoundaries::default())
        .add(Chunker { budget: Budget { max: 7, target: 5, size_weight: 100, overhead: 0 } })
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    let chunks = doc.get::<Chunks>().unwrap();
    assert_eq!(chunks.spans.len(), 2, "over-budget sentence should split into 2 clause chunks");
    assert_eq!(chunks.cut, vec![Cut::DocumentStart, Cut::Clause]);
    assert_eq!(chunks.over_budget, vec![false, false]);

    let first_text = &text[chunks.spans.get(Ix::new(0)).range()];
    let second_text = &text[chunks.spans.get(Ix::new(1)).range()];
    assert_eq!(first_text, "Alpha beta gamma delta,");
    assert_eq!(second_text, "and epsilon zeta eta theta.");
}

#[test]
fn over_budget_sentence_splits_at_constituency_and_phrase_boundaries_and_falls_back_to_tokens() {
    let text = "One two three four five six seven eight nine ten.";
    let mut doc = Document::new(text);
    let (tokens, subwords) = tokenize_words_and_punct(text);

    // Constituency SBAR covering "six seven eight nine ten." (tokens 5..11)
    let sbar_start = text.find("six").unwrap() as u32;
    let tags = TagSet::new("ptb", &["S", "SBAR"]);
    let mut phrase_spans = Spans::new(ViewId::SOURCE);
    phrase_spans.push(Span::new(0, text.len() as u32));
    phrase_spans.push(Span::new(sbar_start, text.len() as u32));
    let phrases = Phrases {
        spans: phrase_spans,
        tokens: vec![IxRange::new(0, 11), IxRange::new(5, 11)],
        kind: vec![Tag(0), Tag(1)],
        kinds: tags,
        head: vec![Ix::new(0), Ix::new(5)],
        parent: vec![None, Some(Ix::new(0))],
    };

    doc.add(&Key::DEFAULT, single_sentence(text), Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, subwords, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, phrases, Provenance::by("t")).unwrap();

    // Stage 1: budget max = 6 fits each of the 2 clauses (5 tokens and 6 tokens).
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Phrases>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(ClauseBoundaries { costs: ClauseCosts::default() })
        .add(Chunker { budget: Budget { max: 6, target: 5, size_weight: 100, overhead: 0 } })
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    let chunks = doc.get::<Chunks>().unwrap();
    assert_eq!(chunks.spans.len(), 2);
    assert_eq!(chunks.cut, vec![Cut::DocumentStart, Cut::Clause]);
    assert_eq!(&text[chunks.spans.get(Ix::new(0)).range()], "One two three four five");
    assert_eq!(&text[chunks.spans.get(Ix::new(1)).range()], "six seven eight nine ten.");

    // Stage 2: budget max = 3 forces each clause to fall back to token windows!
    let mut doc_small = Document::new(text);
    let (tokens2, subwords2) = tokenize_words_and_punct(text);
    let chunk_tags = TagSet::new("chunk", &["NP", "VP"]);
    let mut pc_spans = Spans::new(ViewId::SOURCE);
    pc_spans.push(Span::new(0, text.find("three").unwrap() as u32 - 1));
    pc_spans.push(Span::new(text.find("three").unwrap() as u32, text.len() as u32));
    let phrase_chunks = PhraseChunks {
        spans: pc_spans,
        tokens: vec![IxRange::new(0, 2), IxRange::new(2, 11)],
        kind: vec![Tag(0), Tag(1)],
        kinds: chunk_tags,
    };
    doc_small.add(&Key::DEFAULT, single_sentence(text), Provenance::by("t")).unwrap();
    doc_small.add(&Key::DEFAULT, tokens2, Provenance::by("t")).unwrap();
    doc_small.add(&Key::DEFAULT, subwords2, Provenance::by("t")).unwrap();
    doc_small.add(&Key::DEFAULT, phrase_chunks, Provenance::by("t")).unwrap();

    let fallback_analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<PhraseChunks>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(ClauseBoundaries::default())
        .add(Chunker { budget: Budget { max: 3, target: 3, size_weight: 50, overhead: 0 } })
        .build()
        .unwrap();
    fallback_analyzer.run(std::slice::from_mut(&mut doc_small))[0].as_ref().unwrap();

    let fb_chunks = doc_small.get::<Chunks>().unwrap();
    assert!(fb_chunks.spans.len() >= 4, "expected token-window fallback to produce >= 4 chunks");
    assert_eq!(fb_chunks.cut[0], Cut::DocumentStart);
    assert!(fb_chunks.cut[1..].iter().all(|&c| c == Cut::Clause));
    assert!(fb_chunks.over_budget.iter().all(|&ob| !ob), "all token-window pieces fit within max=3");
}

#[test]
fn coref_boundaries_penalize_chain_cuts_and_reward_clean_topic_shifts() {
    // 4 sentences:
    // S0: "Marie Curie moved to Paris."
    // S1: "She studied radioactivity there." (coreferent with Marie Curie via pronoun "She")
    // S2: "Albert Einstein worked in Bern." (clean break between S1 and S2!)
    // S3: "He published relativity papers." (coreferent with Albert Einstein via pronoun "He")
    let s0 = "Marie Curie moved to Paris.";
    let s1 = "She studied radioactivity there.";
    let s2 = "Albert Einstein worked in Bern.";
    let s3 = "He published relativity papers.";
    let text = format!("{s0} {s1} {s2} {s3}");

    let mut doc = Document::new(text.clone());
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut offset = 0u32;
    for s in [s0, s1, s2, s3] {
        let len = s.len() as u32;
        sent_spans.push(Span::new(offset, offset + len));
        offset += len + 1;
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("t")).unwrap();
    let (_, subwords) = tokenize_words_and_punct(&text);
    doc.add(&Key::DEFAULT, subwords, Provenance::by("t")).unwrap();
    let (tokens, _) = tokenize_words_and_punct(&text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();

    // Mentions:
    // 0: "Marie Curie" in S0 (Proper)
    // 1: "She" in S1 (Pronominal)
    // 2: "Albert Einstein" in S2 (Proper)
    // 3: "He" in S3 (Pronominal)
    let mut mentions = Mentions::new(ViewId::SOURCE);
    let mc_start = text.find("Marie Curie").unwrap() as u32;
    mentions.push(Span::new(mc_start, mc_start + 11), Ix::new(0), IxRange::new(0, 2), Ix::new(1), MentionKind::Proper);
    let she_start = text.find("She").unwrap() as u32;
    mentions.push(
        Span::new(she_start, she_start + 3),
        Ix::new(1),
        IxRange::new(6, 7),
        Ix::new(6),
        MentionKind::Pronominal,
    );
    let ae_start = text.find("Albert Einstein").unwrap() as u32;
    mentions.push(
        Span::new(ae_start, ae_start + 15),
        Ix::new(2),
        IxRange::new(11, 13),
        Ix::new(12),
        MentionKind::Proper,
    );
    let he_start = text.find("He").unwrap() as u32;
    mentions.push(
        Span::new(he_start, he_start + 2),
        Ix::new(3),
        IxRange::new(17, 18),
        Ix::new(17),
        MentionKind::Pronominal,
    );

    let groups = Groups::<Mentions>::new([[Ix::new(0), Ix::new(1)], [Ix::new(2), Ix::new(3)]]);
    let coref_kinds = TagSet::new("coref", &["antecedent"]);
    let edges = Edges::<Mentions, Mentions>::between(
        Key::DEFAULT,
        Key::DEFAULT,
        coref_kinds,
        [
            Edge { from: Ix::new(1), to: Ix::new(0), kind: Tag(0), score: 0.95 },
            Edge { from: Ix::new(3), to: Ix::new(2), kind: Tag(0), score: 0.95 },
        ],
    );

    let on_mentions = || Provenance { reads: vec![Key::<Mentions>::DEFAULT.id()], ..Provenance::by("t") };
    doc.add(&Key::DEFAULT, mentions, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, groups, on_mentions()).unwrap();
    doc.add(&Key::DEFAULT, edges, on_mentions()).unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Mentions>::DEFAULT)
        .given(Key::<Groups<Mentions>>::DEFAULT)
        .given(Key::<Edges<Mentions, Mentions>>::DEFAULT)
        .step(Bound::new(StructureBoundaries { costs: Costs::default() }).write(0, STRUCTURE))
        .step(CorefBoundaries { cohesion: CorefCohesion::default() }.bound())
        .step(
            DistilledBoundaryScorer {
                weights: DistilledWeights {
                    structure: Weight::ONE,
                    drift: Weight(0),
                    sentiment: Weight(0),
                    coref: Weight::ONE,
                    lexical_overlap: 0,
                    entity_continuity: 0,
                    bias: 0,
                },
            }
            .bound(),
        )
        .add(Chunker { budget: Budget { max: 12, target: 8, size_weight: 50, overhead: 0 } })
        .add(ChunkAntecedentBuilder)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    let coref_b = doc.get_key(&Key::<Boundaries>::named("coref")).unwrap();
    // Gap 0 (between S0 and S1) and Gap 2 (between S2 and S3) cross active pronominal chains: positive cost!
    assert!(coref_b.cost[0] > 0, "expected positive coref cost at gap 0, got {}", coref_b.cost[0]);
    assert!(coref_b.cost[2] > 0, "expected positive coref cost at gap 2, got {}", coref_b.cost[2]);
    // Gap 1 (between S1 and S2) has zero active chains crossing between two entity clusters: negative reward + TopicShift!
    assert!(coref_b.cost[1] < 0, "expected negative break reward at gap 1, got {}", coref_b.cost[1]);
    assert_eq!(coref_b.reason[1], Cut::TopicShift);

    // Chunker should cut cleanly between S1 and S2 ([S0, S1] and [S2, S3])!
    let chunks = doc.get::<Chunks>().unwrap();
    assert_eq!(chunks.spans.len(), 2);
    assert_eq!(chunks.sentences[0].start.get()..chunks.sentences[0].end.get(), 0..2);
    assert_eq!(chunks.sentences[1].start.get()..chunks.sentences[1].end.get(), 2..4);
    assert_eq!(chunks.cut[1], Cut::TopicShift);

    // Since no chunk severed a coref chain, ChunkAntecedents has empty headers for both chunks.
    let ant = doc.get::<ChunkAntecedents>().unwrap();
    assert!(ant.headers[0].is_empty());
    assert!(ant.headers[1].is_empty());
}

#[test]
fn forced_cut_across_coref_chain_attaches_antecedent_context_header() {
    let s0 = "Marie Curie discovered polonium in Paris.";
    let s1 = "She later isolated pure radium in her laboratory.";
    let text = format!("{s0} {s1}");

    let mut doc = Document::new(text.clone());
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    sent_spans.push(Span::new(0, s0.len() as u32));
    sent_spans.push(Span::new(s0.len() as u32 + 1, text.len() as u32));
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("t")).unwrap();
    let (tokens, subwords) = tokenize_words_and_punct(&text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, subwords, Provenance::by("t")).unwrap();

    let mut mentions = Mentions::new(ViewId::SOURCE);
    mentions.push(Span::new(0, 11), Ix::new(0), IxRange::new(0, 2), Ix::new(1), MentionKind::Proper);
    let she_start = text.find("She").unwrap() as u32;
    mentions.push(
        Span::new(she_start, she_start + 3),
        Ix::new(1),
        IxRange::new(7, 8),
        Ix::new(7),
        MentionKind::Pronominal,
    );
    let groups = Groups::<Mentions>::new([[Ix::new(0), Ix::new(1)]]);

    let on_mentions = Provenance { reads: vec![Key::<Mentions>::DEFAULT.id()], ..Provenance::by("t") };
    doc.add(&Key::DEFAULT, mentions, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, groups, on_mentions).unwrap();

    // Budget max = 9 forces each sentence into its own chunk despite the coref chain.
    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Mentions>::DEFAULT)
        .given(Key::<Groups<Mentions>>::DEFAULT)
        .add(StructureBoundaries { costs: Costs::default() })
        .add(Chunker { budget: Budget { max: 9, target: 7, size_weight: 100, overhead: 0 } })
        .add(ChunkAntecedentBuilder)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    let chunks = doc.get::<Chunks>().unwrap();
    assert_eq!(chunks.spans.len(), 2);

    let ant = doc.get::<ChunkAntecedents>().unwrap();
    assert!(ant.headers[0].is_empty());
    assert_eq!(ant.headers[1], "[Context: Marie Curie (She)]");
    assert_eq!(ant.links_of(Ix::new(1)).len(), 1);
    assert_eq!(ant.links_of(Ix::new(1))[0].antecedent_sentence, Ix::new(0));
}

#[test]
fn distilled_scorer_and_registry_providers_combine_all_signals() {
    let s0 = "Quantum computing uses superconducting qubits for fast gates.";
    let s1 = "Superconducting qubits require cryogenic dilution refrigerators.";
    let s2 = "Classical databases index relational tables on disk.";
    let text = format!("{s0} {s1} {s2}");

    let mut doc = Document::new(text.clone());
    let mut sent_spans = Spans::new(ViewId::SOURCE);
    let mut offset = 0u32;
    for s in [s0, s1, s2] {
        let len = s.len() as u32;
        sent_spans.push(Span::new(offset, offset + len));
        offset += len + 1;
    }
    doc.add(&Key::DEFAULT, Sentences { spans: sent_spans }, Provenance::by("t")).unwrap();
    let (tokens, subwords) = tokenize_words_and_punct(&text);
    doc.add(&Key::DEFAULT, tokens, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, subwords, Provenance::by("t")).unwrap();

    // Add shared entity "qubits" in S0 and S1.
    let ent_tags = TagSet::new("ner", &["TECH"]);
    let mut ent_spans = Spans::new(ViewId::SOURCE);
    let q0 = text.find("qubits").unwrap() as u32;
    let q1 = text[q0 as usize + 6..].find("qubits").unwrap() as u32 + q0 + 6;
    ent_spans.push(Span::new(q0, q0 + 6));
    ent_spans.push(Span::new(q1, q1 + 6));
    let entities = Entities {
        spans: ent_spans,
        tokens: vec![IxRange::new(4, 5), IxRange::new(9, 10)],
        kind: vec![Tag(0), Tag(0)],
        kinds: ent_tags,
        score: vec![0.99, 0.99],
    };
    let ent_groups = Groups::<Entities>::new([[Ix::new(0), Ix::new(1)]]);
    let on_entities = Provenance { reads: vec![Key::<Entities>::DEFAULT.id()], ..Provenance::by("t") };
    doc.add(&Key::DEFAULT, entities, Provenance::by("t")).unwrap();
    doc.add(&Key::DEFAULT, ent_groups, on_entities).unwrap();
    let mut doc2 = doc.clone();

    // Build steps through the registry providers.
    let s_step = StructureBoundariesProvider.create(&Spec::default().with("instance", "structure")).unwrap();
    let c_step = CorefBoundariesProvider.create(&Spec::default().with("chain_penalty", "700")).unwrap();
    let cl_step = ClauseCutProvider.create(&Spec::default().with("clause", "250")).unwrap();
    let d_step = DistilledBoundariesProvider
        .create(
            &Spec::default()
                .with("structure", "1000")
                .with("coref", "1000")
                .with("lexical", "500")
                .with("entity", "400"),
        )
        .unwrap();

    let analyzer = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<Groups<Entities>>::DEFAULT)
        .boxed(s_step)
        .boxed(c_step)
        .boxed(cl_step)
        .boxed(d_step)
        .build()
        .unwrap();
    analyzer.run(std::slice::from_mut(&mut doc))[0].as_ref().unwrap();

    assert!(doc.get::<ClauseCuts>().is_some());
    let bounds = doc.get::<Boundaries>().unwrap();
    // Gap 0 shares "superconducting", "qubits", and entity "qubits" + coref group -> higher cost than Gap 1!
    assert!(bounds.cost[0] > bounds.cost[1], "expected gap 0 ({}) > gap 1 ({})", bounds.cost[0], bounds.cost[1]);

    // Verify WeighProvider with coref option also runs cleanly and matches weigh_with_coref.
    let w_step = WeighProvider.create(&Spec::default().with("structure", "1000").with("coref", "1000")).unwrap();
    let s_step2 = StructureBoundariesProvider.create(&Spec::default().with("instance", "structure")).unwrap();
    let c_step2 = CorefBoundariesProvider.create(&Spec::default().with("chain_penalty", "700")).unwrap();
    let analyzer2 = Analyzer::builder()
        .given(Key::<View>::DEFAULT)
        .given(Key::<Sentences>::DEFAULT)
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Subwords>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .given(Key::<Groups<Entities>>::DEFAULT)
        .boxed(s_step2)
        .boxed(c_step2)
        .boxed(w_step)
        .build()
        .unwrap();
    analyzer2.run(std::slice::from_mut(&mut doc2))[0].as_ref().unwrap();

    let weighed = doc2.get::<Boundaries>().unwrap();
    let st_b = doc2.get_key(&Key::<Boundaries>::named("structure")).unwrap();
    let cf_b = doc2.get_key(&Key::<Boundaries>::named("coref")).unwrap();
    let zero_b = Boundaries { cost: vec![0; 3], reason: vec![Cut::Sentence; 3] };
    let direct =
        weigh_with_coref(st_b, &zero_b, &zero_b, cf_b, (Weight::ONE, Weight(0), Weight(0), Weight::ONE)).unwrap();
    assert_eq!(weighed.cost, direct.cost);
    assert_eq!(weighed.reason, direct.reason);
}
