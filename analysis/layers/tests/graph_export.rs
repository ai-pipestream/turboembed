//! The shared layers across the document graph boundary (#518): a
//! populated pipeline document round-trips through bytes with its
//! cross-layer indexes, parent trees and discontinuous spans intact,
//! and corrupted payloads (column mismatches, out-of-bounds indexes,
//! parent and head cycles, tags outside their set) are refused with
//! typed errors instead of being smuggled into a document.

use std::sync::Arc;

use turbo_analysis::export::{DocumentGraph, ExportError, Writer, export, import, put_tagset};
use turbo_analysis::{Document, Ix, IxRange, Key, ModelRef, Provenance, Span, Spans, Tag, TagSet, ViewId};
use turbo_analysis_layers::export::codecs;
use turbo_analysis_layers::{
    Aspects, Block, Boundaries, Categories, ChunkTags, Chunks, Confidence, Containment, Currency, Cut, Date,
    Dependencies, DocumentDate, Entities, Granularity, Incivility, IncivilityLabel, IncivilityTarget, Incongruity,
    Language, LanguageSpans, Lemmas, Levels, Matches, Modifier, Money, Multiplier, NumberNotation, Numbers, Opinions,
    OrdinalBase, Ordinals, Origin, PhraseChunks, Phrases, Pii, PiiConfidence, PiiKind, Portion, PosTags, Pragmatics,
    Quantities, RegionCode, Regions, Relation, ScaleMarker, Sentences, Stems, Strs, Structure, Subwords, Temporals,
    Terms, TimeGrain, TimeKind, Times, Tokens, Unit, Utf16Spans, Vectors, WordLemmas, WordPosTags, Words,
};
use turbo_unicode::Script;

const TEXT: &str = "Hello world. Bye now.";

fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn tagset(name: &str, labels: &[&str]) -> Arc<TagSet> {
    TagSet::new(name, labels)
}

/// A document carrying one of most core pipeline layers, added in
/// dependency order.
fn document() -> Document {
    let mut doc = Document::new(TEXT);
    let by = Provenance::by("test");
    doc.add(&Key::DEFAULT, Sentences { spans: spans(&[(0, 12), (13, 21)]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Tokens {
            spans: spans(&[(0, 5), (6, 11), (11, 12), (13, 16), (17, 20), (20, 21)]),
            sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1)]),
        },
        by.clone(),
    )
    .unwrap();
    let mut words = Words::new();
    words.tokens = (0..6).map(|i| IxRange::new(i, i + 1)).collect();
    words.form = Some(vec![Some("hello".into()), None, None, None, None, None]);
    words.subspan = Some(vec![None, Some(Span::new(6, 11)), None, None, None, None]);
    doc.add(&Key::DEFAULT, words, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Subwords {
            spans: spans(&[(0, 3), (3, 5), (6, 11)]),
            ids: vec![10, 11, 12],
            model: ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
            token: Some(vec![Ix::new(0), Ix::new(0), Ix::new(1)]),
        },
        by.clone(),
    )
    .unwrap();
    let pos = tagset("pos", &["NN", "VB", "."]);
    doc.add(
        &Key::DEFAULT,
        PosTags { tagset: Arc::clone(&pos), tags: vec![Tag(0), Tag(0), Tag(2), Tag(1), Tag(0), Tag(2)], prob: None },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        WordPosTags { tagset: pos, tags: vec![Tag(0), Tag(0), Tag(2), Tag(1), Tag(0), Tag(2)], prob: None },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Lemmas { lemma: strs(&["hello", "world", ".", "bye", "now", "."]) }, by.clone()).unwrap();
    doc.add(&Key::DEFAULT, WordLemmas { lemma: strs(&["hello", "world", ".", "bye", "now", "."]) }, by.clone())
        .unwrap();
    doc.add(&Key::DEFAULT, Stems { stem: strs(&["hello", "world", ".", "bye", "now", "."]) }, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Terms {
            surface: strs(&["Hello", "world", ".", "Bye", "now", "."]),
            folded: strs(&["hello", "world", ".", "bye", "now", "."]),
            stem: Some(strs(&["hello", "world", "", "bye", "now", ""])),
            lemma: None,
        },
        by.clone(),
    )
    .unwrap();
    let rels = tagset("rel", &["root", "dep"]);
    doc.add(
        &Key::DEFAULT,
        Dependencies::<Tokens> {
            head: vec![Some(Ix::new(1)), None, Some(Ix::new(1)), Some(Ix::new(4)), None, Some(Ix::new(4))],
            relation: vec![Tag(1), Tag(0), Tag(1), Tag(1), Tag(0), Tag(1)],
            relations: Arc::clone(&rels),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Dependencies::<Words> {
            head: vec![Some(Ix::new(1)), None, Some(Ix::new(1)), Some(Ix::new(4)), None, Some(Ix::new(4))],
            relation: vec![Tag(1), Tag(0), Tag(1), Tag(1), Tag(0), Tag(1)],
            relations: rels,
        },
        by.clone(),
    )
    .unwrap();
    let mut entity_spans = Spans::new(ViewId::SOURCE);
    entity_spans.try_push_fragments(&[Span::new(0, 5), Span::new(17, 20)]).unwrap();
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: entity_spans,
            tokens: vec![IxRange::new(0, 5)],
            kind: vec![Tag(0)],
            kinds: tagset("ent", &["PER"]),
            score: vec![0.5],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        ChunkTags {
            tagset: tagset("chunk", &["B-NP", "I-NP", "O"]),
            tags: vec![Tag(0), Tag(1), Tag(2), Tag(2), Tag(0), Tag(2)],
            prob: Some(vec![0.9; 6]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        PhraseChunks {
            spans: spans(&[(0, 11)]),
            tokens: vec![IxRange::new(0, 2)],
            kind: vec![Tag(0)],
            kinds: tagset("phrase", &["NP"]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Structure {
            spans: spans(&[(0, 21), (0, 12), (13, 21)]),
            kind: vec![Block::Section, Block::Paragraph, Block::Paragraph],
            parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
            depth: vec![1, 0, 0],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Boundaries { cost: vec![1000, -500], reason: vec![Cut::DocumentStart, Cut::Inside(Block::Table)] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: spans(&[(0, 21), (0, 12), (13, 21)]),
            level: vec![1, 0, 0],
            parent: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
            sentences: vec![IxRange::new(0, 2), IxRange::new(0, 1), IxRange::new(1, 2)],
            cut: vec![Cut::DocumentStart, Cut::Sentence, Cut::Paragraph],
            cost: vec![0, 10, -3],
            model_tokens: vec![9, 5, 4],
            over_budget: vec![false, false, true],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Language { code: strs(&["eng"]), score: vec![0.9] }, by.clone()).unwrap();
    let mut lang_spans = LanguageSpans::new(ViewId::SOURCE, tagset("languages", &["eng", "und"]));
    lang_spans.push(Span::new(0, 21), Tag(0), Script::Latin, 0.9);
    doc.add(&Key::DEFAULT, lang_spans, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Temporals {
            spans: spans(&[(13, 16)]),
            granularity: vec![Granularity::Day],
            origin: vec![Origin::Absolute],
            year: vec![2026],
            period: vec![10],
            day: vec![16],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        DocumentDate { date: Some(Date::new(2026, 10, 16).unwrap()), mention: Some(Ix::new(0)) },
        by.clone(),
    )
    .unwrap();
    let mut cats = Categories::<turbo_analysis::View>::new(tagset("cats", &["a", "b"]));
    cats.push(Tag(0), &[0.75, 0.25]);
    doc.add(&Key::DEFAULT, cats, by.clone()).unwrap();
    let mut sent_cats = Categories::<Sentences>::new(tagset("cats", &["a", "b"]));
    sent_cats.push(Tag(0), &[0.75, 0.25]);
    sent_cats.push(Tag(1), &[0.1, 0.9]);
    doc.add(&Key::DEFAULT, sent_cats, by).unwrap();
    doc
}

#[test]
fn core_pipeline_layers_round_trip() {
    let doc = document();
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();

    assert_eq!(back.get::<Tokens>().unwrap().sentence.as_ref().unwrap()[3], Ix::new(1));
    assert_eq!(back.get::<Words>().unwrap().tokens[2], IxRange::new(2, 3));
    let words = back.get::<Words>().unwrap();
    assert_eq!(words.form.as_ref().unwrap()[0].as_deref(), Some("hello"));
    assert_eq!(words.subspan.as_ref().unwrap()[1], Some(Span::new(6, 11)));
    assert_eq!(back.get::<Subwords>().unwrap().ids, [10, 11, 12]);
    assert_eq!(back.get::<Subwords>().unwrap().model.id, "m");
    assert_eq!(back.get::<PosTags>().unwrap().tags[3], Tag(1));
    assert_eq!(back.get::<Lemmas>().unwrap().lemma.get(1), "world");
    assert_eq!(back.get::<Terms>().unwrap().folded.get(0), "hello");
    assert_eq!(back.get::<Dependencies<Tokens>>().unwrap().head[0], Some(Ix::new(1)));
    assert_eq!(back.get::<Dependencies<Words>>().unwrap().head[5], Some(Ix::new(4)));
    let entities = back.get::<Entities>().unwrap();
    assert_eq!(entities.spans.fragments(Ix::new(0)).count(), 2, "the discontinuous entity kept its fragments");
    assert_eq!(entities.tokens[0], IxRange::new(0, 5));
    assert_eq!(back.get::<Structure>().unwrap().parent[2], Some(Ix::new(0)));
    assert_eq!(back.get::<Boundaries>().unwrap().reason[1], Cut::Inside(Block::Table));
    let chunks = back.get::<Chunks>().unwrap();
    assert_eq!(
        (chunks.parent[1], chunks.sentences[2], chunks.over_budget[2]),
        (Some(Ix::new(0)), IxRange::new(1, 2), true)
    );
    assert_eq!(back.get::<Language>().unwrap().best(), Some(("eng", 0.9)));
    assert_eq!(back.get::<LanguageSpans>().unwrap().get(Ix::new(0)).script, Script::Latin);
    assert_eq!(back.get::<Temporals>().unwrap().year, [2026]);
    assert_eq!(back.get::<DocumentDate>().unwrap().mention, Some(Ix::new(0)));
    assert_eq!(back.get::<Categories<turbo_analysis::View>>().unwrap().best_category(0), "a");
    assert_eq!(back.get::<Categories<Sentences>>().unwrap().best_category(1), "b");

    // Re-export is byte stable.
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn sentiment_numeric_and_vector_layers_round_trip() {
    let mut doc = document();
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Opinions {
            tokens: vec![IxRange::new(0, 1), IxRange::new(3, 4)],
            polarity: vec![0.8, -0.4],
            intensity: vec![1.0, 2.0],
            negated: vec![false, true],
            tokens_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Aspects {
            tokens: vec![Some(IxRange::new(0, 1)), None],
            sentence: vec![Some(Ix::new(0)), Some(Ix::new(1))],
            category: vec![Some(Tag(0)), None],
            categories: tagset("aspect-cats", &["food", "service"]),
            positive: vec![0.5, 0.1],
            negative: vec![0.2, 0.6],
            neutral: vec![0.3, 0.3],
            opinions: vec![Ix::new(0)],
            opinion_end: vec![1, 1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            opinions_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Pragmatics {
            score: vec![0.7, 0.1],
            evidence: vec![Some(Incongruity { positive: IxRange::new(0, 1), negative: IxRange::new(1, 2) }), None],
            tokens_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Incivility {
            target: vec![IncivilityTarget::Sentence(Ix::new(0))],
            insult: vec![0.9],
            profanity: vec![0.0],
            threat: vec![0.1],
            identity_attack: vec![0.0],
            intolerance: vec![0.2],
            severe_toxicity: vec![0.3],
            evidence_label: vec![IncivilityLabel::Insult],
            evidence_tokens: vec![IxRange::new(0, 1)],
            evidence_end: vec![1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            chunks_key: Key::DEFAULT,
            structure_key: Key::DEFAULT,
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Numbers {
            spans: spans(&[(0, 5)]),
            numeral_end: vec![5],
            negative: vec![false],
            coefficient: strs(&["42"]),
            scale: vec![0],
            multiplier: vec![Multiplier::One],
            marker: vec![ScaleMarker::None],
            notation: vec![NumberNotation::LatinUs],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Money {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["500"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Ordinals { spans: spans(&[(0, 5)]), base: vec![OrdinalBase::End], offset: vec![-2] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Quantities {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["50"]),
            scale: vec![0],
            unit: strs(&["%"]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Times {
            spans: spans(&[(13, 16)]),
            kind: vec![TimeKind::Date],
            grain: vec![TimeGrain::Day],
            modifier: vec![Modifier { relation: Relation::After, portion: Portion::Start }],
            start: vec![Some(1_800_000_000_000)],
            end: vec![Some(1_800_086_400_000)],
            duration_ms: vec![None],
            ambiguous: vec![true],
            timex: strs(&["2027-01-15"]),
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Pii {
            spans: spans(&[(0, 5)]),
            kind: vec![PiiKind::Email],
            confidence: vec![PiiConfidence::Checksum],
            normalized: strs(&["a@b.c"]),
        },
        by.clone(),
    )
    .unwrap();
    let mut regions = Regions::new();
    regions.push(turbo_analysis_layers::RegionVote::new(RegionCode::parse("US").unwrap(), 0.75).unwrap());
    regions.push(turbo_analysis_layers::RegionVote::new(RegionCode::parse("IE").unwrap(), 0.25).unwrap());
    doc.add(&Key::DEFAULT, regions, by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Matches {
            spans: spans(&[(0, 11)]),
            tokens: vec![IxRange::new(0, 2)],
            label: vec![Tag(0)],
            labels: tagset("match-labels", &["GREET"]),
            rule: vec![7],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Levels { spans: spans(&[(0, 21)]), unit: vec![Unit::Document], parent: vec![None] },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Containment {
            spans: spans(&[(0, 5)]),
            ancestor_id: strs(&["ie", "europe"]),
            ancestor_name: strs(&["Ireland", "Europe"]),
            ancestor_kind: strs(&["country", "continent"]),
            chain_end: vec![2],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Phrases {
            spans: spans(&[(0, 11), (0, 5)]),
            tokens: vec![IxRange::new(0, 2), IxRange::new(0, 1)],
            kind: vec![Tag(0), Tag(0)],
            kinds: tagset("phrase-kinds", &["NP"]),
            head: vec![Ix::new(1), Ix::new(0)],
            parent: vec![None, Some(Ix::new(0))],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            3,
            vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            vec![Ix::new(1)],
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        by.clone(),
    )
    .unwrap();
    doc.add(&Key::DEFAULT, Confidence::<Tokens>::new(vec![0.5; 6]), by.clone()).unwrap();
    doc.add(
        &Key::DEFAULT,
        Utf16Spans::<Tokens>::new(
            vec![0, 6, 11, 13, 17, 20],
            vec![5, 11, 12, 16, 20, 21],
            vec![false, true, false, false, false, false],
        ),
        by,
    )
    .unwrap();

    let mut opinion_cats = Categories::<Opinions>::new(tagset("cats", &["a", "b"]));
    opinion_cats.push(Tag(1), &[0.1, 0.9]);
    opinion_cats.push(Tag(0), &[0.8, 0.2]);
    doc.add(&Key::DEFAULT, opinion_cats, Provenance::by("test")).unwrap();
    let mut aspect_cats = Categories::<Aspects>::new(tagset("cats", &["a", "b"]));
    aspect_cats.push(Tag(0), &[0.6, 0.4]);
    aspect_cats.push(Tag(1), &[0.2, 0.8]);
    doc.add(&Key::DEFAULT, aspect_cats, Provenance::by("test")).unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(back.get::<Categories<Opinions>>().unwrap().best_category(0), "b");
    assert_eq!(back.get::<Categories<Aspects>>().unwrap().best_category(1), "b");
    assert_eq!(back.get::<Opinions>().unwrap().get(Ix::new(1)).effective_polarity(), 0.8);
    assert_eq!(back.get::<Aspects>().unwrap().get(Ix::new(0)).opinions, [Ix::new(0)]);
    assert_eq!(back.get::<Pragmatics>().unwrap().score, [0.7, 0.1]);
    assert_eq!(back.get::<Incivility>().unwrap().target[0], IncivilityTarget::Sentence(Ix::new(0)));
    assert_eq!(back.get::<Numbers>().unwrap().coefficient.get(0), "42");
    assert_eq!(back.get::<Money>().unwrap().currency[0].as_str(), "USD");
    assert_eq!(back.get::<Ordinals>().unwrap().offset, [-2]);
    assert_eq!(back.get::<Quantities>().unwrap().unit.get(0), "%");
    assert_eq!(back.get::<Times>().unwrap().timex.get(0), "2027-01-15");
    assert_eq!(back.get::<Pii>().unwrap().kind, [PiiKind::Email]);
    assert_eq!(back.get::<Regions>().unwrap().share, [0.75, 0.25]);
    assert_eq!(back.get::<Matches>().unwrap().rule, [7]);
    assert_eq!(back.get::<Levels>().unwrap().unit, [Unit::Document]);
    assert_eq!(back.get::<Containment>().unwrap().ancestor_name.get(1), "Europe");
    assert_eq!(back.get::<Phrases>().unwrap().parent[1], Some(Ix::new(0)));
    let vectors = back.get::<Vectors<Sentences>>().unwrap();
    assert_eq!(vectors.row(Ix::new(0)), [1.0, 2.0, 3.0]);
    assert!(vectors.is_empty(Ix::new(1)));
    assert_eq!(back.get::<Confidence<Tokens>>().unwrap().value, [0.5; 6]);
    assert_eq!(back.get::<Utf16Spans<Tokens>>().unwrap().cut, [false, true, false, false, false, false]);
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());
}

#[test]
fn vector_dimension_mismatches_are_refused() {
    let mut doc = document();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            2,
            vec![1.0, 2.0, 3.0, 4.0],
            Vec::new(),
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        Provenance::by("test"),
    )
    .unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    // Claim 5 values for 2 rows of dimension 2.
    let mut w = Writer::new();
    w.u32(2);
    w.usize(2).unwrap();
    w.usize(5).unwrap();
    for _ in 0..5 {
        w.u32(0);
    }
    w.usize(0).unwrap();
    w.str("m").unwrap();
    w.str("r").unwrap();
    w.str("s").unwrap();
    splice(&mut graph, "opennlp:embeddings", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("vector values"), "{detail}"),
        Ok(_) => panic!("a vector block with a value count mismatch imported"),
        Err(other) => panic!("a vector block mismatch failed as {other}"),
    }
}

#[test]
fn sentiment_family_refusals() {
    let mut doc = document();
    let by = Provenance::by("test");
    doc.add(
        &Key::DEFAULT,
        Money {
            spans: spans(&[(0, 5)]),
            negative: vec![false],
            coefficient: strs(&["500"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Numbers {
            spans: spans(&[(0, 5)]),
            numeral_end: vec![5],
            negative: vec![false],
            coefficient: strs(&["42"]),
            scale: vec![0],
            multiplier: vec![Multiplier::One],
            marker: vec![ScaleMarker::None],
            notation: vec![NumberNotation::LatinUs],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Phrases {
            spans: spans(&[(0, 5)]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: tagset("phrase-kinds", &["NP"]),
            head: vec![Ix::new(0)],
            parent: vec![None],
        },
        by.clone(),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Vectors::<Sentences>::new(
            2,
            vec![0.0; 4],
            vec![],
            ModelRef { id: "m".into(), revision: "r".into(), sha256: "s".into() },
        ),
        by,
    )
    .unwrap();

    // A twelve byte vector body claiming billions of values must be
    // a typed refusal, not an allocation.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.u32(1);
    w.usize(u32::MAX as usize).unwrap();
    w.usize(u32::MAX as usize).unwrap();
    splice(&mut graph, "opennlp:embeddings", w.finish(), u32::MAX);
    assert!(import(&graph, &codecs()).is_err(), "a vector body claiming billions of values imported");

    // An extreme number span must be refused, never panicking in the
    // layer's own checks.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(u32::MAX, u32::MAX)]);
    splice(&mut graph, "turbo:numbers", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a number at the end of the address space imported");

    // A phrase head outside its token range.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.str("opennlp:tokens").unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.usize(1).unwrap();
    w.str("NP").unwrap();
    w.usize(1).unwrap();
    w.u16(0);
    w.usize(1).unwrap();
    w.u32(5);
    w.usize(1).unwrap();
    w.u8(0);
    splice(&mut graph, "opennlp:phrases", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a phrase head outside its tokens imported");

    // An unknown currency code.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.usize(1).unwrap();
    w.u8(0);
    w.usize(1).unwrap();
    w.str("1").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.usize(1).unwrap();
    w.str("not-a-currency").unwrap();
    splice(&mut graph, "turbo:money", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "an unknown currency code imported");
}

/// The span column encoding, written out longhand so corrupted
/// payloads can be built without an export context.
fn write_spans(w: &mut Writer, view: &str, spans: &[Span]) {
    w.str(view).unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

fn token_body(instance: &str, sentence: &[u32]) -> Vec<u8> {
    let mut w = Writer::new();
    let spans: Vec<Span> = sentence.iter().enumerate().map(|(i, _)| Span::new(i as u32, i as u32 + 1)).collect();
    write_spans(&mut w, "", &spans);
    w.u8(1);
    w.str("opennlp:sentences").unwrap();
    w.str(instance).unwrap();
    w.usize(sentence.len()).unwrap();
    for s in sentence {
        w.u32(*s);
    }
    w.finish()
}

#[test]
fn layers_on_a_derived_view_round_trip() {
    use turbo_analysis::{Alignment, View, ViewBuilder};
    let mut doc = Document::decoded(
        b"Caf\xe9 moo.".to_vec(),
        "Café moo.",
        Alignment::from_edits(vec![turbo_analysis::text::Edit { parent: Span::new(3, 4), child: Span::new(3, 5) }])
            .unwrap(),
    )
    .unwrap();
    let source = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(source.text());
    builder.keep(5);
    builder.replace(6, "");
    let (clean, alignment) = builder.finish();
    doc.add(&Key::<View>::named("clean"), View::derived(&source, clean, alignment), Provenance::by("test")).unwrap();
    let clean_id = doc.get_key(&Key::<View>::named("clean")).unwrap().id();
    let mut sentences = Spans::new(clean_id);
    sentences.push(Span::new(0, 9));
    doc.add(&Key::DEFAULT, Sentences { spans: sentences }, Provenance::by("test")).unwrap();
    let mut tokens = Spans::new(clean_id);
    tokens.push(Span::new(0, 5));
    tokens.push(Span::new(5, 8));
    tokens.push(Span::new(8, 9));
    doc.add(&Key::DEFAULT, Tokens { spans: tokens, sentence: Some(vec![Ix::new(0); 3]) }, Provenance::by("test"))
        .unwrap();

    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    let tokens = back.get::<Tokens>().unwrap();
    assert_eq!(tokens.spans.view(), back.get_key(&Key::<View>::named("clean")).unwrap().id());
    assert_eq!(back.text(tokens.spans.view()), "Cafémoo.");
    let in_source = back.to_source(tokens.spans.view(), tokens.spans.get(Ix::new(1)));
    assert_eq!(&back.source()[in_source.range()], "moo");
}

#[test]
fn target_identity_and_word_subspan_refusals() {
    // A word subspan split across a UTF-8 character must not import.
    let mut doc = Document::new("é");
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 2)]), sentence: None }, Provenance::by("test")).unwrap();
    let mut words = Words::new();
    words.push_word(IxRange::new(0, 1), None, Some(Span::new(0, 2)));
    doc.add(&Key::DEFAULT, words, Provenance::by("test")).unwrap();
    let mut graph = export(&doc, &codecs()).unwrap();
    let body = graph.layers.iter_mut().find(|l| l.layer == "turbo:words").unwrap().body.as_mut().unwrap();
    let at = body.len() - 4;
    body[at..].copy_from_slice(&1u32.to_le_bytes());
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("boundaries"), "{detail}"),
        Ok(_) => panic!("a subspan splitting a character imported"),
        Err(other) => panic!("a split subspan failed as {other}"),
    }

    // An encoded target instance the provenance does not bind to.
    let doc = document();
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("alt", &[0]), 1);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("a payload naming another target instance imported"),
        Err(other) => panic!("a rebound target failed as {other}"),
    }

    // An empty sentence column naming a missing target.
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("missing", &[]), 0);
    assert!(import(&graph, &codecs()).is_err(), "an empty column naming a missing target imported");

    // A token pointing at a sentence past the sentence layer.
    let mut graph = export(&doc, &codecs()).unwrap();
    splice(&mut graph, "opennlp:tokens", token_body("", &[9]), 1);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::DanglingEndpoint { .. })));

    // A structure block that is its own parent.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    write_spans(&mut w, "", &[Span::new(0, 5)]);
    w.usize(1).unwrap();
    w.u8(2);
    w.usize(1).unwrap();
    w.u8(1);
    w.u32(0);
    w.usize(1).unwrap();
    w.u8(0);
    splice(&mut graph, "turbo:structure", w.finish(), 1);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::Malformed(_))));

    // A document date naming a temporal mention past the temporals.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.u8(0);
    w.u8(1);
    w.str("turbo:temporal").unwrap();
    w.str("").unwrap();
    w.u32(9);
    splice(&mut graph, "turbo:document-date", w.finish(), 1);
    assert!(import(&graph, &codecs()).is_err(), "a document date naming a missing mention imported");
}

#[test]
fn keyed_default_word_target_must_not_rebind_through_provenance() {
    let mut doc = Document::new("ab");
    doc.add(&Key::DEFAULT, Tokens { spans: spans(&[(0, 1)]), sentence: None }, Provenance::by("test")).unwrap();
    doc.add(&Key::named("alt"), Tokens { spans: spans(&[(1, 2)]), sentence: None }, Provenance::by("test")).unwrap();
    let mut words = Words::on(Key::named("alt"));
    words.push_word(IxRange::new(0, 1), None, None);
    let mut provenance = Provenance::by("test");
    provenance.reads = vec![Key::<Tokens>::named("alt").id()];
    doc.add(&Key::DEFAULT, words, provenance).unwrap();

    // The valid document, with its explicit alt key, round-trips.
    let graph = export(&doc, &codecs()).unwrap();
    let back = import(&DocumentGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(), &codecs()).unwrap();
    assert_eq!(export(&back, &codecs()).unwrap().to_bytes().unwrap(), graph.to_bytes().unwrap());

    // The same body with only the encoded target instance emptied
    // would validate against the default tokens and then rebind to
    // alt through provenance, so it must be refused.
    let mut graph = export(&doc, &codecs()).unwrap();
    let mut w = Writer::new();
    w.str("opennlp:tokens").unwrap();
    w.str("").unwrap();
    w.usize(1).unwrap();
    w.u32(0);
    w.u32(1);
    w.u8(0);
    w.u8(0);
    splice(&mut graph, "turbo:words", w.finish(), 1);
    match import(&graph, &codecs()) {
        Err(ExportError::InvalidLayer { detail, .. }) => assert!(detail.contains("alt"), "{detail}"),
        Ok(_) => panic!("an empty keyed target rebound through provenance imported"),
        Err(other) => panic!("an empty keyed target failed as {other}"),
    }
}

fn splice(graph: &mut DocumentGraph, layer_prefix: &str, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer.starts_with(layer_prefix)).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}

#[test]
fn corrupted_payloads_are_refused() {
    let base = export(&document(), &codecs()).unwrap();

    // A dependency head cycle, valid column lengths or not.
    let mut graph = base.clone();
    let mut w = Writer::new();
    put_tagset(&mut w, &tagset("rel", &["root", "dep"])).unwrap();
    w.usize(2).unwrap();
    w.u8(1);
    w.u32(1);
    w.u8(1);
    w.u32(0);
    w.usize(2).unwrap();
    w.u16(1);
    w.u16(1);
    splice(&mut graph, "opennlp:dependencies", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("cycle"), "{detail}"),
        Ok(_) => panic!("a dependency cycle imported"),
        Err(other) => panic!("a dependency cycle failed as {other}"),
    }

    // A tag outside its decoded set.
    let mut graph = base.clone();
    let mut w = Writer::new();
    put_tagset(&mut w, &tagset("pos", &["NN"])).unwrap();
    w.usize(6).unwrap();
    for _ in 0..6 {
        w.u16(9);
    }
    w.u8(0);
    splice(&mut graph, "opennlp:pos", w.finish(), 6);
    assert!(matches!(import(&graph, &codecs()).err(), Some(ExportError::Malformed(_))));

    // A terms column shorter than the surface column.
    let mut graph = base.clone();
    let mut w = Writer::new();
    for col in [2usize, 1] {
        w.usize(col).unwrap();
        for i in 0..col {
            w.str(&format!("t{i}")).unwrap();
        }
    }
    w.u8(0);
    w.u8(0);
    splice(&mut graph, "turbo:terms", w.finish(), 2);
    match import(&graph, &codecs()) {
        Err(ExportError::Malformed(detail)) => assert!(detail.contains("folded"), "{detail}"),
        Ok(_) => panic!("a short folded column imported"),
        Err(other) => panic!("a short folded column failed as {other}"),
    }
}
