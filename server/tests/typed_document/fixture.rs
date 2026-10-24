//! Rich multi-layer `Document` fixture for `typed_document` integration tests.

use std::sync::Arc;

use turbo_analysis::{
    Analyzer, Document, Edge, Edges, Evidence, EvidenceStep, Groups, Ix, IxRange, Key, Layer, ModelRef, Provenance,
    Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId,
};
use turbo_analysis_layers::{
    Aspects, Block, Boilerplate, BoilerplateKind, Boundaries, Categories, ChunkTags, Chunks, Confidence, Containment,
    Currency, Cut, Date, Dependencies, DocumentDate, Entities, Granularity, Incivility, IncivilityLabel,
    IncivilityTarget, Incongruity, Language, Lemmas, Levels, Matches, Modifier, Money, MorphFeatures, Multiplier,
    NumberNotation, Numbers, Opinions, OrdinalBase, Ordinals, Origin, PhraseChunks, Phrases, Pii, PiiConfidence,
    PiiKind, Portion, PosTags, Pragmatics, Quantities, RegionCode, Regions, Relation, ScaleMarker, Sentences, Stems,
    Strs, Structure, Subwords, Temporals, Terms, TimeGrain, TimeKind, Times, Tokens, Unit, Utf16Spans, Vectors,
    WordLemmas, WordPosTags, Words,
};
use turbo_encoding_layers::{Encoding, EncodingEvidence, SourceEncoding};
use turbo_normalize_layers::{
    ArtifactKind, Artifacts, Corrections, Emoji, EmojiKind, Noise, NoiseSeverity, PlaceholderKind, Placeholders,
    ProtectedSpans,
};

pub const TEXT: &str = "Café \u{1f600} 北京 costs $42.50 on 2026-05-01 at 14:30.";

pub fn byte_span(needle: &str) -> Span {
    let at = TEXT.find(needle).unwrap();
    Span::new(at as u32, (at + needle.len()) as u32)
}

pub fn spans<L>(items: &[Span]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &sp in items {
        s.push(sp);
    }
    s
}

pub fn java_utf16(byte: u32) -> u32 {
    TEXT[..byte as usize].encode_utf16().count() as u32
}

pub fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

pub fn tagset(name: &str, labels: &[&str]) -> Arc<TagSet> {
    TagSet::new(name, labels)
}

pub fn model_ref(id: &str) -> ModelRef {
    ModelRef { id: id.into(), revision: "v1".into(), sha256: "abc123".into() }
}

pub fn add<L: Layer>(doc: &mut Document, layer: L, prov: Provenance) {
    doc.add(&Key::DEFAULT, layer, prov).unwrap();
}

pub fn build_rich_document() -> Document {
    let mut doc = Document::new(TEXT);
    let whole = Span::new(0, TEXT.len() as u32);
    let (cafe, emoji, beijing) = (byte_span("Café"), byte_span("\u{1f600}"), byte_span("北京"));
    let (price, date_span, time_span) = (byte_span("$42.50"), byte_span("2026-05-01"), byte_span("14:30"));

    let source = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(source.text());
    vb.keep(TEXT.len());
    let (clean_text, alignment) = vb.finish();
    doc.add(&Key::<View>::named("clean"), View::derived(&source, clean_text, alignment), Provenance::by("test:view"))
        .unwrap();

    add(&mut doc, Sentences { spans: spans(&[whole]) }, Provenance::by("test:sent"));
    add(
        &mut doc,
        Tokens { spans: spans(&[cafe, beijing, price]), sentence: Some(vec![Ix::new(0); 3]) },
        Provenance::by("test:tok"),
    );

    let mut words = Words::new();
    words.tokens = vec![IxRange::new(0, 1), IxRange::new(1, 2)];
    words.form = Some(vec![Some("cafe".into()), None]);
    words.subspan = Some(vec![None, Some(beijing)]);
    add(&mut doc, words, Provenance::by("test:words"));
    add(
        &mut doc,
        Subwords { spans: spans(&[cafe]), ids: vec![101], model: model_ref("sub"), token: Some(vec![Ix::new(0)]) },
        Provenance::by("test:subwords"),
    );

    let pos = tagset("pos", &["NN", "NNP", "SYM"]);
    let mut pos_prov = Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:pos") };
    pos_prov.model = Some(model_ref("pos-en"));
    add(
        &mut doc,
        PosTags { tagset: Arc::clone(&pos), tags: vec![Tag(0), Tag(1), Tag(2)], prob: Some(vec![0.9, 0.95, 0.8]) },
        pos_prov,
    );

    let on_tok = Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:tok-aligned") };
    let on_words = Provenance { on: vec![Key::<Words>::DEFAULT.id()], ..Provenance::by("test:word-aligned") };
    add(&mut doc, WordPosTags { tagset: pos, tags: vec![Tag(0), Tag(1)], prob: None }, on_words.clone());
    add(&mut doc, Lemmas { lemma: strs(&["cafe", "北京", "$42.50"]) }, on_tok.clone());
    add(&mut doc, WordLemmas { lemma: strs(&["cafe", "北京"]) }, on_words.clone());
    add(&mut doc, Stems { stem: strs(&["caf", "北京", "$42.50"]) }, on_tok.clone());
    add(
        &mut doc,
        Terms {
            surface: strs(&["Café", "北京", "$42.50"]),
            folded: strs(&["cafe", "北京", "$42.50"]),
            stem: Some(strs(&["caf", "北京", ""])),
            lemma: None,
        },
        on_tok.clone(),
    );
    add(&mut doc, MorphFeatures::<Tokens>::new(strs(&["Number=Sing", "", ""])), on_tok.clone());
    add(
        &mut doc,
        ChunkTags { tagset: tagset("ck", &["B-NP", "O"]), tags: vec![Tag(0), Tag(0), Tag(1)], prob: None },
        on_tok.clone(),
    );
    add(
        &mut doc,
        PhraseChunks {
            spans: spans(&[cafe]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: tagset("p", &["NP"]),
        },
        Provenance::by("test:pchunk"),
    );
    add(
        &mut doc,
        Phrases {
            spans: spans(&[cafe]),
            tokens: vec![IxRange::new(0, 1)],
            kind: vec![Tag(0)],
            kinds: tagset("pk", &["NP"]),
            head: vec![Ix::new(0)],
            parent: vec![None],
        },
        Provenance::by("test:phrases"),
    );

    let rels = tagset("dep", &["root", "nsubj", "obj"]);
    add(
        &mut doc,
        Dependencies::<Tokens> {
            head: vec![None, Some(Ix::new(0)), Some(Ix::new(0))],
            relation: vec![Tag(0), Tag(1), Tag(2)],
            relations: Arc::clone(&rels),
        },
        on_tok.clone(),
    );
    add(
        &mut doc,
        Dependencies::<Words> { head: vec![None, Some(Ix::new(0))], relation: vec![Tag(0), Tag(1)], relations: rels },
        on_words,
    );

    let mut ent_spans = Spans::<Entities>::new(ViewId::SOURCE);
    ent_spans.try_push_fragments(&[beijing]).unwrap();
    ent_spans.try_push_fragments(&[cafe, price]).unwrap();
    add(
        &mut doc,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(1, 2), IxRange::new(0, 3)],
            kind: vec![Tag(0), Tag(1)],
            kinds: tagset("ner", &["LOC", "PRODUCT"]),
            score: vec![0.92, 0.0],
        },
        Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:ner") },
    );

    let analyzer = Analyzer::builder()
        .given(Key::<Tokens>::DEFAULT)
        .given(Key::<Entities>::DEFAULT)
        .step(EvidenceStep::new(Key::<Entities>::DEFAULT))
        .build()
        .unwrap();
    let mut docs = vec![doc];
    let _ = analyzer.run(&mut docs);
    let mut doc = docs.pop().unwrap();
    assert!(doc.get::<Evidence<Entities>>().is_some());

    add(
        &mut doc,
        Numbers {
            spans: spans(&[price]),
            numeral_start: vec![price.start + 1],
            numeral_end: vec![price.end],
            negative: vec![false],
            coefficient: strs(&["4250"]),
            scale: vec![2],
            multiplier: vec![Multiplier::One],
            marker: vec![ScaleMarker::None],
            notation: vec![NumberNotation::LatinUs],
        },
        Provenance::by("test:num"),
    );
    add(
        &mut doc,
        Ordinals { spans: spans(&[cafe]), base: vec![OrdinalBase::Start], offset: vec![1] },
        Provenance::by("test:ord"),
    );
    add(
        &mut doc,
        Money {
            spans: spans(&[price]),
            negative: vec![false],
            coefficient: strs(&["4250"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
        },
        Provenance::by("test:money"),
    );
    add(
        &mut doc,
        Quantities {
            spans: spans(&[cafe]),
            negative: vec![false],
            coefficient: strs(&["1"]),
            scale: vec![0],
            unit: strs(&["cup"]),
        },
        Provenance::by("test:qty"),
    );
    add(
        &mut doc,
        Temporals {
            spans: spans(&[date_span]),
            granularity: vec![Granularity::Day],
            origin: vec![Origin::Absolute],
            year: vec![2026],
            period: vec![5],
            day: vec![1],
        },
        Provenance::by("test:temp"),
    );
    add(
        &mut doc,
        Times {
            spans: spans(&[time_span]),
            kind: vec![TimeKind::Time],
            grain: vec![TimeGrain::Minute],
            modifier: vec![Modifier { relation: Relation::After, portion: Portion::Start }],
            start: vec![Some(52_200)],
            end: vec![None],
            duration_ms: vec![None],
            ambiguous: vec![false],
            timex: strs(&["T14:30"]),
        },
        Provenance::by("test:times"),
    );
    add(
        &mut doc,
        DocumentDate { date: Some(Date::new(2026, 5, 1).unwrap()), mention: Some(Ix::new(0)) },
        Provenance { reads: vec![Key::<Temporals>::DEFAULT.id()], ..Provenance::by("test:docdate") },
    );
    add(
        &mut doc,
        Containment {
            spans: spans(&[beijing]),
            ancestor_id: strs(&["cn", "asia"]),
            ancestor_name: strs(&["China", "Asia"]),
            ancestor_kind: strs(&["country", "continent"]),
            chain_end: vec![2],
        },
        Provenance::by("test:cont"),
    );
    let mut regions = Regions::new();
    regions.push(turbo_analysis_layers::RegionVote::new(RegionCode::parse("CN").unwrap(), 0.75).unwrap());
    add(&mut doc, regions, Provenance::by("test:regions"));
    add(
        &mut doc,
        Matches {
            spans: spans(&[cafe]),
            tokens: vec![IxRange::new(0, 1)],
            label: vec![Tag(0)],
            labels: tagset("m", &["FOOD"]),
            rule: vec![1],
        },
        Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:match") },
    );

    add(&mut doc, Language { code: strs(&["eng"]), score: vec![0.95] }, Provenance::by("test:lang"));
    let mut cats = Categories::<View>::new(tagset("cats", &["finance", "travel"]));
    cats.push(Tag(0), &[0.75, 0.25]);
    add(&mut doc, cats, Provenance::by("test:cats"));
    add(
        &mut doc,
        Opinions {
            tokens: vec![IxRange::new(0, 1)],
            polarity: vec![0.8],
            intensity: vec![1.0],
            negated: vec![false],
            tokens_key: Key::DEFAULT,
        },
        Provenance { reads: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:op") },
    );
    add(
        &mut doc,
        Aspects {
            tokens: vec![Some(IxRange::new(0, 1))],
            sentence: vec![Some(Ix::new(0))],
            category: vec![Some(Tag(0))],
            categories: tagset("aspect-cats", &["food"]),
            positive: vec![0.7],
            negative: vec![0.1],
            neutral: vec![0.2],
            opinions: vec![Ix::new(0)],
            opinion_end: vec![1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            opinions_key: Key::DEFAULT,
        },
        Provenance {
            reads: vec![Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id(), Key::<Opinions>::DEFAULT.id()],
            ..Provenance::by("test:asp")
        },
    );
    add(
        &mut doc,
        Pragmatics {
            score: vec![0.6],
            evidence: vec![Some(Incongruity { positive: IxRange::new(0, 1), negative: IxRange::new(1, 2) })],
            tokens_key: Key::DEFAULT,
        },
        Provenance {
            on: vec![Key::<Sentences>::DEFAULT.id()],
            reads: vec![Key::<Tokens>::DEFAULT.id()],
            ..Provenance::by("test:prag")
        },
    );
    add(
        &mut doc,
        Incivility {
            target: vec![IncivilityTarget::Sentence(Ix::new(0))],
            insult: vec![0.1],
            profanity: vec![0.0],
            threat: vec![0.0],
            identity_attack: vec![0.0],
            intolerance: vec![0.0],
            severe_toxicity: vec![0.0],
            evidence_label: vec![IncivilityLabel::Insult],
            evidence_tokens: vec![IxRange::new(0, 1)],
            evidence_end: vec![1],
            tokens_key: Key::DEFAULT,
            sentences_key: Key::DEFAULT,
            chunks_key: Key::DEFAULT,
            structure_key: Key::DEFAULT,
        },
        Provenance {
            reads: vec![Key::<Tokens>::DEFAULT.id(), Key::<Sentences>::DEFAULT.id()],
            ..Provenance::by("test:inciv")
        },
    );

    add(
        &mut doc,
        Pii {
            spans: spans(&[price]),
            kind: vec![PiiKind::Email],
            confidence: vec![PiiConfidence::Checksum],
            normalized: strs(&["a@b.c"]),
        },
        Provenance::by("test:pii"),
    );
    add(
        &mut doc,
        Noise { spans: spans(&[emoji]), severity: vec![NoiseSeverity::Damaged], score: vec![0.2] },
        Provenance::by("test:noise"),
    );
    add(
        &mut doc,
        Artifacts { spans: spans(&[date_span]), kind: vec![ArtifactKind::Mojibake] },
        Provenance::by("test:art"),
    );
    add(
        &mut doc,
        Corrections { spans: spans(&[cafe]), suggestion: strs(&["Cafe"]), edit_distance: vec![1] },
        Provenance::by("test:corr"),
    );
    add(
        &mut doc,
        Placeholders { spans: spans(&[price]), kind: vec![PlaceholderKind::Number] },
        Provenance::by("test:ph"),
    );
    add(
        &mut doc,
        Emoji {
            spans: spans(&[emoji]),
            kind: vec![EmojiKind::Basic],
            name: strs(&["grinning face"]),
            region: strs(&[""]),
        },
        Provenance::by("test:emoji"),
    );
    add(&mut doc, ProtectedSpans { spans: spans(&[date_span]) }, Provenance::by("test:prot"));
    add(
        &mut doc,
        SourceEncoding { encoding: Encoding::Utf8, confidence: 1.0, evidence: EncodingEvidence::ValidUtf8 },
        Provenance::by("test:enc"),
    );

    add(
        &mut doc,
        Structure { spans: spans(&[whole]), kind: vec![Block::Paragraph], parent: vec![None], depth: vec![0] },
        Provenance::by("test:struct"),
    );
    let mut chunk_spans = Spans::<Chunks>::new(ViewId::SOURCE);
    chunk_spans.try_push_fragments(&[cafe]).unwrap();
    chunk_spans.try_push_fragments(&[beijing, price]).unwrap();
    add(
        &mut doc,
        Chunks {
            spans: chunk_spans,
            level: vec![0, 0],
            parent: vec![None, None],
            sentences: vec![IxRange::new(0, 1), IxRange::new(0, 1)],
            cut: vec![Cut::DocumentStart, Cut::Sentence],
            cost: vec![0, 3],
            model_tokens: vec![1, 2],
            over_budget: vec![false, false],
            prompt_overhead: 0,
        },
        Provenance { reads: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("test:chunks") },
    );
    add(
        &mut doc,
        Levels {
            spans: spans(&[whole, cafe]),
            unit: vec![Unit::Document, Unit::Chunk(Ix::new(0))],
            parent: vec![None, Some(Ix::new(0))],
        },
        Provenance { reads: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:levels") },
    );
    add(
        &mut doc,
        Boundaries { cost: vec![10], reason: vec![Cut::DocumentStart] },
        Provenance { on: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("test:bounds") },
    );
    add(
        &mut doc,
        Boilerplate::<Chunks>::new(vec![None, Some(BoilerplateKind::Footer)], vec![0.1, 0.9]),
        Provenance { on: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:bp") },
    );
    add(&mut doc, Confidence::<Tokens>::new(vec![0.9, 0.95, 0.85]), on_tok.clone());
    add(
        &mut doc,
        Edges::<Chunks, Chunks>::new(
            tagset("edge", &["next"]),
            [Edge { from: Ix::new(0), to: Ix::new(1), kind: Tag(0), score: 0.75 }],
        ),
        Provenance { reads: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:edges") },
    );
    add(
        &mut doc,
        Groups::<Chunks>::new([vec![Ix::new(0), Ix::new(1)]]),
        Provenance { reads: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:groups") },
    );
    add(
        &mut doc,
        Vectors::<Chunks>::new(2, vec![0.5, -0.5, 1.0, 0.0], vec![], model_ref("embed-chunks")),
        Provenance { on: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:vec") },
    );
    add(
        &mut doc,
        Utf16Spans::<Tokens>::new(
            vec![java_utf16(cafe.start), java_utf16(beijing.start), java_utf16(price.start)],
            vec![java_utf16(cafe.end), java_utf16(beijing.end), java_utf16(price.end)],
            vec![false, false, false],
        ),
        on_tok,
    );
    doc
}
