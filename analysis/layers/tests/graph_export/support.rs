use super::*;

pub(super) fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

pub(super) fn spans<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

pub(super) fn tagset(name: &str, labels: &[&str]) -> Arc<TagSet> {
    TagSet::new(name, labels)
}

/// A document carrying one of most core pipeline layers, added in
/// dependency order.
pub(super) fn document() -> Document {
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

/// The span column encoding, written out longhand so corrupted
/// payloads can be built without an export context.
pub(super) fn write_spans(w: &mut Writer, view: &str, spans: &[Span]) {
    w.str(view).unwrap();
    w.usize(spans.len()).unwrap();
    for s in spans {
        w.usize(1).unwrap();
        w.span(*s);
    }
}

pub(super) fn token_body(instance: &str, sentence: &[u32]) -> Vec<u8> {
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

pub(super) fn splice(graph: &mut DocumentGraph, layer_prefix: &str, body: Vec<u8>, len: u32) {
    let at = graph.layers.iter().position(|l| l.layer.starts_with(layer_prefix)).unwrap();
    graph.layers[at].body = Some(body);
    graph.layers[at].len = len;
}
