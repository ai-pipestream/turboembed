//! Document fixture for `search_projection_sink` integration tests (`#2345`).

use turbo_analysis::{Document, Ix, IxRange, Key, Provenance, Span, Spans, Tag, TagSet, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{
    CONVERTED_MONEY, Categories, Chunks, Containment, Currency, Cut, Date, Dependencies, DocumentDate, Entities,
    Granularity, Money, Origin, Quantities, QuantityMention, Sentences, Strs, Temporals, TokenArcKind, TokenGraph,
    TokenGraphBuilder, Tokens,
};
use turbo_entity_linker::LinkedEntities;
use turbo_geo::{BundledGazetteer, Gazetteer, GeoResolution, Locations};
use turbo_group::{Scratch as SketchScratch, Sketches};
use turbo_normalize_layers::{ArtifactKind, Artifacts, Noise, NoiseSeverity};

pub const RAW_TEXT: &str = "Café 😀 Berlin audit: Acme Corp found no evidence of fraud for €100.00 (500 g) on 2026-05-01. \
Globex LLC allegedly reported fraud.";

fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for x in items {
        s.push(x);
    }
    s
}

fn find_span(haystack: &str, needle: &str) -> Span {
    let start = haystack.find(needle).unwrap();
    Span::new(start as u32, (start + needle.len()) as u32)
}

fn find_last_span(haystack: &str, needle: &str) -> Span {
    let start = haystack.rfind(needle).unwrap();
    Span::new(start as u32, (start + needle.len()) as u32)
}

fn spans_on<L>(view: ViewId, list: &[Span]) -> Spans<L> {
    let mut s = Spans::new(view);
    for &sp in list {
        s.push(sp);
    }
    s
}

/// Builds a document opened from raw UTF-8 BOM bytes (`turbo_encoding::open`),
/// attaches a child normalized view, and populates all search index layers on
/// the child view to prove child->SOURCE projection.
pub fn build_full_search_doc() -> Document {
    let mut raw_bytes = vec![0xEF, 0xBB, 0xBF];
    raw_bytes.extend_from_slice(RAW_TEXT.as_bytes());
    let mut doc = turbo_encoding::open(raw_bytes, None).unwrap();
    assert_eq!(doc.text(ViewId::SOURCE), RAW_TEXT);

    let src_view = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(src_view.text());
    let cafe_pos = RAW_TEXT.find("é").unwrap();
    vb.keep(cafe_pos);
    vb.replace(cafe_pos + "é".len(), "e");
    let (norm_text, align) = vb.finish();
    doc.add(&Key::<View>::named("norm"), View::derived(&src_view, norm_text, align), Provenance::by("test:norm"))
        .unwrap();
    let v_norm = ViewId(1);
    let nt = doc.text(v_norm).to_owned();

    let s1_end = nt.find('.').unwrap() + 1;
    let s2_start = nt.find("Globex").unwrap();
    let sent_spans = [Span::new(0, s1_end as u32), Span::new(s2_start as u32, nt.len() as u32)];
    doc.add(&Key::DEFAULT, Sentences { spans: spans_on(v_norm, &sent_spans) }, Provenance::by("test:sent")).unwrap();

    let tok_words = [
        ("Cafe", 0, find_span(&nt, "Cafe")),
        ("Berlin", 0, find_span(&nt, "Berlin")),
        ("Acme", 0, find_span(&nt, "Acme")),
        ("Corp", 0, find_span(&nt, "Corp")),
        ("found", 0, find_span(&nt, "found")),
        ("no", 0, find_span(&nt, "no")),
        ("evidence", 0, find_span(&nt, "evidence")),
        ("of", 0, find_span(&nt, "of")),
        ("fraud", 0, find_span(&nt, "fraud")),
        ("Globex", 1, find_span(&nt, "Globex")),
        ("LLC", 1, find_span(&nt, "LLC")),
        ("allegedly", 1, find_span(&nt, "allegedly")),
        ("reported", 1, find_span(&nt, "reported")),
        ("fraud2", 1, find_last_span(&nt, "fraud")),
    ];
    let tok_spans: Vec<Span> = tok_words.iter().map(|t| t.2).collect();
    let tok_sents: Vec<Ix<Sentences>> = tok_words.iter().map(|t| Ix::new(t.1)).collect();
    doc.add(
        &Key::DEFAULT,
        Tokens { spans: spans_on(v_norm, &tok_spans), sentence: Some(tok_sents) },
        Provenance::by("test:tok"),
    )
    .unwrap();

    let base_terms: Vec<String> = tok_words
        .iter()
        .map(|(w, _, _)| if *w == "fraud2" { "fraud".to_owned() } else { w.to_ascii_lowercase() })
        .collect();
    let mut tgb = TokenGraphBuilder::from_base_tokens(v_norm, tok_spans.clone(), base_terms);
    let acme_corp_span = Span::new(tok_spans[2].start, tok_spans[3].end);
    tgb.add_span_arc(2..4, acme_corp_span, "acme_corporation", TokenArcKind::Synonym);
    doc.add(&Key::<TokenGraph>::DEFAULT, tgb.finish().unwrap(), Provenance::by("test:tgraph")).unwrap();

    let rels = TagSet::new("ud", &["root", "nsubj", "compound", "obl", "obj", "det", "nmod", "case", "advmod"]);
    let heads = vec![
        Some(Ix::new(1)),
        Some(Ix::new(4)),
        Some(Ix::new(3)),
        Some(Ix::new(4)),
        None,
        Some(Ix::new(6)),
        Some(Ix::new(4)),
        Some(Ix::new(8)),
        Some(Ix::new(6)),
        Some(Ix::new(10)),
        Some(Ix::new(12)),
        Some(Ix::new(12)),
        None,
        Some(Ix::new(12)),
    ];
    let rel_tags = vec![
        Tag(2),
        Tag(3),
        Tag(2),
        Tag(1),
        Tag(0),
        Tag(5),
        Tag(4),
        Tag(7),
        Tag(6),
        Tag(2),
        Tag(1),
        Tag(8),
        Tag(0),
        Tag(4),
    ];
    doc.add(
        &Key::DEFAULT,
        Dependencies::<Tokens> { head: heads, relation: rel_tags, relations: rels },
        Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:dep") },
    )
    .unwrap();

    let globex_span = Span::new(tok_spans[9].start, tok_spans[10].end);
    let ent_spans = spans_on(v_norm, &[acme_corp_span, tok_spans[8], globex_span]);
    doc.add(
        &Key::DEFAULT,
        Entities {
            spans: ent_spans,
            tokens: vec![IxRange::new(2, 4), IxRange::new(8, 9), IxRange::new(9, 11)],
            kind: vec![Tag(0), Tag(1), Tag(0)],
            kinds: TagSet::new("ner", &["ORG", "CONCEPT"]),
            score: vec![0.96, 0.91, 0.89],
        },
        Provenance { on: vec![Key::<Tokens>::DEFAULT.id()], ..Provenance::by("test:ner") },
    )
    .unwrap();

    let mut linked = LinkedEntities::with_capacity(3);
    for (kb, canon, alias, sc) in [
        ("Q_ACME", "Acme Corporation", "Acme Corp", 0.98),
        ("Q_FRAUD", "Financial Fraud", "fraud", 0.94),
        ("Q_GLOBEX", "Globex Corporation", "Globex LLC", 0.90),
    ] {
        linked.kb_id.push(Some(kb.into()));
        linked.provider_id.push("wikidata".into());
        linked.canonical_name.push(canon.into());
        linked.matched_alias.push(alias.into());
        linked.score.push(sc);
        linked.nil_reason.push(None);
        linked.nil_cluster.push(None);
        linked.candidates.push(Vec::new());
    }
    doc.add(&Key::DEFAULT, linked, Provenance { on: vec![Key::<Entities>::DEFAULT.id()], ..Provenance::by("test:el") })
        .unwrap();

    doc.add(
        &Key::DEFAULT,
        Chunks {
            spans: spans_on(v_norm, &sent_spans),
            level: vec![0, 0],
            parent: vec![None, None],
            sentences: vec![IxRange::new(0, 1), IxRange::new(1, 2)],
            cut: vec![Cut::DocumentStart, Cut::Sentence],
            cost: vec![0, 42],
            model_tokens: vec![18, 5],
            over_budget: vec![false, false],
            prompt_overhead: 0,
        },
        Provenance { reads: vec![Key::<Sentences>::DEFAULT.id()], ..Provenance::by("test:chunks") },
    )
    .unwrap();

    let mut scratch = SketchScratch::default();
    let mut doc_sk = Sketches::<View>::empty(3);
    doc_sk.push([nt.as_str()], &mut scratch);
    doc.add(&Key::DEFAULT, doc_sk, Provenance::by("test:doc-sketch")).unwrap();

    let mut chunk_sk = Sketches::<Chunks>::empty(3);
    chunk_sk.push([&nt[sent_spans[0].range()]], &mut scratch);
    chunk_sk.push([&nt[sent_spans[1].range()]], &mut scratch);
    doc.add(
        &Key::DEFAULT,
        chunk_sk,
        Provenance { on: vec![Key::<Chunks>::DEFAULT.id()], ..Provenance::by("test:chunk-sketch") },
    )
    .unwrap();

    let eur_span = find_span(&nt, "€100.00");
    doc.add(
        &Key::DEFAULT,
        Money {
            spans: spans_on(v_norm, &[eur_span]),
            negative: vec![false],
            coefficient: strs(&["10000"]),
            scale: vec![2],
            currency: vec![Currency::parse("EUR").unwrap()],
            per: strs(&[""]),
        },
        Provenance::by("test:money"),
    )
    .unwrap();
    doc.add(
        &CONVERTED_MONEY,
        Money {
            spans: spans_on(v_norm, &[eur_span]),
            negative: vec![false],
            coefficient: strs(&["10850"]),
            scale: vec![2],
            currency: vec![Currency::parse("USD").unwrap()],
            per: strs(&[""]),
        },
        Provenance::by("test:ecb"),
    )
    .unwrap();

    let qty_span = find_span(&nt, "500 g");
    let mut quantities = Quantities::new(v_norm);
    quantities.push(QuantityMention { span: qty_span, negative: false, coefficient: "500", scale: 0, unit: "g" });
    doc.add(&Key::DEFAULT, quantities, Provenance::by("test:qty")).unwrap();

    let date_span = find_span(&nt, "2026-05-01");
    doc.add(
        &Key::DEFAULT,
        Temporals {
            spans: spans_on(v_norm, &[date_span]),
            granularity: vec![Granularity::Day],
            origin: vec![Origin::Absolute],
            year: vec![2026],
            period: vec![5],
            day: vec![1],
        },
        Provenance::by("test:temp"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        DocumentDate { date: Some(Date::new(2026, 5, 1).unwrap()), mention: Some(Ix::new(0)) },
        Provenance::by("test:docdate"),
    )
    .unwrap();

    let berlin_span = find_span(&nt, "Berlin");
    let berlin_entry = BundledGazetteer::get().lookup("Berlin")[0].to_entry();
    let mut locs = Locations::new(v_norm);
    locs.push(GeoResolution::new(berlin_span, berlin_entry, 0.95).unwrap());
    doc.add(&Key::DEFAULT, locs, Provenance::by("test:geo")).unwrap();
    doc.add(
        &Key::DEFAULT,
        Containment {
            spans: spans_on(v_norm, &[berlin_span]),
            ancestor_id: strs(&["de-be", "de"]),
            ancestor_name: strs(&["Berlin", "Germany"]),
            ancestor_kind: strs(&["admin1", "country"]),
            chain_end: vec![2],
        },
        Provenance::by("test:cont"),
    )
    .unwrap();

    let mut sentiment = Categories::<View>::new(TagSet::new("sentiment", &["negative", "neutral", "positive"]));
    sentiment.push(Tag(0), &[0.82, 0.12, 0.06]);
    doc.add(&Key::<Categories<View>>::named("sentiment"), sentiment, Provenance::by("test:sent-cat")).unwrap();

    let n1 = find_span(RAW_TEXT, "Café");
    let n2 = find_span(RAW_TEXT, "😀");
    doc.add(
        &Key::DEFAULT,
        Noise {
            spans: spans_on(ViewId::SOURCE, &[n1, n2]),
            severity: vec![NoiseSeverity::Misspelled, NoiseSeverity::Damaged],
            score: vec![0.25, 0.75],
        },
        Provenance::by("test:noise"),
    )
    .unwrap();
    doc.add(
        &Key::DEFAULT,
        Artifacts { spans: spans_on(ViewId::SOURCE, &[n1]), kind: vec![ArtifactKind::Mojibake] },
        Provenance::by("test:art"),
    )
    .unwrap();

    doc
}
