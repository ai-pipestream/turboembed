use super::*;
use crate::GOLDEN_KEY;
use crate::format::{Column, ColumnData, Golden, GoldenAnchor};
use turbo_analysis::{Document, Provenance, Span, Spans, ViewId};
use turbo_analysis_layers::{Aspects, Incivility, Opinions, Sentences, Strs};

fn doc() -> Document {
    let mut doc = Document::new("a b. c d.");
    let by = Provenance::by("test");
    let sentences = || Sentences { spans: span_list::<Sentences>(&[(0, 4), (5, 9)]) };
    doc.add(&Key::DEFAULT, sentences(), by.clone()).unwrap();
    doc.add(&Key::named(GOLDEN_KEY), sentences(), by.clone()).unwrap();
    let tokens = || Tokens {
        spans: span_list::<Tokens>(&[(0, 1), (2, 3), (3, 4), (5, 6), (7, 8), (8, 9)]),
        sentence: Some(vec![Ix::new(0), Ix::new(0), Ix::new(0), Ix::new(1), Ix::new(1), Ix::new(1)]),
    };
    doc.add(&Key::DEFAULT, tokens(), by.clone()).unwrap();
    doc.add(&Key::named(GOLDEN_KEY), tokens(), by.clone()).unwrap();
    let opinions = Opinions {
        tokens: vec![IxRange::new(0, 1)],
        polarity: vec![0.5],
        intensity: vec![1.0],
        negated: vec![false],
        tokens_key: Key::named(GOLDEN_KEY),
    };
    let mut by_opinions = by.clone();
    by_opinions.reads = vec![Key::<Tokens>::named(GOLDEN_KEY).id()];
    doc.add(&Key::named(GOLDEN_KEY), opinions, by_opinions).unwrap();
    doc
}

fn span_list<L>(pairs: &[(u32, u32)]) -> Spans<L> {
    let mut s = Spans::new(ViewId::SOURCE);
    for &(a, b) in pairs {
        s.push(Span::new(a, b));
    }
    s
}

fn strs(items: &[&str]) -> Strs {
    let mut s = Strs::default();
    for i in items {
        s.push(i);
    }
    s
}

fn col(name: &str, data: ColumnData) -> Column {
    Column { name: name.to_owned(), data }
}

fn aspects_layer(sentence: &str, opinions: &str) -> GoldenLayer {
    GoldenLayer {
        id: "turbo:aspects".to_owned(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            col("token_start", ColumnData::U32(vec![0])),
            col("token_end", ColumnData::U32(vec![1])),
            col("sentence", ColumnData::Str(strs(&[sentence]))),
            col("category", ColumnData::Str(strs(&[""]))),
            col("opinions", ColumnData::Str(strs(&[opinions]))),
            col("positive", ColumnData::F64(vec![0.5])),
            col("negative", ColumnData::F64(vec![0.2])),
            col("neutral", ColumnData::F64(vec![0.3])),
        ],
    }
}

#[test]
fn aspect_indexes_outside_their_layers_are_refused_not_panicked() {
    let doc = doc();
    let golden = Golden::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zurich.golden")).unwrap();
    let key = Key::<Aspects>::named(GOLDEN_KEY).id();
    let cx = LoadCx { doc: &doc, golden: &golden, name: GOLDEN_KEY, key };
    assert!(Aspects::load(&aspects_layer("4294967295", ""), &cx).is_err(), "reserved sentence index loaded");
    assert!(Aspects::load(&aspects_layer("7", ""), &cx).is_err(), "sentence past the sentences loaded");
    assert!(Aspects::load(&aspects_layer("0", "4294967295"), &cx).is_err(), "reserved opinion index loaded");
    assert!(Aspects::load(&aspects_layer("0", "1"), &cx).is_err(), "opinion past the opinions loaded");
    assert!(Aspects::load(&aspects_layer("0", "0"), &cx).is_ok(), "the in-range control was refused");
}

#[test]
fn incivility_targets_outside_their_layers_are_refused_not_panicked() {
    let doc = doc();
    let golden = Golden::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/zurich.golden")).unwrap();
    let key = Key::<Incivility>::named(GOLDEN_KEY).id();
    let cx = LoadCx { doc: &doc, golden: &golden, name: GOLDEN_KEY, key };
    let layer = |index: u32| GoldenLayer {
        id: "turbo:incivility".to_owned(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            col("target_kind", ColumnData::Str(strs(&["sentence"]))),
            col("target_index", ColumnData::U32(vec![index])),
            col("insult", ColumnData::F64(vec![0.0])),
            col("profanity", ColumnData::F64(vec![0.0])),
            col("threat", ColumnData::F64(vec![0.0])),
            col("identity_attack", ColumnData::F64(vec![0.0])),
            col("intolerance", ColumnData::F64(vec![0.0])),
            col("severe_toxicity", ColumnData::F64(vec![0.0])),
            col("evidence", ColumnData::Str(strs(&[""]))),
        ],
    };
    assert!(Incivility::load(&layer(u32::MAX), &cx).is_err(), "reserved target index loaded");
    assert!(Incivility::load(&layer(7), &cx).is_err(), "target past the sentences loaded");
    assert!(Incivility::load(&layer(1), &cx).is_ok(), "the in-range control was refused");
}
