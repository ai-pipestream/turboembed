use turbo_analysis::{Analyzer, Document, Ix, Key, Layer, OffsetTable, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS};
use turbo_grammar::{GrammarAnnotator, JAPANESE_UNITS, Output, Parser, Value};

fn readings(text: &str) -> Vec<(String, String, String)> {
    let mut p = Parser::new(&JAPANESE_UNITS);
    p.parse(text);
    let mut found = Vec::new();
    p.select(text, Output::Quantity, &mut found);
    found
        .into_iter()
        .filter_map(|m| {
            let Value::Quantity(q) = m.value else {
                return None;
            };
            Some((text[m.start as usize..m.end as usize].into(), q.value().to_string(), q.unit().symbol()))
        })
        .collect()
}

#[test]
fn actual_pipeline_keeps_exact_large_fullwidth_and_mixed_amounts() {
    let text = "😀１２３円、202107211321363588510367円、負兩點五メートル、華氏マイナス20度、50元6角3分。";
    let analyzer = Analyzer::builder().step(GrammarAnnotator::new(&JAPANESE_UNITS).bound()).build().unwrap();
    let mut docs = [Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
    let quantities = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
    money.check().unwrap();
    quantities.check().unwrap();
    assert_eq!(money.len(), 3);
    assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "123");
    assert_eq!(money.amount(Ix::new(1)).to_plain_string(), "202107211321363588510367");
    assert_eq!(money.amount(Ix::new(2)).to_plain_string(), "50.63");
    assert_eq!(quantities.len(), 2);
    assert_eq!(quantities.value(Ix::new(0)).to_plain_string(), "-2.5");
    assert_eq!(quantities.value(Ix::new(1)).to_plain_string(), "-20");
    for span in money.spans.iter().map(|(_, s)| s).chain(quantities.spans.iter().map(|(_, s)| s)) {
        let offsets = OffsetTable::build(text);
        for byte in [span.start, span.end] {
            assert_eq!(offsets.from_utf16(text, offsets.to_utf16(text, byte)).unwrap(), byte);
            assert_eq!(offsets.from_code_point(text, offsets.to_code_point(text, byte)).unwrap(), byte);
        }
    }
}

#[test]
fn contextual_ages_ranges_and_currency_minor_units() {
    for input in ["-5歳", "マイナス5歳", "負五歳", "三角形", "七分後", "printf(\"1\\n2\\n\");", "x29cmextra"]
    {
        assert!(readings(input).is_empty(), "{input}: {:?}", readings(input));
    }
    assert_eq!(readings("70-90日齢"), [("90日齢".into(), "90".into(), "days old".into())]);
    for input in ["5ドル3角", "5人民元3銭", "10円5セント", "5ドル100セント"] {
        assert!(!readings(input).iter().any(|(text, _, _)| text == input), "{input}");
    }
    assert_eq!(readings("50元6角3分"), [("50元6角3分".into(), "50.63".into(), "CNY".into())]);
    assert_eq!(readings("100メガバイト"), [("100メガバイト".into(), "100".into(), "MB".into())]);
}

fn verify_projections(input: &str) -> usize {
    let source = format!("😀<p>{input}</p>");
    let mut doc = Document::new(source.as_str());
    let parent = doc.get::<View>().unwrap().clone();
    let mut builder = ViewBuilder::new(&source);
    builder.keep(4);
    builder.replace(7, "");
    builder.keep(7 + input.len());
    builder.replace(source.len(), "");
    let (text, alignment) = builder.finish();
    let key = Key::<View>::named("plain");
    doc.add(&key, View::derived(&parent, text, alignment), Provenance::by("test:strip-markup")).unwrap();
    let child = doc.get_key(&key).unwrap();
    let found = GrammarAnnotator::new(&JAPANESE_UNITS).find(child.text(), child.id());
    found.money.check().unwrap();
    found.quantities.check().unwrap();
    let mut count = 0;
    for span in found.money.spans.iter().map(|(_, s)| s).chain(found.quantities.spans.iter().map(|(_, s)| s)) {
        let original = doc.map_span(child.id(), ViewId::SOURCE, span).unwrap();
        assert_eq!(&source[original.range()], &child.text()[span.range()]);
        assert_eq!(doc.map_span(ViewId::SOURCE, child.id(), original).unwrap(), span);
        for (view, s) in [(parent.clone(), original), (child.clone(), span)] {
            let offsets = OffsetTable::build(view.text());
            for byte in [s.start, s.end] {
                assert_eq!(offsets.from_utf16(view.text(), offsets.to_utf16(view.text(), byte)).unwrap(), byte);
                assert_eq!(
                    offsets.from_code_point(view.text(), offsets.to_code_point(view.text(), byte)).unwrap(),
                    byte
                );
            }
        }
        assert_eq!(original, Span::new(span.start + 3, span.end + 3));
        count += 1;
    }
    count
}

#[test]
fn entire_upstream_corpus_has_bidirectional_layer_projection() {
    let mut cases = 0;
    let mut spans = 0;
    for model in ["Age", "Currency", "Dimension", "Temperature"] {
        let path = format!(
            "{}/tests/data/recognizers-text/Japanese/NumberWithUnit/{model}Model.json",
            env!("CARGO_MANIFEST_DIR")
        );
        let raw = std::fs::read_to_string(path).unwrap();
        let rows: Vec<serde_json::Value> = serde_json::from_str(raw.trim_start_matches('\u{feff}')).unwrap();
        for row in rows {
            spans += verify_projections(row["Input"].as_str().unwrap());
            cases += 1;
        }
    }
    assert_eq!(cases, 188);
    assert!(spans >= 170, "actual layer quantities: {spans}");
    assert_eq!(verify_projections("１２３円、兩點五メートル、四十九兆九千五百九十二億円"), 3);
}

#[test]
fn fullwidth_normalization_maps_the_whole_amount_in_both_directions() {
    let source = "😀価格１２３．５円です";
    let mut doc = Document::new(source);
    let parent = doc.get::<View>().unwrap().clone();
    let mut b = ViewBuilder::new(source);
    for (start, c) in source.char_indices() {
        let end = start + c.len_utf8();
        match c {
            '０'..='９' => b.replace(end, &char::from(b'0' + (c as u32 - '０' as u32) as u8).to_string()),
            '．' => b.replace(end, "."),
            _ => b.keep(end),
        }
    }
    let (text, alignment) = b.finish();
    let key = Key::<View>::named("digits");
    doc.add(&key, View::derived(&parent, text, alignment), Provenance::by("test:digit-width")).unwrap();
    let child = doc.get_key(&key).unwrap();
    let a = GrammarAnnotator::new(&JAPANESE_UNITS).find(source, ViewId::SOURCE);
    let b = GrammarAnnotator::new(&JAPANESE_UNITS).find(child.text(), child.id());
    assert_eq!((a.money.len(), b.money.len()), (1, 1));
    assert_eq!(a.money.amount(Ix::new(0)), b.money.amount(Ix::new(0)));
    let s = a.money.get(Ix::new(0)).span;
    let t = b.money.get(Ix::new(0)).span;
    assert_eq!(&source[s.range()], "１２３．５円");
    assert_eq!(&child.text()[t.range()], "123.5円");
    assert_eq!(doc.map_span(ViewId::SOURCE, child.id(), s).unwrap(), t);
    assert_eq!(doc.map_span(child.id(), ViewId::SOURCE, t).unwrap(), s);
}
