use turbo_grammar::{CHINESE_UNITS, Output, Parser, Value};
#[test]
fn chinese_native_quantity_regression() {
    let text = "三百米";
    let mut parser = Parser::new(&CHINESE_UNITS);
    parser.parse(text);
    let mut out = Vec::new();
    parser.select(text, Output::Quantity, &mut out);
    let values: Vec<_> = out
        .iter()
        .filter_map(|m| {
            if let Value::Quantity(q) = m.value {
                Some((&text[m.start as usize..m.end as usize], q.value().to_f64()))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(values, [(text, 300.0)]);
}

#[test]
fn chinese_pipeline_preserves_fullwidth_source_and_exact_decimal() {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
    use turbo_grammar::annotator::{GRAMMAR_MONEY, GRAMMAR_PERCENTS, GrammarAnnotator};
    let text = "🛒 兑换１．０７５３美元；三百米；人民币 50元6角三分；202107211321363588510367 美元";
    let analyzer = Analyzer::builder().step(GrammarAnnotator::new(&CHINESE_UNITS).bound()).build().unwrap();
    let mut docs = [Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
    let quantities = docs[0].get_key(&GRAMMAR_PERCENTS).unwrap();
    money.check().unwrap();
    quantities.check().unwrap();
    assert_eq!(money.len(), 3);
    assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "1.0753");
    assert_eq!(&text[money.get(Ix::new(0)).span.range()], "１．０７５３美元");
    assert_eq!(money.amount(Ix::new(1)).to_plain_string(), "50.63");
    assert_eq!(money.amount(Ix::new(2)).to_plain_string(), "202107211321363588510367");
    assert_eq!(quantities.len(), 1);
    assert_eq!(&text[quantities.spans.get(Ix::new(0)).range()], "三百米");
    let offsets = OffsetTable::build(text);
    for span in money.spans.iter().map(|(_, s)| s).chain(quantities.spans.iter().map(|(_, s)| s)) {
        for byte in [span.start, span.end] {
            assert_eq!(offsets.from_utf16(text, offsets.to_utf16(text, byte)).unwrap(), byte);
            assert_eq!(offsets.from_code_point(text, offsets.to_code_point(text, byte)).unwrap(), byte);
        }
    }
}

fn readings(text: &str) -> Vec<(String, String, String)> {
    let mut parser = Parser::new(&CHINESE_UNITS);
    parser.parse(text);
    let mut out = Vec::new();
    parser.select(text, Output::Quantity, &mut out);
    out.iter()
        .filter_map(|m| match m.value {
            Value::Quantity(q) => {
                Some((text[m.start as usize..m.end as usize].into(), q.value().to_string(), q.unit().symbol()))
            }
            _ => None,
        })
        .collect()
}
#[test]
fn chinese_rejects_identifiers_and_person_classifiers() {
    for text in [
        "A123美元",
        "一位名叫杰克的男生",
        "一位是存储公司的经理",
        "五角大楼宣布了作战计划",
        "dimension是什么意思",
        "今天2:00 pm",
        "元旦活动",
        "十二月出生",
        "に元",
        "マイナス3元",
    ] {
        assert!(readings(text).is_empty(), "{text}: {:?}", readings(text));
    }
    assert_eq!(readings("一位表示一个比特"), [("一位".into(), "1".into(), "bit".into())]);
}
#[test]
fn chinese_currency_places_do_not_cross_currency_or_repeat() {
    for (text, expected) in [("50元6角三分", "50.63"), ("两美元又三美分", "2.03"), ("10日元5日本銭", "10.05")]
    {
        assert_eq!(readings(text)[0].0, text);
        assert_eq!(readings(text)[0].1, expected);
    }
    for text in ["3美元5角", "3元5日本銭", "3元15角", "3元5角6角", "3元0角6角", "3元0分1角", "3元元", "3欧元欧元"]
    {
        assert!(readings(text).iter().all(|(span, _, _)| span != text), "{text}");
    }
}
#[test]
fn chinese_clause_commas_do_not_join_distinct_prices() {
    assert_eq!(
        readings("$20，300美圆"),
        [("$20".into(), "20".into(), "dollar".into()), ("300美圆".into(), "300".into(), "USD".into())]
    );
}

#[test]
fn chinese_leading_decimals_and_unicode_signs_keep_values_and_source() {
    use turbo_analysis::{Analyzer, Document, Ix, Layer, OffsetTable};
    use turbo_grammar::annotator::{GRAMMAR_MONEY, GrammarAnnotator};
    let analyzer = Analyzer::builder().step(GrammarAnnotator::new(&CHINESE_UNITS).bound()).build().unwrap();
    for (source, expected) in [
        ("．5美元", "0.5"),
        ("．５美元", "0.5"),
        ("０．５美元", "0.5"),
        ("−5美元", "-5"),
        ("‐5美元", "-5"),
        ("－5美元", "-5"),
        ("-5美元", "-5"),
        ("−．５美元", "-0.5"),
        ("‐０．５美元", "-0.5"),
    ] {
        let text = format!("🛒 {source}。结束");
        let mut docs = [Document::new(text.as_str())];
        assert!(analyzer.run(&mut docs).iter().all(Result::is_ok), "{source}");
        let money = docs[0].get_key(&GRAMMAR_MONEY).unwrap();
        money.check().unwrap();
        assert_eq!(money.len(), 1, "{source}");
        assert_eq!(money.amount(Ix::new(0)).to_plain_string(), expected, "{source}");
        let span = money.get(Ix::new(0)).span;
        assert_eq!(&text[span.range()], source);
        let offsets = OffsetTable::build(&text);
        for byte in [span.start, span.end] {
            assert_eq!(offsets.from_utf16(&text, offsets.to_utf16(&text, byte)).unwrap(), byte);
            assert_eq!(offsets.from_code_point(&text, offsets.to_code_point(&text, byte)).unwrap(), byte);
        }
    }
}
