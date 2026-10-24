use super::*;
use turbo_analysis::{Analyzer, Document, Ix, Key, Layer, Provenance};
use turbo_analysis_layers::RegionVote;

fn with_ballot(text: &str, votes: &[(&str, f64)]) -> Document {
    let mut doc = Document::new(text);
    let mut regions = Regions::new();
    for &(code, share) in votes {
        regions.push(RegionVote::parse(code, share).unwrap());
    }
    doc.add(&Key::DEFAULT, regions, Provenance::by("test")).unwrap();
    doc
}

fn run_one(a: &Analyzer, mut doc: Document) -> Document {
    a.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
    doc
}

#[test]
fn region_configs_table_is_sorted_and_valid() {
    assert!(REGION_CONFIGS.windows(2).all(|w| w[0].0 < w[1].0));
    for &(bytes, expected_notation, _) in &REGION_CONFIGS {
        let rc = RegionCode::new(bytes).unwrap();
        assert_eq!(region_notation(rc), expected_notation);
        assert_eq!(NumberScanner::for_region(rc).notation, expected_notation);
        assert_eq!(QuantityExtractor::for_region(rc).notation(), expected_notation);
        let _ = MoneyExtractor::for_region(rc);
    }
}

#[test]
fn locale_notation_matches_opennlp_number_notation_test_vectors() {
    let cases = [
        ("en-US", NumberNotation::LatinUs),
        ("en-GB", NumberNotation::LatinUs),
        ("en-AU", NumberNotation::LatinUs),
        ("ja-JP", NumberNotation::LatinUs),
        ("zh-CN", NumberNotation::LatinUs),
        ("hi-IN", NumberNotation::LatinUs),
        ("en-CA", NumberNotation::LatinUs),
        ("de-DE", NumberNotation::LatinEu),
        ("fr-FR", NumberNotation::LatinEu),
        ("it-IT", NumberNotation::LatinEu),
        ("es-ES", NumberNotation::LatinEu),
        ("nl-NL", NumberNotation::LatinEu),
        ("pt-BR", NumberNotation::LatinEu),
        ("fr-CA", NumberNotation::LatinEu),
        ("", NumberNotation::LatinUs),
        ("DE", NumberNotation::LatinEu),
        ("US", NumberNotation::LatinUs),
    ];
    for (tag, expected) in cases {
        assert_eq!(locale_notation(tag), expected, "locale {tag:?}");
        assert_eq!(NumberScanner::for_locale(tag).notation, expected, "scanner {tag:?}");
        assert_eq!(QuantityExtractor::for_locale(tag).notation(), expected, "quantity {tag:?}");
    }
}

#[test]
fn winning_region_resolves_the_symbol() {
    let doc = with_ballot("the deal is worth $5", &[("AU", 1.0)]);
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out = run_one(&a, doc);
    let money = out.get::<Money>().unwrap();
    assert_eq!(money.len(), 1);
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "AUD");
}

#[test]
fn empty_ballot_falls_back_to_default_table() {
    let doc = with_ballot("costs $5", &[]);
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out = run_one(&a, doc);
    let money = out.get::<Money>().unwrap();
    assert_eq!(money.len(), 1);
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
}

#[test]
fn country_with_minority_language_locales_resolves_symbol() {
    let doc = with_ballot("the grant is $5", &[("CA", 1.0)]);
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out = run_one(&a, doc);
    let money = out.get::<Money>().unwrap();
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "CAD");
}

#[test]
fn locale_scan_skips_unusable_symbol_candidates() {
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let af = run_one(&a, with_ballot("price 100\u{060B} today", &[("AF", 1.0)]));
    let af_money = af.get::<Money>().unwrap();
    assert_eq!(af_money.len(), 1);
    assert_eq!(af_money.get(Ix::new(0)).currency.as_str(), "AFN");
    assert_eq!(af_money.amount(Ix::new(0)).to_plain_string(), "100");

    let cn = run_one(&a, with_ballot("price \u{00A5}100 today", &[("CN", 1.0)]));
    let cn_money = cn.get::<Money>().unwrap();
    assert_eq!(cn_money.len(), 1);
    assert_eq!(cn_money.get(Ix::new(0)).currency.as_str(), "CNY");
    assert_eq!(cn_money.amount(Ix::new(0)).to_plain_string(), "100");
}

#[test]
fn letter_symbol_country_keeps_default_table() {
    let doc = with_ballot("the invoice totals $5", &[("CH", 1.0)]);
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out = run_one(&a, doc);
    let money = out.get::<Money>().unwrap();
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
}

#[test]
fn unknown_country_falls_back_to_default_table() {
    let doc = with_ballot("costs $5", &[("ZZ", 1.0)]);
    let a = Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out = run_one(&a, doc);
    let money = out.get::<Money>().unwrap();
    assert_eq!(money.get(Ix::new(0)).currency.as_str(), "USD");
}

#[test]
fn tie_below_margin_falls_back_to_default_table_and_zero_margin_elects_top_row() {
    let tie_doc = || with_ballot("the deal is worth $5", &[("AU", 0.5), ("CA", 0.5)]);

    let a_strict = Analyzer::builder()
        .add(RegionAwareMoneyAnnotator::with_minimum_margin(0.1).unwrap())
        .given(Key::<Regions>::DEFAULT)
        .build()
        .unwrap();
    let out_strict = run_one(&a_strict, tie_doc());
    assert_eq!(out_strict.get::<Money>().unwrap().get(Ix::new(0)).currency.as_str(), "USD");

    let a_zero =
        Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();
    let out_zero = run_one(&a_zero, tie_doc());
    assert_eq!(out_zero.get::<Money>().unwrap().get(Ix::new(0)).currency.as_str(), "AUD");
}

#[test]
fn margin_outside_unit_interval_is_rejected() {
    for bad in [-0.1, 1.1, f64::NAN] {
        assert!(RegionAwareMoneyAnnotator::with_minimum_margin(bad).is_err());
        assert!(RegionAwareNumberAnnotator::with_minimum_margin(bad).is_err());
        assert!(RegionAwareQuantityAnnotator::with_minimum_margin(bad).is_err());
    }
}

#[test]
fn region_aware_numbers_and_quantities_read_regional_and_mixed_documents() {
    let a = Analyzer::builder()
        .add(RegionAwareNumberAnnotator::new())
        .add(RegionAwareQuantityAnnotator::new())
        .given(Key::<Regions>::DEFAULT)
        .build()
        .unwrap();

    // Separate documents in DE (`1.234,5`) and US (`1,234.5`).
    let mut docs = vec![
        with_ballot("Die Lieferung wiegt 1.234,5 kg und misst 2.500 m.", &[("DE", 1.0)]),
        with_ballot("The shipment weighs 1,234.5 kg and spans 2,500 m.", &[("US", 1.0)]),
    ];
    let results = a.run(&mut docs);
    assert!(results.iter().all(Result::is_ok), "{results:?}");

    let de_nums = docs[0].get::<Numbers>().unwrap();
    assert_eq!(de_nums.len(), 2);
    assert_eq!(de_nums.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(de_nums.notation[0], NumberNotation::LatinEu);
    assert_eq!(de_nums.value(Ix::new(1)).to_plain_string(), "2500");
    assert_eq!(de_nums.notation[1], NumberNotation::LatinEu);

    let de_qty = docs[0].get::<Quantities>().unwrap();
    assert_eq!(de_qty.len(), 2);
    assert_eq!(de_qty.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(de_qty.unit(Ix::new(0)), "kg");
    assert_eq!(de_qty.value(Ix::new(1)).to_plain_string(), "2500");
    assert_eq!(de_qty.unit(Ix::new(1)), "m");

    let us_nums = docs[1].get::<Numbers>().unwrap();
    assert_eq!(us_nums.len(), 2);
    assert_eq!(us_nums.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(us_nums.notation[0], NumberNotation::LatinUs);
    assert_eq!(us_nums.value(Ix::new(1)).to_plain_string(), "2500");
    assert_eq!(us_nums.notation[1], NumberNotation::LatinUs);

    let us_qty = docs[1].get::<Quantities>().unwrap();
    assert_eq!(us_qty.len(), 2);
    assert_eq!(us_qty.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(us_qty.unit(Ix::new(0)), "kg");
    assert_eq!(us_qty.value(Ix::new(1)).to_plain_string(), "2500");
    assert_eq!(us_qty.unit(Ix::new(1)), "m");

    // A single mixed-region document carrying both DE and US votes on the ballot.
    let mixed = run_one(
        &a,
        with_ballot("Berlin meldet 1.234,5 kg und New York reports 1,234.5 kg.", &[("DE", 0.6), ("US", 0.4)]),
    );
    let m_nums = mixed.get::<Numbers>().unwrap();
    assert_eq!(m_nums.len(), 2);
    assert_eq!(m_nums.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(m_nums.notation[0], NumberNotation::LatinEu);
    assert_eq!(m_nums.value(Ix::new(1)).to_plain_string(), "1234.5");
    assert_eq!(m_nums.notation[1], NumberNotation::LatinUs);

    let m_qty = mixed.get::<Quantities>().unwrap();
    assert_eq!(m_qty.len(), 2);
    assert_eq!(m_qty.value(Ix::new(0)).to_plain_string(), "1234.5");
    assert_eq!(m_qty.unit(Ix::new(0)), "kg");
    assert_eq!(m_qty.value(Ix::new(1)).to_plain_string(), "1234.5");
    assert_eq!(m_qty.unit(Ix::new(1)), "kg");
}
