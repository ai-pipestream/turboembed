//! Extended number groupings, exponents, and non-ASCII digit scripts (#931):
//! Indian lakh/crore grouping, thin and no-break space groups, Swiss
//! apostrophe groups, scientific notation exponents, and Unicode `Nd` decimal
//! digits (Arabic-Indic with Arabic separators, Extended Arabic-Indic,
//! Devanagari, Fullwidth), plus the OpenNLP parity toggle.

mod common;

use common::{eu, found, found_with_parity, pairs, us};
use turbo_analysis::{Annotator, Document, Ix, Layer, Span, View, ViewId};
use turbo_numeric::NumberNotation::LatinUs;
use turbo_numeric::{
    Money, MoneyExtractor, NumberScanner, Numbers, Quantities, QuantityExtractor, RegionAwareMoneyAnnotator,
    RegionAwareNumberAnnotator, RegionAwareQuantityAnnotator, RegionCode, RegionVote, Regions,
};

fn scan_numbers(text: &str, scanner: NumberScanner) -> Numbers {
    let mut out = Numbers::new(ViewId::SOURCE);
    scanner.scan(text, &mut out);
    assert_eq!(out.check(), Ok(()), "{text:?}");
    out
}

#[test]
fn indian_lakh_and_crore_grouping() {
    let cases = [
        ("1,00,000", "100000"),
        ("12,34,567", "1234567"),
        ("12,34,567.89", "1234567.89"),
        ("1,23,45,678", "12345678"),
        ("99,99,99,999.5", "999999999.5"),
        ("-5,00,000", "-500000"),
        ("1,00,000k", "100000000"),
    ];
    for (text, expected) in cases {
        assert_eq!(us(text), pairs(&[(text, expected)]), "{text}");
        assert_eq!(found_with_parity(text, LatinUs, true, true), [], "{text} parity");
        assert_eq!(eu(text), [], "{text} in LatinEu");
    }
    // Malformed comma groupings are still rejected whole in LatinUs.
    for bad in ["1,23", "12,34", "123,45,678", "1,000,00", "12,3456,789", "1.00.000"] {
        assert_eq!(us(bad), [], "{bad}");
    }
    for bad in ["123,45,678", "1,000,00", "12,3456,789", "1.00.000"] {
        assert_eq!(eu(bad), [], "{bad}");
    }
    // Region-aware scanner for India (`IN`) reads lakh grouping.
    let in_scanner = NumberScanner::for_region(RegionCode::parse("IN").unwrap());
    let n = scan_numbers("Budget of 12,34,567.50 approved", in_scanner);
    assert_eq!(n.len(), 1);
    assert_eq!(n.value(Ix::new(0)).to_plain_string(), "1234567.50");
}

#[test]
fn thin_and_no_break_space_groups() {
    for sep in ['\u{00A0}', '\u{2009}', '\u{202F}'] {
        let single_group = format!("1{sep}000");
        assert_eq!(us(&single_group), pairs(&[(&single_group, "1000")]));
        assert_eq!(eu(&single_group), pairs(&[(&single_group, "1000")]));
        assert_eq!(found_with_parity(&single_group, LatinUs, true, true), []);

        let multi_group = format!("1{sep}234{sep}567");
        assert_eq!(us(&multi_group), pairs(&[(&multi_group, "1234567")]));
        assert_eq!(eu(&multi_group), pairs(&[(&multi_group, "1234567")]));

        let with_dot_frac = format!("1{sep}234.56");
        assert_eq!(us(&with_dot_frac), pairs(&[(&with_dot_frac, "1234.56")]));

        let with_comma_frac = format!("1{sep}350,75");
        assert_eq!(eu(&with_comma_frac), pairs(&[(&with_comma_frac, "1350.75")]));
        assert_eq!(us(&with_comma_frac), pairs(&[(&with_comma_frac, "1350.75")]));

        let with_scale = format!("1{sep}200{sep}million");
        assert_eq!(us(&with_scale), pairs(&[(&with_scale, "1200000000")]));

        // Non-3-digit group after a thin/no-break space is still rejected.
        for bad in [format!("1{sep}23"), format!("1{sep}2345"), format!("1{sep}234{sep}56"), format!("1234{sep}567")] {
            assert_eq!(us(&bad), [], "{bad:?}");
        }
    }
}

#[test]
fn swiss_apostrophe_grouping() {
    for sep in ['\'', '\u{2019}'] {
        let g1 = format!("1{sep}000");
        assert_eq!(us(&g1), pairs(&[(&g1, "1000")]));
        assert_eq!(eu(&g1), pairs(&[(&g1, "1000")]));
        assert_eq!(found_with_parity(&g1, LatinUs, true, true), []);

        let g2 = format!("1{sep}234{sep}567.89");
        assert_eq!(us(&g2), pairs(&[(&g2, "1234567.89")]));

        let g3 = format!("1{sep}234,56");
        assert_eq!(eu(&g3), pairs(&[(&g3, "1234.56")]));
        assert_eq!(us(&g3), pairs(&[(&g3, "1234.56")]));

        for bad in [format!("1{sep}23"), format!("1{sep}2345"), format!("1{sep}234{sep}56")] {
            assert_eq!(us(&bad), [], "{bad:?}");
        }
    }
    let ch_scanner = NumberScanner::for_region(RegionCode::parse("CH").unwrap());
    let n = scan_numbers("CHF 1'234'567.50 und 2\u{2019}500", ch_scanner);
    assert_eq!(n.len(), 2);
    assert_eq!(n.value(Ix::new(0)).to_plain_string(), "1234567.50");
    assert_eq!(n.value(Ix::new(1)).to_plain_string(), "2500");
}

#[test]
fn scientific_notation_exponents() {
    let us_cases = [
        ("1e3", "1000"),
        ("1E3", "1000"),
        ("1e+3", "1000"),
        ("1e-3", "0.001"),
        ("1E-03", "0.001"),
        ("2.5e-3", "0.0025"),
        ("2,500e+3", "2500000"),
        ("-2.5e-3", "-0.0025"),
        ("0e-3", "0.000"),
        ("0.0e5", "0"),
    ];
    for (text, expected) in us_cases {
        assert_eq!(us(text), pairs(&[(text, expected)]), "{text}");
        assert_eq!(found(text, LatinUs, false), pairs(&[(text, expected)]), "{text} unscaled");
        assert_eq!(found_with_parity(text, LatinUs, true, true), [], "{text} parity");
    }
    assert_eq!(eu("2,5e-3"), pairs(&[("2,5e-3", "0.0025")]));
    assert_eq!(eu("2.500e+3"), pairs(&[("2.500e+3", "2500000")]));

    // Malformed or chained exponents are rejected whole without splitting.
    for bad in ["1.e-3", "1,e3", "1e-3e-4", "4e5x", "1e3k"] {
        assert_eq!(us(bad), [], "{bad}");
        assert_eq!(eu(bad), [], "{bad}");
    }
}

#[test]
fn non_ascii_digits_and_arabic_separators() {
    // Arabic-Indic (U+0660..=U+0669) and Arabic separators (U+066B, U+066C).
    let cases = [
        ("\u{0660}\u{0661}\u{0662}\u{0663}", "123"),
        ("\u{0661}\u{0662}\u{0663}", "123"),
        ("-\u{0664}\u{0665}", "-45"),
        ("1\u{066C}234", "1234"),
        ("1\u{066B}5", "1.5"),
        ("1\u{066C}234\u{066B}56", "1234.56"),
        ("\u{0661}\u{066C}\u{0662}\u{0663}\u{0664}\u{066B}\u{0665}\u{0666}", "1234.56"),
        // Extended Arabic-Indic (U+06F0..=U+06F9).
        ("\u{06F1}\u{06F2}\u{06F3}\u{066B}\u{06F4}", "123.4"),
        // Devanagari (U+0966..=U+096F), including Indian lakh grouping.
        ("\u{0967}\u{0968}\u{0969}", "123"),
        ("\u{0967},\u{0966}\u{0966},\u{0966}\u{0966}\u{0966}", "100000"),
        // Fullwidth (U+FF10..=U+FF19) with ASCII and fullwidth separators.
        ("\u{FF11}\u{FF12}\u{FF13}", "123"),
        ("\u{FF11},\u{FF12}\u{FF13}\u{FF14}", "1234"),
        ("\u{FF11}\u{FF0C}\u{FF12}\u{FF13}\u{FF14}\u{FF0E}\u{FF15}", "1234.5"),
    ];
    for (text, expected) in cases {
        let n = scan_numbers(text, NumberScanner::default());
        assert_eq!(n.len(), 1, "{text:?}");
        assert_eq!(n.spans.get(Ix::new(0)), Span::new(0, text.len() as u32), "{text:?}");
        assert_eq!(n.value(Ix::new(0)).to_plain_string(), expected, "{text:?}");
        assert_eq!(found_with_parity(text, LatinUs, true, true), [], "{text:?} parity");
    }

    // ASCII digits separated by fullwidth comma/period (`U+FF0C`, `U+FF0E`) in CJK text
    // keep both numbers in both default and OpenNLP-parity modes.
    for text in ["5\u{FF0C}3", "a 5\u{FF0C}3 b", "5\u{FF0E}3", "a 5\u{FF0E}3 b"] {
        assert_eq!(us(text), pairs(&[("5", "5"), ("3", "3")]), "{text:?} default");
        assert_eq!(found_with_parity(text, LatinUs, true, true), pairs(&[("5", "5"), ("3", "3")]), "{text:?} parity");
    }
    // Malformed fullwidth numeral with chained fullwidth periods is still rejected whole.
    assert_eq!(us("\u{FF11}\u{FF0E}\u{FF12}\u{FF0E}\u{FF13}"), []);
}

#[test]
fn quantities_and_money_support_extended_numbers_and_regions() {
    // Quantities with lakh grouping, thin spaces, Swiss apostrophes, exponents, and non-ASCII digits.
    let q_ext = QuantityExtractor::default();
    for (text, val, unit) in [
        ("1,00,000 kg", "100000", "kg"),
        ("1\u{202F}250 GB", "1250", "GB"),
        ("1'500 km", "1500", "km"),
        ("2.5e-3 kg", "0.0025", "kg"),
        ("\u{0968}\u{096B} kg", "25", "kg"),
        ("\u{0665}\u{0660}%", "50", "%"),
    ] {
        let mut q = Quantities::new(ViewId::SOURCE);
        q_ext.scan(text, &mut q);
        assert_eq!(q.check(), Ok(()), "{text:?}");
        assert_eq!(q.len(), 1, "{text:?}");
        assert_eq!(q.value(Ix::new(0)).to_plain_string(), val, "{text:?}");
        assert_eq!(q.unit(Ix::new(0)), unit, "{text:?}");
    }

    // Money with lakh grouping, Swiss apostrophes, thin spaces, exponents, and non-ASCII digits.
    let m_ext = MoneyExtractor::default();
    for (text, val, cur) in [
        ("\u{20B9}12,34,567.50", "1234567.50", "INR"),
        ("CHF 1'234.50", "1234.50", "CHF"),
        ("1\u{00A0}350.75 euros", "1350.75", "EUR"),
        ("$1.5e6", "1500000", "USD"),
        ("$\u{FF11}\u{FF10}\u{FF10}", "100", "USD"),
    ] {
        let mut m = Money::new(ViewId::SOURCE);
        m_ext.scan(text, &mut m);
        assert_eq!(m.check(), Ok(()), "{text:?}");
        assert_eq!(m.len(), 1, "{text:?}");
        assert_eq!(m.amount(Ix::new(0)).to_plain_string(), val, "{text:?}");
        assert_eq!(m.currency[0].as_str(), cur, "{text:?}");
    }

    // Full region-aware annotators on a document with Swiss (`CH`) ballot.
    let doc = Document::new("Paid CHF 1'250.50 for 2\u{202F}500 kg.");
    let view = doc.get::<View>().unwrap();
    let mut regions = Regions::new();
    regions.push(RegionVote::new(RegionCode::parse("CH").unwrap(), 1.0).unwrap());
    let (nums,) = RegionAwareNumberAnnotator::new().annotate((view, &regions)).unwrap();
    let (quants,) = RegionAwareQuantityAnnotator::new().annotate((view, &regions)).unwrap();
    let (money,) = RegionAwareMoneyAnnotator::new().annotate((view, &regions)).unwrap();
    assert_eq!(nums.len(), 2);
    assert_eq!(quants.len(), 1);
    assert_eq!(quants.value(Ix::new(0)).to_plain_string(), "2500");
    assert_eq!(money.len(), 1);
    assert_eq!(money.amount(Ix::new(0)).to_plain_string(), "1250.50");
}

#[test]
fn all_golden_inputs_satisfy_layer_and_offset_invariants_in_default_extended_mode() {
    use turbo_analysis::Analyzer;
    use turbo_goldens::data_dir;
    use turbo_goldens::invariants::assert_invariants;
    use turbo_numeric::NumberNotation::{LatinEu, LatinUs};

    for notation in [LatinUs, LatinEu] {
        let analyzer = Analyzer::builder()
            .add(NumberScanner::new(notation))
            .add(QuantityExtractor::new(notation))
            .add(MoneyExtractor::new(notation))
            .build()
            .unwrap();
        for set in ["num", "quantity", "money", "multi", "edge", "formats", "surrogate"] {
            let dir = data_dir().join("inputs").join(set);
            for entry in std::fs::read_dir(&dir).unwrap() {
                let path = entry.unwrap().path();
                if path.extension().and_then(|e| e.to_str()) != Some("txt") {
                    continue;
                }
                let text = std::fs::read_to_string(&path).unwrap();
                let mut doc = Document::new(text.as_str());
                for res in analyzer.run(std::slice::from_mut(&mut doc)) {
                    res.unwrap();
                }
                assert_invariants(&doc, &format!("{} ({notation:?})", path.display()));
            }
        }
    }
}

#[test]
#[ignore]
fn timing_receipt_interleaved_release() {
    use std::hint::black_box;
    use std::time::Instant;
    use turbo_goldens::data_dir;

    let mut ascii_corpus = String::new();
    for name in ["prose-english.txt", "prose-german.txt", "decimals-us.txt", "eu-grouping.txt", "both-notations.txt"] {
        let s = std::fs::read_to_string(data_dir().join("inputs/num").join(name)).unwrap();
        for _ in 0..200 {
            ascii_corpus.push_str(&s);
            ascii_corpus.push('\n');
        }
    }

    let mut non_latin_corpus = String::new();
    let non_latin_sample = concat!(
        "भारत में कुल जनसंख्या और बजट अनुमान 125,000 करोड़ रुपये तथा 24.5 प्रतिशत वृद्धि दर्ज की गई।\n",
        "ประเทศไทยมีประชากร लगभग 66 ล้านคน และงบประมาณ 3,500.75 ล้านบาทในปีนี้\n",
        "بلغ إجمالي الإنتاج 12,345.67 طن متري في الربع الثالث من العام.\n",
        "東京都の統計データ（2026年度）：売上高は1,250.5億円、前年比＋12.4％を達成しました！😀🚀📦\n",
    );
    for _ in 0..2_500 {
        non_latin_corpus.push_str(non_latin_sample);
    }

    let mut non_latin_nd_corpus = String::new();
    let non_latin_nd_sample = concat!(
        "भारत में कुल जनसंख्या और बजट अनुमान १,२५,००० करोड़ रुपये तथा २४.५ प्रतिशत वृद्धि दर्ज की गई।\n",
        "ประเทศไทยมีประชากร लगभग 66 ล้านคน และงบประมาณ 3,500.75 ล้านบาทในปีนี้\n",
        "بلغ إجمالي الإنتاج ١٢٬٣٤٥٫٦٧ طن متري في الربع الثالث من العام.\n",
        "東京都の統計データ（２０２６年度）：売上高は１，２５０．５億円、前年比＋12.4％を達成しました！😀🚀📦\n",
    );
    for _ in 0..2_500 {
        non_latin_nd_corpus.push_str(non_latin_nd_sample);
    }

    let baseline = NumberScanner::new(LatinUs).with_opennlp_parity(true);
    let extended = NumberScanner::new(LatinUs);
    let mut out = Numbers::new(ViewId::SOURCE);
    let iters = 100;
    if let Ok(which) = std::env::var("TURBO_BENCH_CORPUS") {
        let corpus = match which.as_str() {
            "non_latin" => &non_latin_corpus,
            "non_latin_nd" => &non_latin_nd_corpus,
            _ => &ascii_corpus,
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            out.spans = turbo_analysis::Spans::new(ViewId::SOURCE);
            out.numeral_end.clear();
            out.negative.clear();
            out.coefficient = Default::default();
            out.scale.clear();
            out.multiplier.clear();
            out.marker.clear();
            out.notation.clear();
            extended.scan(black_box(corpus.as_str()), &mut out);
            black_box(out.len());
        }
        let d = t0.elapsed();
        println!(
            "corpus={which} bytes={} iters={iters} ms={:.2} mb_s={:.1} mentions={}",
            corpus.len(),
            d.as_secs_f64() * 1000.0,
            (corpus.len() as f64 * iters as f64 / 1e6) / d.as_secs_f64(),
            out.len(),
        );
        return;
    }
    for (label, corpus) in [
        ("ascii", &ascii_corpus),
        ("non_latin", &non_latin_corpus),
        ("non_latin_nd", &non_latin_nd_corpus),
    ] {
        for round in 1..=6 {
            let t0 = Instant::now();
            for _ in 0..iters {
                out.spans = turbo_analysis::Spans::new(ViewId::SOURCE);
                out.numeral_end.clear();
                out.negative.clear();
                out.coefficient = Default::default();
                out.scale.clear();
                out.multiplier.clear();
                out.marker.clear();
                out.notation.clear();
                baseline.scan(black_box(corpus.as_str()), &mut out);
                black_box(out.len());
            }
            let d_base = t0.elapsed();

            let t1 = Instant::now();
            for _ in 0..iters {
                out.spans = turbo_analysis::Spans::new(ViewId::SOURCE);
                out.numeral_end.clear();
                out.negative.clear();
                out.coefficient = Default::default();
                out.scale.clear();
                out.multiplier.clear();
                out.marker.clear();
                out.notation.clear();
                extended.scan(black_box(corpus.as_str()), &mut out);
                black_box(out.len());
            }
            let d_ext = t1.elapsed();
            eprintln!(
                "[{label}] round {round}: parity={:.2}ms ({:.1} MB/s) | extended={:.2}ms ({:.1} MB/s)",
                d_base.as_secs_f64() * 1000.0,
                (corpus.len() as f64 * iters as f64 / 1e6) / d_base.as_secs_f64(),
                d_ext.as_secs_f64() * 1000.0,
                (corpus.len() as f64 * iters as f64 / 1e6) / d_ext.as_secs_f64(),
            );
        }
    }
}
