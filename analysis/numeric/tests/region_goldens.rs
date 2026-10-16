//! Compares `MoneyExtractor::for_region` and `RegionAwareMoneyAnnotator`
//! against `tests/vectors/region_for_region.tsv`, written by
//! `java/RegionMoneyVectors.java` on the pinned JDK 21 image
//! (`Temurin-21.0.12.1+1`, OpenNLP commit `cdf33c653d5d61d5af7d37ffee189a83bd8392ea`)
//! across all 249 `Locale.getISOCountries()` plus the unassigned code `ZZ`.

use turbo_analysis::{Analyzer, Document, Ix, Key, Layer, Provenance};
use turbo_analysis_layers::{Money, NumberNotation, RegionCode, RegionVote, Regions};
use turbo_numeric::money::{MoneyExtractor, RegionAwareMoneyAnnotator};

const PROBE_TEXT: &str = "$1,234.56 $1.234,56 \u{20AC}1,234.56 \u{20AC}1.234,56 \u{00A3}50 \u{00A5}100 \
\u{060B}20 \u{058F}30 \u{20BC}40 \u{09F3}50 \u{20A1}60 \u{20BE}70 \u{20AA}80 \
\u{20B9}90 \u{17DB}10 \u{20A9}20 \u{20B8}30 \u{20AD}40 \u{20AE}50 \u{20A6}60 \
\u{20B1}70 \u{20BD}80 \u{0E3F}90 \u{20BA}10 \u{20B4}20 \u{20AB}30";

const VECTORS: &str = include_str!("vectors/region_for_region.tsv");

fn format_mentions(money: &Money) -> String {
    (0..money.len())
        .map(|i| {
            let ix = Ix::new(i as u32);
            let m = money.get(ix);
            format!("{}..{}:{}:{}", m.span.start, m.span.end, money.amount(ix).to_plain_string(), m.currency.as_str())
        })
        .collect::<Vec<_>>()
        .join(";")
}

fn format_symbols(extractor: &MoneyExtractor) -> String {
    let mut pairs: Vec<(u32, String)> =
        extractor.symbols().iter().map(|&(ch, cur)| (ch as u32, cur.as_str().to_owned())).collect();
    pairs.sort_by_key(|&(cp, _)| cp);
    pairs.into_iter().map(|(cp, cur)| format!("U+{cp:04X}={cur}")).collect::<Vec<_>>().join(",")
}

#[test]
fn every_iso_country_matches_the_jdk21_for_region_golden() {
    let mut lines = VECTORS.lines();
    let header = lines.next().expect("vector file has a header");
    assert!(header.starts_with("# generator=RegionMoneyVectors.java\topennlp.commit="));

    let analyzer =
        Analyzer::builder().add(RegionAwareMoneyAnnotator::new()).given(Key::<Regions>::DEFAULT).build().unwrap();

    let mut count = 0usize;
    for (line_no, line) in lines.enumerate() {
        let fields: Vec<&str> = line.split('\t').collect();
        assert_eq!(fields.len(), 5, "line {}: expected 5 fields", line_no + 2);
        let (cc_str, _locale, notation_str, expected_symbols, expected_mentions) =
            (fields[0], fields[1], fields[2], fields[3], fields[4]);

        let rc = RegionCode::parse(cc_str).unwrap_or_else(|e| panic!("{cc_str}: {e}"));
        let expected_notation = match notation_str {
            "LATIN_US" => NumberNotation::LatinUs,
            "LATIN_EU" => NumberNotation::LatinEu,
            other => panic!("{cc_str}: unexpected notation {other}"),
        };

        let extractor = MoneyExtractor::for_region(rc);
        assert_eq!(extractor.notation(), expected_notation, "{cc_str}: notation mismatch");
        assert_eq!(format_symbols(&extractor), expected_symbols, "{cc_str}: symbol table mismatch");

        let mut doc = Document::new(PROBE_TEXT);
        let mut regions = Regions::new();
        regions.push(RegionVote::new(rc, 1.0).unwrap());
        doc.add(&Key::DEFAULT, regions, Provenance::by("test")).unwrap();

        analyzer.run(std::slice::from_mut(&mut doc)).into_iter().next().unwrap().unwrap();
        let money = doc.get::<Money>().unwrap();
        assert_eq!(format_mentions(money), expected_mentions, "{cc_str}: extracted mentions mismatch");
        count += 1;
    }
    assert_eq!(count, 250, "expected 249 ISO countries plus ZZ");
}
