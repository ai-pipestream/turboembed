use turbo_analysis::{Annotator, Ix, Layer, Span, ViewId};
use turbo_numeric::{NumberNotation, Quantities, QuantityExtractor};

#[test]
fn parity_selects_legacy_money_exclusion_without_changing_native_defaults() {
    for notation in [NumberNotation::LatinUs, NumberNotation::LatinEu] {
        let native = QuantityExtractor::new(notation);
        assert!(!native.opennlp_parity());
        let parity = native.clone().with_opennlp_parity(true);
        assert!(parity.opennlp_parity());
        for (text, start) in [("A$2m", 2), ("US$2m", 3), ("HK$2m", 3)] {
            let mut actual = Quantities::new(ViewId::SOURCE);
            native.scan(text, &mut actual);
            assert!(actual.spans.is_empty(), "native money exclusion: {text}");
            parity.scan(text, &mut actual);
            actual.check().unwrap();
            assert_eq!(actual.spans.iter().map(|(_, span)| span).collect::<Vec<_>>(), [Span::new(start, start + 2)]);
            let value = actual.value(Ix::new(0));
            assert_eq!((value.negative, value.digits, value.scale, actual.unit(Ix::new(0))), (false, "2", 0, "m"));
        }
        let mut ordinary_money = Quantities::new(ViewId::SOURCE);
        parity.scan("$2m", &mut ordinary_money);
        assert!(ordinary_money.spans.is_empty());
    }
}

#[test]
fn parity_identity_is_distinct_and_native_identity_is_preserved() {
    // Captured from the real default extractor before adding the option.
    for (notation, expected) in
        [(NumberNotation::LatinUs, 10060711523185091347), (NumberNotation::LatinEu, 5786200242830044788)]
    {
        let native = QuantityExtractor::new(notation);
        assert_eq!(native.config_hash(), expected);
        let parity = native.clone().with_opennlp_parity(true);
        assert_ne!(parity.config_hash(), native.config_hash());
        let restored = parity.with_opennlp_parity(false);
        assert!(!restored.opennlp_parity());
        assert_eq!(restored.config_hash(), native.config_hash());
    }
}
