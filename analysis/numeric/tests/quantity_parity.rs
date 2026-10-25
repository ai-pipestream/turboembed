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
fn parity_identity_preserves_current_main_modes_and_units() {
    // Actual public API values captured on incoming main f3d10cb7.
    let cases = [
        (
            NumberNotation::LatinUs,
            [15676901307674456763, 6775125584621861070, 6040691781845600814, 18243062135851126261],
        ),
        (
            NumberNotation::LatinEu,
            [6241264112553325702, 10819214315860654899, 17423958123388179277, 8689324141470668998],
        ),
    ];
    for (notation, expected) in cases {
        let native = QuantityExtractor::new(notation);
        let parity = native.clone().with_opennlp_parity(true);
        let custom = QuantityExtractor::with_units(["kg", "km"], notation).unwrap();
        let actual = [
            native.config_hash(),
            parity.config_hash(),
            custom.config_hash(),
            custom.with_opennlp_parity(true).config_hash(),
        ];
        assert_eq!(actual, expected);
        assert_ne!(parity.config_hash(), native.config_hash());
        let restored = parity.with_opennlp_parity(false);
        assert!(!restored.opennlp_parity());
        assert_eq!(restored.config_hash(), native.config_hash());
    }
}

#[test]
fn one_parity_setting_controls_numeric_scanning_and_money_exclusion() {
    let native = QuantityExtractor::default();
    let parity = native.clone().with_opennlp_parity(true);
    let restored = parity.clone().with_opennlp_parity(false);
    for text in ["1,00,000 kg", "١٢ kg"] {
        let mut normal = Quantities::new(ViewId::SOURCE);
        native.scan(text, &mut normal);
        assert_eq!(normal.spans.len(), 1, "native: {text}");
        let mut strict = Quantities::new(ViewId::SOURCE);
        parity.scan(text, &mut strict);
        assert!(strict.spans.is_empty(), "parity: {text}");
        let mut again = Quantities::new(ViewId::SOURCE);
        restored.scan(text, &mut again);
        assert_eq!(
            again.spans.iter().map(|(_, s)| s).collect::<Vec<_>>(),
            normal.spans.iter().map(|(_, s)| s).collect::<Vec<_>>()
        );
        assert_eq!(again.value(Ix::new(0)), normal.value(Ix::new(0)));
    }
}
