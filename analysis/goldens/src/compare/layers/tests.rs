use turbo_analysis::{Document, IxRange, Key, Layer, Provenance, Span, View, ViewBuilder, ViewId};
use turbo_analysis_layers::{
    Currency, Money, MoneyMention, Multiplier, NumberMention, NumberNotation, Numbers, Opinion, Opinions, Quantities,
    QuantityMention, ScaleMarker, Strs,
};

use super::super::{Cx, GoldenCompare, Mismatch, Tolerance};
use crate::format::{
    AnnotatorRef, Column, ColumnData, Generator, Golden, GoldenAnchor, GoldenLayer, Header, Input, Jdk, OpenNlp,
    UnicodeVersion,
};
use crate::sha::sha256;

fn dummy_golden(text: &str, layers: Vec<GoldenLayer>) -> Golden {
    Golden {
        header: Header {
            generator: Generator { name: "test".into(), version: 1 },
            opennlp: OpenNlp { commit: [0; 20], version: "2.5.7".into() },
            jdk: Jdk { version: "21".into(), vm_version: "21".into(), vendor: "test".into() },
            unicode: UnicodeVersion { major: 15, minor: 0 },
            input: Input {
                set: "test".into(),
                id: "0".into(),
                utf8_len: text.len() as u32,
                utf16_len: text.encode_utf16().count() as u32,
                sha256: sha256(text.as_bytes()),
            },
            models: Vec::new(),
            annotator: AnnotatorRef { class: "test".into(), method: "test".into() },
            config: Vec::new(),
            error: None,
        },
        layers,
    }
}

fn strs_col(values: &[&str]) -> Strs {
    let mut out = Strs::default();
    for v in values {
        out.push(v);
    }
    out
}

#[test]
fn numbers_compare_maps_numeral_end_from_derived_view_to_source() {
    // Source text has a 3-byte prefix replaced by a 1-byte space in the derived view:
    // Source: "___12k" (len 6; "12k" is 3..6, numeral "12" is 3..5)
    // View:   " 12k"   (len 4; "12k" is 1..4, numeral "12" is 1..3)
    let source = "___12k";
    let mut doc = Document::new(source);
    let source_view = doc.get::<View>().unwrap().clone();
    let mut vb = ViewBuilder::new(source);
    vb.replace(3, " ");
    let (view_text, alignment) = vb.finish();
    let view = View::derived(&source_view, view_text, alignment);
    let view_key: Key<View> = Key::named("norm");
    doc.add(&view_key, view, Provenance::by("test")).unwrap();
    let derived_view_id = doc.get_key(&view_key).unwrap().id();

    let mut numbers = Numbers::new(derived_view_id);
    numbers.push(NumberMention {
        span: Span::new(1, 4),
        numeral_end: 3,
        negative: false,
        coefficient: "12000",
        scale: 0,
        multiplier: Multiplier::Thousand,
        marker: ScaleMarker::Suffix,
        notation: NumberNotation::LatinUs,
    });
    doc.add(&Key::<Numbers>::DEFAULT, numbers, Provenance::by("test")).unwrap();

    let golden_layer = GoldenLayer {
        id: Numbers::ID.into(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            Column { name: "start".into(), data: ColumnData::U32(vec![3]) },
            Column { name: "end".into(), data: ColumnData::U32(vec![6]) },
            Column { name: "numeral_end".into(), data: ColumnData::U32(vec![5]) },
            Column { name: "negative".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "coefficient".into(), data: ColumnData::Str(strs_col(&["12000"])) },
            Column { name: "scale".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "plain".into(), data: ColumnData::Str(strs_col(&["12000"])) },
        ],
    };
    let golden = dummy_golden(source, vec![golden_layer]);
    let cx = Cx { doc: &doc, key: Key::<Numbers>::DEFAULT.id(), golden: &golden, layer: Numbers::ID.into() };
    let rust_numbers = doc.get::<Numbers>().unwrap();
    assert_eq!(rust_numbers.compare(&cx, &golden.layers[0], Tolerance::exact()), Ok(()));
}

#[test]
fn compare_multi_values_does_not_attach_later_row_float_column_to_earlier_mismatch() {
    let source = "good bad";
    let doc = Document::new(source);

    let mut rust_ops = Opinions::default();
    rust_ops.push(Opinion { tokens: IxRange::new(0, 1), polarity: 0.8, intensity: 0.5, negated: false });
    rust_ops.push(Opinion { tokens: IxRange::new(1, 2), polarity: -0.7, intensity: 0.4, negated: false });

    // Row 0 differs in token_end (1 vs 2), while polarity & intensity match at row 0.
    // Row 1 differs in intensity (0.4 vs 0.9), while polarity matches at row 1.
    let golden_layer = GoldenLayer {
        id: Opinions::ID.into(),
        anchor: Opinions::golden_anchor(),
        columns: vec![
            Column { name: "token_start".into(), data: ColumnData::U32(vec![0, 1]) },
            Column { name: "token_end".into(), data: ColumnData::U32(vec![2, 2]) },
            Column { name: "polarity".into(), data: ColumnData::F64(vec![0.8, -0.7]) },
            Column { name: "intensity".into(), data: ColumnData::F64(vec![0.5, 0.9]) },
            Column { name: "negated".into(), data: ColumnData::U32(vec![0, 0]) },
        ],
    };
    let golden = dummy_golden(source, vec![golden_layer]);
    let cx = Cx { doc: &doc, key: Key::<Opinions>::DEFAULT.id(), golden: &golden, layer: Opinions::ID.into() };
    let err = rust_ops.compare(&cx, &golden.layers[0], Tolerance::exact()).unwrap_err();
    match err {
        Mismatch::Item { index, what, rust, golden, .. } => {
            assert_eq!(index, 0);
            assert_eq!(what, "tokens");
            // Row 0 should display the primary value column ("polarity"), not row 1's mismatched "intensity".
            assert_eq!(rust.value, Some(("polarity", 0.8)));
            assert_eq!(golden.value, Some(("polarity", 0.8)));
        }
        other => panic!("expected Mismatch::Item, got {other:?}"),
    }
}

#[test]
fn numeric_layers_reject_invalid_boolean_negative_in_golden() {
    let source = "10 $10 10kg";
    let doc = Document::new(source);

    let mut numbers = Numbers::new(ViewId::SOURCE);
    numbers.push(NumberMention {
        span: Span::new(0, 2),
        numeral_end: 2,
        negative: false,
        coefficient: "10",
        scale: 0,
        multiplier: Multiplier::One,
        marker: ScaleMarker::None,
        notation: NumberNotation::LatinUs,
    });
    let bad_numbers_golden = GoldenLayer {
        id: Numbers::ID.into(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            Column { name: "start".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "end".into(), data: ColumnData::U32(vec![2]) },
            Column { name: "numeral_end".into(), data: ColumnData::U32(vec![2]) },
            Column { name: "negative".into(), data: ColumnData::U32(vec![2]) },
            Column { name: "coefficient".into(), data: ColumnData::Str(strs_col(&["10"])) },
            Column { name: "scale".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "plain".into(), data: ColumnData::Str(strs_col(&["10"])) },
        ],
    };

    let mut money = Money::new(ViewId::SOURCE);
    money.push(MoneyMention {
        span: Span::new(3, 6),
        negative: false,
        coefficient: "10",
        scale: 0,
        currency: Currency::parse("USD").unwrap(),
    });
    let bad_money_golden = GoldenLayer {
        id: Money::ID.into(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            Column { name: "start".into(), data: ColumnData::U32(vec![3]) },
            Column { name: "end".into(), data: ColumnData::U32(vec![6]) },
            Column { name: "coefficient".into(), data: ColumnData::Str(strs_col(&["10"])) },
            Column { name: "negative".into(), data: ColumnData::U32(vec![3]) },
            Column { name: "scale".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "plain".into(), data: ColumnData::Str(strs_col(&["10"])) },
            Column { name: "currency".into(), data: ColumnData::Str(strs_col(&["USD"])) },
        ],
    };

    let mut quantities = Quantities::new(ViewId::SOURCE);
    quantities.push(QuantityMention {
        span: Span::new(7, 11),
        negative: false,
        coefficient: "10",
        scale: 0,
        unit: "kg",
    });
    let bad_quantities_golden = GoldenLayer {
        id: Quantities::ID.into(),
        anchor: GoldenAnchor::Text,
        columns: vec![
            Column { name: "start".into(), data: ColumnData::U32(vec![7]) },
            Column { name: "end".into(), data: ColumnData::U32(vec![11]) },
            Column { name: "coefficient".into(), data: ColumnData::Str(strs_col(&["10"])) },
            Column { name: "negative".into(), data: ColumnData::U32(vec![5]) },
            Column { name: "scale".into(), data: ColumnData::U32(vec![0]) },
            Column { name: "plain".into(), data: ColumnData::Str(strs_col(&["10"])) },
            Column { name: "unit".into(), data: ColumnData::Str(strs_col(&["kg"])) },
        ],
    };

    let golden = dummy_golden(source, vec![bad_numbers_golden, bad_money_golden, bad_quantities_golden]);

    let cx_num = Cx { doc: &doc, key: Key::<Numbers>::DEFAULT.id(), golden: &golden, layer: Numbers::ID.into() };
    assert!(matches!(
        numbers.compare(&cx_num, &golden.layers[0], Tolerance::exact()),
        Err(Mismatch::BadGolden { ref detail, .. }) if detail.contains("negative 2")
    ));

    let cx_mon = Cx { doc: &doc, key: Key::<Money>::DEFAULT.id(), golden: &golden, layer: Money::ID.into() };
    assert!(matches!(
        money.compare(&cx_mon, &golden.layers[1], Tolerance::exact()),
        Err(Mismatch::BadGolden { ref detail, .. }) if detail.contains("negative 3")
    ));

    let cx_qty = Cx { doc: &doc, key: Key::<Quantities>::DEFAULT.id(), golden: &golden, layer: Quantities::ID.into() };
    assert!(matches!(
        quantities.compare(&cx_qty, &golden.layers[2], Tolerance::exact()),
        Err(Mismatch::BadGolden { ref detail, .. }) if detail.contains("negative 5")
    ));
}
