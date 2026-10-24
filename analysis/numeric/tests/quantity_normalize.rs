//! Tests for quantity normalization to SI / stated base units (#932) across
//! length, mass, volume, temperature (affine), and speed, using the shared
//! unit table in `turbo-analysis-layers::unit`.

use turbo_analysis::{Ix, Layer, ViewId};
use turbo_numeric::NumberNotation::LatinUs;
use turbo_numeric::unit::{Length, Per, Speed, Temperature, Unit, Volume, Weight};
use turbo_numeric::{Dec, Quantities, QuantityExtractor};

fn d(s: &str) -> Dec {
    let neg = s.starts_with('-');
    let s = s.trim_start_matches('-');
    let (i, f) = s.split_once('.').unwrap_or((s, ""));
    let v = Dec::from_digits(i.as_bytes(), f.as_bytes());
    if neg { v.negated() } else { v }
}

fn scan_with(extractor: &QuantityExtractor, text: &str) -> Quantities {
    let mut q = Quantities::new(ViewId::SOURCE);
    extractor.scan(text, &mut q);
    assert_eq!(q.check(), Ok(()));
    q
}

#[test]
fn length_quantities_normalize_to_meters_and_compare_across_units() {
    let q = scan_with(
        &QuantityExtractor::default(),
        "Bridge span is 2.5km (2,500 m), cable thickness 50 cm (500 mm), clearance 100 ft (1 mi approach).",
    );
    assert_eq!(q.spans.len(), 6);

    // 2.5km and 2,500 m have identical normalized values and base unit ("m").
    assert_eq!((q.value(Ix::new(0)).to_plain_string(), q.unit(Ix::new(0))), ("2.5".to_string(), "km"));
    assert_eq!((q.normalized_value(Ix::new(0)).to_plain_string(), q.base_unit(Ix::new(0))), ("2500".to_string(), "m"));
    assert_eq!((q.value(Ix::new(1)).to_plain_string(), q.unit(Ix::new(1))), ("2500".to_string(), "m"));
    assert_eq!(q.normalized_value(Ix::new(0)), q.normalized_value(Ix::new(1)));
    assert_eq!(q.normalized_dec(Ix::new(0)), Some(d("2500")));

    // 50 cm and 500 mm both normalize to 0.5 m.
    assert_eq!((q.normalized_value(Ix::new(2)).to_plain_string(), q.base_unit(Ix::new(2))), ("0.5".to_string(), "m"));
    assert_eq!(q.normalized_value(Ix::new(2)), q.normalized_value(Ix::new(3)));

    // 100 ft -> 30.48 m, 1 mi -> 1609.344 m.
    assert_eq!((q.normalized_value(Ix::new(4)).to_plain_string(), q.base_unit(Ix::new(4))), ("30.48".to_string(), "m"));
    assert_eq!(
        (q.normalized_value(Ix::new(5)).to_plain_string(), q.base_unit(Ix::new(5))),
        ("1609.344".to_string(), "m")
    );
    assert!(q.normalized_dec(Ix::new(2)).unwrap() < q.normalized_dec(Ix::new(4)).unwrap());
    assert!(q.normalized_dec(Ix::new(4)).unwrap() < q.normalized_dec(Ix::new(5)).unwrap());
    assert!(q.normalized_dec(Ix::new(5)).unwrap() < q.normalized_dec(Ix::new(0)).unwrap());
}

#[test]
fn mass_quantities_normalize_to_kilograms_and_compare_across_units() {
    let q = scan_with(
        &QuantityExtractor::default(),
        "Shipment weighs 1.5 t, pallet is 80 kg, sample is 500 g (500,000 mg), parcel is 10 lb (160 oz).",
    );
    assert_eq!(q.spans.len(), 6);

    for i in 0..q.spans.len() {
        assert_eq!(q.base_unit(Ix::new(i as u32)), "kg");
    }
    // 1.5 t -> 1500 kg
    assert_eq!(q.normalized_value(Ix::new(0)).to_plain_string(), "1500");
    // 80 kg -> 80 kg
    assert_eq!(q.normalized_value(Ix::new(1)).to_plain_string(), "80");
    // 500 g and 500,000 mg -> 0.5 kg
    assert_eq!(q.normalized_value(Ix::new(2)).to_plain_string(), "0.5");
    assert_eq!(q.normalized_value(Ix::new(2)), q.normalized_value(Ix::new(3)));
    // 10 lb and 160 oz -> 4.5359237 kg
    assert_eq!(q.normalized_value(Ix::new(4)).to_plain_string(), "4.5359237");
    assert_eq!(q.normalized_value(Ix::new(4)), q.normalized_value(Ix::new(5)));
}

#[test]
fn volume_quantities_normalize_to_cubic_meters() {
    let extractor = QuantityExtractor::with_units(["L", "mL", "ml", "gal", "bbl"], LatinUs).unwrap();
    let q = scan_with(&extractor, "Tank holds 250 L, vial has 500 mL (500 ml), reserve is 10 gal or 1 bbl.");
    assert_eq!(q.spans.len(), 5);

    for i in 0..q.spans.len() {
        assert_eq!(q.base_unit(Ix::new(i as u32)), "m³");
    }
    assert_eq!(q.normalized_value(Ix::new(0)).to_plain_string(), "0.25");
    assert_eq!(q.normalized_value(Ix::new(1)).to_plain_string(), "0.0005");
    assert_eq!(q.normalized_value(Ix::new(1)), q.normalized_value(Ix::new(2)));
    assert_eq!(q.normalized_value(Ix::new(3)).to_plain_string(), "0.03785411784");
    assert_eq!(q.normalized_value(Ix::new(4)).to_plain_string(), "0.158987294928");
}

#[test]
fn temperature_quantities_normalize_affinely_to_kelvin() {
    let extractor = QuantityExtractor::with_units(["C", "F", "K"], LatinUs).unwrap();
    let q = scan_with(
        &extractor,
        "Freezing is 0 C or 32 F (273.15 K); boiling is 100 C or 212 F (373.15 K); crossover is -40 C or -40 F.",
    );
    assert_eq!(q.spans.len(), 8);

    for i in 0..q.spans.len() {
        assert_eq!(q.base_unit(Ix::new(i as u32)), "K");
    }
    // 0 C == 32 F == 273.15 K
    assert_eq!(q.normalized_value(Ix::new(0)).to_plain_string(), "273.15");
    assert_eq!(q.normalized_value(Ix::new(0)), q.normalized_value(Ix::new(1)));
    assert_eq!(q.normalized_value(Ix::new(1)), q.normalized_value(Ix::new(2)));

    // 100 C == 212 F == 373.15 K
    assert_eq!(q.normalized_value(Ix::new(3)).to_plain_string(), "373.15");
    assert_eq!(q.normalized_value(Ix::new(3)), q.normalized_value(Ix::new(4)));
    assert_eq!(q.normalized_value(Ix::new(4)), q.normalized_value(Ix::new(5)));

    // -40 C == -40 F == 233.15 K
    assert_eq!(q.normalized_value(Ix::new(6)).to_plain_string(), "233.15");
    assert_eq!(q.normalized_value(Ix::new(6)), q.normalized_value(Ix::new(7)));
}

#[test]
fn speed_quantities_normalize_to_meters_per_second() {
    let q = scan_with(&QuantityExtractor::default(), "Wind gusts reached 36 kph and 100 mph.");
    assert_eq!(q.spans.len(), 2);

    assert_eq!(q.base_unit(Ix::new(0)), "m/s");
    assert_eq!(q.normalized_value(Ix::new(0)).to_plain_string(), "10");

    assert_eq!(q.base_unit(Ix::new(1)), "m/s");
    assert_eq!(q.normalized_value(Ix::new(1)).to_plain_string(), "44.704");
}

#[test]
fn shared_unit_table_parses_written_symbols_back_to_typed_units() {
    assert_eq!(Unit::from_symbol("km"), Some(Unit::Length(Length::Kilometer)));
    assert_eq!(Unit::from_symbol("lbs"), Some(Unit::Weight(Weight::Pound)));
    assert_eq!(Unit::from_symbol("mL"), Some(Unit::Volume(Volume::Milliliter)));
    assert_eq!(Unit::from_symbol("°C"), Some(Unit::Temperature(Temperature::Celsius)));
    assert_eq!(Unit::from_symbol("km/h"), Some(Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour))));
    assert_eq!(Unit::from_symbol("mph"), Some(Unit::Speed(Speed::Per(Length::Mile, Per::Hour))));
}

#[test]
fn extracted_quantity_outside_dec_scale_range_returns_none_from_normalized_dec() {
    let zeros = "0".repeat(39_999);
    let text = format!("0.{zeros}1 kg and 0.{zeros}1 g");
    let q = scan_with(&QuantityExtractor::default(), &text);
    assert_eq!(q.spans.len(), 2);

    assert_eq!(q.unit(Ix::new(0)), "kg");
    assert_eq!(q.scale[0], 40_000);
    assert_eq!(q.base_unit(Ix::new(0)), "");
    assert_eq!(q.normalized_dec(Ix::new(0)), None);

    assert_eq!(q.unit(Ix::new(1)), "g");
    assert_eq!(q.scale[1], 40_000);
    assert_eq!(q.base_unit(Ix::new(1)), "");
    assert_eq!(q.normalized_dec(Ix::new(1)), None);
}
