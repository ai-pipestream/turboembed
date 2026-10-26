use turbo_analysis_layers::{
    dec::Dec,
    unit::{Area, Energy, Length, Unit, Weight},
};

#[test]
fn modern_mainland_units_normalize_without_premature_rounding() {
    let square_meter = Unit::Area(Area::Square(Length::Meter));
    for (amount, meters) in [(3, 2000), (15, 10000), (3870000, 2580000000)] {
        assert_eq!(Unit::Area(Area::Mu).normalize(Dec::from_i64(amount)), Some((Dec::from_i64(meters), square_meter)));
    }
    assert_eq!(Unit::Weight(Weight::Jin).normalize(Dec::from_i64(2)), Some((Dec::ONE, Unit::Weight(Weight::Kilogram))));
    assert_eq!(
        Unit::Weight(Weight::Liang).normalize(Dec::from_i64(20)),
        Some((Dec::ONE, Unit::Weight(Weight::Kilogram)))
    );
    assert_eq!(
        Unit::Energy(Energy::Kilocalorie).normalize(Dec::ONE),
        Some((Dec::from_i64(4184), Unit::Energy(Energy::Joule)))
    );
}
#[test]
fn added_unit_symbols_round_trip_without_aliasing_old_units() {
    for (unit, symbol, name) in [
        (Unit::Area(Area::Mu), "mu", "Mu"),
        (Unit::Weight(Weight::Jin), "jin", "Jin"),
        (Unit::Weight(Weight::Liang), "liang", "Liang"),
        (Unit::Energy(Energy::Kilocalorie), "kcal", "Kilocalorie"),
    ] {
        assert_eq!(Unit::from_symbol(symbol), Some(unit));
        assert_eq!(unit.to_string(), name);
    }
}
