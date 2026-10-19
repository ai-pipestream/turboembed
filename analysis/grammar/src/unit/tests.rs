use super::*;

use turbo_analysis_layers::Currency;

fn symbol(u: Unit) -> String {
    let mut s = String::new();
    u.write_symbol(&mut s);
    s
}

#[test]
fn names_follow_recognizers_text() {
    assert_eq!(Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour)).to_string(), "Kilometer per hour");
    assert_eq!(Unit::Area(Area::Square(Length::Meter)).to_string(), "Square meter");
    assert_eq!(Unit::Volume(Volume::Cubic(Length::Foot)).to_string(), "Cubic foot");
    assert_eq!(Unit::Area(Area::Square(Length::LightYear)).to_string(), "Square light year");
    assert_eq!(Unit::Temperature(Temperature::Celsius).to_string(), "C");
    assert_eq!(Unit::Age(Age::Unstated).to_string(), "Age");
    assert_eq!(Unit::Weight(Weight::MetricTon).to_string(), "Metric ton");
    assert_eq!(Unit::Information(Information::Megabyte).to_string(), "Megabyte");
    assert_eq!(Unit::Angle(Angle::Turn).to_string(), "Turn");
    assert_eq!(Unit::Speed(Speed::Knot).to_string(), "Knot");
    let usd = Currency::parse("USD").unwrap();
    assert_eq!(Unit::Currency(CurrencyUnit::Code(usd)).to_string(), "United States dollar");
}

#[test]
fn symbols_are_short_forms() {
    assert_eq!(symbol(Unit::Speed(Speed::Per(Length::Kilometer, Per::Hour))), "km/h");
    assert_eq!(symbol(Unit::Area(Area::Square(Length::Meter))), "m²");
    assert_eq!(symbol(Unit::Volume(Volume::Cubic(Length::Centimeter))), "cm³");
    assert_eq!(symbol(Unit::Temperature(Temperature::Fahrenheit)), "°F");
    assert_eq!(symbol(Unit::Age(Age::Year)), "years old");
    assert_eq!(symbol(Unit::Information(Information::Gigabyte)), "GB");
    assert_eq!(symbol(Unit::Currency(CurrencyUnit::Generic(Generic::Dollar))), "dollar");
}

#[test]
fn tons_and_imperial_measures_are_their_own_units() {
    assert_eq!(symbol(Unit::Weight(Weight::MetricTon)), "t");
    assert_eq!(symbol(Unit::Weight(Weight::ShortTon)), "sh tn");
    assert_eq!(symbol(Unit::Weight(Weight::LongTon)), "long tn");
    assert_eq!(Unit::Weight(Weight::ShortTon).to_string(), "Short ton (US)");
    assert_eq!(Unit::Weight(Weight::LongTon).to_string(), "Long ton (British)");
    assert_eq!(symbol(Unit::Volume(Volume::Gallon)), "gal");
    assert_eq!(symbol(Unit::Volume(Volume::ImperialGallon)), "imp gal");
    assert_eq!(symbol(Unit::Volume(Volume::ImperialFluidOunce)), "imp fl oz");
    assert_eq!(symbol(Unit::Length(Length::NauticalMile)), "nmi");
    assert_eq!(Unit::Volume(Volume::ImperialPint).dimension(), Some(Dimension::Volume));
}

#[test]
fn kinds_and_subtypes() {
    assert_eq!(Unit::Weight(Weight::Pound).measure(), Measure::Dimension);
    assert_eq!(Unit::Weight(Weight::Pound).dimension(), Some(Dimension::Weight));
    assert_eq!(Unit::Temperature(Temperature::Degree).dimension(), None);
    assert_eq!(Unit::Age(Age::Day).measure().name(), "age");
    assert_eq!(Dimension::Information.name(), "Information");
    assert_eq!(Information::of(9, true), Some(Information::Gigabyte));
    assert_eq!(Information::of(9, false), Some(Information::Gigabit));
    assert_eq!(Information::of(2, true), None);
    let pound = Unit::Currency(CurrencyUnit::Generic(Generic::Pound));
    assert_eq!(pound.measure(), Measure::Currency);
    assert!(pound.currency().is_some());
    assert_eq!(Unit::Length(Length::Inch).currency(), None);
}

#[test]
fn a_unit_is_small() {
    assert!(std::mem::size_of::<Unit>() <= 6);
}
