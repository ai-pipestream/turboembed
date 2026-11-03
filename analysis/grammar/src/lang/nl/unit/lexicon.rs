use super::*;
use crate::engine::Entry;
use crate::unit::{
    Age, Angle, Area, CurrencyUnit, Generic, Information, Length, Minor, Per, Volume, Weight,
    currency::{CODES, code},
};
pub(super) struct Word {
    pub word: &'static str,
    pub unit: Unit,
    pub flags: u8,
}
const fn w(word: &'static str, unit: Unit, flags: u8) -> Word {
    Word { word, unit, flags }
}
const fn l(word: &'static str, unit: Length) -> Word {
    w(word, Unit::Length(unit), 0)
}
const fn v(word: &'static str, unit: Volume) -> Word {
    w(word, Unit::Volume(unit), 0)
}
const fn g(word: &'static str, unit: Weight) -> Word {
    w(word, Unit::Weight(unit), 0)
}
const fn c(word: &'static str, unit: Generic) -> Word {
    w(word, Unit::Currency(CurrencyUnit::Generic(unit)), 0)
}
const fn iso(word: &'static str, code: [u8; 3]) -> Word {
    w(word, Unit::Currency(CurrencyUnit::Code(turbo_analysis_layers::Currency::new(code).unwrap())), 0)
}
const fn t(word: &'static str, unit: Temperature, flags: u8) -> Word {
    w(word, Unit::Temperature(unit), flags)
}
pub(super) const WORDS: &[Word] = &[
    l("km", Length::Kilometer),
    l("kilometer", Length::Kilometer),
    l("kilometers", Length::Kilometer),
    l("hm", Length::Hectometer),
    l("hectometer", Length::Hectometer),
    l("dam", Length::Decameter),
    l("decameter", Length::Decameter),
    l("m", Length::Meter),
    l("meter", Length::Meter),
    l("meters", Length::Meter),
    l("dm", Length::Decimeter),
    l("decimeter", Length::Decimeter),
    l("cm", Length::Centimeter),
    l("centimeter", Length::Centimeter),
    l("mm", Length::Millimeter),
    l("millimeter", Length::Millimeter),
    l("nm", Length::Nanometer),
    w("nanometer", Unit::Length(Length::Nanometer), BARE),
    w("millimicron", Unit::Length(Length::Nanometer), BARE),
    l("inch", Length::Inch),
    l("inches", Length::Inch),
    l("voet", Length::Foot),
    l("voeten", Length::Foot),
    l("ft", Length::Foot),
    l("mijl", Length::Mile),
    l("mijlen", Length::Mile),
    l("yard", Length::Yard),
    l("lichtjaar", Length::LightYear),
    v("l", Volume::Liter),
    v("liter", Volume::Liter),
    v("liters", Volume::Liter),
    v("ml", Volume::Milliliter),
    v("milliliter", Volume::Milliliter),
    v("cl", Volume::Centiliter),
    v("centiliter", Volume::Centiliter),
    v("dl", Volume::Deciliter),
    v("deciliter", Volume::Deciliter),
    v("cc", Volume::Cubic(Length::Centimeter)),
    v("kuub", Volume::Cubic(Length::Meter)),
    v("vat", Volume::Barrel),
    v("vaten", Volume::Barrel),
    v("theelepel", Volume::Teaspoon),
    v("theelepels", Volume::Teaspoon),
    v("eetlepel", Volume::Tablespoon),
    v("eetlepels", Volume::Tablespoon),
    g("g", Weight::Gram),
    g("gram", Weight::Gram),
    g("grammen", Weight::Gram),
    g("kg", Weight::Kilogram),
    g("kilogram", Weight::Kilogram),
    g("kilo", Weight::Kilogram),
    g("mg", Weight::Milligram),
    g("milligram", Weight::Milligram),
    g("ton", Weight::MetricTon),
    g("t", Weight::MetricTon),
    // Modern Dutch colloquial units; the amount builder converts them to grams.
    g("ons", Weight::Gram),
    g("pond", Weight::Gram),
    g("ponden", Weight::Gram),
    w("ha", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("hectare", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("are", Unit::Area(Area::Square(Length::Decameter)), 0),
    w("aren", Unit::Area(Area::Square(Length::Decameter)), 0),
    w("bit", Unit::Information(Information::Bit), 0),
    w("bits", Unit::Information(Information::Bit), 0),
    w("byte", Unit::Information(Information::Byte), 0),
    w("bytes", Unit::Information(Information::Byte), 0),
    w("kb", Unit::Information(Information::Kilobit), 0),
    w("mb", Unit::Information(Information::Megabit), 0),
    w("gb", Unit::Information(Information::Gigabit), 0),
    w("terabyte", Unit::Information(Information::Terabyte), 0),
    w("kilobyte", Unit::Information(Information::Kilobyte), 0),
    w("megabyte", Unit::Information(Information::Megabyte), 0),
    w("gigabyte", Unit::Information(Information::Gigabyte), 0),
    w("megabit", Unit::Information(Information::Megabit), 0),
    w("rad", Unit::Angle(Angle::Radian), 0),
    w("radiaal", Unit::Angle(Angle::Radian), 0),
    w("radialen", Unit::Angle(Angle::Radian), 0),
    w("draai", Unit::Angle(Angle::Turn), 0),
    w("slag", Unit::Angle(Angle::Turn), 0),
    w("slagen", Unit::Angle(Angle::Turn), 0),
    t("graad", Temperature::Degree, 0),
    t("graden", Temperature::Degree, 0),
    t("gr", Temperature::Degree, 0),
    t("°", Temperature::Degree, 0),
    t("celsius", Temperature::Celsius, BARE),
    t("fahrenheit", Temperature::Fahrenheit, BARE),
    t("kelvin", Temperature::Kelvin, BARE),
    c("$", Generic::Dollar),
    c("dollar", Generic::Dollar),
    c("dollars", Generic::Dollar),
    c("£", Generic::Pound),
    iso("€", *b"EUR"),
    iso("euro", *b"EUR"),
    iso("euros", *b"EUR"),
    iso("yuan", *b"CNY"),
    iso("kuai", *b"CNY"),
    iso("元", *b"CNY"),
    iso("gulden", *b"NLG"),
    iso("guldens", *b"NLG"),
    w("markka", Unit::Currency(CurrencyUnit::Code(turbo_analysis_layers::Currency::new(*b"FIM").unwrap())), BARE),
    iso("penni", *b"FIM"),
    c("cent", Generic::Cent),
    c("centen", Generic::Cent),
    c("kwartje", Generic::Cent),
    c("kwartjes", Generic::Cent),
    c("dubbeltje", Generic::Cent),
    c("dubbeltjes", Generic::Cent),
    c("stuiver", Generic::Cent),
    c("stuivers", Generic::Cent),
    w("mao", Unit::Currency(CurrencyUnit::Minor(Minor::Jiao)), 0),
    w("fen", Unit::Currency(CurrencyUnit::Minor(Minor::Fen)), 0),
    w("jaar", Unit::Age(Age::Year), 0),
    w("jaren", Unit::Age(Age::Year), 0),
    w("jarige", Unit::Age(Age::Year), 0),
    w("maand", Unit::Age(Age::Month), 0),
    w("maanden", Unit::Age(Age::Month), 0),
    w("week", Unit::Age(Age::Week), 0),
    w("weken", Unit::Age(Age::Week), 0),
    w("dag", Unit::Age(Age::Day), 0),
    w("dagen", Unit::Age(Age::Day), 0),
];
pub(super) const TIMES: &[(&str, Per)] = &[
    ("s", Per::Second),
    ("seconde", Per::Second),
    ("seconden", Per::Second),
    ("min", Per::Minute),
    ("minuut", Per::Minute),
    ("minuten", Per::Minute),
    ("u", Per::Hour),
    ("uur", Per::Hour),
    ("uren", Per::Hour),
    ("dag", Per::Day),
    ("week", Per::Week),
    ("maand", Per::Month),
    ("jaar", Per::Year),
];
pub(super) const COUNTRIES: &[(&str, Generic, &str)] = &[
    ("amerikaanse", Generic::Dollar, "usd"),
    ("canadese", Generic::Dollar, "cad"),
    ("australische", Generic::Dollar, "aud"),
];
pub(super) fn entries() -> Vec<Entry> {
    let e = |word, dim, value, flags| Entry { word, dim, value, flags };
    let mut out: Vec<_> = WORDS.iter().enumerate().map(|(i, w)| e(w.word, RAW, i as i64, w.flags)).collect();
    out.extend(CODES.iter().enumerate().map(|(i, (c, _))| e(c, CODE, i as i64, PREFIX)));
    out.extend(TIMES.iter().enumerate().map(|(i, (w, _))| e(w, TIME, i as i64, 0)));
    out.extend(COUNTRIES.iter().enumerate().map(|(i, (w, _, _))| e(w, COUNTRY, i as i64, 0)));
    out.extend([("c", 0), ("f", 1)].map(|(w, i)| e(w, SCALE_WORD, i, 0)));
    out
}
pub(super) fn coded(s: &str) -> Option<Unit> {
    Some(Unit::Currency(CurrencyUnit::Code(code(s)?)))
}
