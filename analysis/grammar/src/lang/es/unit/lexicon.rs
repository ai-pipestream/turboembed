use super::*;
use crate::engine::Entry;
use crate::unit::{
    Age, Area, Information, Length, Minor, Per, Volume,
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
    w(word, Unit::Currency(CurrencyUnit::Generic(unit)), BARE | PREFIX)
}
const fn iso(word: &'static str, code: [u8; 3]) -> Word {
    w(word, Unit::Currency(CurrencyUnit::Code(turbo_analysis_layers::Currency::new(code).unwrap())), BARE | PREFIX)
}
const fn t(word: &'static str, unit: Temperature, flags: u8) -> Word {
    w(word, Unit::Temperature(unit), flags)
}
pub(super) const WORDS: &[Word] = &[
    l("km", Length::Kilometer),
    l("kilometro", Length::Kilometer),
    l("kilometros", Length::Kilometer),
    l("hm", Length::Hectometer),
    l("hectometro", Length::Hectometer),
    l("hectometros", Length::Hectometer),
    l("dam", Length::Decameter),
    l("decametro", Length::Decameter),
    l("decametros", Length::Decameter),
    l("m", Length::Meter),
    l("metro", Length::Meter),
    l("metros", Length::Meter),
    l("dm", Length::Decimeter),
    l("decimetro", Length::Decimeter),
    l("decimetros", Length::Decimeter),
    l("cm", Length::Centimeter),
    l("centimetro", Length::Centimeter),
    l("centimetros", Length::Centimeter),
    l("mm", Length::Millimeter),
    l("milimetro", Length::Millimeter),
    l("milimetros", Length::Millimeter),
    l("nm", Length::Nanometer),
    l("nanometro", Length::Nanometer),
    l("nanometros", Length::Nanometer),
    l("pulgada", Length::Inch),
    l("pulgadas", Length::Inch),
    l("in", Length::Inch),
    l("pie", Length::Foot),
    l("pies", Length::Foot),
    l("ft", Length::Foot),
    l("milla", Length::Mile),
    l("millas", Length::Mile),
    l("yarda", Length::Yard),
    l("yardas", Length::Yard),
    v("l", Volume::Liter),
    v("litro", Volume::Liter),
    v("litros", Volume::Liter),
    v("ml", Volume::Milliliter),
    v("mililitro", Volume::Milliliter),
    v("mililitros", Volume::Milliliter),
    v("cl", Volume::Centiliter),
    v("centilitro", Volume::Centiliter),
    v("centilitros", Volume::Centiliter),
    v("dl", Volume::Deciliter),
    v("decilitro", Volume::Deciliter),
    v("decilitros", Volume::Deciliter),
    v("cc", Volume::Cubic(Length::Centimeter)),
    v("galon", Volume::Gallon),
    v("galones", Volume::Gallon),
    w("barril", Unit::Volume(Volume::Barrel), BARE),
    w("barriles", Unit::Volume(Volume::Barrel), BARE),
    g("g", Weight::Gram),
    g("gramo", Weight::Gram),
    g("gramos", Weight::Gram),
    g("kg", Weight::Kilogram),
    g("kilo", Weight::Kilogram),
    g("kilos", Weight::Kilogram),
    g("kilogramo", Weight::Kilogram),
    g("kilogramos", Weight::Kilogram),
    g("mg", Weight::Milligram),
    g("miligramo", Weight::Milligram),
    g("miligramos", Weight::Milligram),
    g("t", Weight::MetricTon),
    g("tonelada", Weight::MetricTon),
    g("toneladas", Weight::MetricTon),
    g("libra", Weight::Pound),
    g("libras", Weight::Pound),
    g("lb", Weight::Pound),
    g("lbs", Weight::Pound),
    g("onza", Weight::Ounce),
    g("onzas", Weight::Ounce),
    g("oz", Weight::Ounce),
    w("acre", Unit::Area(Area::Acre), 0),
    w("acres", Unit::Area(Area::Acre), 0),
    w("hectarea", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("hectareas", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("bit", Unit::Information(Information::Bit), 0),
    w("bits", Unit::Information(Information::Bit), 0),
    w("byte", Unit::Information(Information::Byte), 0),
    w("bytes", Unit::Information(Information::Byte), 0),
    w("mb", Unit::Information(Information::Megabit), 0),
    w("gb", Unit::Information(Information::Gigabit), 0),
    w("kb", Unit::Information(Information::Kilobit), 0),
    w("rad", Unit::Angle(Angle::Radian), 0),
    w("radian", Unit::Angle(Angle::Radian), 0),
    w("radianes", Unit::Angle(Angle::Radian), 0),
    w("vuelta", Unit::Angle(Angle::Turn), 0),
    w("vueltas", Unit::Angle(Angle::Turn), 0),
    t("grado", Temperature::Degree, 0),
    t("grados", Temperature::Degree, 0),
    t("°", Temperature::Degree, 0),
    t("o", Temperature::Degree, 0),
    t("oc", Temperature::Celsius, 0),
    t("of", Temperature::Fahrenheit, 0),
    t("ok", Temperature::Kelvin, 0),
    t("celsius", Temperature::Celsius, BARE),
    t("centigrado", Temperature::Celsius, BARE),
    t("centigrados", Temperature::Celsius, BARE),
    t("fahrenheit", Temperature::Fahrenheit, BARE),
    t("kelvin", Temperature::Kelvin, BARE),
    t("kelvins", Temperature::Kelvin, BARE),
    t("k", Temperature::Kelvin, 0),
    c("$", Generic::Dollar),
    c("dolar", Generic::Dollar),
    c("dolares", Generic::Dollar),
    c("£", Generic::Pound),
    c("libra", Generic::Pound),
    c("libras", Generic::Pound),
    c("franco", Generic::Franc),
    c("francos", Generic::Franc),
    c("peso", Generic::Peso),
    c("pesos", Generic::Peso),
    c("marco", Generic::Mark),
    c("marcos", Generic::Mark),
    iso("€", *b"EUR"),
    iso("euro", *b"EUR"),
    iso("euros", *b"EUR"),
    iso("yen", *b"JPY"),
    iso("yenes", *b"JPY"),
    iso("¥", *b"JPY"),
    iso("yuan", *b"CNY"),
    iso("yuanes", *b"CNY"),
    iso("peseta", *b"ESP"),
    iso("pesetas", *b"ESP"),
    iso("florin", *b"NLG"),
    iso("florines", *b"NLG"),
    w("centavo", Unit::Currency(CurrencyUnit::Minor(Minor::Centavo)), BARE),
    w("centavos", Unit::Currency(CurrencyUnit::Minor(Minor::Centavo)), BARE),
    w("centimo", Unit::Currency(CurrencyUnit::Minor(Minor::Cent)), BARE),
    w("centimos", Unit::Currency(CurrencyUnit::Minor(Minor::Cent)), BARE),
    w("penique", Unit::Currency(CurrencyUnit::Minor(Minor::Penny)), BARE),
    w("peniques", Unit::Currency(CurrencyUnit::Minor(Minor::Penny)), BARE),
    w("año", Unit::Age(Age::Year), 0),
    w("años", Unit::Age(Age::Year), 0),
    w("ano", Unit::Age(Age::Year), 0),
    w("anos", Unit::Age(Age::Year), 0),
    w("mes", Unit::Age(Age::Month), 0),
    w("meses", Unit::Age(Age::Month), 0),
    w("semana", Unit::Age(Age::Week), 0),
    w("semanas", Unit::Age(Age::Week), 0),
    w("dia", Unit::Age(Age::Day), 0),
    w("dias", Unit::Age(Age::Day), 0),
];
pub(super) const TIMES: &[(&str, Per)] = &[
    ("s", Per::Second),
    ("segundo", Per::Second),
    ("segundos", Per::Second),
    ("minuto", Per::Minute),
    ("minutos", Per::Minute),
    ("min", Per::Minute),
    ("hora", Per::Hour),
    ("horas", Per::Hour),
    ("h", Per::Hour),
    ("dia", Per::Day),
    ("dias", Per::Day),
    ("semana", Per::Week),
    ("semanas", Per::Week),
    ("mes", Per::Month),
    ("meses", Per::Month),
    ("año", Per::Year),
    ("años", Per::Year),
    ("ano", Per::Year),
    ("anos", Per::Year),
];
pub(super) const COUNTRIES: &[(&str, Generic, &str)] = &[
    ("canadiense", Generic::Dollar, "cad"),
    ("canadienses", Generic::Dollar, "cad"),
    ("estadounidense", Generic::Dollar, "usd"),
    ("estadounidenses", Generic::Dollar, "usd"),
    ("australiano", Generic::Dollar, "aud"),
    ("australianos", Generic::Dollar, "aud"),
    ("neozelandes", Generic::Dollar, "nzd"),
    ("neozelandeses", Generic::Dollar, "nzd"),
    ("finlandes", Generic::Mark, "fim"),
    ("finlandeses", Generic::Mark, "fim"),
    ("suizo", Generic::Franc, "chf"),
    ("suizos", Generic::Franc, "chf"),
    ("mexicano", Generic::Peso, "mxn"),
    ("mexicanos", Generic::Peso, "mxn"),
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
