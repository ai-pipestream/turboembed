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
    l("chilometro", Length::Kilometer),
    l("chilometri", Length::Kilometer),
    l("hm", Length::Hectometer),
    l("ettometro", Length::Hectometer),
    l("ettometri", Length::Hectometer),
    l("dam", Length::Decameter),
    l("decametro", Length::Decameter),
    l("decametri", Length::Decameter),
    l("m", Length::Meter),
    l("metro", Length::Meter),
    l("metri", Length::Meter),
    l("dm", Length::Decimeter),
    l("decimetro", Length::Decimeter),
    l("decimetri", Length::Decimeter),
    l("cm", Length::Centimeter),
    l("centimetro", Length::Centimeter),
    l("centimetri", Length::Centimeter),
    l("mm", Length::Millimeter),
    l("millimetro", Length::Millimeter),
    l("millimetri", Length::Millimeter),
    l("nm", Length::Nanometer),
    l("nanometro", Length::Nanometer),
    l("nanometri", Length::Nanometer),
    l("pollice", Length::Inch),
    l("pollici", Length::Inch),
    l("piede", Length::Foot),
    l("piedi", Length::Foot),
    l("ft", Length::Foot),
    l("mi", Length::Mile),
    l("micrometro", Length::Micrometer),
    l("micrometri", Length::Micrometer),
    l("miglio", Length::Mile),
    l("miglia", Length::Mile),
    l("iarda", Length::Yard),
    l("iarde", Length::Yard),
    v("l", Volume::Liter),
    v("litro", Volume::Liter),
    v("litri", Volume::Liter),
    v("ml", Volume::Milliliter),
    v("millilitro", Volume::Milliliter),
    v("millilitri", Volume::Milliliter),
    v("cl", Volume::Centiliter),
    v("centilitro", Volume::Centiliter),
    v("centilitri", Volume::Centiliter),
    v("dl", Volume::Deciliter),
    v("decilitro", Volume::Deciliter),
    v("decilitri", Volume::Deciliter),
    v("cc", Volume::Cubic(Length::Centimeter)),
    v("gallone", Volume::Gallon),
    v("galloni", Volume::Gallon),
    w("barile", Unit::Volume(Volume::Barrel), 0),
    w("barili", Unit::Volume(Volume::Barrel), 0),
    g("g", Weight::Gram),
    g("grammo", Weight::Gram),
    g("grammi", Weight::Gram),
    g("kg", Weight::Kilogram),
    g("kilo", Weight::Kilogram),
    g("chili", Weight::Kilogram),
    g("chilogrammo", Weight::Kilogram),
    g("chilogrammi", Weight::Kilogram),
    g("mg", Weight::Milligram),
    g("milligrammo", Weight::Milligram),
    g("milligrammi", Weight::Milligram),
    g("t", Weight::MetricTon),
    g("tonnellata", Weight::MetricTon),
    g("tonnellate", Weight::MetricTon),
    g("libbra", Weight::Pound),
    g("libbre", Weight::Pound),
    g("lb", Weight::Pound),
    g("lbs", Weight::Pound),
    g("oncia", Weight::Ounce),
    g("once", Weight::Ounce),
    g("oz", Weight::Ounce),
    w("acro", Unit::Area(Area::Acre), 0),
    w("acri", Unit::Area(Area::Acre), 0),
    w("ettaro", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("ettari", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("bit", Unit::Information(Information::Bit), 0),
    w("bits", Unit::Information(Information::Bit), 0),
    w("byte", Unit::Information(Information::Byte), 0),
    w("bytes", Unit::Information(Information::Byte), 0),
    w("mb", Unit::Information(Information::Megabit), 0),
    w("gb", Unit::Information(Information::Gigabit), 0),
    w("kb", Unit::Information(Information::Kilobit), 0),
    w("rad", Unit::Angle(Angle::Radian), 0),
    w("radiante", Unit::Angle(Angle::Radian), 0),
    w("radianti", Unit::Angle(Angle::Radian), 0),
    w("giro", Unit::Angle(Angle::Turn), 0),
    w("giri", Unit::Angle(Angle::Turn), 0),
    t("grado", Temperature::Degree, 0),
    t("gradi", Temperature::Degree, 0),
    t("°", Temperature::Degree, 0),
    t("o", Temperature::Degree, 0),
    t("oc", Temperature::Celsius, 0),
    t("of", Temperature::Fahrenheit, 0),
    t("ok", Temperature::Kelvin, 0),
    t("celsius", Temperature::Celsius, BARE),
    t("centigrado", Temperature::Celsius, BARE),
    t("centigradi", Temperature::Celsius, BARE),
    t("farenheit", Temperature::Fahrenheit, BARE),
    t("fahrenheit", Temperature::Fahrenheit, BARE),
    t("kelvin", Temperature::Kelvin, BARE),
    t("kelvins", Temperature::Kelvin, BARE),
    t("k", Temperature::Kelvin, 0),
    c("$", Generic::Dollar),
    c("dollaro", Generic::Dollar),
    c("dollari", Generic::Dollar),
    c("£", Generic::Pound),
    c("sterlina", Generic::Pound),
    c("sterline", Generic::Pound),
    c("franco", Generic::Franc),
    c("franchi", Generic::Franc),
    c("peso", Generic::Peso),
    c("pesos", Generic::Peso),
    c("marco", Generic::Mark),
    c("marchi", Generic::Mark),
    iso("btc", *b"XBT"),
    iso("bitcoin", *b"XBT"),
    iso("€", *b"EUR"),
    iso("euro", *b"EUR"),
    iso("yen", *b"JPY"),
    iso("¥", *b"JPY"),
    iso("yuan", *b"CNY"),
    iso("peseta", *b"ESP"),
    iso("pesetas", *b"ESP"),
    iso("fiorino", *b"NLG"),
    iso("fiorini", *b"NLG"),
    w("centavo", Unit::Currency(CurrencyUnit::Minor(Minor::Centavo)), BARE),
    w("centavos", Unit::Currency(CurrencyUnit::Minor(Minor::Centavo)), BARE),
    w("cent", Unit::Currency(CurrencyUnit::Minor(Minor::Cent)), BARE),
    w("centesimo", Unit::Currency(CurrencyUnit::Minor(Minor::Cent)), BARE),
    w("centesimi", Unit::Currency(CurrencyUnit::Minor(Minor::Cent)), BARE),
    w("penny", Unit::Currency(CurrencyUnit::Minor(Minor::Penny)), BARE),
    w("pence", Unit::Currency(CurrencyUnit::Minor(Minor::Pence)), BARE),
    w("anno", Unit::Age(Age::Year), 0),
    w("anni", Unit::Age(Age::Year), 0),
    w("mese", Unit::Age(Age::Month), 0),
    w("mesi", Unit::Age(Age::Month), 0),
    w("settimana", Unit::Age(Age::Week), 0),
    w("settimane", Unit::Age(Age::Week), 0),
    w("giorno", Unit::Age(Age::Day), 0),
    w("giorni", Unit::Age(Age::Day), 0),
];
pub(super) const TIMES: &[(&str, Per)] = &[
    ("s", Per::Second),
    ("secondo", Per::Second),
    ("secondi", Per::Second),
    ("minuto", Per::Minute),
    ("minuti", Per::Minute),
    ("min", Per::Minute),
    ("ora", Per::Hour),
    ("ore", Per::Hour),
    ("h", Per::Hour),
    ("giorno", Per::Day),
    ("giorni", Per::Day),
    ("settimana", Per::Week),
    ("settimane", Per::Week),
    ("mese", Per::Month),
    ("mesi", Per::Month),
    ("anno", Per::Year),
    ("anni", Per::Year),
];
pub(super) const COUNTRIES: &[(&str, Generic, &str)] = &[
    ("canadese", Generic::Dollar, "cad"),
    ("canadesi", Generic::Dollar, "cad"),
    ("americano", Generic::Dollar, "usd"),
    ("americani", Generic::Dollar, "usd"),
    ("australiano", Generic::Dollar, "aud"),
    ("australiani", Generic::Dollar, "aud"),
    ("neozelandese", Generic::Dollar, "nzd"),
    ("neozelandesi", Generic::Dollar, "nzd"),
    ("finlandese", Generic::Mark, "fim"),
    ("finlandesi", Generic::Mark, "fim"),
    ("svizzero", Generic::Franc, "chf"),
    ("svizzeri", Generic::Franc, "chf"),
    ("messicano", Generic::Peso, "mxn"),
    ("messicani", Generic::Peso, "mxn"),
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
