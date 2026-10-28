use super::*;
use crate::engine::Entry;
use crate::unit::{Age, Area, CurrencyUnit, Generic, Information, Length, Minor, Temperature, Volume, Weight};

pub(super) struct Word {
    pub text: &'static str,
    pub unit: Unit,
    pub bare: bool,
}
const fn w(text: &'static str, unit: Unit) -> Word {
    Word { text, unit, bare: false }
}
const fn b(text: &'static str, unit: Unit) -> Word {
    Word { text, unit, bare: true }
}
const fn c(text: &'static str, code: [u8; 3]) -> Word {
    b(text, Unit::Currency(CurrencyUnit::Code(turbo_analysis_layers::Currency::new(code).unwrap())))
}
pub(super) const WORDS: &[Word] = &[
    w("歳", Unit::Age(Age::Year)),
    w("才", Unit::Age(Age::Year)),
    w("ヶ月", Unit::Age(Age::Month)),
    w("か月", Unit::Age(Age::Month)),
    w("箇月", Unit::Age(Age::Month)),
    w("週間", Unit::Age(Age::Week)),
    w("週", Unit::Age(Age::Week)),
    w("日齢", Unit::Age(Age::Day)),
    w("日", Unit::Age(Age::Day)),
    c("円", *b"JPY"),
    c("人民元", *b"CNY"),
    c("元", *b"CNY"),
    c("ドル", *b"USD"),
    c("米ドル", *b"USD"),
    c("ユーロ", *b"EUR"),
    c("スイスフラン", *b"CHF"),
    c("香港ドル", *b"HKD"),
    c("ニュー台湾ドル", *b"TWD"),
    c("台湾ドル", *b"TWD"),
    c("ナイラ", *b"NGN"),
    c("€", *b"EUR"),
    c("¥", *b"JPY"),
    c("￥", *b"JPY"),
    c("\\", *b"JPY"),
    b("どる", Unit::Currency(CurrencyUnit::Generic(Generic::Dollar))),
    b("$", Unit::Currency(CurrencyUnit::Generic(Generic::Dollar))),
    b("£", Unit::Currency(CurrencyUnit::Generic(Generic::Pound))),
    w("銭", Unit::Currency(CurrencyUnit::Minor(Minor::Sen))),
    w("セント", Unit::Currency(CurrencyUnit::Minor(Minor::Cent))),
    w("角", Unit::Currency(CurrencyUnit::Minor(Minor::Jiao))),
    w("分", Unit::Currency(CurrencyUnit::Minor(Minor::Fen))),
    w("ポンド", Unit::Weight(Weight::Pound)),
    w("オンス", Unit::Weight(Weight::Ounce)),
    w("トン", Unit::Weight(Weight::MetricTon)),
    w("メートルトン", Unit::Weight(Weight::MetricTon)),
    w("キログラム", Unit::Weight(Weight::Kilogram)),
    w("グラム", Unit::Weight(Weight::Gram)),
    w("ミリリットル", Unit::Volume(Volume::Milliliter)),
    w("リットル", Unit::Volume(Volume::Liter)),
    w("ガロン", Unit::Volume(Volume::Gallon)),
    w("立方フィート", Unit::Volume(Volume::Cubic(Length::Foot))),
    w("インチ", Unit::Length(Length::Inch)),
    w("マイル", Unit::Length(Length::Mile)),
    w("フィート", Unit::Length(Length::Foot)),
    w("メートル", Unit::Length(Length::Meter)),
    w("キロメートル", Unit::Length(Length::Kilometer)),
    w("キロ", Unit::Length(Length::Kilometer)),
    w("ミリ", Unit::Length(Length::Millimeter)),
    w("ミリメートル", Unit::Length(Length::Millimeter)),
    w("センチ", Unit::Length(Length::Centimeter)),
    w("センチメートル", Unit::Length(Length::Centimeter)),
    w("cm", Unit::Length(Length::Centimeter)),
    w("デシメートル", Unit::Length(Length::Decimeter)),
    w("pm", Unit::Length(Length::Picometer)),
    w("ピクトメーター", Unit::Length(Length::Picometer)),
    w("ピコメートル", Unit::Length(Length::Picometer)),
    w("ムー", Unit::Area(Area::Mu)),
    w("エーカー", Unit::Area(Area::Acre)),
    w("平方メートル", Unit::Area(Area::Square(Length::Meter))),
    w("平方キロメートル", Unit::Area(Area::Square(Length::Kilometer))),
    w("ビット", Unit::Information(Information::Bit)),
    w("メガバイト", Unit::Information(Information::Megabyte)),
    w("度", Unit::Temperature(Temperature::Degree)),
    b("摂氏", Unit::Temperature(Temperature::Celsius)),
    b("摂氏温度", Unit::Temperature(Temperature::Celsius)),
    b("華氏", Unit::Temperature(Temperature::Fahrenheit)),
    b("華氏温度", Unit::Temperature(Temperature::Fahrenheit)),
    b("℃", Unit::Temperature(Temperature::Celsius)),
    b("℉", Unit::Temperature(Temperature::Fahrenheit)),
];
pub(super) fn entries() -> Vec<Entry> {
    let mut entries: Vec<_> = WORDS
        .iter()
        .enumerate()
        .map(|(value, w)| Entry { word: w.text, dim: RAW, value: value as i64, flags: 0 })
        .collect();
    for word in ["と", "の", "時速", "毎秒", "スイス", "フラン", "パナマ", "バルボア", "us"] {
        entries.push(Entry { word, dim: MARKER, value: 0, flags: 0 });
    }
    entries
}
