use super::*;
use crate::engine::Entry;
use crate::unit::{Age, Area, Energy, Frequency, Generic, Information, Length, Minor, Power, Weight};

pub(super) struct Word {
    pub word: &'static str,
    pub unit: Unit,
    pub bare: bool,
    pub prefix: bool,
}
const fn w(word: &'static str, unit: Unit) -> Word {
    Word { word, unit, bare: false, prefix: false }
}
const fn b(word: &'static str, unit: Unit) -> Word {
    Word { word, unit, bare: true, prefix: false }
}
const fn p(word: &'static str, unit: Unit) -> Word {
    Word { word, unit, bare: true, prefix: true }
}
const fn currency(code: [u8; 3]) -> Unit {
    Unit::Currency(CurrencyUnit::Code(turbo_analysis_layers::Currency::new(code).unwrap()))
}
pub(super) const WORDS: &[Word] = &[
    w("岁", Unit::Age(Age::Year)),
    w("歲", Unit::Age(Age::Year)),
    w("周岁", Unit::Age(Age::Year)),
    w("週歲", Unit::Age(Age::Year)),
    w("个月大", Unit::Age(Age::Month)),
    w("個月大", Unit::Age(Age::Month)),
    w("月大", Unit::Age(Age::Month)),
    w("周大", Unit::Age(Age::Week)),
    w("週大", Unit::Age(Age::Week)),
    w("天大", Unit::Age(Age::Day)),
    w("千米", Unit::Length(Length::Kilometer)),
    w("公里", Unit::Length(Length::Kilometer)),
    w("km", Unit::Length(Length::Kilometer)),
    w("米", Unit::Length(Length::Meter)),
    w("公尺", Unit::Length(Length::Meter)),
    w("m", Unit::Length(Length::Meter)),
    w("公分", Unit::Length(Length::Centimeter)),
    w("厘米", Unit::Length(Length::Centimeter)),
    w("cm", Unit::Length(Length::Centimeter)),
    w("毫米", Unit::Length(Length::Millimeter)),
    w("mm", Unit::Length(Length::Millimeter)),
    w("公斤", Unit::Weight(Weight::Kilogram)),
    w("千克", Unit::Weight(Weight::Kilogram)),
    w("kg", Unit::Weight(Weight::Kilogram)),
    w("克", Unit::Weight(Weight::Gram)),
    w("公克", Unit::Weight(Weight::Gram)),
    w("g", Unit::Weight(Weight::Gram)),
    w("微克", Unit::Weight(Weight::Microgram)),
    w("µg", Unit::Weight(Weight::Microgram)),
    w("吨", Unit::Weight(Weight::MetricTon)),
    w("噸", Unit::Weight(Weight::MetricTon)),
    w("斤", Unit::Weight(Weight::Jin)),
    w("市斤", Unit::Weight(Weight::Jin)),
    w("两", Unit::Weight(Weight::Liang)),
    w("兩", Unit::Weight(Weight::Liang)),
    w("市两", Unit::Weight(Weight::Liang)),
    w("市兩", Unit::Weight(Weight::Liang)),
    w("亩", Unit::Area(Area::Mu)),
    w("畝", Unit::Area(Area::Mu)),
    w("位", Unit::Information(Information::Bit)),
    w("比特", Unit::Information(Information::Bit)),
    w("千比特", Unit::Information(Information::Kilobit)),
    w("千字节", Unit::Information(Information::Kilobyte)),
    w("千字節", Unit::Information(Information::Kilobyte)),
    w("兆字节", Unit::Information(Information::Megabyte)),
    w("兆字節", Unit::Information(Information::Megabyte)),
    w("字节", Unit::Information(Information::Byte)),
    w("字節", Unit::Information(Information::Byte)),
    w("赫兹", Unit::Frequency(Frequency::Hertz)),
    w("赫茲", Unit::Frequency(Frequency::Hertz)),
    w("兆赫兹", Unit::Frequency(Frequency::Megahertz)),
    w("兆赫茲", Unit::Frequency(Frequency::Megahertz)),
    w("千瓦", Unit::Power(Power::Kilowatt)),
    w("千卡", Unit::Energy(Energy::Kilocalorie)),
    w("元", currency(*b"CNY")),
    w("圆", currency(*b"CNY")),
    w("圓", currency(*b"CNY")),
    w("块", currency(*b"CNY")),
    w("塊", currency(*b"CNY")),
    p("人民币", currency(*b"CNY")),
    p("人民幣", currency(*b"CNY")),
    b("美元", currency(*b"USD")),
    b("美圆", currency(*b"USD")),
    b("美圓", currency(*b"USD")),
    b("欧元", currency(*b"EUR")),
    b("歐元", currency(*b"EUR")),
    b("瑞士法郎", currency(*b"CHF")),
    b("港元", currency(*b"HKD")),
    b("港币", currency(*b"HKD")),
    b("港幣", currency(*b"HKD")),
    b("台币", currency(*b"TWD")),
    b("臺幣", currency(*b"TWD")),
    b("台幣", currency(*b"TWD")),
    b("日元", currency(*b"JPY")),
    b("日圆", currency(*b"JPY")),
    b("日圓", currency(*b"JPY")),
    b("奈拉", currency(*b"NGN")),
    b("比特币", currency(*b"XBT")),
    b("比特幣", currency(*b"XBT")),
    p("￥", currency(*b"CNY")),
    p("¥", currency(*b"CNY")),
    p("$", Unit::Currency(CurrencyUnit::Generic(Generic::Dollar))),
    w("角", Unit::Currency(CurrencyUnit::Minor(Minor::Jiao))),
    w("毛", Unit::Currency(CurrencyUnit::Minor(Minor::Jiao))),
    w("分", Unit::Currency(CurrencyUnit::Minor(Minor::Fen))),
    w("日本銭", Unit::Currency(CurrencyUnit::Minor(Minor::Sen))),
    w("日本钱", Unit::Currency(CurrencyUnit::Minor(Minor::Sen))),
    w("美分", Unit::Currency(CurrencyUnit::Minor(Minor::Cent))),
    w("度", Unit::Temperature(Temperature::Degree)),
    b("摄氏度", Unit::Temperature(Temperature::Celsius)),
    b("攝氏度", Unit::Temperature(Temperature::Celsius)),
    b("华氏度", Unit::Temperature(Temperature::Fahrenheit)),
    b("華氏度", Unit::Temperature(Temperature::Fahrenheit)),
    b("开尔文", Unit::Temperature(Temperature::Kelvin)),
    b("開爾文", Unit::Temperature(Temperature::Kelvin)),
];
pub(super) fn entries() -> Vec<Entry> {
    let mut entries: Vec<_> =
        WORDS.iter().enumerate().map(|(i, w)| Entry { word: w.word, dim: RAW, value: i as i64, flags: 0 }).collect();
    for word in ["华氏温度", "華氏溫度", "摄氏温度", "攝氏溫度", "半", "个", "個", "又", "和"] {
        entries.push(Entry { word, dim: Dim(cjk::DIMS + 3), value: 0, flags: 0 });
    }
    entries
}
