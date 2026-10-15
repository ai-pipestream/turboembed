//! The English words for units, after the unit lists of Recognizers-Text
//! and common usage.
//!
//! A unit word's lexicon entry carries its index in [`UNITS`] or
//! [`MONEY`]; a code's its index in [`crate::unit::currency::CODES`]; a
//! country's the index of its first row in [`NATIONS`].

use crate::engine::{Dim, Entry};
use crate::unit::currency::{CODES, code};
use crate::unit::{
    Angle, Area, CurrencyUnit, Generic, Information, Length, Minor, Speed, Temperature, Unit, Volume, Weight,
};

use super::{
    ABBR, AMBIG, BITCASE, CODE, CURW, DEG, DOT, INCHW, LOWER, MCODE, MULT, NATION, NATION_ONLY, ONLY_DOLLAR, PREFIX,
    SIPRE, TIMEW, TSCALE, UWORD,
};

/// A word and the unit it names.
#[derive(Copy, Clone, Debug)]
pub(crate) struct Word {
    pub word: &'static str,
    pub unit: Unit,
    pub flags: u8,
}

const fn w(word: &'static str, unit: Unit, flags: u8) -> Word {
    Word { word, unit, flags }
}

const fn len(word: &'static str, l: Length, flags: u8) -> Word {
    w(word, Unit::Length(l), flags)
}

const fn vol(word: &'static str, v: Volume, flags: u8) -> Word {
    w(word, Unit::Volume(v), flags)
}

const fn wt(word: &'static str, x: Weight, flags: u8) -> Word {
    w(word, Unit::Weight(x), flags)
}

const fn info(word: &'static str, i: Information, flags: u8) -> Word {
    w(word, Unit::Information(i), flags)
}

/// Measure words. A unit written with a symbol is `ABBR`; one that is
/// also a common English word or abbreviation is `AMBIG`, and stands as
/// a unit only after a number.
pub(crate) static UNITS: &[Word] = &[
    len("km", Length::Kilometer, ABBR),
    len("kilometer", Length::Kilometer, 0),
    len("kilometers", Length::Kilometer, 0),
    len("kilometre", Length::Kilometer, 0),
    len("kilometres", Length::Kilometer, 0),
    len("hm", Length::Hectometer, ABBR | AMBIG | LOWER),
    len("hectometer", Length::Hectometer, 0),
    len("hectometers", Length::Hectometer, 0),
    len("hectometre", Length::Hectometer, 0),
    len("hectometres", Length::Hectometer, 0),
    len("dam", Length::Decameter, ABBR | AMBIG | LOWER),
    len("decameter", Length::Decameter, 0),
    len("decameters", Length::Decameter, 0),
    len("decametre", Length::Decameter, 0),
    len("decametres", Length::Decameter, 0),
    len("dekameter", Length::Decameter, 0),
    len("dekameters", Length::Decameter, 0),
    len("m", Length::Meter, ABBR | AMBIG | LOWER),
    len("meter", Length::Meter, 0),
    len("meters", Length::Meter, 0),
    len("metre", Length::Meter, 0),
    len("metres", Length::Meter, 0),
    len("dm", Length::Decimeter, ABBR | AMBIG | LOWER),
    len("decimeter", Length::Decimeter, 0),
    len("decimeters", Length::Decimeter, 0),
    len("decimetre", Length::Decimeter, 0),
    len("decimetres", Length::Decimeter, 0),
    len("cm", Length::Centimeter, ABBR),
    len("centimeter", Length::Centimeter, 0),
    len("centimeters", Length::Centimeter, 0),
    len("centimetre", Length::Centimeter, 0),
    len("centimetres", Length::Centimeter, 0),
    len("mm", Length::Millimeter, ABBR),
    len("millimeter", Length::Millimeter, 0),
    len("millimeters", Length::Millimeter, 0),
    len("millimetre", Length::Millimeter, 0),
    len("millimetres", Length::Millimeter, 0),
    // U+03BC GREEK SMALL LETTER MU and U+00B5 MICRO SIGN.
    len("μm", Length::Micrometer, ABBR),
    len("µm", Length::Micrometer, ABBR),
    len("micrometer", Length::Micrometer, 0),
    len("micrometers", Length::Micrometer, 0),
    len("micrometre", Length::Micrometer, 0),
    len("micrometres", Length::Micrometer, 0),
    len("micron", Length::Micrometer, 0),
    len("microns", Length::Micrometer, 0),
    len("nm", Length::Nanometer, ABBR | LOWER),
    len("nanometer", Length::Nanometer, 0),
    len("nanometers", Length::Nanometer, 0),
    len("nanometre", Length::Nanometer, 0),
    len("nanometres", Length::Nanometer, 0),
    len("picometer", Length::Picometer, 0),
    len("picometers", Length::Picometer, 0),
    len("picometre", Length::Picometer, 0),
    len("picometres", Length::Picometer, 0),
    len("mile", Length::Mile, 0),
    len("miles", Length::Mile, 0),
    len("nmi", Length::NauticalMile, ABBR),
    len("yard", Length::Yard, AMBIG),
    len("yards", Length::Yard, AMBIG),
    len("yd", Length::Yard, ABBR | AMBIG),
    len("yds", Length::Yard, ABBR | AMBIG),
    len("foot", Length::Foot, AMBIG),
    len("feet", Length::Foot, AMBIG),
    len("ft", Length::Foot, ABBR | AMBIG),
    len("inch", Length::Inch, 0),
    len("inches", Length::Inch, 0),
    w("acre", Unit::Area(Area::Acre), 0),
    w("acres", Unit::Area(Area::Acre), 0),
    w("hectare", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("hectares", Unit::Area(Area::Square(Length::Hectometer)), 0),
    w("sqft", Unit::Area(Area::Square(Length::Foot)), ABBR),
    vol("l", Volume::Liter, ABBR | AMBIG),
    vol("liter", Volume::Liter, 0),
    vol("liters", Volume::Liter, 0),
    vol("litre", Volume::Liter, 0),
    vol("litres", Volume::Liter, 0),
    vol("hectoliter", Volume::Hectoliter, 0),
    vol("hectoliters", Volume::Hectoliter, 0),
    vol("hectolitre", Volume::Hectoliter, 0),
    vol("hectolitres", Volume::Hectoliter, 0),
    vol("decaliter", Volume::Decaliter, 0),
    vol("decaliters", Volume::Decaliter, 0),
    vol("decalitre", Volume::Decaliter, 0),
    vol("decalitres", Volume::Decaliter, 0),
    vol("dl", Volume::Deciliter, ABBR | AMBIG | LOWER),
    vol("deciliter", Volume::Deciliter, 0),
    vol("deciliters", Volume::Deciliter, 0),
    vol("decilitre", Volume::Deciliter, 0),
    vol("decilitres", Volume::Deciliter, 0),
    vol("cl", Volume::Centiliter, ABBR | AMBIG | LOWER),
    vol("centiliter", Volume::Centiliter, 0),
    vol("centiliters", Volume::Centiliter, 0),
    vol("centilitre", Volume::Centiliter, 0),
    vol("centilitres", Volume::Centiliter, 0),
    vol("ml", Volume::Milliliter, ABBR),
    vol("mls", Volume::Milliliter, ABBR),
    vol("milliliter", Volume::Milliliter, 0),
    vol("milliliters", Volume::Milliliter, 0),
    vol("millilitre", Volume::Milliliter, 0),
    vol("millilitres", Volume::Milliliter, 0),
    vol("cc", Volume::Milliliter, ABBR | AMBIG),
    vol("teaspoon", Volume::Teaspoon, 0),
    vol("teaspoons", Volume::Teaspoon, 0),
    vol("teaspoonful", Volume::Teaspoon, 0),
    vol("teaspoonfuls", Volume::Teaspoon, 0),
    vol("tsp", Volume::Teaspoon, ABBR | DOT),
    vol("tspn", Volume::Teaspoon, ABBR | DOT),
    vol("tablespoon", Volume::Tablespoon, 0),
    vol("tablespoons", Volume::Tablespoon, 0),
    vol("tablespoonful", Volume::Tablespoon, 0),
    vol("tablespoonfuls", Volume::Tablespoon, 0),
    vol("tbsp", Volume::Tablespoon, ABBR | DOT),
    vol("tbs", Volume::Tablespoon, ABBR | DOT),
    vol("tbl", Volume::Tablespoon, ABBR | AMBIG | DOT),
    vol("tbls", Volume::Tablespoon, ABBR | DOT),
    vol("pint", Volume::Pint, 0),
    vol("pints", Volume::Pint, 0),
    vol("quart", Volume::Quart, AMBIG),
    vol("quarts", Volume::Quart, AMBIG),
    vol("cup", Volume::Cup, AMBIG),
    vol("cups", Volume::Cup, AMBIG),
    vol("gill", Volume::Gill, AMBIG),
    vol("gills", Volume::Gill, AMBIG),
    vol("pinch", Volume::Pinch, AMBIG),
    vol("pinches", Volume::Pinch, AMBIG),
    vol("fluidram", Volume::FluidDram, 0),
    vol("fluidrams", Volume::FluidDram, 0),
    vol("barrel", Volume::Barrel, AMBIG),
    vol("barrels", Volume::Barrel, AMBIG),
    vol("bbl", Volume::Barrel, ABBR),
    vol("minim", Volume::Minim, AMBIG),
    vol("cord", Volume::Cord, AMBIG),
    vol("cords", Volume::Cord, AMBIG),
    vol("peck", Volume::Peck, AMBIG),
    vol("pecks", Volume::Peck, AMBIG),
    vol("bushel", Volume::Bushel, 0),
    vol("bushels", Volume::Bushel, 0),
    vol("hogshead", Volume::Hogshead, 0),
    vol("hogsheads", Volume::Hogshead, 0),
    vol("gallon", Volume::Gallon, 0),
    vol("gallons", Volume::Gallon, 0),
    vol("gal", Volume::Gallon, ABBR | AMBIG),
    wt("kg", Weight::Kilogram, ABBR),
    wt("kgs", Weight::Kilogram, ABBR),
    wt("kilogram", Weight::Kilogram, 0),
    wt("kilograms", Weight::Kilogram, 0),
    wt("kilogramme", Weight::Kilogram, 0),
    wt("kilogrammes", Weight::Kilogram, 0),
    wt("kilo", Weight::Kilogram, AMBIG),
    wt("kilos", Weight::Kilogram, AMBIG),
    wt("g", Weight::Gram, ABBR | AMBIG | LOWER),
    wt("gram", Weight::Gram, 0),
    wt("grams", Weight::Gram, 0),
    wt("gramme", Weight::Gram, 0),
    wt("grammes", Weight::Gram, 0),
    wt("mg", Weight::Milligram, ABBR),
    wt("milligram", Weight::Milligram, 0),
    wt("milligrams", Weight::Milligram, 0),
    wt("milligramme", Weight::Milligram, 0),
    wt("milligrammes", Weight::Milligram, 0),
    wt("μg", Weight::Microgram, ABBR),
    wt("µg", Weight::Microgram, ABBR),
    wt("mcg", Weight::Microgram, ABBR),
    wt("microgram", Weight::Microgram, 0),
    wt("micrograms", Weight::Microgram, 0),
    wt("microgramme", Weight::Microgram, 0),
    wt("microgrammes", Weight::Microgram, 0),
    wt("ton", Weight::ShortTon, 0),
    wt("tons", Weight::ShortTon, 0),
    wt("tonne", Weight::MetricTon, 0),
    wt("tonnes", Weight::MetricTon, 0),
    wt("t", Weight::MetricTon, ABBR | AMBIG | LOWER),
    wt("pound", Weight::Pound, AMBIG),
    wt("pounds", Weight::Pound, AMBIG),
    wt("lb", Weight::Pound, ABBR),
    wt("lbs", Weight::Pound, ABBR),
    wt("ounce", Weight::Ounce, 0),
    wt("ounces", Weight::Ounce, 0),
    wt("oz", Weight::Ounce, ABBR | DOT),
    wt("grain", Weight::Grain, AMBIG),
    wt("grains", Weight::Grain, AMBIG),
    wt("pennyweight", Weight::Pennyweight, 0),
    wt("pennyweights", Weight::Pennyweight, 0),
    wt("stone", Weight::Stone, AMBIG),
    wt("dram", Weight::Dram, AMBIG),
    wt("drams", Weight::Dram, AMBIG),
    wt("drachm", Weight::Dram, AMBIG),
    wt("drachma", Weight::Dram, AMBIG),
    w("mph", Unit::Speed(Speed::Per(Length::Mile, crate::unit::Per::Hour)), ABBR),
    w("kph", Unit::Speed(Speed::Per(Length::Kilometer, crate::unit::Per::Hour)), ABBR),
    w("kmph", Unit::Speed(Speed::Per(Length::Kilometer, crate::unit::Per::Hour)), ABBR),
    w("knot", Unit::Speed(Speed::Knot), AMBIG),
    w("knots", Unit::Speed(Speed::Knot), AMBIG),
    info("bit", Information::Bit, 0),
    info("bits", Information::Bit, 0),
    info("byte", Information::Byte, 0),
    info("bytes", Information::Byte, 0),
    info("kilobit", Information::Kilobit, 0),
    info("kilobits", Information::Kilobit, 0),
    info("megabit", Information::Megabit, 0),
    info("megabits", Information::Megabit, 0),
    info("gigabit", Information::Gigabit, 0),
    info("gigabits", Information::Gigabit, 0),
    info("terabit", Information::Terabit, 0),
    info("terabits", Information::Terabit, 0),
    info("petabit", Information::Petabit, 0),
    info("petabits", Information::Petabit, 0),
    info("kilobyte", Information::Kilobyte, 0),
    info("kilobytes", Information::Kilobyte, 0),
    info("megabyte", Information::Megabyte, 0),
    info("megabytes", Information::Megabyte, 0),
    info("gigabyte", Information::Gigabyte, 0),
    info("gigabytes", Information::Gigabyte, 0),
    info("terabyte", Information::Terabyte, 0),
    info("terabytes", Information::Terabyte, 0),
    info("petabyte", Information::Petabyte, 0),
    info("petabytes", Information::Petabyte, 0),
    info("kbit", Information::Kilobit, ABBR),
    info("mbit", Information::Megabit, ABBR),
    info("gbit", Information::Gigabit, ABBR),
    info("tbit", Information::Terabit, ABBR),
    info("kbyte", Information::Kilobyte, ABBR),
    info("mbyte", Information::Megabyte, ABBR),
    info("gbyte", Information::Gigabyte, ABBR),
    info("tbyte", Information::Terabyte, ABBR),
    // Bits or bytes by the case of the `b` as written: `Mb`, `MB`.
    info("kb", Information::Kilobit, ABBR | BITCASE),
    info("mb", Information::Megabit, ABBR | BITCASE),
    info("gb", Information::Gigabit, ABBR | BITCASE),
    info("tb", Information::Terabit, ABBR | BITCASE),
    info("pb", Information::Petabit, ABBR | BITCASE),
    w("radian", Unit::Angle(Angle::Radian), 0),
    w("radians", Unit::Angle(Angle::Radian), 0),
    w("rad", Unit::Angle(Angle::Radian), ABBR | AMBIG | LOWER),
    w("turn", Unit::Angle(Angle::Turn), AMBIG),
    w("turns", Unit::Angle(Angle::Turn), AMBIG),
];

const fn cur(word: &'static str, c: CurrencyUnit, flags: u8) -> Word {
    w(word, Unit::Currency(c), flags)
}

const fn generic(word: &'static str, g: Generic, flags: u8) -> Word {
    cur(word, CurrencyUnit::Generic(g), flags)
}

const fn minor(word: &'static str, m: Minor, flags: u8) -> Word {
    cur(word, CurrencyUnit::Minor(m), flags)
}

/// A currency code known to be in the table, by its lowercase form.
const fn iso(word: &'static str, lower: &[u8; 3], flags: u8) -> Word {
    let c = match turbo_analysis_layers::Currency::new([
        lower[0].to_ascii_uppercase(),
        lower[1].to_ascii_uppercase(),
        lower[2].to_ascii_uppercase(),
    ]) {
        Some(c) => c,
        None => panic!("a code of three letters"),
    };
    cur(word, CurrencyUnit::Code(c), flags)
}

/// Money words. `PREFIX` ones may come before the amount (`$5`, `€ 2`);
/// `NATION_ONLY` ones name money only after a country (`Korean won`,
/// `Brazilian real`).
pub(crate) static MONEY: &[Word] = &[
    generic("$", Generic::Dollar, PREFIX),
    generic("dollar", Generic::Dollar, 0),
    generic("dollars", Generic::Dollar, 0),
    iso("€", b"eur", PREFIX),
    iso("euro", b"eur", 0),
    iso("euros", b"eur", 0),
    generic("£", Generic::Pound, PREFIX),
    generic("pound", Generic::Pound, AMBIG),
    generic("pounds", Generic::Pound, AMBIG),
    iso("sterling", b"gbp", 0),
    iso("¥", b"jpy", PREFIX),
    iso("yen", b"jpy", 0),
    iso("yens", b"jpy", 0),
    // U+FFE5 FULLWIDTH YEN SIGN, the yuan as Chinese text writes it.
    iso("￥", b"cny", PREFIX),
    iso("元", b"cny", 0),
    iso("yuan", b"cny", 0),
    iso("renminbi", b"cny", 0),
    iso("rmb", b"cny", 0),
    iso("kuai", b"cny", 0),
    iso("₹", b"inr", PREFIX),
    generic("rupee", Generic::Rupee, 0),
    generic("rupees", Generic::Rupee, 0),
    iso("₩", b"krw", PREFIX),
    iso("₽", b"rub", PREFIX),
    iso("₺", b"try", PREFIX),
    iso("₪", b"ils", PREFIX),
    iso("฿", b"thb", PREFIX),
    iso("₫", b"vnd", PREFIX),
    iso("₡", b"crc", PREFIX),
    iso("₾", b"gel", PREFIX),
    iso("₭", b"lak", PREFIX),
    iso("₦", b"ngn", PREFIX),
    iso("₱", b"php", PREFIX),
    iso("₴", b"uah", PREFIX),
    iso("₲", b"pyg", PREFIX),
    iso("₵", b"ghs", PREFIX),
    iso("₮", b"mnt", PREFIX),
    iso("৳", b"bdt", PREFIX),
    iso("៛", b"khr", PREFIX),
    iso("₸", b"kzt", PREFIX),
    iso("₼", b"azn", PREFIX),
    // The rupee in Tamil and Sinhala.
    iso("ரூ", b"lkr", 0),
    iso("රු", b"lkr", 0),
    iso("₿", b"xbt", PREFIX),
    iso("bitcoin", b"xbt", 0),
    iso("bitcoins", b"xbt", 0),
    iso("btc", b"xbt", PREFIX),
    iso("xbt", b"xbt", PREFIX),
    generic("franc", Generic::Franc, 0),
    generic("francs", Generic::Franc, 0),
    generic("peso", Generic::Peso, 0),
    generic("pesos", Generic::Peso, 0),
    generic("krone", Generic::Krone, 0),
    generic("kroner", Generic::Krone, 0),
    generic("krona", Generic::Krone, 0),
    generic("kronor", Generic::Krone, 0),
    generic("króna", Generic::Krone, 0),
    generic("dinar", Generic::Dinar, 0),
    generic("dinars", Generic::Dinar, 0),
    generic("riyal", Generic::Riyal, 0),
    generic("riyals", Generic::Riyal, 0),
    generic("rial", Generic::Riyal, 0),
    generic("rials", Generic::Riyal, 0),
    generic("ruble", Generic::Ruble, 0),
    generic("rubles", Generic::Ruble, 0),
    generic("rouble", Generic::Ruble, 0),
    generic("roubles", Generic::Ruble, 0),
    generic("shilling", Generic::Shilling, 0),
    generic("shillings", Generic::Shilling, 0),
    generic("dirham", Generic::Dirham, 0),
    generic("dirhams", Generic::Dirham, 0),
    generic("manat", Generic::Manat, 0),
    generic("manats", Generic::Manat, 0),
    generic("leu", Generic::Leu, 0),
    generic("lira", Generic::Lira, NATION_ONLY),
    generic("liras", Generic::Lira, NATION_ONLY),
    generic("lire", Generic::Lira, NATION_ONLY),
    generic("mark", Generic::Mark, NATION_ONLY),
    generic("marks", Generic::Mark, NATION_ONLY),
    generic("won", Generic::Won, NATION_ONLY),
    iso("real", b"brl", NATION_ONLY),
    iso("reais", b"brl", NATION_ONLY),
    iso("reals", b"brl", NATION_ONLY),
    iso("dong", b"vnd", NATION_ONLY),
    iso("peseta", b"esp", 0),
    iso("pesetas", b"esp", 0),
    iso("markka", b"fim", 0),
    iso("markkaa", b"fim", 0),
    iso("guilder", b"nlg", 0),
    iso("guilders", b"nlg", 0),
    iso("gulden", b"nlg", 0),
    iso("rand", b"zar", 0),
    iso("baht", b"thb", 0),
    iso("zloty", b"pln", 0),
    iso("złoty", b"pln", 0),
    iso("zlotys", b"pln", 0),
    iso("forint", b"huf", 0),
    iso("forints", b"huf", 0),
    iso("naira", b"ngn", 0),
    iso("shekel", b"ils", 0),
    iso("shekels", b"ils", 0),
    iso("rupiah", b"idr", 0),
    iso("ringgit", b"myr", 0),
    iso("hryvnia", b"uah", 0),
    iso("hryvnias", b"uah", 0),
    iso("koruna", b"czk", 0),
    iso("korunas", b"czk", 0),
    iso("tenge", b"kzt", 0),
    iso("cedi", b"ghs", 0),
    iso("cedis", b"ghs", 0),
    minor("cent", Minor::Cent, 0),
    minor("cents", Minor::Cent, 0),
    minor("penny", Minor::Penny, 0),
    minor("pennies", Minor::Penny, 0),
    minor("pence", Minor::Pence, 0),
    minor("satoshi", Minor::Satoshi, AMBIG),
    minor("satoshis", Minor::Satoshi, AMBIG),
    minor("millibitcoin", Minor::Millibitcoin, 0),
    minor("millibitcoins", Minor::Millibitcoin, 0),
    minor("jiao", Minor::Jiao, 0),
    minor("mao", Minor::Jiao, AMBIG),
    minor("fen", Minor::Fen, AMBIG),
    minor("kopek", Minor::Kopek, 0),
    minor("kopeks", Minor::Kopek, 0),
    minor("kopeck", Minor::Kopek, 0),
    minor("kopecks", Minor::Kopek, 0),
    minor("centavo", Minor::Centavo, 0),
    minor("centavos", Minor::Centavo, 0),
    minor("centime", Minor::Centime, 0),
    minor("centimes", Minor::Centime, 0),
    minor("paisa", Minor::Paisa, 0),
    minor("paise", Minor::Paisa, 0),
];

/// Codes that are also English words or common abbreviations. They name
/// money only in capitals and after the amount (`10 TRY`, never `ALL 5`
/// or `try 5`).
const AMBIGUOUS_CODES: &[&str] = &[
    "all", "bam", "bob", "cad", "cop", "cup", "dem", "dop", "ern", "esp", "gel", "gip", "imp", "kgs", "lak", "mad",
    "mnt", "mop", "nad", "pen", "php", "ron", "rub", "sar", "sek", "sos", "try",
];

pub(crate) fn ambiguous_code(lower: &str) -> bool {
    AMBIGUOUS_CODES.contains(&lower)
}

/// Codes after an `M` for millions: `30 MUSD`, `MEUR 120`.
pub(crate) static MILLION_CODES: &[(&str, &str)] = &[
    ("maud", "aud"),
    ("mcad", "cad"),
    ("mchf", "chf"),
    ("mcny", "cny"),
    ("mdkk", "dkk"),
    ("meur", "eur"),
    ("mgbp", "gbp"),
    ("mhkd", "hkd"),
    ("minr", "inr"),
    ("mjpy", "jpy"),
    ("mnok", "nok"),
    ("mnzd", "nzd"),
    ("msek", "sek"),
    ("msgd", "sgd"),
    ("musd", "usd"),
];

/// A country word and a currency it issues: with `generic`, the currency
/// of that name (`Swiss` and `franc` give `CHF`); with none, a currency
/// a following word names already (`Japanese yen`).
#[derive(Copy, Clone, Debug)]
pub(crate) struct Nation {
    pub word: &'static str,
    pub generic: Option<Generic>,
    pub code: &'static str,
    pub flags: u8,
}

const fn n(word: &'static str, generic: Generic, code: &'static str) -> Nation {
    Nation { word, generic: Some(generic), code, flags: 0 }
}

const fn named(word: &'static str, code: &'static str) -> Nation {
    Nation { word, generic: None, code, flags: 0 }
}

/// A short form that only goes before `$`: `C$`, `HK$`.
const fn dollar(word: &'static str, code: &'static str) -> Nation {
    Nation { word, generic: Some(Generic::Dollar), code, flags: ONLY_DOLLAR }
}

/// Country words, a word's rows together. A word with a space is read by
/// a rule, not the lexicon.
pub(crate) static NATIONS: &[Nation] = &[
    n("us", Generic::Dollar, "usd"),
    n("u.s.", Generic::Dollar, "usd"),
    n("united states", Generic::Dollar, "usd"),
    n("american", Generic::Dollar, "usd"),
    n("australian", Generic::Dollar, "aud"),
    n("australia", Generic::Dollar, "aud"),
    n("aussie", Generic::Dollar, "aud"),
    dollar("au", "aud"),
    n("canadian", Generic::Dollar, "cad"),
    n("canada", Generic::Dollar, "cad"),
    dollar("c", "cad"),
    dollar("can", "cad"),
    dollar("ca", "cad"),
    n("singapore", Generic::Dollar, "sgd"),
    n("singaporean", Generic::Dollar, "sgd"),
    dollar("s", "sgd"),
    dollar("sg", "sgd"),
    n("hk", Generic::Dollar, "hkd"),
    n("hong kong", Generic::Dollar, "hkd"),
    n("nz", Generic::Dollar, "nzd"),
    n("new zealand", Generic::Dollar, "nzd"),
    n("jamaican", Generic::Dollar, "jmd"),
    dollar("j", "jmd"),
    n("bahamian", Generic::Dollar, "bsd"),
    n("barbadian", Generic::Dollar, "bbd"),
    n("belize", Generic::Dollar, "bzd"),
    n("bermudian", Generic::Dollar, "bmd"),
    n("brunei", Generic::Dollar, "bnd"),
    n("fijian", Generic::Dollar, "fjd"),
    n("fiji", Generic::Dollar, "fjd"),
    n("guyanese", Generic::Dollar, "gyd"),
    n("liberian", Generic::Dollar, "lrd"),
    n("namibian", Generic::Dollar, "nad"),
    n("surinamese", Generic::Dollar, "srd"),
    n("taiwan", Generic::Dollar, "twd"),
    n("taiwanese", Generic::Dollar, "twd"),
    n("new taiwan", Generic::Dollar, "twd"),
    dollar("nt", "twd"),
    n("trinidadian", Generic::Dollar, "ttd"),
    n("trinidad and tobago", Generic::Dollar, "ttd"),
    n("cayman islands", Generic::Dollar, "kyd"),
    n("solomon islands", Generic::Dollar, "sbd"),
    n("mexican", Generic::Peso, "mxn"),
    n("mexico", Generic::Peso, "mxn"),
    dollar("mex", "mxn"),
    n("philippine", Generic::Peso, "php"),
    n("argentine", Generic::Peso, "ars"),
    n("argentinian", Generic::Peso, "ars"),
    n("chilean", Generic::Peso, "clp"),
    n("colombian", Generic::Peso, "cop"),
    n("cuban", Generic::Peso, "cup"),
    n("dominican", Generic::Peso, "dop"),
    n("uruguayan", Generic::Peso, "uyu"),
    dollar("r", "brl"),
    named("brazilian", "brl"),
    n("british", Generic::Pound, "gbp"),
    n("uk", Generic::Pound, "gbp"),
    n("egyptian", Generic::Pound, "egp"),
    n("lebanese", Generic::Pound, "lbp"),
    n("syrian", Generic::Pound, "syp"),
    n("sudanese", Generic::Pound, "sdg"),
    n("gibraltar", Generic::Pound, "gip"),
    n("swiss", Generic::Franc, "chf"),
    n("french", Generic::Franc, "frf"),
    n("congolese", Generic::Franc, "cdf"),
    n("rwandan", Generic::Franc, "rwf"),
    n("guinean", Generic::Franc, "gnf"),
    n("burundian", Generic::Franc, "bif"),
    n("djiboutian", Generic::Franc, "djf"),
    n("comorian", Generic::Franc, "kmf"),
    n("indian", Generic::Rupee, "inr"),
    n("pakistani", Generic::Rupee, "pkr"),
    n("nepalese", Generic::Rupee, "npr"),
    n("nepali", Generic::Rupee, "npr"),
    n("mauritian", Generic::Rupee, "mur"),
    n("seychellois", Generic::Rupee, "scr"),
    n("sri lankan", Generic::Rupee, "lkr"),
    n("russian", Generic::Ruble, "rub"),
    n("belarusian", Generic::Ruble, "byn"),
    n("danish", Generic::Krone, "dkk"),
    n("norwegian", Generic::Krone, "nok"),
    n("swedish", Generic::Krone, "sek"),
    n("icelandic", Generic::Krone, "isk"),
    n("kuwaiti", Generic::Dinar, "kwd"),
    n("iraqi", Generic::Dinar, "iqd"),
    n("jordanian", Generic::Dinar, "jod"),
    n("bahraini", Generic::Dinar, "bhd"),
    n("libyan", Generic::Dinar, "lyd"),
    n("tunisian", Generic::Dinar, "tnd"),
    n("algerian", Generic::Dinar, "dzd"),
    n("serbian", Generic::Dinar, "rsd"),
    n("saudi", Generic::Riyal, "sar"),
    n("qatari", Generic::Riyal, "qar"),
    n("omani", Generic::Riyal, "omr"),
    n("iranian", Generic::Riyal, "irr"),
    n("yemeni", Generic::Riyal, "yer"),
    n("kenyan", Generic::Shilling, "kes"),
    n("ugandan", Generic::Shilling, "ugx"),
    n("tanzanian", Generic::Shilling, "tzs"),
    n("somali", Generic::Shilling, "sos"),
    n("moroccan", Generic::Dirham, "mad"),
    n("emirati", Generic::Dirham, "aed"),
    n("uae", Generic::Dirham, "aed"),
    n("romanian", Generic::Leu, "ron"),
    n("moldovan", Generic::Leu, "mdl"),
    n("azerbaijani", Generic::Manat, "azn"),
    n("turkmen", Generic::Manat, "tmt"),
    n("turkish", Generic::Lira, "try"),
    n("italian", Generic::Lira, "itl"),
    n("german", Generic::Mark, "dem"),
    n("deutsche", Generic::Mark, "dem"),
    n("korean", Generic::Won, "krw"),
    n("south korean", Generic::Won, "krw"),
    n("north korean", Generic::Won, "kpw"),
    named("japanese", "jpy"),
    named("chinese", "cny"),
    named("dutch", "nlg"),
    named("finnish", "fim"),
    named("spanish", "esp"),
    named("thai", "thb"),
    named("polish", "pln"),
    named("hungarian", "huf"),
    named("israeli", "ils"),
    named("nigerian", "ngn"),
    named("indonesian", "idr"),
    named("malaysian", "myr"),
    named("ukrainian", "uah"),
    named("czech", "czk"),
    named("ghanaian", "ghs"),
    named("kazakh", "kzt"),
    named("kazakhstani", "kzt"),
    named("vietnamese", "vnd"),
    named("south african", "zar"),
];

/// The first row of `word` in [`NATIONS`].
pub(crate) fn nation(word: &str) -> Option<usize> {
    NATIONS.iter().position(|r| r.word == word)
}

/// The rows of the word whose first row is `first`.
pub(crate) fn nation_rows(first: usize) -> &'static [Nation] {
    let word = NATIONS[first].word;
    let n = NATIONS[first..].iter().take_while(|r| r.word == word).count();
    &NATIONS[first..first + n]
}

/// SI prefixes written as words, with their powers of ten: `kilo meter`,
/// `mega bytes`.
const PREFIXES: &[(&str, i64)] = &[
    ("kilo", 3),
    ("hecto", 2),
    ("deca", 1),
    ("deka", 1),
    ("deci", -1),
    ("centi", -2),
    ("milli", -3),
    ("micro", -6),
    ("nano", -9),
    ("pico", -12),
    ("mega", 6),
    ("giga", 9),
    ("tera", 12),
    ("peta", 15),
];

/// Time words: what a speed is per and what an age is counted in. The
/// value is a [`super::Time`] as a number.
const TIMES: &[(&str, super::Time, u8)] = {
    use super::Time::*;
    &[
        ("ms", Millisecond, ABBR),
        ("millisecond", Millisecond, 0),
        ("milliseconds", Millisecond, 0),
        ("s", Second, ABBR),
        ("sec", Second, ABBR),
        ("second", Second, 0),
        ("seconds", Second, 0),
        ("min", Minute, ABBR),
        ("minute", Minute, 0),
        ("minutes", Minute, 0),
        ("h", Hour, ABBR),
        ("hr", Hour, ABBR),
        ("hour", Hour, 0),
        ("hours", Hour, 0),
        ("day", Day, 0),
        ("days", Day, 0),
        ("week", Week, 0),
        ("weeks", Week, 0),
        ("month", Month, 0),
        ("months", Month, 0),
        ("year", Year, 0),
        ("years", Year, 0),
        ("yr", Year, ABBR),
        ("yrs", Year, ABBR),
    ]
};

/// Temperature scales: names, and letters that name a scale only after a
/// degree (`deg c`, `°F`).
const SCALES: &[(&str, Temperature, u8)] = &[
    ("celsius", Temperature::Celsius, 0),
    ("celcius", Temperature::Celsius, 0),
    ("centigrade", Temperature::Celsius, 0),
    ("centigrate", Temperature::Celsius, 0),
    ("fahrenheit", Temperature::Fahrenheit, 0),
    ("farenheit", Temperature::Fahrenheit, 0),
    ("kelvin", Temperature::Kelvin, 0),
    ("kelvins", Temperature::Kelvin, 0),
    ("rankine", Temperature::Rankine, 0),
    ("delisle", Temperature::Delisle, 0),
    ("c", Temperature::Celsius, ABBR),
    ("f", Temperature::Fahrenheit, ABBR),
    ("k", Temperature::Kelvin, ABBR),
];

/// The temperature scale of a [`TSCALE`] entry's value.
pub(crate) fn scale(i: i64) -> Temperature {
    SCALES[i as usize].1
}

/// Multipliers glued to an amount after a currency: `$3.5m`, `€2bn`.
const MULTIPLIERS: &[(&str, i64)] = &[("m", 6), ("mn", 6), ("mm", 6), ("bn", 9), ("b", 9), ("tn", 12), ("k", 3)];

const fn e(word: &'static str, dim: Dim, value: i64, flags: u8) -> Entry {
    Entry { word, dim, value, flags }
}

/// Every entry of the unit words, to add to the number grammar's.
pub(crate) fn entries() -> Vec<Entry> {
    let mut out = Vec::new();
    out.extend(UNITS.iter().enumerate().map(|(i, x)| e(x.word, UWORD, i as i64, x.flags)));
    out.extend(MONEY.iter().enumerate().map(|(i, x)| e(x.word, CURW, i as i64, x.flags)));
    for (i, (c, _)) in CODES.iter().enumerate() {
        out.push(e(c, CODE, i as i64, if ambiguous_code(c) { AMBIG } else { PREFIX }));
    }
    for (m, c) in MILLION_CODES {
        let i = CODES.binary_search_by(|(k, _)| k.cmp(c)).expect("a code in the table");
        debug_assert!(code(c).is_some());
        out.push(e(m, MCODE, i as i64, 0));
    }
    for (i, r) in NATIONS.iter().enumerate() {
        let first = i == 0 || NATIONS[i - 1].word != r.word;
        if first && !r.word.contains([' ', '.']) {
            out.push(e(r.word, NATION, i as i64, r.flags));
        }
    }
    out.extend(PREFIXES.iter().map(|&(p, v)| e(p, SIPRE, v, 0)));
    out.extend(TIMES.iter().map(|&(t, v, f)| e(t, TIMEW, v as i64, f)));
    out.extend(SCALES.iter().enumerate().map(|(i, &(s, _, f))| e(s, TSCALE, i as i64, f)));
    for (d, f) in [("degree", 0), ("degrees", 0), ("deg", DOT), ("degs", 0), ("°", 0)] {
        out.push(e(d, DEG, 0, f));
    }
    out.extend(MULTIPLIERS.iter().map(|&(m, v)| e(m, MULT, v, 0)));
    // `in` only after feet (`5 ft 2 in`); `"` after a number outside a
    // quotation.
    out.push(e("in", INCHW, 0, AMBIG));
    out.push(e("\"", INCHW, 1, AMBIG));
    // U+2033 DOUBLE PRIME.
    out.push(e("″", INCHW, 2, 0));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_are_lowercase_and_codes_resolve() {
        for x in UNITS.iter().chain(MONEY) {
            assert_eq!(x.word.to_lowercase(), x.word, "{}", x.word);
        }
        for r in NATIONS {
            assert!(code(r.code).is_some(), "{}", r.code);
        }
        for (_, c) in MILLION_CODES {
            assert!(code(c).is_some(), "{c}");
        }
        for c in AMBIGUOUS_CODES {
            assert!(code(c).is_some(), "{c}");
        }
    }

    #[test]
    fn a_nation_word_has_its_rows_together() {
        for (i, r) in NATIONS.iter().enumerate() {
            let first = nation(r.word).unwrap();
            assert!(nation_rows(first).len() > i - first, "{} is split", r.word);
        }
        assert_eq!(nation_rows(nation("us").unwrap()).len(), 1);
        assert_eq!(nation("atlantis"), None);
    }

    #[test]
    fn entries_cover_every_table() {
        let all = entries();
        assert!(all.iter().any(|x| x.word == "musd" && x.dim == MCODE));
        assert!(all.iter().any(|x| x.word == "try" && x.dim == CODE && x.flags & AMBIG != 0));
        assert!(all.iter().any(|x| x.word == "usd" && x.dim == CODE && x.flags & PREFIX != 0));
        assert!(!all.iter().any(|x| x.word.contains(' ')));
        assert_eq!(scale(0), Temperature::Celsius);
    }
}
