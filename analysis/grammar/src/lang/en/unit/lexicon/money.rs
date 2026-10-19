use super::*;

pub(super) const fn cur(word: &'static str, c: CurrencyUnit, flags: u8) -> Word {
    w(word, Unit::Currency(c), flags)
}

pub(super) const fn generic(word: &'static str, g: Generic, flags: u8) -> Word {
    cur(word, CurrencyUnit::Generic(g), flags)
}

pub(super) const fn minor(word: &'static str, m: Minor, flags: u8) -> Word {
    cur(word, CurrencyUnit::Minor(m), flags)
}

/// A currency code known to be in the table, by its lowercase form.
pub(super) const fn iso(word: &'static str, lower: &[u8; 3], flags: u8) -> Word {
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
pub(in crate::lang) static MONEY: &[Word] = &[
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
pub(super) const AMBIGUOUS_CODES: &[&str] = &[
    "all", "bam", "bob", "cad", "cop", "cup", "dem", "dop", "ern", "esp", "gel", "gip", "imp", "kgs", "lak", "mad",
    "mnt", "mop", "nad", "pen", "php", "ron", "rub", "sar", "sek", "sos", "try",
];

pub(in crate::lang) fn ambiguous_code(lower: &str) -> bool {
    AMBIGUOUS_CODES.contains(&lower)
}

/// Codes after an `M` for millions: `30 MUSD`, `MEUR 120`.
pub(in crate::lang) static MILLION_CODES: &[(&str, &str)] = &[
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
