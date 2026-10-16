//! Units of money: a currency with a code, a name many currencies share
//! (`dollar`, `franc`), or a minor unit (`cent`, `pence`).
//!
//! Codes are [`Currency`] values. The table here lists the ISO 4217
//! codes Recognizers-Text names, a few withdrawn ones (`FIM`, `ESP`,
//! `NLG`, `DEM`, `FRF`, `ITL`) and `XBT` for bitcoin, which ISO 4217 does
//! not assign.

use std::fmt;

use turbo_analysis_layers::Currency;

/// A currency unit.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CurrencyUnit {
    /// One currency: `USD`, `JPY`.
    Code(Currency),
    /// A name that several currencies share, with no country said:
    /// `$`, `pounds`, `francs`.
    Generic(Generic),
    /// A fraction of a major unit: `cents`, `pence`, `satoshis`.
    Minor(Minor),
}

/// A currency name that several currencies share.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Generic {
    Dollar,
    Pound,
    Franc,
    Peso,
    Rupee,
    Krone,
    Dinar,
    Riyal,
    Ruble,
    Shilling,
    Dirham,
    Manat,
    Leu,
    Lira,
    Mark,
    Won,
}

impl Generic {
    fn name(self) -> &'static str {
        match self {
            Generic::Dollar => "Dollar",
            Generic::Pound => "Pound",
            Generic::Franc => "Franc",
            Generic::Peso => "Peso",
            Generic::Rupee => "Rupee",
            Generic::Krone => "Krone",
            Generic::Dinar => "Dinar",
            Generic::Riyal => "Riyal",
            Generic::Ruble => "Ruble",
            Generic::Shilling => "Shilling",
            Generic::Dirham => "Dirham",
            Generic::Manat => "Manat",
            Generic::Leu => "Leu",
            Generic::Lira => "Lira",
            Generic::Mark => "Mark",
            Generic::Won => "Won",
        }
    }

    /// The currency the name denotes by convention when no country is
    /// said, as `turbo-numeric` reads it: `$` and `dollar` are `USD`,
    /// `£` and `pound` `GBP`, `rupee` `INR`, `ruble` `RUB`. The others
    /// have none.
    pub fn conventional(self) -> Option<Currency> {
        match self {
            Generic::Dollar => code("usd"),
            Generic::Pound => code("gbp"),
            Generic::Rupee => code("inr"),
            Generic::Ruble => code("rub"),
            _ => None,
        }
    }
}

/// A minor unit of money.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Minor {
    Cent,
    Penny,
    Pence,
    Satoshi,
    Millibitcoin,
    Jiao,
    Fen,
    Kopek,
    Centavo,
    Centime,
    Paisa,
}

impl Minor {
    fn name(self) -> &'static str {
        match self {
            Minor::Cent => "Cent",
            Minor::Penny => "Penny",
            Minor::Pence => "Pence",
            Minor::Satoshi => "Satoshi",
            Minor::Millibitcoin => "Millibitcoin",
            Minor::Jiao => "Jiao",
            Minor::Fen => "Fen",
            Minor::Kopek => "Kopek",
            Minor::Centavo => "Centavo",
            Minor::Centime => "Centime",
            Minor::Paisa => "Paisa",
        }
    }

    /// The major unit and the power of ten it is divided by: a cent is a
    /// dollar over 10^2, a satoshi a bitcoin over 10^8.
    pub fn major(self) -> (CurrencyUnit, u8) {
        let generic = |g| CurrencyUnit::Generic(g);
        let coded = |c| CurrencyUnit::Code(code(c).expect("a code in the table"));
        match self {
            Minor::Cent => (generic(Generic::Dollar), 2),
            Minor::Penny | Minor::Pence => (generic(Generic::Pound), 2),
            Minor::Satoshi => (coded("xbt"), 8),
            Minor::Millibitcoin => (coded("xbt"), 3),
            Minor::Jiao => (coded("cny"), 1),
            Minor::Fen => (coded("cny"), 2),
            Minor::Kopek => (generic(Generic::Ruble), 2),
            Minor::Centavo => (generic(Generic::Peso), 2),
            Minor::Centime => (generic(Generic::Franc), 2),
            Minor::Paisa => (generic(Generic::Rupee), 2),
        }
    }
}

impl CurrencyUnit {
    /// The currency an amount in this unit is in, and the power of ten
    /// the amount is divided by to state it in that currency: `USD` and
    /// 0 for `US dollars`, `USD` and 2 for `cents`, `None` for a name
    /// with no conventional currency (`francs`).
    pub fn currency(self) -> Option<(Currency, u8)> {
        match self {
            CurrencyUnit::Code(c) => Some((c, 0)),
            CurrencyUnit::Generic(g) => g.conventional().map(|c| (c, 0)),
            CurrencyUnit::Minor(m) => {
                let (major, e) = m.major();
                major.currency().map(|(c, e0)| (c, e0 + e))
            }
        }
    }

    /// Whether the unit's major unit is divided into hundredths, so that
    /// `15 dollars 50` is 15.50. True for every unit but a few codes.
    pub fn has_hundredths(self) -> bool {
        match self {
            CurrencyUnit::Code(c) => !NO_HUNDREDTHS.contains(&c.code()),
            CurrencyUnit::Generic(_) => true,
            CurrencyUnit::Minor(_) => false,
        }
    }

    pub(crate) fn write_symbol(self, out: &mut String) {
        match self {
            CurrencyUnit::Code(c) => out.push_str(c.as_str()),
            CurrencyUnit::Generic(g) => out.extend(g.name().chars().map(|c| c.to_ascii_lowercase())),
            CurrencyUnit::Minor(m) => out.extend(m.name().chars().map(|c| c.to_ascii_lowercase())),
        }
    }
}

impl fmt::Display for CurrencyUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            CurrencyUnit::Code(c) => match name(c) {
                Some(n) => f.write_str(n),
                None => f.write_str(c.as_str()),
            },
            CurrencyUnit::Generic(g) => f.write_str(g.name()),
            CurrencyUnit::Minor(m) => f.write_str(m.name()),
        }
    }
}

/// Codes whose currency has no hundredths in use.
const NO_HUNDREDTHS: &[[u8; 3]] = &[*b"JPY", *b"KRW", *b"KPW", *b"VND", *b"CLP", *b"ISK", *b"PYG", *b"XBT"];

/// The code written in lowercase as `lower`, when the table has it.
pub fn code(lower: &str) -> Option<Currency> {
    let i = CODES.binary_search_by(|(c, _)| c.cmp(&lower)).ok()?;
    let b = CODES[i].0.as_bytes();
    Currency::new([b[0].to_ascii_uppercase(), b[1].to_ascii_uppercase(), b[2].to_ascii_uppercase()])
}

/// The English name of a code in the table.
pub fn name(c: Currency) -> Option<&'static str> {
    let low = c.code().map(|b| b.to_ascii_lowercase());
    CODES.binary_search_by(|(k, _)| k.as_bytes().cmp(&low[..])).ok().map(|i| CODES[i].1)
}

/// Every code the grammar knows, lowercase and sorted, with the English
/// name Recognizers-Text gives it.
pub static CODES: &[(&str, &str)] = &[
    ("aed", "United Arab Emirates dirham"),
    ("afn", "Afghan afghani"),
    ("all", "Albanian lek"),
    ("amd", "Armenian dram"),
    ("aoa", "Angolan kwanza"),
    ("ars", "Argentine peso"),
    ("aud", "Australian dollar"),
    ("awg", "Aruban florin"),
    ("azn", "Azerbaijani manat"),
    ("bam", "Bosnia and Herzegovina convertible mark"),
    ("bbd", "Barbadian dollar"),
    ("bdt", "Bangladeshi taka"),
    ("bgn", "Bulgarian lev"),
    ("bhd", "Bahraini dinar"),
    ("bif", "Burundian franc"),
    ("bmd", "Bermudian dollar"),
    ("bnd", "Brunei dollar"),
    ("bob", "Bolivian boliviano"),
    ("brl", "Brazilian real"),
    ("bsd", "Bahamian dollar"),
    ("btn", "Bhutanese ngultrum"),
    ("bwp", "Botswana pula"),
    ("byn", "New Belarusian ruble"),
    ("bzd", "Belize dollar"),
    ("cad", "Canadian dollar"),
    ("cdf", "Congolese franc"),
    ("chf", "Swiss franc"),
    ("clp", "Chilean peso"),
    ("cny", "Chinese yuan"),
    ("cop", "Colombian peso"),
    ("crc", "Costa Rican colón"),
    ("cuc", "Cuban convertible peso"),
    ("cup", "Cuban peso"),
    ("cve", "Cape Verdean escudo"),
    ("czk", "Czech koruna"),
    ("dem", "Deutsche mark"),
    ("djf", "Djiboutian franc"),
    ("dkk", "Danish krone"),
    ("dop", "Dominican peso"),
    ("dzd", "Algerian dinar"),
    ("egp", "Egyptian pound"),
    ("ern", "Eritrean nakfa"),
    ("esp", "Peseta"),
    ("etb", "Ethiopian birr"),
    ("eur", "Euro"),
    ("fim", "Finnish markka"),
    ("fjd", "Fijian dollar"),
    ("fkp", "Falkland Islands pound"),
    ("frf", "French franc"),
    ("gbp", "British pound"),
    ("gel", "Georgian lari"),
    ("ghs", "Ghanaian cedi"),
    ("gip", "Gibraltar pound"),
    ("gmd", "Gambian dalasi"),
    ("gnf", "Guinean franc"),
    ("gtq", "Guatemalan quetzal"),
    ("gyd", "Guyanese dollar"),
    ("hkd", "Hong Kong dollar"),
    ("hnl", "Honduran lempira"),
    ("hrk", "Croatian kuna"),
    ("htg", "Haitian gourde"),
    ("huf", "Hungarian forint"),
    ("idr", "Indonesian rupiah"),
    ("ils", "Israeli new shekel"),
    ("imp", "Manx pound"),
    ("inr", "Indian rupee"),
    ("iqd", "Iraqi dinar"),
    ("irr", "Iranian rial"),
    ("isk", "Icelandic króna"),
    ("itl", "Italian lira"),
    ("jep", "Jersey pound"),
    ("jmd", "Jamaican dollar"),
    ("jod", "Jordanian dinar"),
    ("jpy", "Japanese yen"),
    ("kes", "Kenyan shilling"),
    ("kgs", "Kyrgyzstani som"),
    ("khr", "Cambodian riel"),
    ("kmf", "Comorian franc"),
    ("kpw", "North Korean won"),
    ("krw", "South Korean won"),
    ("kwd", "Kuwaiti dinar"),
    ("kyd", "Cayman Islands dollar"),
    ("kzt", "Kazakhstani tenge"),
    ("lak", "Lao kip"),
    ("lbp", "Lebanese pound"),
    ("lkr", "Sri Lankan rupee"),
    ("lrd", "Liberian dollar"),
    ("lsl", "Lesotho loti"),
    ("lyd", "Libyan dinar"),
    ("mad", "Moroccan dirham"),
    ("mdl", "Moldovan leu"),
    ("mga", "Malagasy ariary"),
    ("mkd", "Macedonian denar"),
    ("mmk", "Burmese kyat"),
    ("mnt", "Mongolian tögrög"),
    ("mop", "Macanese pataca"),
    ("mro", "Mauritanian ouguiya"),
    ("mur", "Mauritian rupee"),
    ("mvr", "Maldivian rufiyaa"),
    ("mwk", "Malawian kwacha"),
    ("mxn", "Mexican peso"),
    ("myr", "Malaysian ringgit"),
    ("mzn", "Mozambican metical"),
    ("nad", "Namibian dollar"),
    ("ngn", "Nigerian naira"),
    ("nio", "Nicaraguan córdoba"),
    ("nlg", "Netherlands guilder"),
    ("nok", "Norwegian krone"),
    ("npr", "Nepalese rupee"),
    ("nzd", "New Zealand dollar"),
    ("omr", "Omani rial"),
    ("pab", "Panamanian balboa"),
    ("pen", "Peruvian sol"),
    ("pgk", "Papua New Guinean kina"),
    ("php", "Philippine peso"),
    ("pkr", "Pakistani rupee"),
    ("pln", "Polish złoty"),
    ("prb", "Transnistrian ruble"),
    ("pyg", "Paraguayan guaraní"),
    ("qar", "Qatari riyal"),
    ("ron", "Romanian leu"),
    ("rsd", "Serbian dinar"),
    ("rub", "Russian ruble"),
    ("rwf", "Rwandan franc"),
    ("sar", "Saudi riyal"),
    ("sbd", "Solomon Islands dollar"),
    ("scr", "Seychellois rupee"),
    ("sdg", "Sudanese pound"),
    ("sek", "Swedish krona"),
    ("sgd", "Singapore dollar"),
    ("shp", "Saint Helena pound"),
    ("sll", "Sierra Leonean leone"),
    ("sos", "Somali shilling"),
    ("srd", "Surinamese dollar"),
    ("ssp", "South Sudanese pound"),
    ("stn", "São Tomé and Príncipe dobra"),
    ("syp", "Syrian pound"),
    ("szl", "Swazi lilangeni"),
    ("thb", "Thai baht"),
    ("tjs", "Tajikistani somoni"),
    ("tmt", "Turkmenistan manat"),
    ("tnd", "Tunisian dinar"),
    ("try", "Turkish lira"),
    ("ttd", "Trinidad and Tobago dollar"),
    ("tvd", "Tuvaluan dollar"),
    ("twd", "New Taiwan dollar"),
    ("tzs", "Tanzanian shilling"),
    ("uah", "Ukrainian hryvnia"),
    ("ugx", "Ugandan shilling"),
    ("usd", "United States dollar"),
    ("uyu", "Uruguayan peso"),
    ("vef", "Venezuelan bolívar"),
    ("vnd", "Vietnamese dong"),
    ("vuv", "Vanuatu vatu"),
    ("wst", "Samoan tālā"),
    ("xaf", "Central African CFA franc"),
    ("xbt", "Bitcoin"),
    ("xof", "West African CFA franc"),
    ("xpf", "CFP franc"),
    ("yer", "Yemeni rial"),
    ("zar", "South African rand"),
    ("zmw", "Zambian kwacha"),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_and_well_formed() {
        assert!(CODES.windows(2).all(|w| w[0].0 < w[1].0));
        for (c, _) in CODES {
            assert!(c.len() == 3 && c.bytes().all(|b| b.is_ascii_lowercase()), "{c}");
            assert_eq!(code(c).map(|x| x.as_str().to_ascii_lowercase()), Some(c.to_string()));
        }
    }

    #[test]
    fn codes_and_names() {
        let usd = code("usd").unwrap();
        assert_eq!(usd.as_str(), "USD");
        assert_eq!(name(usd), Some("United States dollar"));
        assert_eq!(code("xyz"), None);
        assert_eq!(name(Currency::parse("XYZ").unwrap()), None);
        assert_eq!(CurrencyUnit::Code(Currency::parse("XYZ").unwrap()).to_string(), "XYZ");
        assert_eq!(CurrencyUnit::Generic(Generic::Franc).to_string(), "Franc");
        assert_eq!(CurrencyUnit::Minor(Minor::Pence).to_string(), "Pence");
    }

    #[test]
    fn amounts_go_to_a_currency() {
        let usd = code("usd").unwrap();
        assert_eq!(CurrencyUnit::Code(usd).currency(), Some((usd, 0)));
        assert_eq!(CurrencyUnit::Generic(Generic::Dollar).currency(), Some((usd, 0)));
        assert_eq!(CurrencyUnit::Minor(Minor::Cent).currency(), Some((usd, 2)));
        assert_eq!(CurrencyUnit::Minor(Minor::Satoshi).currency(), Some((code("xbt").unwrap(), 8)));
        assert_eq!(CurrencyUnit::Generic(Generic::Franc).currency(), None);
        assert_eq!(CurrencyUnit::Minor(Minor::Centime).currency(), None);
    }

    #[test]
    fn hundredths() {
        assert!(CurrencyUnit::Generic(Generic::Dollar).has_hundredths());
        assert!(!CurrencyUnit::Code(code("jpy").unwrap()).has_hundredths());
        assert!(CurrencyUnit::Code(code("eur").unwrap()).has_hundredths());
        assert!(!CurrencyUnit::Minor(Minor::Cent).has_hundredths());
    }
}
