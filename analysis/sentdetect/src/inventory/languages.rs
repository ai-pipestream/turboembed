//! Non-English tables, loaded once per language from licensed static data.
use std::sync::OnceLock;

use super::{Abbreviation, Bundle, ContextRule};

struct Language {
    code: &'static str,
    data: &'static str,
    tables: OnceLock<[Vec<Abbreviation>; 4]>,
    bundle: OnceLock<Bundle>,
}

impl Language {
    const fn new(code: &'static str, data: &'static str) -> Self {
        Self { code, data, tables: OnceLock::new(), bundle: OnceLock::new() }
    }

    fn get(&'static self) -> &'static Bundle {
        self.bundle.get_or_init(|| {
            let tables = self.tables.get_or_init(|| {
                let mut tables: [Vec<Abbreviation>; 4] = std::array::from_fn(|_| Vec::new());
                for line in self.data.lines() {
                    let mut fields = line.splitn(3, '\t');
                    let class = match fields.next().expect("bundled class") {
                        "General" => 0,
                        "Titles" => 1,
                        "Academic" => 2,
                        "LegalTechnical" => 3,
                        _ => unreachable!("checked bundled class"),
                    };
                    let rule = match fields.next().expect("bundled rule") {
                        "Prepositive" => ContextRule::Prepositive,
                        "Numeric" => ContextRule::Numeric,
                        "Trailing" => ContextRule::Trailing,
                        _ => unreachable!("checked bundled rule"),
                    };
                    tables[class].push(Abbreviation { text: fields.next().expect("bundled text"), rule });
                }
                tables
            });
            Bundle { code: self.code, classes: std::array::from_fn(|i| tables[i].as_slice()), penultimates: &[] }
        })
    }
}

macro_rules! language {
    ($code:literal) => {
        Language::new($code, include_str!(concat!("../../data/inventory/", $code, ".tsv")))
    };
}

static LANGUAGES: [Language; 13] = [
    language!("ar"),
    language!("bg"),
    language!("da"),
    language!("de"),
    language!("es"),
    language!("fr"),
    language!("it"),
    language!("kk"),
    language!("nl"),
    language!("pl"),
    language!("pt"),
    language!("ru"),
    language!("sk"),
];

pub(super) fn bundle(code: &str) -> Option<&'static Bundle> {
    LANGUAGES.iter().find(|l| code.eq_ignore_ascii_case(l.code)).map(Language::get)
}
