//! The county names of the US Census Bureau's 2020 county list
//! (`national_county2020.txt`, public domain, from
//! <https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt>,
//! SHA-256 `9f6e5f6eb6ac2f5e9a36d5fd01dec77991bddc75118f748a069441a4782970d6`):
//! the names whose last word is `County`, `Parish`, `Borough`, `Area`
//! (`Census Area`), `Municipality` or `Municipio`, each once, one per
//! line of `us_counties.txt`, sorted by their characters in upper case.
//! The independent cities (`Baltimore city`) are cities, and the
//! District of Columbia and the island territories are no county.

use std::cmp::Ordering;
use std::sync::OnceLock;

static TEXT: &str = include_str!("us_counties.txt");

/// The words a county name ends with, in upper case.
pub(crate) const SUFFIXES: [&str; 6] = ["COUNTY", "PARISH", "BOROUGH", "AREA", "MUNICIPALITY", "MUNICIPIO"];

/// The most words a name holds, its suffix among them.
pub(crate) const MAX_WORDS: usize = 5;

fn names() -> &'static [&'static str] {
    static NAMES: OnceLock<Vec<&'static str>> = OnceLock::new();
    NAMES.get_or_init(|| TEXT.lines().collect())
}

/// `c` in upper case, a right single quotation mark read as an apostrophe.
fn upper(c: char) -> impl Iterator<Item = char> {
    (if c == '\u{2019}' { '\'' } else { c }).to_uppercase()
}

fn cmp_upper(a: &str, b: &str) -> Ordering {
    a.chars().flat_map(upper).cmp(b.chars().flat_map(upper))
}

/// The table's spelling of `name`, compared in upper case, when it is a
/// county name.
pub(crate) fn county(name: &str) -> Option<&'static str> {
    let names = names();
    names.binary_search_by(|n| cmp_upper(n, name)).ok().map(|i| names[i])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_sorted_once_each_and_ends_in_a_suffix() {
        let n = names();
        assert_eq!(n.len(), 1918);
        assert!(n.windows(2).all(|w| cmp_upper(w[0], w[1]) == Ordering::Less));
        for name in n {
            let last = name.rsplit(' ').next().unwrap().to_uppercase();
            assert!(SUFFIXES.contains(&last.as_str()), "{name}");
            assert!(name.split(' ').count() <= MAX_WORDS, "{name}");
            assert!(turbo_unicode::normalize::is_normalized(turbo_unicode::normalize::Form::Nfc, name), "{name}");
        }
        assert_eq!(county("COOK COUNTY"), Some("Cook County"));
        assert_eq!(county("Prince George\u{2019}s County"), Some("Prince George's County"));
        assert_eq!(county("MAYAGÜEZ MUNICIPIO"), Some("Mayagüez Municipio"));
        assert_eq!(county("Baltimore city"), None);
    }
}
