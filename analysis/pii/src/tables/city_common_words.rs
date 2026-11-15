//! The one-word place names of the bundled gazetteer that are also common
//! English words or common given or family names (`mobile`, `reading`,
//! `jackson`, `florence`), which the bare city reading (#2097) leaves
//! out: one match key (`turbo_geo::Folder`) per line of
//! `city_common_words.txt`, sorted by bytes.
//!
//! The list is the gazetteer's one-token match keys (names and alternate
//! names) of four letters or
//! more that are in either of two lists, intersected once and checked in,
//! so the scanner reads only the 268 keys that collide:
//!
//! - the lower-case entries of SCOWL's `english-words` and
//!   `american-words` at sizes 10 to 50 (SCOWL 2020.12.07, Kevin
//!   Atkinson's spell-checker word lists, <http://wordlist.aspell.net/>,
//!   `scowl-2020.12.07.tar.gz` SHA-256
//!   `5587667caa20c4891390c2d42dbb4d5c4c3f41bee77af1457ece3ba23fb859cc`,
//!   under the permissive notice in docs/licences.md): a name SCOWL writes
//!   in lower case is a common word;
//! - the 1,000 most frequent female first names, male first names and
//!   surnames of the US Census Bureau's 1990 name files (public domain,
//!   <https://www2.census.gov/topics/genealogy/1990surnames/>,
//!   `dist.female.first` SHA-256
//!   `bd2f310fc4e5d5e5ea122c9d4342c9821145823118eb20db1647f305ec77b358`,
//!   `dist.male.first`
//!   `0a5078ef6effe3b483d15b0f7f95047662126c9bfb624ecd5e5b978fc0f2470b`,
//!   `dist.all.last`
//!   `b0e2b3743ccbad641ca48b344c24cdebcd1d9a1f76dc6dbf05986f2919f0b4e1`).

use std::sync::OnceLock;

static TEXT: &str = include_str!("city_common_words.txt");

/// The keys, in byte order.
pub(crate) fn keys() -> &'static [&'static str] {
    static KEYS: OnceLock<Vec<&'static str>> = OnceLock::new();
    KEYS.get_or_init(|| TEXT.lines().collect())
}

/// Whether the match key `key` is a common word or name.
pub(crate) fn contains(key: &str) -> bool {
    keys().binary_search(&key).is_ok()
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use turbo_geo::{BUNDLED_TABLE, Folder};

    use super::*;

    #[test]
    fn every_key_is_a_one_word_gazetteer_key_of_four_letters_or_more_in_order() {
        let mut folder = Folder::new();
        let gazetteer: HashSet<String> = BUNDLED_TABLE
            .lines()
            .filter(|l| !l.starts_with('#'))
            .flat_map(|l| {
                let f: Vec<&str> = l.split(';').collect();
                let alternates = f[3].split('|').filter(|n| !n.is_empty());
                std::iter::once(f[2]).chain(alternates).map(|n| folder.fold(n)).collect::<Vec<_>>()
            })
            .collect();
        let all = keys();
        assert_eq!(all.len(), 268);
        assert!(all.windows(2).all(|w| w[0] < w[1]), "sorted and unique");
        for &k in all {
            assert!(k.chars().count() >= 4 && !k.contains(' '), "{k:?}");
            assert!(gazetteer.contains(k), "{k:?} is no gazetteer key");
        }
    }

    #[test]
    fn common_words_and_names_are_in_and_plain_city_names_are_not() {
        for k in ["mobile", "reading", "phoenix", "price", "guide", "jackson", "florence", "austin", "nice"] {
            assert!(contains(k), "{k}");
        }
        for k in ["rouen", "boston", "paris", "london", "chicago", "pasadena", "cambridge", "Mobile"] {
            assert!(!contains(k), "{k}");
        }
    }
}
