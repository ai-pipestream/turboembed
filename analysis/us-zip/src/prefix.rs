//! Whether a three-digit ZIP prefix belongs to a state: the states of its
//! ZCTAs (`zip3-states.txt`) and the assignments that hold no ZCTA
//! (`zip3-extra.txt`). Reading these needs neither the ZCTA nor the place
//! table, so a caller that asks only this does not parse them.

use std::sync::OnceLock;

/// The bundled prefixes that hold no ZCTA (`data/zip3-extra.txt`).
pub const PREFIX_EXTRA_TABLE: &str = include_str!("../data/zip3-extra.txt");

/// For each prefix `000` to `999`, its states, in code order.
fn prefixes() -> &'static [Vec<[u8; 2]>] {
    static TABLE: OnceLock<Vec<Vec<[u8; 2]>>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = vec![Vec::new(); 1000];
        for text in [crate::PREFIX_TABLE, PREFIX_EXTRA_TABLE] {
            for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
                let (p, states) = line.split_once('\t').expect("a prefix row has a prefix and states");
                let p = crate::parse::digits(p, 3).expect("a prefix is three digits") as usize;
                for s in states.split(',') {
                    let code = s.split(':').next().unwrap_or("").as_bytes();
                    let code: [u8; 2] = code.try_into().expect("a state code is two letters");
                    t[p].push(code);
                }
            }
        }
        for s in &mut t {
            s.sort_unstable();
            s.dedup();
        }
        t
    })
}

/// Whether the ZIP prefix `zip3` (three ASCII digits, `627`) belongs to the
/// state, territory or armed forces designation `state` (a USPS code,
/// `IL`): one of the prefix's ZCTAs lies mostly in that state, or the
/// Postal Service assigns the prefix there and it holds no ZCTA of it
/// (`AE 090`, `UM 968`). Anything else, a lower case code among it, is
/// `false`.
pub fn zip3_in_state(zip3: &[u8], state: &[u8]) -> bool {
    let (Ok(code), [a, b, c]) = (<[u8; 2]>::try_from(state), zip3) else {
        return false;
    };
    if ![a, b, c].iter().all(|d| d.is_ascii_digit()) {
        return false;
    }
    let p = usize::from(a - b'0') * 100 + usize::from(b - b'0') * 10 + usize::from(c - b'0');
    prefixes()[p].binary_search(&code).is_ok()
}
