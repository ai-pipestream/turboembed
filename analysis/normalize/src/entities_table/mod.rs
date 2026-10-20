//! The WHATWG named character reference table, split across
//! sorted chunks for binary search.

mod a;
mod b;
mod c;
mod d;
mod e;
mod f;

/// The longest key in the table, in bytes: no match attempt
/// needs to look past this many bytes after the `&`.
pub(super) const MAX_KEY: usize = 32;

static CHUNKS: &[&[(&str, &str)]] =
    &[a::ENTRIES_A, b::ENTRIES_B, c::ENTRIES_C, d::ENTRIES_D, e::ENTRIES_E, f::ENTRIES_F];

/// The replacement text for the exact key `name` (with its
/// trailing semicolon where the table has one).
pub(super) fn lookup(name: &str) -> Option<&'static str> {
    for chunk in CHUNKS {
        if let Ok(i) = chunk.binary_search_by(|(k, _)| (*k).cmp(name)) {
            return Some(chunk[i].1);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_is_complete_and_sorted() {
        let total: usize = CHUNKS.iter().map(|c| c.len()).sum();
        assert_eq!(total, 2231);
        let mut previous = "";
        for chunk in CHUNKS {
            for (k, _) in *chunk {
                assert!(previous < k, "keys out of order at {k}");
                assert!(k.len() <= MAX_KEY, "{k} is longer than MAX_KEY");
                previous = k;
            }
        }
        assert_eq!(lookup("amp;"), Some("&"));
        assert_eq!(lookup("amp"), Some("&"));
        assert_eq!(lookup("CounterClockwiseContourIntegral;"), Some("\u{2233}"));
        assert_eq!(lookup("NotAName;"), None);
    }
}
