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

/// The longest semicolon-less (legacy) key in the table, in
/// bytes: no legacy match attempt looks past this.
pub(super) const MAX_LEGACY: usize = 6;

static CHUNKS: &[&[(&str, &str)]] =
    &[a::ENTRIES_A, b::ENTRIES_B, c::ENTRIES_C, d::ENTRIES_D, e::ENTRIES_E, f::ENTRIES_F];

/// The replacement text for the exact key `name` (with its
/// trailing semicolon where the table has one). The chunks
/// are contiguous ranges of one globally sorted table, so
/// the chunk whose range could hold `name` is found from
/// the chunk first keys and only that chunk is searched.
pub(super) fn lookup(name: &str) -> Option<&'static str> {
    let chunk = CHUNKS[chunk_of(name)];
    chunk.binary_search_by(|(k, _)| (*k).cmp(name)).ok().map(|i| chunk[i].1)
}

/// The chunk whose sorted range contains `name`, if any
/// chunk does: the last chunk whose first key is at most
/// `name` (the first chunk when `name` precedes every key).
fn chunk_of(name: &str) -> usize {
    CHUNKS.partition_point(|c| c.first().is_some_and(|(k, _)| *k <= name)).saturating_sub(1)
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
                // The fast lookup path relies on this key shape.
                let body = k.strip_suffix(';').unwrap_or(k);
                assert!(body.as_bytes().first().is_some_and(u8::is_ascii_alphabetic), "{k}");
                assert!(body.bytes().all(|b| b.is_ascii_alphanumeric()), "{k}");
                if body == *k {
                    assert!(k.len() <= MAX_LEGACY, "legacy key {k} is longer than MAX_LEGACY");
                }
                previous = k;
            }
        }
        assert_eq!(lookup("amp;"), Some("&"));
        assert_eq!(lookup("amp"), Some("&"));
        assert_eq!(lookup("CounterClockwiseContourIntegral;"), Some("\u{2233}"));
        assert_eq!(lookup("NotAName;"), None);
    }
}
