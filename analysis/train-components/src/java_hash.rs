//! `String.hashCode`, `StringList.hashCode` and the order a
//! `java.util.HashMap` iterates its entries in, which decides the order
//! of a serialized dictionary's entries (`Dictionary` holds a
//! `HashSet<StringList>`) and of a model file's zip entries
//! (`BaseModel` holds a `HashMap<String, Object>`).
//!
//! The map is `turbo_jvm::hash_map`'s model of the JDK's table: a table of
//! 16 buckets, doubled when the number of entries passes three quarters of
//! the table, each bucket a list in insertion order that a resize splits
//! without reordering, or a red-black tree once it reaches nine entries in
//! a table of 64 buckets or more (the table is doubled instead when it is
//! smaller).

use turbo_jvm::hash_map::{HashMapOrder, Insert, spread};
use turbo_unicode::java::code_point::to_lower_case;

use crate::error::{Error, Result};

/// `String.hashCode` over UTF-16 code units.
pub fn string_hash_code(units: impl IntoIterator<Item = u16>) -> i32 {
    let mut h: i32 = 0;
    for u in units {
        h = h.wrapping_mul(31).wrapping_add(i32::from(u));
    }
    h
}

/// `String.hashCode` of a Rust string.
pub fn str_hash_code(s: &str) -> i32 {
    string_hash_code(s.encode_utf16())
}

/// `StringList.hashCode`: the tokens joined by `,`, lowercased per code
/// point with `Character.toLowerCase` (`StringUtil.toLowerCase`), hashed as
/// a string.
pub fn string_list_hash_code<'t>(tokens: impl IntoIterator<Item = &'t str>) -> i32 {
    let mut h: i32 = 0;
    let mut buf = [0u16; 2];
    for (i, token) in tokens.into_iter().enumerate() {
        if i > 0 {
            h = h.wrapping_mul(31).wrapping_add(i32::from(b','));
        }
        for c in token.chars() {
            let lowered = char::from_u32(to_lower_case(c as u32)).unwrap_or(c);
            for &u in lowered.encode_utf16(&mut buf).iter() {
                h = h.wrapping_mul(31).wrapping_add(i32::from(u));
            }
        }
    }
    h
}

/// The order a `HashMap` iterates entries inserted in this order with
/// these hashes, as indices into `hashes`. The keys must be distinct, as
/// the entries of a set are; two keys with one hash are two entries.
///
/// A bucket that a table of 64 or more turns into a tree comes in the
/// tree's order ([`turbo_jvm::hash_map`]). Two entries with one hash code
/// in such a bucket are refused ([`Error::TreeBucketTie`]): `StringList`
/// is not `Comparable`, so the JVM orders them by
/// `System.identityHashCode`. `String` keys would be ordered by
/// `compareTo`, but only their hash codes are given here, and a model's
/// few artifact names never fill a bucket.
pub fn hash_map_order(hashes: &[i32]) -> Result<Vec<usize>> {
    let spreads: Vec<u32> = hashes.iter().map(|&h| spread(h)).collect();
    let mut order = Vec::with_capacity(hashes.len());
    if !HashMapOrder::new().order(&spreads, Insert::Put, &mut order) {
        return Err(Error::TreeBucketTie);
    }
    Ok(order.into_iter().map(|i| i as usize).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn string_hash_codes_are_javas() {
        assert_eq!(str_hash_code(""), 0);
        assert_eq!(str_hash_code("a"), 97);
        assert_eq!(str_hash_code("manifest.properties"), 1339103826);
        assert_eq!(str_hash_code("sent.model"), 734307667);
        assert_eq!(str_hash_code("token.model"), 1737130740);
        assert_eq!(str_hash_code("abbreviations.dictionary"), 730939199);
        // Code units, not code points: an emoji is two.
        assert_eq!(str_hash_code("\u{1F600}"), (0xD83D_i32).wrapping_mul(31).wrapping_add(0xDE00));
    }

    #[test]
    fn string_list_hash_codes_lowercase_and_join_with_commas() {
        assert_eq!(string_list_hash_code(["Dr."]), str_hash_code("dr."));
        assert_eq!(string_list_hash_code(["et", "al."]), str_hash_code("et,al."));
        // Character.toLowerCase per code point: U+0130 lowercases to i.
        assert_eq!(string_list_hash_code(["\u{0130}"]), str_hash_code("i"));
    }

    #[test]
    fn a_small_map_iterates_by_bucket_then_insertion() {
        // Hashes 2, 17 and 1 land in buckets 2, 1 and 1 of a 16 table: the
        // two in bucket 1 come first, in their insertion order.
        assert_eq!(hash_map_order(&[2, 17, 1]).unwrap(), vec![1, 2, 0]);
        // A negative hash is spread as unsigned: -16 is 0xFFFFFFF0, spread
        // to 0xFFFF000F, bucket 15.
        assert_eq!(hash_map_order(&[-16, 0]).unwrap(), vec![1, 0]);
    }

    #[test]
    fn the_table_doubles_past_twelve_entries_and_splits_buckets_in_order() {
        // 13 entries, all in bucket 0 of a 16 table, half of them in bucket
        // 16 of the doubled table. The doubling happens when the 13th goes
        // in, and keeps each half's order.
        let hashes: Vec<i32> = (0..13).map(|i| if i % 2 == 0 { 32 * i } else { 32 * i + 16 }).collect();
        let order = hash_map_order(&hashes).unwrap();
        let (lo, hi): (Vec<usize>, Vec<usize>) = (0..13).partition(|i| i % 2 == 0);
        assert_eq!(order, [lo, hi].concat());
    }

    #[test]
    fn nine_in_one_bucket_doubles_a_small_table_once() {
        // Nine hashes in bucket 0 of a 16 table: the ninth doubles the
        // table to 32, where they are still one bucket, and the JDK does
        // not look again. A tenth with hash 16 then has bucket 16.
        let mut hashes: Vec<i32> = (0..9).map(|i| 32 * i).collect();
        hashes.push(16);
        assert_eq!(hash_map_order(&hashes).unwrap(), (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn a_tree_bucket_with_one_hash_twice_is_refused() {
        // 30 distinct small hashes take the table to 64 (doubled at the 13th
        // and the 25th entry). Eight more with hash 64 join hash 0 in bucket
        // 0: the ninth entry there makes it a tree, which orders entries
        // with one hash by identity.
        let mut hashes: Vec<i32> = (0..30).collect();
        hashes.extend([64; 7]);
        assert!(hash_map_order(&hashes).is_ok());
        hashes.push(64);
        assert!(matches!(hash_map_order(&hashes), Err(Error::TreeBucketTie)));
    }

    #[test]
    fn a_tree_bucket_of_distinct_hashes_comes_in_tree_order() {
        // As above with eight distinct hashes: the tree's root (hash 192)
        // comes first in the bucket, then the rest in insertion order, as
        // JDK 21 iterates a `HashSet<Integer>` of these.
        let mut hashes: Vec<i32> = (0..30).collect();
        hashes.extend((1..=8).map(|i| 64 * i));
        let order = hash_map_order(&hashes).unwrap();
        let bucket: Vec<i32> = order.iter().map(|&i| hashes[i]).filter(|h| h % 64 == 0).collect();
        assert_eq!(bucket, [192, 0, 64, 128, 256, 320, 384, 448, 512]);
    }
}
