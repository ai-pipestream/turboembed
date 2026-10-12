//! The iteration order of a `java.util.HashMap` (JDK 8 and later), for
//! keys put into an empty map with the default capacity and load factor.
//!
//! `LanguageDetectorME` hands its n-grams to the model in the order of a
//! `HashMap` keyed by `CharBuffer`, and the model adds up their parameters
//! in that order. Floating point addition is not associative, so the
//! probabilities match the JVM's bit for bit only when the n-grams are
//! summed in the same order.
//!
//! The map's buckets are an array whose length (the capacity) is a power
//! of two, starting at 16. A key goes into bucket `spread(hash) &
//! (capacity - 1)`, appended to that bucket's list. The capacity doubles
//! when the map holds more than three quarters of it in keys, and when a
//! list reaches nine keys while the capacity is below 64. A resize splits
//! every list into two and keeps the relative order of its keys, so a
//! bucket always holds its keys in the order they were put. Iteration
//! walks the buckets in index order, each list front to back.
//!
//! A list that reaches nine keys at a capacity of 64 or more becomes a
//! red-black tree, and a tree bucket is iterated in an order that depends
//! on the tree's shape and, for keys with equal hashes, on
//! `System.identityHashCode`. That order is not reproduced: the bucket
//! keeps put order here, and [`iteration_order`] reports that the result
//! is not exact.

/// A list that grows to this many keys becomes a tree, or makes the table
/// grow while it is smaller than [`MIN_TREEIFY_CAPACITY`]
/// (`TREEIFY_THRESHOLD` is 8, checked as the ninth key is appended).
const TREEIFY_LEN: u32 = 9;

/// `MIN_TREEIFY_CAPACITY`: below this capacity a long list makes the
/// table grow instead of becoming a tree.
const MIN_TREEIFY_CAPACITY: usize = 64;

/// `HashMap.DEFAULT_INITIAL_CAPACITY`.
const INITIAL_CAPACITY: usize = 16;

/// `HashMap.hash`: the key's `hashCode` with its high half folded into the
/// low half, which is what the bucket index is taken from.
#[inline]
pub(crate) fn spread(hash: i32) -> u32 {
    let h = hash as u32;
    h ^ (h >> 16)
}

/// Buffers for [`iteration_order`], kept between calls.
#[derive(Debug, Default)]
pub(crate) struct OrderScratch {
    counts: Vec<u32>,
}

/// Writes into `out` the keys of `put` (indexes into `spread`) in the order
/// a `HashMap` iterates them after they were put in that order, each key
/// once, into an empty map. Returns `false` when a bucket would have
/// become a tree, whose order is not reproduced (see the module docs); the
/// keys are then still all in `out`, with that bucket in put order.
pub(crate) fn iteration_order(spread: &[u32], put: &[u32], scratch: &mut OrderScratch, out: &mut Vec<u32>) -> bool {
    out.clear();
    if put.is_empty() {
        return true;
    }
    let counts = &mut scratch.counts;
    let mut capacity = INITIAL_CAPACITY;
    counts.clear();
    counts.resize(capacity, 0);
    let mut exact = true;
    for (size, &key) in put.iter().enumerate() {
        let bucket = spread[key as usize] as usize & (capacity - 1);
        counts[bucket] += 1;
        if counts[bucket] >= TREEIFY_LEN {
            if capacity < MIN_TREEIFY_CAPACITY {
                capacity *= 2;
                recount(spread, &put[..=size], capacity, counts);
            } else {
                exact = false;
            }
        }
        // `++size > threshold`, where the threshold is three quarters of
        // the capacity.
        if size + 1 > capacity / 4 * 3 {
            capacity *= 2;
            recount(spread, &put[..=size], capacity, counts);
        }
    }

    // Each bucket holds its keys in put order, so the iteration order is a
    // stable sort of the put order by bucket index.
    counts.clear();
    counts.resize(capacity + 1, 0);
    for &key in put {
        counts[(spread[key as usize] as usize & (capacity - 1)) + 1] += 1;
    }
    for i in 1..counts.len() {
        counts[i] += counts[i - 1];
    }
    out.resize(put.len(), 0);
    for &key in put {
        let slot = &mut counts[spread[key as usize] as usize & (capacity - 1)];
        out[*slot as usize] = key;
        *slot += 1;
    }
    exact
}

/// The number of keys in each bucket at `capacity`.
fn recount(spread: &[u32], keys: &[u32], capacity: usize, counts: &mut Vec<u32>) {
    counts.clear();
    counts.resize(capacity, 0);
    for &key in keys {
        counts[spread[key as usize] as usize & (capacity - 1)] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn order(hashes: &[i32]) -> (Vec<u32>, bool) {
        let spread: Vec<u32> = hashes.iter().map(|&h| spread(h)).collect();
        let put: Vec<u32> = (0..hashes.len() as u32).collect();
        let mut out = Vec::new();
        let exact = iteration_order(&spread, &put, &mut OrderScratch::default(), &mut out);
        (out, exact)
    }

    #[test]
    fn an_empty_map_has_no_keys() {
        assert_eq!(order(&[]), (vec![], true));
    }

    #[test]
    fn keys_are_visited_by_bucket_then_put_order() {
        // Capacity 16: buckets 3, 1, 3, 0 and 1 (17 & 15).
        assert_eq!(order(&[3, 1, 19, 0, 17]), (vec![3, 1, 4, 0, 2], true));
    }

    #[test]
    fn the_high_half_of_the_hash_is_folded_in() {
        assert_eq!(spread(0x0001_0000), 0x0001_0001);
        assert_eq!(spread(-1), 0xFFFF_0000);
        // 0x10000 lands in bucket 1, before bucket 2.
        assert_eq!(order(&[2, 0x0001_0000]), (vec![1, 0], true));
    }

    #[test]
    fn the_table_grows_past_three_quarters() {
        // Twelve keys fit in 16 buckets; the thirteenth doubles the table,
        // so 16 and 0 are no longer in one bucket.
        let mut hashes: Vec<i32> = vec![16, 0];
        hashes.extend(1..=10);
        assert_eq!(order(&hashes).0, (0..12).collect::<Vec<u32>>());
        // At capacity 32, the hash 16 has a bucket of its own after 0 to 11.
        hashes.push(11);
        assert_eq!(order(&hashes).0, (1..13).chain([0]).collect::<Vec<u32>>());
    }

    #[test]
    fn a_long_list_in_a_small_table_makes_it_grow() {
        // Nine keys in bucket 0 at capacity 16 double it to 32, where 16,
        // 48 and 80 move to bucket 16.
        let hashes: Vec<i32> = (0..9).map(|i| i * 16).collect();
        let (out, exact) = order(&hashes);
        assert!(exact);
        assert_eq!(out, [0, 2, 4, 6, 8, 1, 3, 5, 7]);
        // Nine keys of one bucket at every capacity up to 64 grow it to 64
        // and then become a tree, which is reported.
        let hashes: Vec<i32> = (0..9).map(|i| i * 64).collect();
        let (_, exact) = order(&hashes);
        assert!(exact, "the ninth key grows the table from 16 to 32");
        let hashes: Vec<i32> = (0..11).map(|i| i * 64).collect();
        let (out, exact) = order(&hashes);
        assert!(!exact, "the eleventh key reaches a list of nine at capacity 64");
        assert_eq!(out.len(), 11);
    }
}
