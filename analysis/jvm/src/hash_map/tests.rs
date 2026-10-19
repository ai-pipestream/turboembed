use super::*;

fn order(hashes: &[i32], insert: Insert) -> (Vec<u32>, bool, usize) {
    let spread: Vec<u32> = hashes.iter().map(|&h| spread(h)).collect();
    let mut map = HashMapOrder::new();
    let mut out = Vec::new();
    let exact = map.order(&spread, insert, &mut out);
    (out, exact, map.capacity())
}

#[test]
fn hash_codes_match_java() {
    assert_eq!("".java_hash(), 0);
    assert_eq!("ab".java_hash(), 3105);
    assert_eq!("Aa".java_hash(), 2112);
    assert_eq!("BB".java_hash(), 2112);
    assert_eq!("EMAIL".java_hash(), 66081660);
    assert_eq!("😀".java_hash(), 1_772_899);
    assert_eq!(concat_hash(string_hash("ab"), string_hash("c</w>"), 5), string_hash("abc</w>"));
    assert_eq!(("ab", "c").java_hash(), 96354);
    assert_eq!(vec!["a", "b"].java_hash(), 31 * (31 + 97) + 98);
    assert_eq!(CharBuffer(&[b'a' as u16, b'b' as u16]).java_hash(), 31 * (31 + 98) + 97);
    assert_eq!(None::<&str>.java_hash(), 0);
    assert_eq!(Some("ab").java_hash(), 3105);
    assert_eq!(42i32.java_hash(), 42);
}

#[test]
fn put_keeps_put_order_in_a_bin_and_merge_reverses_it() {
    assert_eq!(order(&[3, 1, 19, 0, 17], Insert::Put).0, [3, 1, 4, 0, 2]);
    assert_eq!(order(&[3, 1, 19, 0, 17], Insert::Merge { again: false }).0, [3, 4, 1, 2, 0]);
}

#[test]
fn put_grows_after_the_thirteenth_key_and_merge_before_the_fourteenth() {
    let hashes: Vec<i32> = (0..13).collect();
    assert_eq!(order(&hashes, Insert::Put).2, 32);
    assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 16);
    let hashes: Vec<i32> = (0..14).collect();
    assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
    let hashes: Vec<i32> = (0..13).collect();
    assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 32);
    let hashes: Vec<i32> = (0..12).collect();
    assert_eq!(order(&hashes, Insert::Merge { again: true }).2, 16);
}

#[test]
fn a_long_bin_grows_a_small_table_and_becomes_a_tree_in_a_large_one() {
    let hashes: Vec<i32> = (0..8).map(|i| i * 64).collect();
    assert_eq!(order(&hashes, Insert::Merge { again: false }).2, 32);
    assert_eq!(order(&hashes, Insert::Put).2, 16);
    let hashes: Vec<i32> = (0..9).map(|i| i * 64).collect();
    assert_eq!(order(&hashes, Insert::Put).2, 32);
    // The orders JDK 21 gives: the tree's root first, then the rest in
    // list order.
    let hashes: Vec<i32> = (0..10).map(|i| i * 64).collect();
    assert_eq!(order(&hashes, Insert::Merge { again: false }), (vec![6, 9, 8, 7, 5, 4, 3, 2, 1, 0], true, 64));
    let hashes: Vec<i32> = (0..12).map(|i| i * 64).collect();
    assert_eq!(order(&hashes, Insert::Put), (vec![3, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11], true, 64));
    // Equal hash codes in a tree are ordered by identity on the JVM.
    let (out, exact, capacity) = order(&[64; 10], Insert::Merge { again: false });
    assert_eq!((exact, capacity, out.len()), (false, 64, 10));
}

#[test]
fn order_keys_keeps_every_item_when_a_bin_treeifies() {
    let spreads: Vec<u32> = (0..12).map(|i| spread(i * 64)).collect();
    let keys = [0u32, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let mut orderer = HashMapOrder::new();
    let mut out = Vec::new();
    let exact = orderer.order_keys(&spreads, &keys, Insert::Put, &mut out);
    assert!(!exact);
    assert_eq!(out.len(), keys.len());
    let mut sorted = out.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, keys);
}

#[test]
fn strings_with_one_hash_code_in_a_tree_are_ordered_by_compare_to() {
    // "Aa" and "BB" share a hash code, so these ten strings all have
    // one. The ninth turns their bin of a table of 64 into a tree.
    let keys = [
        "BBAaAaBB", "BBAaAaAa", "AaBBBBBB", "AaBBBBAa", "AaBBAaBB", "AaBBAaAa", "AaAaBBBB", "AaAaBBAa", "AaAaAaBB",
        "AaAaAaAa",
    ];
    let mut map = JavaHashMap::with_capacity(64);
    for k in keys {
        map.put(k, ());
    }
    assert_eq!((map.tree_bins(), map.is_jvm_exact()), (1, true));
    // The order JDK 21 gives.
    let order: Vec<&str> = map.keys().copied().collect();
    assert_eq!(
        order,
        [
            "AaBBBBAa", "BBAaAaBB", "BBAaAaAa", "AaBBBBBB", "AaBBAaBB", "AaBBAaAa", "AaAaBBBB", "AaAaBBAa", "AaAaAaBB",
            "AaAaAaAa"
        ]
    );
}

#[test]
fn keys_in_a_tree_bin_are_found_by_descending_the_tree() {
    // 4096 strings with one hash code, all in one tree bin.
    let keys: Vec<String> =
        (0..1u32 << 12).map(|i| (0..12).map(|b| if i >> b & 1 == 1 { "Aa" } else { "BB" }).collect()).collect();
    let mut map = JavaHashMap::new();
    for (i, k) in keys.iter().enumerate() {
        map.merge(k.as_str(), i, |v, n| *v += n);
    }
    assert_eq!((map.len(), map.tree_bins(), map.is_jvm_exact()), (keys.len(), 1, true));
    for (i, k) in keys.iter().enumerate() {
        assert_eq!(map.get(k.as_str()), Some(&i), "{k}");
    }
    // Same hash code, not a key; and a key merged again stays one entry.
    assert_eq!(map.get("Aa".repeat(11).as_str()), None);
    assert_eq!(map.get(format!("{}Ab", "Aa".repeat(11)).as_str()), None);
    map.merge(keys[7].as_str(), 1, |v, n| *v += n);
    assert_eq!((map.len(), map.get(keys[7].as_str())), (keys.len(), Some(&8)));
}

#[test]
fn keys_the_tree_cannot_order_are_found_on_either_side() {
    /// One hash code and no order, as `CharBuffer` keys of one hash.
    #[derive(PartialEq, Eq)]
    struct Unordered(u32);
    impl JavaHash for Unordered {
        fn java_hash(&self) -> i32 {
            7
        }
    }
    let mut map = JavaHashMap::new();
    for i in 0..200 {
        map.put(Unordered(i), i);
    }
    assert_eq!((map.tree_bins(), map.is_jvm_exact()), (1, false));
    for i in 0..200 {
        assert_eq!(map.get(&Unordered(i)), Some(&i));
    }
    assert_eq!(map.get(&Unordered(200)), None);
}

#[test]
fn a_cleared_map_keeps_its_capacity_and_reset_starts_fresh() {
    let mut map = HashMapOrder::new();
    let mut out = Vec::new();
    let spread: Vec<u32> = (0..20).collect();
    map.order(&spread, Insert::Put, &mut out);
    assert_eq!(map.capacity(), 32);
    map.order(&[16, 0], Insert::Put, &mut out);
    assert_eq!(out, [1, 0]);
    map.reset();
    map.order(&[16, 0], Insert::Put, &mut out);
    assert_eq!(map.capacity(), 16);
    assert_eq!(out, [0, 1]);
}

#[test]
fn java_hash_map_put_all_and_replace_match_jdk_rules() {
    let mut m = JavaHashMap::<String, u8>::new();
    m.put_all([("a".into(), 1)].into_iter());
    assert_eq!((m.capacity(), m.threshold()), (2, 1));
    m.put_all([("b".into(), 2)].into_iter());
    assert_eq!((m.capacity(), m.threshold()), (4, 3));
    m.put("c".into(), 3);
    m.put("d".into(), 4);
    assert_eq!((m.capacity(), m.threshold()), (8, 6));
    m.put("a".into(), 9);
    assert_eq!(m.get("a"), Some(&9));
    let keys: Vec<String> = m.into_ordered().into_iter().map(|(k, _)| k).collect();
    assert_eq!(keys, ["a", "b", "c", "d"]);
}

#[test]
fn maximum_capacity_threshold_is_integer_max_value() {
    let mut m = JavaHashMap::<i32, ()>::with_capacity(MAXIMUM_CAPACITY);
    assert_eq!(m.threshold(), MAXIMUM_CAPACITY);
    let mut orderer = HashMapOrder::new();
    assert_eq!(orderer.threshold(), 0);
    let _ = &mut m;
    let _ = &mut orderer;
}
