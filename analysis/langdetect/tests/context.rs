//! The context generator: n-grams, their order and their bytes, on texts
//! small enough to work out by hand.

mod common;

use turbo_langdetect::ContextGenerator;

use common::{modified_utf8, mutf8};

/// The features of a prepared text, as strings where they are UTF-8.
fn grams(generator: &mut ContextGenerator, text: &str) -> Vec<Vec<u8>> {
    generator.generate_prepared(text);
    generator.features().map(<[u8]>::to_vec).collect()
}

fn strs(items: &[&str]) -> Vec<Vec<u8>> {
    items.iter().map(|s| mutf8(s)).collect()
}

#[test]
fn n_grams_come_in_the_jvm_hash_map_order() {
    // CharBuffer hashes (each unit from the last, 31 * h + unit, from 1):
    // a 128, b 129, ab 4096, ba 4066, aba 126143, bab 127074. With the
    // high half folded in, the buckets of a 16 bucket table are 0, 1, 0,
    // 2, 14 and 3, so the order is a, ab, b, ba, bab, aba.
    let mut g = ContextGenerator::default();
    assert_eq!(grams(&mut g, "abab"), strs(&["a", "ab", "b", "ba", "bab", "aba"]));
    assert!(g.exact_order());
    assert_eq!(g.len(), 6);
}

#[test]
fn each_distinct_n_gram_is_listed_once() {
    let mut g = ContextGenerator::default();
    let mut found = grams(&mut g, "aaaa");
    found.sort();
    assert_eq!(found, strs(&["a", "aa", "aaa"]));
}

#[test]
fn the_text_is_lowercased_per_code_point_with_the_simple_mapping() {
    let mut g = ContextGenerator::new(1, 1);
    let mut found = grams(&mut g, "AbC");
    found.sort();
    assert_eq!(found, strs(&["a", "b", "c"]));
    // U+0130 lowercases to a plain "i" in Character.toLowerCase(int); the
    // full mapping would add U+0307.
    assert_eq!(grams(&mut g, "\u{130}"), strs(&["i"]));
    // Final sigma is not applied: every capital sigma becomes U+03C3.
    assert_eq!(grams(&mut g, "\u{3A3}"), strs(&["\u{3C3}"]));
    // A capital outside the BMP lowercases to a character outside the BMP:
    // U+10400 DESERET CAPITAL LONG I to U+10428, two code units.
    assert_eq!(grams(&mut g, "\u{10400}"), [modified_utf8(&[0xD801]), modified_utf8(&[0xDC28])]);
}

#[test]
fn no_simple_lowercase_mapping_changes_the_utf16_length() {
    // StringUtil.toLowerCaseCharBuffer allocates as many code units as the
    // text has; a mapping that changed the length would leave zeros at the
    // end or overflow. No code point has one.
    for c in (0..=0x10_FFFFu32).filter_map(char::from_u32) {
        let lower = turbo_unicode::java::code_point::to_lower_case(u32::from(c));
        assert_eq!(lower > 0xFFFF, u32::from(c) > 0xFFFF, "U+{:04X} -> U+{lower:04X}", u32::from(c));
    }
}

#[test]
fn n_grams_split_surrogate_pairs() {
    // Without the normalizers, a character outside the BMP is two code
    // units and the n-grams cut between them: each half alone is a
    // feature, written as Java writes a lone surrogate (three bytes).
    let mut g = ContextGenerator::default();
    let mut found = grams(&mut g, "a\u{1F600}");
    found.sort();
    let mut expected = vec![
        modified_utf8(&[0x61]),
        modified_utf8(&[0xD83D]),
        modified_utf8(&[0xDE00]),
        modified_utf8(&[0x61, 0xD83D]),
        modified_utf8(&[0xD83D, 0xDE00]),
        modified_utf8(&[0x61, 0xD83D, 0xDE00]),
    ];
    expected.sort();
    assert_eq!(found, expected);
    assert!(found.contains(&vec![0xED, 0xA0, 0xBD]), "the lone high surrogate");
    assert!(std::str::from_utf8(&modified_utf8(&[0xD83D])).is_err());
}

#[test]
fn u0000_is_written_as_two_bytes() {
    let mut g = ContextGenerator::new(1, 1);
    let mut found = grams(&mut g, "a\0");
    found.sort();
    assert_eq!(found, [vec![0x61], vec![0xC0, 0x80]]);
}

#[test]
fn the_lengths_are_a_range_of_code_units() {
    let mut g = ContextGenerator::new(2, 2);
    let mut found = grams(&mut g, "abc");
    found.sort();
    assert_eq!(found, strs(&["ab", "bc"]));
    // A text shorter than the longest n-gram has only the shorter ones.
    let mut g = ContextGenerator::default();
    let mut found = grams(&mut g, "ab");
    found.sort();
    assert_eq!(found, strs(&["a", "ab", "b"]));
    let mut g = ContextGenerator::new(4, 5);
    assert!(grams(&mut g, "abc").is_empty());
}

#[test]
#[should_panic(expected = "need 1 <= min <= max")]
fn a_zero_length_is_refused() {
    ContextGenerator::new(0, 3);
}

#[test]
#[should_panic(expected = "need 1 <= min <= max")]
fn a_range_that_ends_before_it_starts_is_refused() {
    ContextGenerator::new(3, 2);
}

#[test]
fn the_normalizers_run_first() {
    let mut g = ContextGenerator::default();
    // Empty, whitespace, digits, a URL, an e-mail address, emoji and
    // social media markup: nothing is left once the normalizers ran.
    for text in
        ["", " \t\n ", "12345 678", "http://example.com/a?b=c", "me@example.org", "\u{1F600}\u{1F389}", "#tag @user"]
    {
        g.generate(text);
        assert!(g.is_empty(), "{text:?}: {:?}", g.features().collect::<Vec<_>>());
    }
    // An emoji between two letters becomes a space, so no n-gram holds a
    // surrogate.
    g.generate("a\u{1F600}b");
    let mut found: Vec<Vec<u8>> = g.features().map(<[u8]>::to_vec).collect();
    found.sort();
    assert_eq!(found, strs(&[" ", " b", "a", "a ", "a b", "b"]));
    // Digits become a space and the spaces shrink to one.
    g.generate("x 9 y");
    let mut found: Vec<Vec<u8>> = g.features().map(<[u8]>::to_vec).collect();
    found.sort();
    assert_eq!(found, strs(&[" ", " y", "x", "x ", "x y", "y"]));
    // A run of one character shrinks to two.
    g.generate("zzzzzz");
    let mut found: Vec<Vec<u8>> = g.features().map(<[u8]>::to_vec).collect();
    found.sort();
    assert_eq!(found, strs(&["z", "zz"]));
}

#[test]
fn a_generator_is_reused_without_leftovers() {
    let mut g = ContextGenerator::default();
    let long = "the quick brown fox jumps over the lazy dog ".repeat(50);
    grams(&mut g, &long);
    assert_eq!(grams(&mut g, "abab"), strs(&["a", "ab", "b", "ba", "bab", "aba"]));
    g.generate("");
    assert!(g.is_empty());
    assert_eq!(g.features().count(), 0);
}

#[test]
fn a_large_text_keeps_every_distinct_n_gram() {
    // Every BMP letter from U+4E00 on, so the slot table grows many times
    // and every n-gram is distinct.
    let text: String = (0x4E00u32..0x4E00 + 3000).filter_map(char::from_u32).collect();
    let mut g = ContextGenerator::default();
    g.generate_prepared(&text);
    assert_eq!(g.len(), 3000 + 2999 + 2998);
    let mut found: Vec<Vec<u8>> = g.features().map(<[u8]>::to_vec).collect();
    found.sort();
    found.dedup();
    assert_eq!(found.len(), g.len());
}

#[test]
fn a_bucket_that_java_turns_into_a_tree_comes_in_the_jvms_order() {
    // These eleven characters all have a CharBuffer hash (31 plus the code
    // unit) that is a multiple of 64, so as unigrams they share one bucket
    // of a table of 64. The ninth and tenth grow the table from 16 to 64;
    // the eleventh makes the JVM turn the bucket into a red-black tree,
    // which iterates its root first. Their hashes differ, so the order is
    // the JVM's.
    let text = "!a\u{A1}\u{E1}\u{121}\u{161}\u{1A1}\u{1E1}\u{221}\u{261}\u{2A1}";
    for c in text.chars() {
        assert!((31 + u32::from(c)).is_multiple_of(64), "{c:?}");
    }
    let mut unigrams = ContextGenerator::new(1, 1);
    // The order JDK 21 gives for the two maps of `getContext` and
    // `predict`.
    let jvm =
        ["\u{E1}", "!", "a", "\u{A1}", "\u{121}", "\u{161}", "\u{1A1}", "\u{1E1}", "\u{221}", "\u{261}", "\u{2A1}"];
    assert_eq!(grams(&mut unigrams, text), strs(&jvm));
    assert!(unigrams.exact_order());
    // Ten of them stay a list, in put order.
    let shorter: String = text.chars().take(10).collect();
    assert_eq!(
        grams(&mut unigrams, &shorter),
        strs(&["!", "a", "\u{A1}", "\u{E1}", "\u{121}", "\u{161}", "\u{1A1}", "\u{1E1}", "\u{221}", "\u{261}"])
    );
    assert!(unigrams.exact_order());
    let mut g = ContextGenerator::default();
    g.generate_prepared(text);
    assert!(g.exact_order());
    assert_eq!(g.len(), 11 + 10 + 9);
}

// Differences from OpenNLP

#[test]
fn characters_new_since_unicode_15_lowercase_with_the_unicode_18_tables() {
    // U+A7CB LATIN CAPITAL LETTER RAMS HORN was added in Unicode 16.0 with
    // the lowercase U+0264. JDK 21 implements Unicode 15.0 and leaves it as
    // it is; this port lowercases it.
    let mut g = ContextGenerator::new(1, 1);
    assert_eq!(grams(&mut g, "\u{A7CB}"), strs(&["\u{264}"]));
}
