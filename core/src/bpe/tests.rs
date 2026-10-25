use super::*;

/// The pattern's classes are the regular expressions': \\s, \\p{L}
/// and \\p{N} for every character.
#[test]
fn classes_are_the_regular_expressions() {
    let space = regex::Regex::new(r"^\s$").unwrap();
    let letter = regex::Regex::new(r"^\p{L}$").unwrap();
    let number = regex::Regex::new(r"^\p{N}$").unwrap();
    let mut wrong = Vec::new();
    for c in (0..=0x10ffffu32).filter_map(char::from_u32) {
        let mut buf = [0u8; 4];
        let s = &*c.encode_utf8(&mut buf);
        let want = if space.is_match(s) {
            Class::Space
        } else if letter.is_match(s) {
            Class::Letter
        } else if number.is_match(s) {
            Class::Number
        } else {
            Class::Other
        };
        if class(c) != want {
            wrong.push(c);
        }
    }
    assert!(wrong.is_empty(), "{} characters: {:?}", wrong.len(), &wrong[..wrong.len().min(40)]);
}

/// Regenerates the split Unicode class tables.
#[test]
#[ignore = "writes the tables; run by hand when Unicode moves"]
fn write_classes() {
    crate::classes::tests::write_classes();
}

/// Each byte has a character and each character its byte back.
#[test]
fn byte_chars_go_both_ways() {
    let chars = byte_chars();
    let mut seen = std::collections::HashSet::new();
    for b in 0..=255u8 {
        assert!(seen.insert(chars[b as usize]));
        assert_eq!(char_byte(chars[b as usize]), Some(b));
    }
    assert_eq!(chars[b' ' as usize], '\u{120}');
    assert_eq!(chars[b'\n' as usize], '\u{10a}');
    assert_eq!(char_byte('\u{144}'), None);
}

/// Merging by scanning and by the heap give the same symbols, on
/// random merge lists and words.
#[test]
fn scanning_and_the_heap_merge_alike() {
    let mut x: u64 = 0x1234_5678_9abc_def1;
    let mut next = |n: u64| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x % n
    };
    for _ in 0..300 {
        let mut merges = Merges::default();
        let count = next(40) as u32;
        for (rank, id) in (0..count).zip(8u32..) {
            let (l, r) = (next(id as u64) as u32, next(id as u64) as u32);
            merges.entry(key(l, r)).or_insert((rank, id));
        }
        let len = 1 + next(120) as usize;
        let symbols: Vec<Symbol> =
            (0..len).map(|i| Symbol { id: next(8) as u32, start: i as u32, end: i as u32 + 1 }).collect();
        let mut a = Word { symbols: symbols.clone(), ..Default::default() };
        let mut b = Word { symbols, ..Default::default() };
        a.merge_scanning(&merges);
        b.merge_heap(&merges);
        let ids = |w: &Word| w.symbols.iter().map(|s| (s.id, s.start, s.end)).collect::<Vec<_>>();
        assert_eq!(ids(&a), ids(&b));
    }
}

/// Upstream's Split pre-tokenizer with `pattern`: each word's byte range
/// in a text.
fn upstream_words(pattern: &str) -> impl Fn(&str) -> Vec<(usize, usize)> {
    use tokenizers::pre_tokenizers::split::{Split, SplitPattern};
    use tokenizers::{OffsetReferential, OffsetType, PreTokenizedString, PreTokenizer, SplitDelimiterBehavior};
    let split = Split::new(SplitPattern::Regex(pattern.into()), SplitDelimiterBehavior::Isolated, false).unwrap();
    move |text| {
        let mut p = PreTokenizedString::from(text);
        split.pre_tokenize(&mut p).unwrap();
        p.get_splits(OffsetReferential::Original, OffsetType::Byte).into_iter().map(|(_, o, _)| o).collect()
    }
}

/// Llama-3's and Qwen2's patterns, matched by hand, cut text where the
/// regular expressions do: on the corners of each alternative and on
/// random strings of the characters that tell them apart.
#[test]
fn the_split_patterns_cut_where_upstream_does() {
    let alphabet: Vec<char> =
        "aZé\u{17f}'sStTrReEvVmMlLdD 1234①٣\u{301}!?.,-\t\r\n\u{a0}\u{3000}\u{b}日本😀_\"".chars().collect();
    let mut texts: Vec<String> = [
        "it's IT'S they're WE'VE I'm you'll he'd 'tis '\u{17f} 'ſt 'K ''s",
        "12345 1234567 ① ②③④⑤ ١٢٣٤ x1y22z333",
        "  hello   world\t\ttab \n\n line\r\n crlf \n  \n  ",
        "!!! ... ?!\n\n\n --- ,abc .def 'ghi \tjkl \u{3000}mno",
        "trailing   ",
        "   ",
        "a\u{301}b\u{302}\u{303} \u{301}c",
    ]
    .map(str::to_owned)
    .to_vec();
    let mut x: u64 = 0x2545_f491_4f6c_dd1d;
    for _ in 0..3000 {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        let len = 1 + (x % 12) as usize;
        let mut s = String::new();
        for k in 0..len {
            s.push(alphabet[((x >> (k * 5)) % alphabet.len() as u64) as usize]);
        }
        texts.push(s);
    }
    for (pattern, digits) in [(LLAMA3, 3), (QWEN2, 1)] {
        let theirs = upstream_words(pattern);
        for text in &texts {
            let mut ours = Vec::new();
            split_digits(text, digits, |a, b| ours.push((a, b)));
            assert_eq!(ours, theirs(text), "{text:?} with {digits} digits");
        }
    }
}
