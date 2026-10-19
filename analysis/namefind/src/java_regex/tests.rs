use crate::error::Feature;

use super::*;

fn find(p: &str, text: &str) -> Option<(usize, usize)> {
    let r = JavaRegex::compile(p).unwrap();
    let mut scratch = Scratch::default();
    r.find_at(text, 0, &mut scratch)
}

fn feature(p: &str) -> Feature {
    match JavaRegex::compile(p) {
        Err(Error::Unsupported { feature, .. }) => feature,
        other => panic!("{p}: {other:?}"),
    }
}

fn syntax(p: &str) {
    assert!(matches!(JavaRegex::compile(p), Err(Error::Syntax { .. })), "{p}");
}

#[test]
fn classes_are_ascii() {
    assert_eq!(find(r"\d", "٠7"), Some(("٠".len(), "٠7".len())));
    assert_eq!(find(r"[\w-]", "é-"), Some(("é".len(), "é-".len())));
    assert_eq!(find("a.b", "a\nb a\u{85}b axb"), Some((9, 12)));
    assert_eq!(find("(?s).", "\n"), Some((0, 1)));
}

#[test]
fn case_folds_ascii_letters_only() {
    assert_eq!(find("(?i)aé[x-z]", "AéZ"), Some((0, 4)));
    assert_eq!(find("(?i)aé[x-z]", "AÉZ"), None);
    let r = JavaRegex::compile_with("[k]", Flags::CASE_INSENSITIVE).unwrap();
    let mut scratch = Scratch::default();
    assert_eq!(r.find_at("K", 0, &mut scratch), Some((0, 1)));
    assert_eq!(r.find_at("\u{212A}", 0, &mut scratch), None);
    // Flags end with their group.
    assert_eq!(find("((?i)a)a", "Aa"), Some((0, 2)));
    assert_eq!(find("((?i)a)a", "AA"), None);
}

#[test]
fn quoting_and_escapes() {
    assert_eq!(find(r"\Qa.b\E*", "a.bb"), Some((0, 4)));
    assert_eq!(find(r"\Qa.b\E*", "axb"), None);
    assert_eq!(find(r"[\Q]\E]", "]"), Some((0, 1)));
    assert_eq!(find(r"\0101\x41\u0041\cA", "AAA\x01"), Some((0, 4)));
    assert_eq!(find(r"\0400", " 0"), Some((0, 2)));
    assert_eq!(find(r"\uD83D\uDE00", "😀"), Some((0, 4)));
    assert_eq!(find(r"[\v-]", "\x0B-"), Some((0, 1)));
    assert_eq!(find("[]a]", "]"), Some((0, 1)));
    assert_eq!(find("{1}", "abc"), Some((0, 0)));
}

#[test]
fn refusals() {
    assert_eq!(feature("(?=a)"), Feature::Lookahead);
    assert_eq!(feature("(?<!a)b"), Feature::Lookbehind);
    assert_eq!(feature("(?>a)"), Feature::AtomicGroup);
    assert_eq!(feature(r"(a)\1"), Feature::Backreference);
    assert_eq!(feature("a*+"), Feature::PossessiveQuantifier);
    assert_eq!(feature("(a*)*"), Feature::EmptyLoop);
    assert_eq!(feature("^*"), Feature::QuantifiedAssertion);
    assert_eq!(feature("[a[b]]"), Feature::NestedClass);
    assert_eq!(feature("[a&&b]"), Feature::ClassIntersection);
    assert_eq!(feature(r"\p{L}"), Feature::Property);
    assert_eq!(feature(r"\uD800"), Feature::Surrogate);
    assert_eq!(feature("(?m)a"), Feature::Flag('m'));
    assert_eq!(feature(r"\R"), Feature::Escape('R'));
    assert_eq!(feature(r"\N{LATIN SMALL LETTER A}"), Feature::Escape('N'));
    assert_eq!(feature(r"[\N{LATIN SMALL LETTER A}]"), Feature::Escape('N'));
    for p in ["{", "a{", "a{1", "x{,2}", "a{2,1}", "[]", "(?i)*", "a**", r"\y", r"\E", "(?<a>x)(?<a>y)"] {
        syntax(p);
    }
    for p in [r"\0", r"\08", r"\c", r"\x4", r"\x{110000}", "(?<1a>x)", "x{2147483648}", "(a", "a)", "[a-\\d]"] {
        syntax(p);
    }
    for p in [r"\b{g", r"\b{gx}", r"\N", r"\N{", r"\N{}", r"\N{LATIN", r"[\N]", r"[\N{}]", r"[\N{LATIN]"] {
        syntax(p);
    }
}

#[test]
fn limits() {
    assert_eq!(JavaRegex::compile("a$").unwrap().limits(), Limits { dollar: true, ..Limits::default() });
    let b = JavaRegex::compile(r"\bx").unwrap().limits();
    assert!(b.word_boundary && !b.mid_pair);
    assert!(JavaRegex::compile(r"\b|x").unwrap().limits().mid_pair);
    assert!(JavaRegex::compile(r"\Bx").unwrap().limits().mid_pair);
    assert!(JavaRegex::compile(r"[\x00-\uFFFF]").unwrap().limits().mid_pair);
    assert!(!JavaRegex::compile(r"[^a]").unwrap().limits().mid_pair);
    assert!(JavaRegex::compile(r"[^\x{10000}-\x{10FFFF}]").unwrap().limits().mid_pair);
}

#[test]
fn compile_limit_and_backtracking_memoization() {
    assert!(matches!(JavaRegex::compile("a{2000000}"), Err(Error::Compile { .. })));
    assert!(matches!(JavaRegex::compile("a{2000000,}"), Err(Error::Compile { .. })));
    // Triggers the visited bitset (> 64 backtracks at a single start position) without exponential blowup.
    assert_eq!(find("(a|a)+b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaac"), None);
    assert_eq!(find("(a|a)+b", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaab"), Some((0, 31)));
    // The failed states are kept across start positions, so a search
    // that fails at every start is linear, not quadratic, in the text.
    let long = "a".repeat(20_000);
    assert_eq!(find("(a|aa)*c", &long), None);
    assert_eq!(find(".*x", &long), None);
    assert_eq!(find("(a|aa)*c|a$", &long), Some((19_999, 20_000)));
    let mut scratch = Scratch::default();
    for (p, want) in [("(a|aa)*c", None), (".*x", None), ("(a|aa)*c|a$", Some((19_999, 20_000)))] {
        let r = JavaRegex::compile(p).unwrap();
        assert_eq!(r.find_at_with(&long, 0, &mut scratch, Engine::Pike), want, "{p}");
    }
}

#[test]
fn find_at_rejects_non_char_boundary_and_overflow_table_fits() {
    let r = JavaRegex::compile("a").unwrap();
    let mut scratch = Scratch::default();
    // Byte 1 is inside the 2-byte UTF-8 encoding of 'é'.
    assert_eq!(r.find_at_with("éa", 1, &mut scratch, Engine::Backtrack), None);
    assert_eq!(r.find_at_with("éa", 1, &mut scratch, Engine::Pike), None);
    assert_eq!(r.find_at("éa", 2, &mut scratch), Some((2, 3)));
    assert!(!r.table_fits(usize::MAX));
}

#[test]
fn large_tables_run_on_the_pike_vm() {
    // 200,001 instructions over 2,001 positions is past the table's
    // limit, so the search keeps no table at all.
    let r = JavaRegex::compile("a{0,100000}!").unwrap();
    assert!(r.table_fits(1_000) && !r.table_fits(2_000));
    let text = "x".repeat(2_000);
    let mut scratch = Scratch::default();
    assert_eq!(r.find_at(&text, 0, &mut scratch), None);
    assert!(scratch.visited.is_empty());
    let text = format!("{}aa!", "x".repeat(2_000));
    assert_eq!(r.find_at(&text, 0, &mut scratch), Some((2_000, 2_003)));
    assert_eq!(r.find_at(&text, 2_001, &mut scratch), Some((2_001, 2_003)));
}

#[test]
fn the_table_is_cleared_between_searches() {
    // Each search turns the table on; a state left from the one before
    // would hide the later matches.
    let r = JavaRegex::compile("(a|a)+b").unwrap();
    let text = format!("{}c aab ", "a".repeat(30)).repeat(3);
    let mut scratch = Scratch::default();
    let mut from = 0;
    let mut found = Vec::new();
    while let Some((s, e)) = r.find_at(&text, from, &mut scratch) {
        found.push(s);
        from = e;
    }
    assert_eq!(found, [32, 68, 104]);
    assert!(!scratch.touched.is_empty());
}

#[test]
fn nesting_is_limited() {
    let nested = |n: usize| format!("{}a{}", "(".repeat(n), ")".repeat(n));
    assert_eq!(find(&nested(MAX_DEPTH), "a"), Some((0, 1)));
    assert!(matches!(JavaRegex::compile(&nested(MAX_DEPTH + 1)), Err(Error::Compile { .. })));
    assert!(matches!(JavaRegex::compile(&"(".repeat(100_000)), Err(Error::Compile { .. })));
}
