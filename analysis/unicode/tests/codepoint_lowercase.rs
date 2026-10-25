use turbo_unicode::java::{
    eq_lowercase_code_points, lowercase_code_points, lowercase_code_points_changes, to_lowercase,
    to_lowercase_code_points,
};

#[test]
fn stringutil_lowercases_code_points_without_context_or_expansion() {
    for (text, expected, full) in [
        ("İstanbul", "istanbul", "i\u{307}stanbul"),
        ("ΟΔΟΣ", "οδοσ", "οδος"),
        ("\u{10400}", "\u{10428}", "\u{10428}"),
        ("ǅ", "ǆ", "ǆ"),
        ("", "", ""),
        ("already lower 12", "already lower 12", "already lower 12"),
        ("*UNK*", "*unk*", "*unk*"),
    ] {
        assert_eq!(lowercase_code_points(text), expected);
        assert_eq!(lowercase_code_points_changes(text), text != expected);
        assert!(eq_lowercase_code_points(text, expected));
        let mut out = String::from("prefix:");
        to_lowercase_code_points(text, &mut out);
        assert_eq!(out, format!("prefix:{expected}"));
        let mut full_result = String::new();
        to_lowercase(text, &mut full_result);
        assert_eq!(full_result, full);
    }
    // Comparison folds only its first argument, even on the ASCII path.
    assert!(!eq_lowercase_code_points("UPPER", "UPPER"));
    assert!(!eq_lowercase_code_points("İ", "i\u{307}"));
    assert!(!eq_lowercase_code_points("", "a"));
    assert!(!eq_lowercase_code_points("ab", "a"));
    assert!(!eq_lowercase_code_points("ΟΔΟΣ", "οδος"));
}

#[test]
fn ascii_runs_resume_after_each_unicode_character_without_losing_bytes() {
    for n in 0..25 {
        let upper = "A".repeat(n);
        let lower = "a".repeat(n);
        let text = format!("{upper}İ{upper}ΟΔΟΣ{upper}\u{10400}{upper}ǅ{upper}\0");
        let expected = format!("{lower}i{lower}οδοσ{lower}\u{10428}{lower}ǆ{lower}\0");
        assert_eq!(lowercase_code_points(&text), expected);
        assert!(eq_lowercase_code_points(&text, &expected));
        assert!(lowercase_code_points_changes(&text));
        assert!(!lowercase_code_points_changes(&expected));
    }
}
