use turbo_unicode::{java::equals_ignore_case, utf16::utf16_len};

// The same shared function must work in generated Snowball constants.
const LENGTHS: [usize; 5] =
    [utf16_len("😀a"), utf16_len("ab"), utf16_len(""), utf16_len("İſK"), utf16_len("\u{10400}")];

#[test]
fn utf16_length_counts_supplementary_text_at_compile_time_and_runtime() {
    assert_eq!(LENGTHS, [3, 2, 0, 3, 2]);
    for text in ["😀a", "ab", "", "İſK", "\u{10400}", "a\0é東京😀"] {
        assert_eq!(utf16_len(text), text.encode_utf16().count());
    }
}

#[test]
fn java_case_comparison_uses_upper_then_lower_without_expansion() {
    for (a, b, expected) in [
        ("REG\u{131}ON", "region", true),
        ("\u{17f}", "s", true),
        ("\u{212a}elvin", "kelvin", true),
        ("İ", "i", true),
        ("ΟΔΟΣ", "οδος", true),
        ("ΟΔΟΣ", "οδοσ", true),
        ("\u{10400}", "\u{10428}", true),
        ("😀a", "😀A", true),
        ("😀a", "ab", false),
        ("😀a", "abc", false),
        ("ß", "SS", false),
        ("İ", "i\u{307}", false),
        ("ab", "a", false),
        ("", "", true),
        ("a", "", false),
    ] {
        assert_eq!(equals_ignore_case(a, b), expected, "{a:?} / {b:?}");
        assert_eq!(equals_ignore_case(b, a), expected, "{b:?} / {a:?}");
    }
}
