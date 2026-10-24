use turbo_unicode::java;
use turbo_unicode::whitespace::{WhitespaceMode, is_unicode_blank, trim_unicode_whitespace};

#[test]
fn unicode_legacy_and_java_keep_their_distinct_blank_definitions() {
    for (text, unicode, legacy, java_blank, unicode_trim, legacy_trim) in [
        ("\u{a0}", true, true, false, "", ""),
        ("\u{3000} a\u{85}", false, false, false, "a", "a\u{85}"),
        ("\u{1c}", false, true, true, "\u{1c}", ""),
        ("", true, true, true, "", ""),
        ("\u{85}", true, false, false, "", "\u{85}"),
        ("\u{200b}", false, false, false, "\u{200b}", "\u{200b}"),
    ] {
        assert_eq!(WhitespaceMode::Unicode.is_blank(text), unicode, "Unicode {text:?}");
        assert_eq!(WhitespaceMode::Legacy.is_blank(text), legacy, "Legacy {text:?}");
        assert_eq!(java::is_blank(text), java_blank, "Java {text:?}");
        assert_eq!(WhitespaceMode::Unicode.trim(text), unicode_trim);
        assert_eq!(WhitespaceMode::Legacy.trim(text), legacy_trim);
        assert_eq!(is_unicode_blank(text), unicode);
        assert_eq!(trim_unicode_whitespace(text), unicode_trim);
    }
}

#[test]
fn trim_returns_the_original_subslice_without_touching_internal_spaces() {
    let text = "\u{3000}  a\u{a0}b \u{85}";
    let trimmed = WhitespaceMode::Unicode.trim(text);
    assert_eq!(trimmed, "a\u{a0}b");
    assert_eq!(trimmed.as_ptr(), text[5..].as_ptr());
    for mode in [WhitespaceMode::Unicode, WhitespaceMode::Legacy] {
        assert_eq!(mode.trim(mode.trim(text)), mode.trim(text));
        let blank = " \u{a0}";
        assert_eq!(mode.trim(blank).as_ptr(), blank[blank.len()..].as_ptr());
    }
}
