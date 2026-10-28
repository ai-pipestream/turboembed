use turbo_unicode::java;
use turbo_unicode::whitespace::{is_blank, trim};

#[test]
fn unicode_and_java_keep_their_distinct_blank_definitions() {
    for (text, blank, java_blank, trimmed) in [
        ("\u{a0}", true, false, ""),
        ("\u{3000} a\u{85}", false, false, "a"),
        ("\u{1c}", false, true, "\u{1c}"),
        ("", true, true, ""),
        ("\u{85}", true, false, ""),
        ("\u{200b}", false, false, "\u{200b}"),
    ] {
        assert_eq!(is_blank(text), blank, "{text:?}");
        assert_eq!(java::is_blank(text), java_blank, "Java {text:?}");
        assert_eq!(trim(text), trimmed, "{text:?}");
    }
}

#[test]
fn information_separators_are_text_and_next_line_is_whitespace() {
    // U+001C to U+001F are not White_Space: they are neither trimmed nor
    // blank, so a word that holds one keeps it.
    for sep in '\u{1c}'..='\u{1f}' {
        let text = format!("{sep}a{sep}b{sep}");
        assert_eq!(trim(&text), text);
        assert!(!is_blank(&sep.to_string()));
    }
    // U+0085 is White_Space: trimmed at either end, kept inside.
    assert_eq!(trim("\u{85}a\u{85}b\u{85}"), "a\u{85}b");
    assert!(is_blank("\u{85} \u{85}"));
}

#[test]
fn trim_returns_the_original_subslice_without_touching_internal_spaces() {
    let text = "\u{3000}  a\u{a0}b \u{85}";
    let trimmed = trim(text);
    assert_eq!(trimmed, "a\u{a0}b");
    assert_eq!(trimmed.as_ptr(), text[5..].as_ptr());
    assert_eq!(trim(trim(text)), trim(text));
    let blank = " \u{a0}";
    assert_eq!(trim(blank).as_ptr(), blank[blank.len()..].as_ptr());
}
