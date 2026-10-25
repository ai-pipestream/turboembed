use turbo_normalize::{Whitespace, apply, tables};
use turbo_unicode::whitespace::{UNICODE_WHITESPACE, WhitespaceMode};

#[test]
fn normalization_membership_matches_shared_unicode_for_all_code_points() {
    assert_eq!(tables::WHITESPACE.iter().map(|&c| c as u32).collect::<Vec<_>>(), UNICODE_WHITESPACE);
    for cp in 0..=0x10ffff {
        if let Some(c) = char::from_u32(cp) {
            assert_eq!(tables::is_whitespace(c), WhitespaceMode::Unicode.is_whitespace(cp), "U+{cp:04X}");
        }
    }
}

#[test]
fn real_text_keeps_information_separators_and_collapses_unicode_spaces() {
    let input = "Dr. Smith\u{b}paid\u{85}10\u{a0}euros\u{3000}yesterday.\u{1c}Fine!";
    assert_eq!(apply(&Whitespace::collapse(), input), "Dr. Smith paid 10 euros yesterday.\u{1c}Fine!");
    assert_eq!(apply(&Whitespace::keep_line_breaks(), input), "Dr. Smith\npaid\n10 euros yesterday.\u{1c}Fine!");
}
