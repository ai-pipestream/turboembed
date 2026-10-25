use super::extended::next_candidate_pos;
use turbo_unicode::java;

#[test]
fn candidate_search_matches_the_unicode_reference_on_every_platform() {
    let mut text = String::new();
    let mut expected = Vec::new();
    for value in 0..=0x10_FFFF {
        if let Some(c) = char::from_u32(value) {
            if c == '-' || java::is_digit(c) {
                expected.push(text.len());
            }
            text.push(c);
            text.push(' ');
        }
    }
    let mut actual = Vec::new();
    let mut start = 0;
    while let Some(pos) = next_candidate_pos(&text, start, false) {
        actual.push(pos);
        start = pos + text[pos..].chars().next().unwrap().len_utf8();
    }
    assert_eq!(actual, expected);
}

#[test]
fn candidate_search_keeps_ascii_and_unicode_digits_at_each_word_boundary() {
    for c in "-0123456789١９१𑁦".chars() {
        for padding in 0..48 {
            let text = format!("{}{c} été 中文", "x".repeat(padding));
            assert_eq!(next_candidate_pos(&text, 0, false), Some(padding), "{c:?} at {padding}");
            assert_eq!(next_candidate_pos(&text, padding + c.len_utf8(), false), None);
        }
    }
}
