use super::candidate::next_avx2;
use super::extended::next_candidate_pos;
use turbo_unicode::java;

#[test]
fn wide_candidates_match_the_unicode_reference_over_every_code_point() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
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
    while let Some(pos) = unsafe { next_avx2(&text, start) } {
        actual.push(pos);
        start = pos + text[pos..].chars().next().unwrap().len_utf8();
    }
    assert_eq!(actual, expected);
}

#[test]
fn wide_candidates_keep_every_nd_digit_across_block_edges_and_tails() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let digits: Vec<_> = (0x80..=0x10_FFFF).filter_map(char::from_u32).filter(|&c| java::is_digit(c)).collect();
    for c in digits {
        for padding in 0..96 {
            let mut text = "x".repeat(padding);
            text.push(c);
            text.push_str("😀漢字 xyz ");
            assert_eq!(unsafe { next_avx2(&text, 0) }, Some(padding), "{c:?} at {padding}");
            assert_eq!(next_candidate_pos(&text, 0, false), Some(padding));
            assert_eq!(next_candidate_pos(&text, 0, true), None);
            assert_eq!(unsafe { next_avx2(&text, padding + c.len_utf8()) }, None);
        }
    }
}

#[test]
fn wide_candidates_keep_ascii_digits_and_signs_at_every_tail_size() {
    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    for digit in "-0123456789".chars() {
        for padding in 0..96 {
            for tail in [0, 1, 2, 31, 32, 33, 34] {
                let text = format!("{}{digit}{}", "x".repeat(padding), "x".repeat(tail));
                assert_eq!(unsafe { next_avx2(&text, 0) }, Some(padding));
                assert_eq!(next_candidate_pos(&text, 0, false), Some(padding));
                assert_eq!(next_candidate_pos(&text, 0, true), Some(padding));
            }
        }
    }
}
