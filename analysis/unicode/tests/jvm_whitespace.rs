//! Retains the exhaustive JDK 21 vectors formerly checked by turbo-jvm.
use turbo_unicode::java;

#[test]
fn both_java_whitespace_entry_points_match_jdk_for_every_code_point() {
    let expected: Vec<u32> = include_str!("../../jvm/tests/vectors/whitespace.tsv")
        .lines()
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| u32::from_str_radix(line.trim(), 16).unwrap())
        .collect();
    assert_eq!(expected.len(), 25);
    assert!(expected.windows(2).all(|w| w[0] < w[1]));
    for cp in 0..=0x10ffff {
        let is_space = expected.binary_search(&cp).is_ok();
        assert_eq!(java::code_point::is_whitespace(cp), is_space, "U+{cp:04X}");
        if let Some(c) = char::from_u32(cp) {
            assert_eq!(java::is_whitespace(c), is_space, "char U+{cp:04X}");
        }
    }
}
