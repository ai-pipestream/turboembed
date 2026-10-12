//! Every case of Unicode's WordBreakTest.txt (Unicode 18.0) through
//! `word_boundaries`. The file is the unchanged UCD copy turbo-normalize
//! tests with (analysis/normalize/tests/data/README.md).

use std::path::Path;

use turbo_tokenize::word_boundaries;

#[test]
fn every_case_of_word_break_test() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../normalize/tests/data/WordBreakTest-18.0.0.txt");
    let file = std::fs::read_to_string(&path).unwrap();
    let mut cases = 0;
    let mut out = Vec::new();
    for (n, line) in file.lines().enumerate() {
        let case = line.split('#').next().unwrap().trim();
        if case.is_empty() {
            continue;
        }
        let mut text = String::new();
        let mut expected = Vec::new();
        for field in case.split_whitespace() {
            match field {
                "\u{f7}" => expected.push(text.len() as u32),
                "\u{d7}" => {}
                hex => text.push(char::from_u32(u32::from_str_radix(hex, 16).unwrap()).unwrap()),
            }
        }
        out.clear();
        word_boundaries(&text, &mut out);
        assert_eq!(out, expected, "line {}: {line}", n + 1);
        cases += 1;
    }
    // The 18.0 file's case count, so a reader that skipped lines cannot pass.
    assert_eq!(cases, 1944);
}
