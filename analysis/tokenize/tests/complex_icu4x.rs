//! ICU4X's own word segmentation tests with dictionaries and LSTMs
//! (`tests/icu4x/WordBreakDictionary.txt` and `tests/icu4x/WordBreakLstm.txt`,
//! from ICU4X's `components/segmenter/tests/testdata/`), run through the
//! complex segmenter: every line whose text has no Han, Hiragana, Katakana or
//! Hangul must give exactly ICU4X's segments. Chinese and Japanese need
//! ICU4X's CJ dictionary, which is not bundled (#191, #192 cover those
//! languages), so their lines are skipped and counted.

use turbo_tokenize::complex::{ComplexSegmenter, bundled, bundled_dictionaries, bundled_lstms};
use turbo_unicode::Script;
use turbo_unicode::script::script;

/// The text of a test line with `\u{XXXX}` escapes resolved.
fn unescape(text: &str) -> String {
    let mut out = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("\\u{") {
        out.push_str(&rest[..at]);
        let close = rest[at..].find('}').expect("an escape ends with }") + at;
        let cp = u32::from_str_radix(&rest[at + 3..close], 16).expect("a hexadecimal escape");
        out.push(char::from_u32(cp).expect("a character"));
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

fn is_cjk(c: char) -> bool {
    matches!(script(c), Script::Han | Script::Hiragana | Script::Katakana | Script::Hangul)
}

fn run_suite(segmenter: &ComplexSegmenter<'_>, file: &str) -> (usize, usize) {
    let (mut checked, mut skipped) = (0, 0);
    for line in file.lines() {
        let data = line.split('#').next().unwrap();
        if data.trim().is_empty() {
            continue;
        }
        // Trailing spaces before a comment are not part of the case.
        let data = data.trim_end();
        let expected: Vec<String> = data.split('|').map(unescape).collect();
        let text: String = expected.concat();
        if text.chars().any(is_cjk) {
            skipped += 1;
            continue;
        }
        let mut actual = Vec::new();
        segmenter.for_each_segment(&text, |s, e| actual.push(text[s as usize..e as usize].to_owned()));
        assert_eq!(actual, expected, "{line}");
        checked += 1;
    }
    (checked, skipped)
}

#[test]
fn word_break_dictionary() {
    let file = include_str!("icu4x/WordBreakDictionary.txt");
    assert_eq!(run_suite(bundled_dictionaries(), file), (81, 7));
}

#[test]
fn word_break_lstm() {
    let file = include_str!("icu4x/WordBreakLstm.txt");
    assert_eq!(run_suite(bundled_lstms(), file), (77, 7));
}

/// ICU4X's `dictionary_terminal_break` test: for every prefix of these
/// texts, some of which end inside a dictionary word, the last boundary is
/// the end of the prefix.
#[test]
fn terminal_break() {
    for segmenter in [bundled(), bundled_dictionaries(), bundled_lstms()] {
        for text in [
            "ผู้ใช้สามารถติดตั้งโปรแกรมคอมพิวเตอร์ได้ด้วยตนเอง",
            "ປະເທດລາວມີພົນລະເມືອງປະມານເຈັດລ້ານຄົນ",
            "ព្រះរាជាណាចក្រកម្ពុជាជាប្រទេសមួយនៅអាស៊ីអាគ្នេយ៍",
            "မြန်မာနိုင်ငံသည်အရှေ့တောင်အာရှတွင်တည်ရှိသည်",
        ] {
            for (i, c) in text.char_indices() {
                let prefix = &text[..i + c.len_utf8()];
                let mut last = None;
                segmenter.for_each_segment(prefix, |_, end| last = Some(end as usize));
                assert_eq!(last, Some(prefix.len()), "{prefix}");
            }
        }
    }
}
