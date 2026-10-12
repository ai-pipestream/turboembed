//! The newline detector against `NewlineSentenceDetector` on the JVM: every
//! file of the `newline` and `newline-legacy` golden directories, spans
//! compared exactly in UTF-8 bytes and in UTF-16 code units.

mod common;

use turbo_sentdetect::{WhitespaceMode, newline_spans};
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

fn check_dir(dir: &str, mode: WhitespaceMode) -> usize {
    let files = common::files(dir);
    let mut out = Vec::new();
    for path in &files {
        let g = common::read(path);
        let name = path.display();
        assert_eq!(g.component, "newline", "{name}");
        assert_eq!(g.whitespace, mode, "{name}");
        // The golden offsets agree with each other on the decoded input,
        // so the reader got the text right.
        for s in &g.spans {
            assert_eq!(utf16_to_byte(&g.input, s.utf16.0), Some(s.utf8.0 as usize), "{name}");
            assert_eq!(utf16_to_byte(&g.input, s.utf16.1), Some(s.utf8.1 as usize), "{name}");
        }
        out.clear();
        newline_spans(&g.input, mode, &mut out);
        let got: Vec<common::GoldenSpan> = out
            .iter()
            .map(|s| common::GoldenSpan {
                utf8: (s.start, s.end),
                utf16: (
                    byte_to_utf16(&g.input, s.start as usize).unwrap(),
                    byte_to_utf16(&g.input, s.end as usize).unwrap(),
                ),
            })
            .collect();
        assert_eq!(got, g.spans, "{name}");
    }
    files.len()
}

#[test]
fn newline_unicode() {
    assert_eq!(check_dir("newline", WhitespaceMode::Unicode), 34);
}

#[test]
fn newline_legacy() {
    assert_eq!(check_dir("newline-legacy", WhitespaceMode::Legacy), 34);
}
