//! Native newline contract over the existing 34 Unicode edge input texts.
//! Expected boundaries come from CR/LF separation and Unicode trimming,
//! independently of the detector's scanning implementation.

use turbo_analysis::Span;
use turbo_sentdetect::newline_spans;
use turbo_unicode::props::white_space;
use turbo_unicode::utf16::{byte_to_utf16, utf16_to_byte};

#[test]
fn newline_unicode() {
    let directory = turbo_goldens::data_dir().join("inputs/edge");
    let mut files: Vec<_> = std::fs::read_dir(directory).unwrap().map(|e| e.unwrap().path()).collect();
    files.sort();
    assert_eq!(files.len(), 34);
    for path in files {
        let text = std::fs::read_to_string(&path).unwrap();
        let mut expected = Vec::new();
        let mut base = 0;
        for line in text.split(['\r', '\n']) {
            let trimmed = line.trim_matches(white_space);
            if !trimmed.is_empty() {
                let leading = line.len() - line.trim_start_matches(white_space).len();
                let start = base + leading;
                expected.push(Span::new(start as u32, (start + trimmed.len()) as u32));
            }
            base += line.len() + 1;
        }
        let mut got = Vec::new();
        newline_spans(&text, &mut got);
        assert_eq!(got, expected, "{}", path.display());
        for span in got {
            assert!(!text[span.range()].is_empty());
            for edge in [span.start, span.end] {
                let byte = edge as usize;
                let units = text[..byte].encode_utf16().count();
                assert_eq!(byte_to_utf16(&text, byte), Some(units));
                assert_eq!(utf16_to_byte(&text, units), Some(byte));
                let points = text[..byte].chars().count();
                assert_eq!(text.char_indices().map(|(b, _)| b).chain([text.len()]).nth(points), Some(byte));
            }
        }
    }
}
