//! Deterministic generated-input properties independent of any external model.

use turbo_analysis::{OffsetTable, Span};
use turbo_sentdetect::{LineBreaks, RuleScratch, RuleSplitter, Rules};
use turbo_unicode::props::white_space;

fn valid_partition(text: &str, spans: &[Span]) {
    let table = OffsetTable::build(text);
    let mut cursor = 0;
    for &span in spans {
        let (start, end) = (span.start as usize, span.end as usize);
        assert!(cursor <= start && start < end && end <= text.len(), "{text:?}: {spans:?}");
        assert!(text.is_char_boundary(start) && text.is_char_boundary(end));
        assert!(text[cursor..start].chars().all(white_space), "lost input in {text:?}: {spans:?}");
        assert!(!white_space(text[start..end].chars().next().unwrap()));
        assert!(!white_space(text[start..end].chars().next_back().unwrap()));
        let utf16 = table.span_to_utf16_typed(text, span);
        let points = table.span_to_code_point_typed(text, span);
        assert_eq!(utf16.start.0 as usize, text[..start].encode_utf16().count());
        assert_eq!(utf16.end.0 as usize, text[..end].encode_utf16().count());
        assert_eq!(points.start.0 as usize, text[..start].chars().count());
        assert_eq!(points.end.0 as usize, text[..end].chars().count());
        assert_eq!(table.span_from_utf16(text, utf16), Ok(span));
        assert_eq!(table.span_from_code_point(text, points), Ok(span));
        cursor = end;
    }
    assert!(text[cursor..].chars().all(white_space), "lost tail in {text:?}: {spans:?}");
}

#[test]
fn generated_unicode_partitions_keep_all_content_and_reusable_state() {
    let atoms = [
        "", "A", "z", "3.5", ".", "!", "?", "(", ")", "\"", "\n", "\r\n", "\u{2029}", "\u{3000}", "e\u{301}", "😀",
        "猫。", "\0",
    ];
    for splitter in [
        RuleSplitter::new("en"),
        RuleSplitter::new("ja"),
        RuleSplitter::new("und").with_rules(Rules::NONE).with_line_breaks(LineBreaks::Sentence),
    ] {
        let mut scratch = RuleScratch::default();
        for first in atoms {
            for second in atoms {
                for third in atoms {
                    let text = format!("{first}{second}{third}");
                    let mut fresh = Vec::new();
                    splitter.split(&text, &mut fresh);
                    valid_partition(&text, &fresh);
                    // split_with appends; an existing caller prefix must remain intact.
                    let sentinel = Span::new(123, 456);
                    let mut reused = vec![sentinel];
                    splitter.split_with(&text, &mut scratch, &mut reused);
                    assert_eq!(reused[0], sentinel);
                    assert_eq!(&reused[1..], fresh, "{text:?}");
                    let mut fallible = Vec::new();
                    splitter.try_split(&text, &mut fallible).unwrap();
                    assert_eq!(fallible, fresh, "{text:?}");
                }
            }
        }
    }
}
