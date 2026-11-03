//! Synthetic context regressions, separate from the authentic PDF gold set.

use turbo_analysis::Span;
use turbo_normalize::{Artifact, ArtifactDetector};
use turbo_normalize_layers::ArtifactKind;

const COMPACT: &str = "a\u{200b}p\u{200b}p\u{200b}r\u{200b}o\u{200b}v\u{200b}e\u{200b}d";
const WRAPPED: &str =
    "a\u{200b}\n    p\u{200b}p\u{200b}\n       r\u{200b}o\u{200b}\n          v\u{200b}e\u{200b}\n             d";

fn spans(text: &str) -> Vec<Span> {
    text.match_indices('\u{200b}').map(|(i, c)| Span::new(i as u32, (i + c.len()) as u32)).collect()
}

#[test]
fn extraction_layout_changes_reporting_without_losing_characters() {
    for detector in [ArtifactDetector::default(), ArtifactDetector::builder().only(&ArtifactKind::ALL).build()] {
        // The prefix ensures byte offsets cannot accidentally be character indices.
        let compact = format!("東京: {COMPACT}");
        let wrapped = format!("東京: {WRAPPED}");
        assert_eq!(spans(&compact).len(), 7);
        let all = spans(&wrapped);
        assert_eq!(all.len(), 7);
        assert!(detector.detect(&compact).is_empty());
        let expected: Vec<_> =
            [0, 2, 4, 6].into_iter().map(|i| Artifact { span: all[i], kind: ArtifactKind::ZeroWidth }).collect();
        assert_eq!(detector.detect(&wrapped), expected);
        for artifact in expected {
            assert_eq!(&wrapped[artifact.span.range()], "\u{200b}");
        }
        // Detection borrows the text: reporting neither removes nor repairs it.
        assert_eq!(spans(&wrapped), all);
    }
}

#[test]
fn layout_whitespace_ends_the_between_letters_context() {
    let detector = ArtifactDetector::default();
    for separator in [" ", "\t", "\n", "\r\n", "\u{00a0}"] {
        let text = format!("a\u{200b}{separator}p");
        assert_eq!(detector.detect(&text), [Artifact { span: Span::new(1, 4), kind: ArtifactKind::ZeroWidth }]);
    }
}

#[test]
fn between_letters_exception_and_adjacent_run_rule_are_distinct() {
    let detector = ArtifactDetector::default();
    for text in ["a\u{200b}p", "e\u{0301}\u{200b}x", "ก\u{200b}ข", "ب\u{200c}ب", "👩\u{200d}💻"] {
        assert!(detector.detect(text).is_empty(), "{text:?}");
    }
    assert_eq!(
        detector.detect("a\u{200b}\u{200b}p"),
        [Artifact { span: Span::new(1, 7), kind: ArtifactKind::ZeroWidth }],
    );
}
