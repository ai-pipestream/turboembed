//! Raw-text VADER follows Python str.split, including legacy separators.
use turbo_doccat::sentiment::LexiconSentiment;

#[test]
fn vader_scores_python_separator_variants_identically() {
    let scorer = LexiconSentiment::new();
    assert!(scorer.score_text("good movie").compound > 0.0);
    for separator in ['\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', '\u{85}', '\u{a0}', '\u{3000}'] {
        for plain in ["good movie", "not good", "VERY good!", "bad but great", "good :) movie"] {
            let expected = scorer.score_text(plain);
            let altered = format!("{separator}{}{separator}", plain.replace(' ', &separator.to_string().repeat(2)));
            let actual = scorer.score_text(&altered);
            assert_eq!(actual, expected, "{altered:?}");
        }
        assert_eq!(scorer.score_text(&separator.to_string().repeat(3)), scorer.score_text(""));
    }
}

#[test]
fn vader_does_not_split_format_characters() {
    let scorer = LexiconSentiment::new();
    for c in ['\u{180e}', '\u{200b}', '\u{feff}'] {
        assert_eq!(scorer.score_text(&format!("good{c}movie")).compound, 0.0);
    }
}
