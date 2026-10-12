//! The noise scorer. The expectations of OpenNLP's noise tests
//! (StructuralNoiseScorerTest, NoiseScorerBoundaryTest, NoiseSpanTest,
//! NoiseManualExampleTest, NoiseAnnotatorTest, NoiseConcurrencyTest) come
//! first, with OpenNLP's UTF-16 offsets written as UTF-8 byte offsets and
//! the asset layer replaced by the `Placeholders` layer. Then the
//! document statistics and property loops over generated text.

use std::collections::HashSet;

use turbo_analysis::{Analyzer, Annotator, Document, Key, Layer, Provenance, Span, View, ViewId};
use turbo_normalize::noise::ExclusionOutOfBounds;
use turbo_normalize::{NoiseFinding, NoiseScorer, NoiseStats};
use turbo_normalize_layers::NoiseSeverity::{self, *};
use turbo_normalize_layers::{Noise, PlaceholderKind, Placeholders};

fn score(text: &str) -> Vec<NoiseFinding> {
    NoiseScorer::new().score(text, &[]).unwrap()
}

fn excluding(text: &str, exclude: &[Span]) -> Vec<NoiseFinding> {
    NoiseScorer::new().score(text, exclude).unwrap()
}

fn words(list: &[&str]) -> NoiseScorer {
    NoiseScorer::with_dictionary(list.iter().map(|w| w.to_string()).collect::<HashSet<String>>())
}

fn f(start: usize, end: usize, severity: NoiseSeverity, score: f64) -> NoiseFinding {
    NoiseFinding { span: Span::new(start as u32, end as u32), severity, score }
}

fn sp(start: u32, end: u32) -> Span {
    Span::new(start, end)
}

fn severities(found: &[NoiseFinding]) -> Vec<NoiseSeverity> {
    found.iter().map(|f| f.severity).collect()
}

fn text_of<'t>(text: &'t str, found: &NoiseFinding) -> &'t str {
    &text[found.span.range()]
}

// ---------------------------------------------------------------------------
// StructuralNoiseScorerTest

// testRepresentativeWordsProduceNoNoise
#[test]
fn representative_english_words_are_clean() {
    for word in ["strengths", "catchphrase", "rhythms", "bookkeeper", "latchstring", "twelfths"] {
        assert_eq!(score(&format!("before {word} after")), vec![], "{word}");
    }
}

// testOrdinaryProseStaysClean
#[test]
fn ordinary_prose_is_clean() {
    assert_eq!(score("The quick brown fox jumps over the lazy dog, twelfth night approaches."), vec![]);
}

// testOtherScriptsAreNeverFlagged
#[test]
fn other_scripts_are_never_flagged() {
    for text in ["می\u{200C}خواهم", "กรุงเทพมหานคร", "東京都千代田区"] {
        assert_eq!(score(text), vec![], "{text}");
    }
}

// testCamelCaseIdentifierStaysClean, testLongIdentifierWithOneDigitStaysClean
#[test]
fn identifiers_are_clean() {
    assert_eq!(score("see AbstractSingletonProxyFactoryBean docs"), vec![]);
    assert_eq!(score("DocumentationVersion2Api"), vec![]);
}

// testMisspelledBoundaryRequiresARepairToAKnownWord
#[test]
fn misspelled_needs_a_repair_to_a_known_word() {
    let scorer = words(&["modern"]);
    assert_eq!(scorer.score("rxodern", &[]).unwrap(), vec![]);
    assert_eq!(severities(&scorer.score("rnodern", &[]).unwrap()), [Misspelled]);
}

// testDamagedBoundaryStartsAtSevenConsonants
#[test]
fn damage_starts_at_seven_consonants() {
    assert_eq!(score("astrchmo"), vec![]);
    assert_eq!(severities(&score("astrchmfo")), [Damaged]);
}

// testGibberishBoundaryStartsAtTwoSignals
#[test]
fn gibberish_starts_at_two_signals() {
    assert_eq!(severities(&score("asdkfjqwza")), [Damaged]);
    assert_eq!(severities(&score("asdkfjqwzx")), [Gibberish]);
}

// testBinaryishBoundaryStartsAtTwentyFourCharacters
#[test]
fn binaryish_starts_at_twenty_four_characters() {
    assert_eq!(score("AbAbAbAbAbAbAbAbAbAbAbA"), vec![]);
    assert_eq!(severities(&score("AbAbAbAbAbAbAbAbAbAbAbAb")), [Binaryish]);
}

// testTwoAgreeingSignalsAreGibberish
#[test]
fn two_agreeing_signals_are_gibberish() {
    for token in ["asdkfjqwzx", "xxxxxxxx", "zxkcvbnmsdfg"] {
        let text = format!("start {token} end");
        let found = score(&text);
        assert_eq!(severities(&found), [Gibberish], "{token}");
        assert_eq!(text_of(&text, &found[0]), token);
    }
}

// testSingleSignalIsDamage
#[test]
fn one_signal_is_damage() {
    assert_eq!(severities(&score("a astrchmfko z")), [Damaged]);
}

// testDigitInterleavingIsDamage
#[test]
fn letter_digit_interleaving_is_damage() {
    assert_eq!(severities(&score("see c0mput3r there")), [Damaged]);
}

// testBase64ShapedRunIsBinaryish
#[test]
fn ascii_punctuation_is_trimmed_from_a_base64_run() {
    let token = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==";
    let text = format!("payload ({token}).");
    let found = score(&text);
    assert_eq!(severities(&found), [Binaryish]);
    assert_eq!(text_of(&text, &found[0]), token);
}

// testBase64ShapedRunInUnicodePunctuationIsBinaryish
#[test]
fn unicode_punctuation_is_trimmed_from_a_base64_run() {
    let token = "QWxhZGRpbjpvcGVuIHNlc2FtZQ==";
    let wrappers = [
        ("\u{201C}", "\u{201D}"),
        ("\u{AB}", "\u{BB}"),
        ("\u{300C}", "\u{300D}"),
        ("\u{FF08}", "\u{FF09}"),
        ("\u{2018}", "\u{2019}\u{3002}"),
        ("", "\u{2026}"),
    ];
    for (open, close) in wrappers {
        let text = format!("payload {open}{token}{close} here");
        let found = score(&text);
        assert_eq!(severities(&found), [Binaryish], "{text}");
        assert_eq!(text_of(&text, &found[0]), token);
    }
}

// testExcludedRegionIsNotScored
#[test]
fn an_excluded_region_is_not_scored() {
    let token = "QWxhZGRpbjF2cGVuNHNlc2FtZQ";
    let text = format!("x {token} zxkcvbnmsdfg");
    let found = excluding(&text, &[sp(2, 2 + token.len() as u32)]);
    assert_eq!(severities(&found), [Gibberish]);
}

// testEmptyExclusionDoesNotSuppressAToken
#[test]
fn an_empty_exclusion_excludes_nothing() {
    assert_eq!(excluding("zxkcvbnmsdfg", &[sp(3, 3)]), score("zxkcvbnmsdfg"));
}

// testFindingsDoNotMergeAcrossExcludedWhitespace
#[test]
fn findings_do_not_join_across_excluded_whitespace() {
    assert_eq!(excluding("bcdfg \t bcdfg", &[sp(6, 7)]), [f(0, 5, Damaged, 0.5), f(8, 13, Damaged, 0.5)]);
}

// testExclusionsMustFitTheText
#[test]
fn exclusions_must_fit_the_text() {
    assert_eq!(
        NoiseScorer::new().score("text", &[sp(3, 5)]),
        Err(ExclusionOutOfBounds { span: sp(3, 5), text_len: 4 })
    );
    assert!(NoiseScorer::new().score("", &[sp(1, 1)]).is_err());
}

// testUnsortedOverlappingExclusionsAreAccepted
#[test]
fn unsorted_overlapping_exclusions() {
    let text = "bcdfg hello bcdfg end bcdfg";
    let exclude = [sp(12, 16), sp(0, 3), sp(11, 14), sp(0, 3)];
    assert_eq!(excluding(text, &exclude), [f(22, 27, Damaged, 0.5)]);
}

// testTouchingExclusionsDoNotSuppressAToken
#[test]
fn touching_exclusions_do_not_suppress_a_token() {
    assert_eq!(excluding(" bcdfg ", &[sp(0, 1), sp(6, 7)]), score(" bcdfg "));
}

// testAdjacentFindingsMergeToTheWorseSeverity
#[test]
fn adjacent_findings_join_to_the_worse_severity() {
    let text = "xxxxxxxx c0mput3r";
    let found = score(text);
    assert_eq!(severities(&found), [Gibberish]);
    assert_eq!(text_of(text, &found[0]), text);
}

// testWhitespaceSplittingIgnoresTheWhitespaceMode
#[test]
fn next_line_separates_and_joins_tokens() {
    let text = "xxxxxxxx\u{85}c0mput3r";
    let found = score(text);
    assert_eq!(severities(&found), [Gibberish]);
    assert_eq!(text_of(text, &found[0]), text);
}

// testMergedSpanCarriesTheSurvivingSeveritysScore
#[test]
fn a_joined_span_carries_the_worse_severitys_score() {
    let text = "rnodern zxkcvbnmsdfg";
    assert_eq!(words(&["modern"]).score(text, &[]).unwrap(), [f(0, 20, Gibberish, 0.5)]);
}

// testMergedSpansOfOneSeverityKeepTheHigherScore
#[test]
fn joined_spans_of_one_severity_keep_the_higher_score() {
    assert_eq!(score("zxkcvbnmsdfg zxkcvbnmsdfgggg"), [f(0, 28, Gibberish, 0.75)]);
}

// testConfusionRepairsAreMisspelledWithADictionary
#[test]
fn confusion_repairs_are_misspelled() {
    let text = "rnodern tirnes and vvord";
    let found = words(&["modern", "times", "word"]).score(text, &[]).unwrap();
    assert_eq!(severities(&found), [Misspelled, Misspelled]);
    assert_eq!(text_of(text, &found[0]), "rnodern tirnes");
    assert_eq!(text_of(text, &found[1]), "vvord");
}

// testConfusionsWithoutADictionaryAreNotReported
#[test]
fn confusions_without_a_dictionary_are_not_reported() {
    assert_eq!(score("rnodern tirnes and vvord"), vec![]);
}

// testDictionaryAcceptedTokenOverridesStructure
#[test]
fn a_dictionary_word_is_never_flagged() {
    assert_eq!(words(&["zxkcvbnmsdfg"]).score("zxkcvbnmsdfg", &[]).unwrap(), vec![]);
}

// testLongPayloadSaturatesTheScore
#[test]
fn a_long_payload_saturates_the_score() {
    let found = score("QWxh1ZGRp2bjF2c3BlbjRzZXNhbWU5QWxh1ZGRp2bjF2c3BlbjRzZXNhbWU5");
    assert_eq!(found.len(), 1);
    assert_eq!((found[0].severity, found[0].score), (Binaryish, 1.0));
}

// testRejectsContractViolations: null arguments have no counterpart.

// ---------------------------------------------------------------------------
// NoiseScorerBoundaryTest

/// A threshold case: its name, the token, and the expected severity and
/// score, if any.
type Threshold = (&'static str, &'static str, Option<(NoiseSeverity, f64)>);

// testStructuralThreshold
#[test]
fn structural_thresholds() {
    let cases: [Threshold; 19] = [
        ("consonant run 6", "astrchmo", None),
        ("consonant run 7", "astrchmfo", Some((Damaged, 0.5))),
        ("repeat run 3", "aaab", None),
        ("repeat run 4", "aaaab", Some((Damaged, 0.5))),
        ("vowelless letters 4", "bcdf", None),
        ("vowelless letters 5", "bcdfg", Some((Damaged, 0.5))),
        ("vowel ratio above 0.10", "bcdafghjk", None),
        ("vowel ratio 0.10", "bcdafghjkl", Some((Damaged, 0.5))),
        ("vowel ratio with 7 letters", "bcdafgh", None),
        ("vowel ratio with 8 letters", "bcdafghj", None),
        ("alternations 3", "a1b23456", None),
        ("alternations 4", "a1b2edoo", Some((Damaged, 0.5))),
        ("alternating token length 7", "a1b2edo", None),
        ("alternating token length 8", "a1b2edoo", Some((Damaged, 0.5))),
        ("signals 0", "hello", None),
        ("signals 1", "bcdfg", Some((Damaged, 0.5))),
        ("signals 2", "bcdfghj", Some((Gibberish, 0.5))),
        ("signals 3", "bbbbbbb", Some((Gibberish, 0.75))),
        ("signals 4", "bbbbbbb1c2d3f4g", Some((Gibberish, 1.0))),
    ];
    for (name, token, expected) in cases {
        let expected: Vec<NoiseFinding> = expected.map(|(s, x)| f(0, token.len(), s, x)).into_iter().collect();
        assert_eq!(score(token), expected, "{name}");
    }
}

// testBinaryThreshold
#[test]
fn binary_thresholds() {
    let cases: [(&str, String, bool); 10] = [
        ("length 23", format!("{}A", "Ab".repeat(11)), false),
        ("length 24", "Ab".repeat(12), true),
        ("letter case changes 11", format!("{}{}", "Ab".repeat(6), "ab".repeat(6)), false),
        ("letter case changes 12", format!("{}{}", "Ab".repeat(6), "AB".repeat(6)), true),
        ("digit and case changes 7", format!("{}1{}a", "Ab".repeat(4), "ab".repeat(7)), false),
        ("digit and case changes 8", format!("{}A1{}", "Ab".repeat(4), "ab".repeat(7)), true),
        ("base64 plus", format!("{}+", "ab".repeat(12)), true),
        ("base64 slash", format!("{}/", "ab".repeat(12)), true),
        ("base64 padding", format!("{}=", "ab".repeat(12)), true),
        ("non-base64 character", format!("{}_", "Ab".repeat(12)), false),
    ];
    for (name, token, binary) in cases {
        let expected = if binary { vec![f(0, token.len(), Binaryish, token.len() as f64 / 48.0)] } else { vec![] };
        assert_eq!(score(&token), expected, "{name}");
    }
}

// testBinaryScoreSaturation
#[test]
fn binary_score_saturation() {
    for length in [47, 48, 49] {
        let token = &"Ab".repeat(25)[..length];
        assert_eq!(score(token), [f(0, length, Binaryish, (length as f64 / 48.0).min(1.0))]);
    }
}

// testMinimumCoreLengthForDictionaryRepair
#[test]
fn dictionary_repair_needs_three_characters() {
    let scorer = words(&["m", "mx"]);
    assert_eq!(scorer.score("rn", &[]).unwrap(), vec![]);
    assert_eq!(scorer.score("rnx", &[]).unwrap(), [f(0, 3, Misspelled, 0.9)]);
}

// testLowercaseDictionaryAcceptancePrecedesStructuralSignals
#[test]
fn the_dictionary_is_asked_in_lowercase_first() {
    assert_eq!(words(&["xxxxxxxx"]).score("XXXXXXXX", &[]).unwrap(), vec![]);
}

// testUnscoredInputs
#[test]
fn unscored_inputs() {
    for text in ["", " \t\n", "()[]{}", "bcdfg\u{E9}", "bcdfg\u{1F600}"] {
        assert_eq!(score(text), vec![], "{text:?}");
    }
}

// testOriginalOffsetsAfterSupplementaryCharacterAndPunctuation: UTF-16
// [4, 9) is UTF-8 [6, 11).
#[test]
fn offsets_after_a_supplementary_character_and_punctuation() {
    assert_eq!(score("\u{1F600} (bcdfg),\u{2003}end"), [f(6, 11, Damaged, 0.5)]);
}

// testWhitespaceSeparatedFindingsMerge
#[test]
fn findings_join_across_every_kind_of_whitespace() {
    for separator in [" ", "\t", "\r\n", "\u{A0}", "\u{2003}"] {
        let text = format!("bcdfg{separator}bcdfg");
        assert_eq!(score(&text), [f(0, text.len(), Damaged, 0.5)], "{separator:?}");
    }
}

// ---------------------------------------------------------------------------
// NoiseManualExampleTest

const MANUAL: &str = "rnodern tirnes and zxkcvbnmsdfg";

// testScoringExamplePrintsTheStatedLines
#[test]
fn manual_scoring_example() {
    let found = words(&["modern", "times"]).score(MANUAL, &[]).unwrap();
    let lines: Vec<String> =
        found.iter().map(|f| format!("{} [{}..{})", f.severity.name(), f.span.start, f.span.end)).collect();
    assert_eq!(lines, ["misspelled [0..14)", "gibberish [19..31)"]);
}

// testWithoutADictionaryOnlyTheGibberishRemains
#[test]
fn manual_example_without_a_dictionary() {
    assert_eq!(severities(&score(MANUAL)), [Gibberish]);
}

// testDocumentExample, testStandaloneDocumentExample
#[test]
fn manual_document_example() {
    let text = "The scan contains zxkcvbnmsdfg here.";
    let analyzer = Analyzer::builder().add(NoiseScorer::new()).build().unwrap();
    let mut docs = vec![Document::new(text)];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    let noise = docs[0].get::<Noise>().unwrap();
    assert_eq!(noise.len(), 1);
    assert_eq!(&text[noise.spans.iter().next().unwrap().1.range()], "zxkcvbnmsdfg");
    assert_eq!(noise.severity, [Gibberish]);
}

// ---------------------------------------------------------------------------
// NoiseAnnotatorTest and NoiseConcurrencyTest, with placeholders in place
// of OpenNLP's assets

fn annotate(scorer: &NoiseScorer, text: &str, placeholders: Option<&[Span]>) -> Noise {
    let doc = Document::new(text);
    let view = doc.get::<View>().unwrap();
    let layer = placeholders.map(|spans| {
        let mut p = Placeholders::new(view.id());
        for &s in spans {
            p.spans.push(s);
            p.kind.push(PlaceholderKind::Asset);
        }
        p
    });
    let (noise,) = scorer.annotate((view, layer.as_ref())).unwrap();
    noise.check().unwrap();
    noise
}

fn layer_findings(noise: &Noise) -> Vec<NoiseFinding> {
    noise
        .spans
        .iter()
        .map(|(i, s)| NoiseFinding { span: s, severity: noise.severity[i.get()], score: noise.score[i.get()] })
        .collect()
}

// testModesDeclareTheirRequirements
#[test]
fn annotator_id_and_layer() {
    assert_eq!(<NoiseScorer as Annotator>::ID, "turbo:noise");
    assert_eq!(Noise::ID, "turbo:noise");
    assert_ne!(NoiseScorer::new().config_hash(), words(&["a"]).config_hash());
}

// testDetectedAssetsAreNotReportedAsNoise
#[test]
fn placeholder_spans_are_not_reported_as_noise() {
    let payload = "iVBORw0KGgoAAAANSUhEUgAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    let text = format!("report zxkcvbnmsdfg here {payload}");
    let start = text.len() - payload.len();
    let all = annotate(&NoiseScorer::new(), &text, None);
    assert_eq!(all.len(), 2);
    let noise = annotate(&NoiseScorer::new(), &text, Some(&[sp(start as u32, text.len() as u32)]));
    assert_eq!(layer_findings(&noise), [f(7, 19, Gibberish, 0.5)]);
}

// testStandaloneModeScoresTheWholeText
#[test]
fn without_placeholders_the_whole_text_is_scored() {
    let noise = annotate(&NoiseScorer::new(), "QWxhZGRpbjF2cGVuNHNlc2FtZQ here", None);
    assert_eq!(noise.severity, [Binaryish]);
}

// testDefaultModeRejectsADocumentWithoutTheAssetLayer: the placeholder
// layer is an optional input here, so a document without one is scored
// whole (see docs/normalizers.md). Placeholders in another view are
// refused.
#[test]
fn placeholders_of_another_view_are_refused() {
    let doc = Document::new("bcdfg");
    let view = doc.get::<View>().unwrap();
    let mut p = Placeholders::new(ViewId(1));
    p.spans.push(sp(0, 1));
    p.kind.push(PlaceholderKind::Url);
    assert!(NoiseScorer::new().annotate((view, Some(&p))).is_err());
}

// testEmptyInputAddsAnEmptyNoiseLayer
#[test]
fn empty_input_gets_an_empty_layer() {
    let analyzer = Analyzer::builder().add(NoiseScorer::new()).build().unwrap();
    let mut docs = vec![Document::new("")];
    assert!(analyzer.run(&mut docs).iter().all(Result::is_ok));
    assert!(docs[0].get::<Noise>().unwrap().is_empty());
}

// testScorerReceivesFullAssetSpansAndPreservesInputLayers
#[test]
fn the_whole_placeholder_span_is_excluded() {
    let noise = annotate(&NoiseScorer::new(), "abc bcdfg", Some(&[sp(0, 3)]));
    assert_eq!(layer_findings(&noise), [f(4, 9, Damaged, 0.5)]);
    let noise = annotate(&NoiseScorer::new(), "bcdfg bcdfg", Some(&[sp(1, 2)]));
    assert_eq!(layer_findings(&noise), [f(6, 11, Damaged, 0.5)]);
}

// testSharedInstancesAreSafeForConcurrentUse
#[test]
fn one_scorer_shared_between_threads() {
    let scorer = words(&["hello", "modern"]);
    let cases: [(&str, Vec<Span>, Vec<NoiseFinding>); 4] = [
        ("hello bcdfg", vec![], vec![f(6, 11, Damaged, 0.5)]),
        ("rnodern", vec![], vec![f(0, 7, Misspelled, 0.9)]),
        ("hello bcdfg zxkcvbnmsdfg", vec![sp(6, 11)], vec![f(12, 24, Gibberish, 0.5)]),
        ("\u{1F600} (bcdfg),\u{2003}end", vec![], vec![f(6, 11, Damaged, 0.5)]),
    ];
    let documents: [(&str, Span, Vec<NoiseFinding>); 4] = [
        ("hello bcdfg", sp(0, 5), vec![f(6, 11, Damaged, 0.5)]),
        ("rnodern bcdfg", sp(8, 13), vec![f(0, 7, Misspelled, 0.9)]),
        ("zxkcvbnmsdfg hello", sp(13, 18), vec![f(0, 12, Gibberish, 0.5)]),
        ("\u{1F600} (bcdfg),\u{2003}end", sp(14, 17), vec![f(6, 11, Damaged, 0.5)]),
    ];
    std::thread::scope(|s| {
        let scored: Vec<_> = cases.iter().map(|(t, e, _)| s.spawn(|| scorer.score(t, e).unwrap())).collect();
        let annotated: Vec<_> =
            documents.iter().map(|(t, p, _)| s.spawn(|| layer_findings(&annotate(&scorer, t, Some(&[*p]))))).collect();
        for (h, (_, _, expected)) in scored.into_iter().zip(&cases) {
            assert_eq!(&h.join().unwrap(), expected);
        }
        for (h, (_, _, expected)) in annotated.into_iter().zip(&documents) {
            assert_eq!(&h.join().unwrap(), expected);
        }
    });
}

// ---------------------------------------------------------------------------
// NoiseSpanTest: the layer's own checks

#[test]
fn severity_names_are_opennlp_strings() {
    let all = [Misspelled, Damaged, Gibberish, Binaryish];
    assert_eq!(all.map(NoiseSeverity::name), ["misspelled", "damaged", "gibberish", "binaryish"]);
    for s in all {
        assert_eq!(NoiseSeverity::from_name(s.name()), Some(s));
    }
    assert_eq!(NoiseSeverity::from_name("ocr:suspect"), None);
}

#[test]
fn the_layer_refuses_scores_outside_zero_to_one() {
    let layer = |score: f64| {
        let mut n = Noise::new(ViewId::SOURCE);
        n.spans.push(sp(0, 1));
        n.severity.push(Damaged);
        n.score.push(score);
        n.check()
    };
    for good in [f64::MIN_POSITIVE, 5e-324, 0.5, 1.0] {
        assert!(layer(good).is_ok(), "{good}");
    }
    for bad in [0.0, -0.0, -0.1, 1.0000000000000002, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(layer(bad).is_err(), "{bad}");
    }
    let mut n = Noise::new(ViewId::SOURCE);
    n.spans.push(sp(0, 1));
    assert!(n.check().is_err(), "a severity and a score for every span");
}

// ---------------------------------------------------------------------------
// Differences from OpenNLP

// Punctuation outside the Basic Multilingual Plane is trimmed: OpenNLP
// reads UTF-16 code units, and a surrogate is not punctuation.
#[test]
fn supplementary_punctuation_is_trimmed() {
    assert_eq!(score("\u{1E95E}bcdfg\u{1E95F}"), [f(4, 9, Damaged, 0.5)]);
}

// ---------------------------------------------------------------------------
// Document statistics

#[test]
fn statistics_of_a_small_document() {
    let text = "# Title\nThe cat sat... on the mat.\nThe cat sat... on the mat.\n  #tag \u{2026}\n";
    let s = NoiseStats::of(text);
    assert_eq!(s.chars, text.chars().count() as u32);
    assert_eq!(s.lines, 4);
    // "#" "Title" "The" "cat" "sat..." "on" "the" "mat." twice, "#tag", "…":
    // "#" is a word (a symbol, not surrounding punctuation), "…" is not.
    assert_eq!(s.words, 1 + 1 + 2 * 6 + 1);
    assert_eq!(s.symbols, 2 + 2 + 1);
    assert_eq!(s.symbol_to_word_ratio, 5.0 / 15.0);
    let word_chars = 1 + 5 + 2 * (3 + 3 + 3 + 2 + 3 + 3) + 4;
    assert_eq!(s.mean_word_length, f64::from(word_chars) / 15.0);
    assert_eq!(s.terminal_punctuation_fraction, 3.0 / 4.0);
    assert_eq!(s.duplicate_line_fraction, 1.0 / 4.0);
    assert_eq!(s.duplicate_line_char_fraction, 26.0 / f64::from(s.chars));
    assert_eq!(NoiseScorer::new().stats(text), s);
}

#[test]
fn statistics_of_an_empty_text_are_zero() {
    let s = NoiseStats::of(" \n\t ");
    assert_eq!((s.words, s.lines, s.symbols), (0, 0, 0));
    assert_eq!(
        [s.symbol_to_word_ratio, s.mean_word_length, s.terminal_punctuation_fraction, s.duplicate_line_fraction],
        [0.0; 4]
    );
}

#[test]
fn statistics_count_runs_of_full_stops_once() {
    assert_eq!(NoiseStats::of("a.. b... c...... d.").symbols, 2);
    assert_eq!(NoiseStats::of("wait...").symbols, 1);
}

// ---------------------------------------------------------------------------
// Property loops

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

const TOKENS: &[&str] = &[
    "hello",
    "the",
    "bcdfg",
    "zxkcvbnmsdfg",
    "xxxxxxxx",
    "c0mput3r",
    "rnodern",
    "tirnes",
    "vvord",
    "QWxhZGRpbjpvcGVuIHNlc2FtZQ==",
    "AbAbAbAbAbAbAbAbAbAbAbAb",
    "(bcdfg),",
    "\u{201C}zxkcvbnm\u{201D}",
    "caf\u{E9}",
    "\u{1F600}",
    "...",
    "ab",
    "a1b2c3d4",
    "strengths",
    "\u{AB}hello\u{BB}",
    "\u{1E95E}bcdfg",
    "+/=",
];
const SPACES: &[&str] = &[" ", "  ", "\t", "\n", "\r\n", "\u{A0}", "\u{2003}", "\u{85}", " , "];

fn generate(rng: &mut Rng) -> String {
    let mut text = String::new();
    for i in 0..rng.below(12) {
        if i > 0 || rng.below(3) == 0 {
            text.push_str(SPACES[rng.below(SPACES.len())]);
        }
        text.push_str(TOKENS[rng.below(TOKENS.len())]);
    }
    text
}

/// Random exclusions on character boundaries, some empty, some
/// overlapping.
fn exclusions(rng: &mut Rng, text: &str) -> Vec<Span> {
    let bounds: Vec<u32> = text.char_indices().map(|(i, _)| i as u32).chain([text.len() as u32]).collect();
    (0..rng.below(3))
        .map(|_| {
            let a = bounds[rng.below(bounds.len())];
            let b = bounds[rng.below(bounds.len())];
            Span::new(a.min(b), a.max(b))
        })
        .collect()
}

#[test]
fn generated_text() {
    let mut rng = Rng(0x0123_4567_89AB_CDEF);
    let scorer = words(&["hello", "modern", "times", "word", "the"]);
    for _ in 0..4000 {
        let text = generate(&mut rng);
        let exclude = exclusions(&mut rng, &text);
        let found = scorer.score(&text, &exclude).unwrap();
        assert!(found.windows(2).all(|w| w[0].span.end <= w[1].span.start), "{text:?} {found:?}");
        for x in &found {
            let s = &text[x.span.range()];
            assert!(!s.is_empty() && !s.starts_with(char::is_whitespace) && !s.ends_with(char::is_whitespace));
            assert!(x.score > 0.0 && x.score <= 1.0);
            assert!(exclude.iter().all(|e| e.is_empty() || e.end <= x.span.start || x.span.end <= e.start), "{text:?}");
        }
        // A token scored alone is noise exactly when its core lies inside
        // a finding of the whole text (tokens that touch an exclusion are
        // never scored).
        let mut at = 0;
        for token in text.split(|c: char| turbo_unicode::props::white_space(c)) {
            let (start, end) = (at, at + token.len());
            at = end + text[end..].chars().next().map_or(0, char::len_utf8);
            if token.is_empty()
                || exclude.iter().any(|e| !e.is_empty() && (e.start as usize) < end && start < e.end as usize)
            {
                continue;
            }
            let alone = scorer.score(token, &[]).unwrap();
            let inside = alone.first().is_some_and(|a| {
                let core = Span::new(start as u32 + a.span.start, start as u32 + a.span.end);
                found.iter().any(|x| x.span.contains(core))
            });
            assert_eq!(!alone.is_empty(), inside, "{token:?} in {text:?}");
            // A clean token overlaps no finding: a joined finding spans only
            // noisy cores and the whitespace between them.
            if alone.is_empty() {
                let overlaps = |x: &NoiseFinding| (x.span.start as usize) < end && start < x.span.end as usize;
                assert!(!found.iter().any(overlaps), "{token:?} in {text:?}");
            }
        }
        // The layer passes the document's checks.
        let mut doc = Document::new(text.as_str());
        let view = doc.get::<View>().unwrap().clone();
        let (noise,) = scorer.annotate((&view, None)).unwrap();
        doc.add(&Key::<Noise>::DEFAULT, noise, Provenance::by("test")).unwrap();
    }
}
