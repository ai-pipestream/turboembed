//! Comprehensive matcher, oracle, normalization, insertion, and annotator tests
//! ported from OpenNLP's `opennlp.tools.glossary` test suite.

use std::sync::Arc;

use turbo_analysis::{Analyzer, Bound, Document, Span, ViewBuilder};
use turbo_glossary::selection::spans_intersect;
use turbo_glossary::{
    AhoCorasickGlossaryMatcher, CompositeGlossaryMatcher, Dimension, GlossaryAnnotator, GlossaryEntry, GlossaryMatch,
    GlossaryMatcher, GlossaryTerms, TermAnalyzer, TermAnalyzingGlossaryMatcher, TokenGlossaryAnnotator,
};
use turbo_normalize::{AccentFold, CaseFold, Chain, Dashes, Invisible, Normalizer, Shrink, UnicodeForm, Whitespace};
use turbo_stem::{Algorithm, PorterStemmer, SnowballAlgorithm, SnowballStemmer};
use turbo_tokenize::uax29::word_boundaries;
use turbo_tokenize::{UntypedWordTokenizer, WhitespaceTokenizer};
use turbo_unicode::java::code_point;

fn entry(id: &str, term: &str) -> GlossaryEntry {
    GlossaryEntry::new(id, term).unwrap()
}

fn matched(start: u32, end: u32, id: &str, term: &str) -> GlossaryMatch {
    GlossaryMatch::new(Span::new(start, end), id, term).unwrap()
}

// ============================================================================
// GlossaryRecordsTest
// ============================================================================

#[test]
fn entry_rejects_empty_or_whitespace_id_and_term() {
    for blank in ["", " ", "\t\n", "\u{00A0}", "\u{202F}", "\u{3000}"] {
        assert!(GlossaryEntry::new(blank, "term").is_err());
        assert!(GlossaryEntry::new("ID-1", blank).is_err());
        assert!(GlossaryMatch::new(Span::new(0, 4), blank, "term").is_err());
        assert!(GlossaryMatch::new(Span::new(0, 4), "ID-1", blank).is_err());
    }
}

#[test]
fn entry_preserves_exact_id_and_term_without_trimming() {
    let e = GlossaryEntry::new(" ID-1 ", " spaced term ").unwrap();
    assert_eq!(e.id(), " ID-1 ");
    assert_eq!(e.term(), " spaced term ");

    let m = GlossaryMatch::new(Span::new(2, 13), " ID-1 ", " spaced term ").unwrap();
    assert_eq!(m.span(), Span::new(2, 13));
    assert_eq!(m.id(), " ID-1 ");
    assert_eq!(m.term(), " spaced term ");
}

// ============================================================================
// AhoCorasickGlossaryMatcherTest
// ============================================================================

#[test]
fn exact_matching_and_word_boundaries() {
    let entries = vec![entry("C1", "cat"), entry("C2", "new york"), entry("C3", "new york city")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();

    let text = "A cat in new york city and scatter in new york.";
    let matches = matcher.match_text(text);

    assert_eq!(
        matches,
        vec![matched(2, 5, "C1", "cat"), matched(9, 22, "C3", "new york city"), matched(38, 46, "C2", "new york"),]
    );
}

#[test]
fn case_insensitive_matching() {
    let entries = vec![entry("AI-1", "Machine Learning"), entry("AI-2", "Deep Learning")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, true).unwrap();

    let text = "MACHINE LEARNING and deep learning are related.";
    let matches = matcher.match_text(text);

    assert_eq!(matches, vec![matched(0, 16, "AI-1", "Machine Learning"), matched(21, 34, "AI-2", "Deep Learning"),]);
}

#[test]
fn case_sensitive_rejects_mismatched_case() {
    let entries = vec![entry("AI-1", "Machine Learning")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();

    assert!(matcher.match_text("machine learning").is_empty());
    assert_eq!(matcher.match_text("Machine Learning").len(), 1);
}

#[test]
fn tie_breaking_prefers_earlier_entry_for_identical_spans() {
    let entries = vec![entry("FIRST", "alpha"), entry("SECOND", "alpha"), entry("THIRD", "ALPHA")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, true).unwrap();

    let matches = matcher.match_text("alpha");
    assert_eq!(matches, vec![matched(0, 5, "FIRST", "alpha")]);
}

#[test]
fn overlapping_candidates_across_positions_prefer_earliest_start_then_longest() {
    let entries = vec![entry("AB", "a b"), entry("BC", "b c"), entry("ABC", "a b c")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();

    assert_eq!(matcher.match_text("a b c"), vec![matched(0, 5, "ABC", "a b c")]);

    let entries2 = vec![entry("AB", "a b"), entry("BCD", "b c d")];
    let matcher2 = AhoCorasickGlossaryMatcher::new(&entries2, false).unwrap();

    assert_eq!(matcher2.match_text("a b c d"), vec![matched(0, 3, "AB", "a b")]);
}

#[test]
fn supplementary_code_points_and_apostrophes() {
    let entries = vec![entry("E1", "can't"), entry("E2", "🚀")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();

    let text = "We can't stop 🚀 now";
    let rocket_start = text.find('🚀').unwrap() as u32;
    let rocket_end = rocket_start + '🚀'.len_utf8() as u32;
    let matches = matcher.match_text(text);

    assert_eq!(matches, vec![matched(3, 8, "E1", "can't"), matched(rocket_start, rocket_end, "E2", "🚀"),]);
}

#[test]
fn empty_text_and_constructor_validation() {
    let entries = vec![entry("C1", "cat")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();
    assert!(matcher.match_text("").is_empty());

    assert!(AhoCorasickGlossaryMatcher::new(&[], false).is_err());
}

#[test]
fn normalized_matching_reports_original_spans_across_length_changing_edits() {
    let normalizer =
        Arc::new(Chain::new().then(Invisible).then(Whitespace::collapse()).then(Dashes::new()).then(AccentFold::new()));
    let entries = vec![entry("LOC-1", "New York"), entry("HYP-1", "co-operate"), entry("CAFE-1", "cafe")];
    let matcher = AhoCorasickGlossaryMatcher::with_normalizer(&entries, true, normalizer).unwrap();

    // "Visit  New   York, co\u{2014}operate at the caf\u{00E9}\u{200B}!"
    let text = "Visit  New   York, co\u{2014}operate at the caf\u{00E9}\u{200B}!";
    let matches = matcher.match_text(text);

    assert_eq!(matches.len(), 3);
    assert_eq!(matches[0].id(), "LOC-1");
    assert_eq!(&text[matches[0].span().range()], "New   York");
    assert_eq!(matches[1].id(), "HYP-1");
    assert_eq!(&text[matches[1].span().range()], "co\u{2014}operate");
    assert_eq!(matches[2].id(), "CAFE-1");
    assert_eq!(&text[matches[2].span().range()], "caf\u{00E9}");
}

#[test]
fn normalized_matching_rejects_term_that_normalizes_to_blank() {
    let entries = vec![entry("INV", "\u{200B}")];
    let res = AhoCorasickGlossaryMatcher::with_normalizer(&entries, false, Arc::new(Invisible));
    assert!(res.is_err());
}

#[test]
fn ascii_boundary_fast_path_agrees_with_uax29_context_sensitive_mid_characters() {
    let entries = vec![entry("N1", "3"), entry("W1", "cat"), entry("C1", "can")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, false).unwrap();

    // WB6/WB7: 'can' inside "can't" or "can:t" must not match because "'" and ":"
    // are MidNumLet / MidLetter between letters.
    assert!(matcher.match_text("can't can:t").is_empty());

    // WB11/WB12: '3' inside "3.14" or "3,000" must not match.
    assert!(matcher.match_text("3.14 3,000").is_empty());

    // Trailing period/comma/quote after a word IS a UAX #29 boundary.
    let matches = matcher.match_text("cat. 'cat' (3)");
    assert_eq!(matches, vec![matched(0, 3, "W1", "cat"), matched(6, 9, "W1", "cat"), matched(12, 13, "N1", "3"),]);
}

#[test]
fn case_insensitive_matching_handles_cross_byte_length_unicode_lowercase() {
    // U+212A KELVIN SIGN (3 UTF-8 bytes) has Character.toLowerCase == 'k' (1 UTF-8 byte).
    // U+1E9E CAPITAL SHARP S (3 UTF-8 bytes) has Character.toLowerCase == U+00DF 'ß' (2 UTF-8 bytes).
    let entries = vec![entry("K", "k"), entry("SS", "ß")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, true).unwrap();

    let text = "\u{212A} and \u{1E9E}";
    let matches = matcher.match_text(text);
    assert_eq!(matches, vec![matched(0, 3, "K", "k"), matched(8, 11, "SS", "ß"),]);
}

// ============================================================================
// AhoCorasickGlossaryOracleTest
// ============================================================================

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u32(&mut self, bound: u32) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32) % bound
    }

    fn next_bool(&mut self) -> bool {
        self.next_u32(2) == 1
    }
}

fn naive_aho_corasick_oracle(
    entries: &[GlossaryEntry],
    ignore_case: bool,
    normalizer: Option<&Arc<dyn turbo_normalize::DynNormalizer>>,
    text: &str,
) -> Vec<GlossaryMatch> {
    if text.is_empty() {
        return Vec::new();
    }
    let norm_pair = normalizer.map(|n| {
        let mut vb = ViewBuilder::new(text);
        n.normalize_dyn(text, &mut vb);
        vb.finish()
    });
    let scan_text = match &norm_pair {
        Some((s, _)) => s.as_str(),
        None => text,
    };
    if scan_text.is_empty() {
        return Vec::new();
    }

    let mut boundaries = Vec::new();
    word_boundaries(text, &mut boundaries);

    let fold_str = |s: &str| -> Vec<u32> {
        s.chars().map(|c| if ignore_case { code_point::to_lower_case(c as u32) } else { c as u32 }).collect()
    };

    let scan_chars: Vec<(usize, usize, u32)> = scan_text
        .char_indices()
        .map(|(i, c)| {
            let cp = if ignore_case { code_point::to_lower_case(c as u32) } else { c as u32 };
            (i, i + c.len_utf8(), cp)
        })
        .collect();

    let mut candidates: Vec<(u32, u32, usize)> = Vec::new();

    for (entry_idx, e) in entries.iter().enumerate() {
        let pattern_str = if let Some(n) = normalizer {
            let mut vb = ViewBuilder::new(e.term());
            n.normalize_dyn(e.term(), &mut vb);
            vb.finish().0
        } else {
            e.term().to_owned()
        };
        let pat_cps = fold_str(&pattern_str);
        if pat_cps.is_empty() || pat_cps.len() > scan_chars.len() {
            continue;
        }
        for start_cp in 0..=(scan_chars.len() - pat_cps.len()) {
            let matches_here = (0..pat_cps.len()).all(|k| scan_chars[start_cp + k].2 == pat_cps[k]);
            if matches_here {
                let norm_start = scan_chars[start_cp].0 as u32;
                let norm_end = scan_chars[start_cp + pat_cps.len() - 1].1 as u32;
                let orig = match &norm_pair {
                    Some((_, align)) => align.span_to_parent(Span::new(norm_start, norm_end)),
                    None => Span::new(norm_start, norm_end),
                };
                if boundaries.binary_search(&orig.start).is_ok() && boundaries.binary_search(&orig.end).is_ok() {
                    candidates.push((orig.start, orig.end, entry_idx));
                }
            }
        }
    }

    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));

    let mut accepted: Vec<GlossaryMatch> = Vec::new();
    for (s, e, idx) in candidates {
        let span = Span::new(s, e);
        if accepted.iter().all(|prev| !spans_intersect(prev.span(), span)) {
            accepted.push(GlossaryMatch::from_entry(span, &entries[idx]));
        }
    }
    accepted
}

const ATOM_POOL: &[&str] = &[
    "a",
    "b",
    "ab",
    "ba",
    "aba",
    "cat",
    "Can",
    "can't",
    "3",
    "14",
    "3.14",
    "caf\u{00E9}",
    "cafe",
    "co-operate",
    "co\u{2014}operate",
    "\u{03B1}\u{03B2}",
    "\u{1F680}",
    "\u{10400}",
];

const SEPARATOR_POOL: &[&str] =
    &[" ", "  ", "\t", "\n", ". ", ", ", " - ", " \u{2014} ", "'", ":", ".", ",", "\u{200B}", " \u{00A0} "];

fn random_glossary(rng: &mut Lcg, normalizer: Option<&Arc<dyn turbo_normalize::DynNormalizer>>) -> Vec<GlossaryEntry> {
    let count = 1 + rng.next_u32(8) as usize;
    let mut entries: Vec<GlossaryEntry> = Vec::with_capacity(count);
    for i in 0..count {
        let term = loop {
            let words = 1 + rng.next_u32(3) as usize;
            let mut s = String::new();
            for w in 0..words {
                if w > 0 {
                    s.push_str(if rng.next_bool() { " " } else { "-" });
                }
                s.push_str(ATOM_POOL[rng.next_u32(ATOM_POOL.len() as u32) as usize]);
            }
            if let Some(n) = normalizer {
                let mut vb = ViewBuilder::new(&s);
                n.normalize_dyn(&s, &mut vb);
                if turbo_glossary::is_blank(&vb.finish().0) {
                    continue;
                }
            }
            break s;
        };
        let id = if i > 0 && rng.next_u32(4) == 0 { entries[(i - 1) / 2].id().to_owned() } else { format!("ID-{i}") };
        entries.push(entry(&id, &term));
    }
    entries
}

fn random_text(rng: &mut Lcg, entries: &[GlossaryEntry]) -> String {
    let segments = rng.next_u32(14) as usize;
    let mut sb = String::new();
    for s in 0..segments {
        if s > 0 {
            sb.push_str(SEPARATOR_POOL[rng.next_u32(SEPARATOR_POOL.len() as u32) as usize]);
        }
        if !entries.is_empty() && rng.next_u32(3) == 0 {
            let raw = entries[rng.next_u32(entries.len() as u32) as usize].term();
            if rng.next_bool() {
                sb.push_str(&raw.to_uppercase());
            } else {
                sb.push_str(raw);
            }
        } else {
            sb.push_str(ATOM_POOL[rng.next_u32(ATOM_POOL.len() as u32) as usize]);
        }
    }
    sb
}

#[test]
fn matches_naive_oracle_across_random_glossaries_and_texts() {
    let mut rng = Lcg::new(0x600D_BEEF_CAFE_F00D);
    for trial in 0..200 {
        let ignore_case = rng.next_bool();
        let entries = random_glossary(&mut rng, None);
        let matcher = AhoCorasickGlossaryMatcher::new(&entries, ignore_case).unwrap();

        for sample in 0..10 {
            let text = random_text(&mut rng, &entries);
            let expected = naive_aho_corasick_oracle(&entries, ignore_case, None, &text);
            let actual = matcher.match_text(&text);
            assert_eq!(actual, expected, "Mismatch on trial={trial} sample={sample}");
        }
    }
}

#[test]
fn matches_naive_oracle_with_offset_aware_normalizer() {
    let normalizers: Vec<Arc<dyn turbo_normalize::DynNormalizer>> = vec![
        Arc::new(Whitespace::collapse()),
        Arc::new(Dashes::new()),
        Arc::new(AccentFold::new()),
        Arc::new(Invisible),
        Arc::new(Chain::new().then(Invisible).then(Whitespace::collapse()).then(Dashes::new()).then(AccentFold::new())),
    ];
    let mut rng = Lcg::new(0x1234_5678_9ABC_DEF0);
    for (n_idx, normalizer) in normalizers.iter().enumerate() {
        for trial in 0..50 {
            let ignore_case = rng.next_bool();
            let entries = random_glossary(&mut rng, Some(normalizer));
            let matcher =
                AhoCorasickGlossaryMatcher::with_normalizer(&entries, ignore_case, Arc::clone(normalizer)).unwrap();

            for sample in 0..8 {
                let text = random_text(&mut rng, &entries);
                let expected = naive_aho_corasick_oracle(&entries, ignore_case, Some(normalizer), &text);
                let actual = matcher.match_text(&text);
                assert_eq!(actual, expected, "Normalized mismatch on normalizer={n_idx} trial={trial} sample={sample}");
            }
        }
    }
}

// ============================================================================
// TermAnalyzingGlossaryMatcherTest
// ============================================================================

#[test]
fn stemmed_matching_finds_inflected_multi_word_terms() {
    let analyzer = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .stem_algorithm(Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English)))
        .build();
    let entries =
        vec![entry("ML-1", "neural network"), entry("ML-2", "deep neural network"), entry("ML-3", "gradient descent")];
    let matcher = TermAnalyzingGlossaryMatcher::new(&entries, analyzer).unwrap();

    let text = "Training Deep Neural Networks with stochastic Gradient Descents.";
    let matches = matcher.match_text(text);

    assert_eq!(
        matches,
        vec![matched(9, 29, "ML-2", "deep neural network"), matched(46, 63, "ML-3", "gradient descent"),]
    );
    assert_eq!(&text[matches[0].span().range()], "Deep Neural Networks");
    assert_eq!(&text[matches[1].span().range()], "Gradient Descents");
}

#[test]
fn hyphenated_and_whitespace_separated_compounds_match_same_token_sequence() {
    let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
    let entries = vec![entry("STATE-1", "state of the art")];
    let matcher = TermAnalyzingGlossaryMatcher::new(&entries, analyzer).unwrap();

    let text = "A state-of-the-art system.";
    let matches = matcher.match_text(text);

    assert_eq!(matches, vec![matched(2, 18, "STATE-1", "state of the art")]);
    assert_eq!(&text[matches[0].span().range()], "state-of-the-art");
}

#[test]
fn pre_tokenizer_normalizer_maps_token_spans_back_to_original_text() {
    let analyzer = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .accent_fold()
        .stem_algorithm(Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English)))
        .build();
    let entries = vec![entry("COOP-1", "cooperate")];
    let matcher = TermAnalyzingGlossaryMatcher::with_normalizer(&entries, analyzer, Arc::new(Invisible)).unwrap();

    let text = "They co\u{00AD}operated yesterday.";
    let matches = matcher.match_text(text);

    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].id(), "COOP-1");
    assert_eq!(&text[matches[0].span().range()], "co\u{00AD}operated");
}

struct StripDigitsNormalizer;

impl Normalizer for StripDigitsNormalizer {
    const ID: &'static str = "test:strip-digits";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        for (i, ch) in text.char_indices() {
            if ch.is_ascii_digit() {
                out.keep(i);
                out.replace(i + ch.len_utf8(), "");
            }
        }
    }
}

#[test]
fn rejects_lemma_dimension_and_invalid_entries() {
    let lemma_analyzer = TermAnalyzer::builder().lemmatize().build();
    let entries = vec![entry("C1", "cat")];
    assert!(TermAnalyzingGlossaryMatcher::new(&entries, lemma_analyzer).is_err());

    let default_analyzer = TermAnalyzer::builder().build();
    let punct_only = vec![entry("P1", "...")];
    assert!(TermAnalyzingGlossaryMatcher::new(&punct_only, default_analyzer.clone()).is_err());
    assert!(TermAnalyzingGlossaryMatcher::new(&[], default_analyzer).is_err());

    let digit_stripping =
        TermAnalyzer::builder().transform(Dimension::Nfc, Arc::new(StripDigitsNormalizer)).unwrap().build();
    assert!(TermAnalyzingGlossaryMatcher::new(&[entry("D1", "123")], digit_stripping.clone()).is_err());
    assert!(TermAnalyzingGlossaryMatcher::new(&[entry("D2", "foo 123 bar")], digit_stripping).is_err());
}

#[test]
fn skips_blank_normalized_tokens_in_input_text() {
    let digit_stripping =
        TermAnalyzer::builder().transform(Dimension::Nfc, Arc::new(StripDigitsNormalizer)).unwrap().build();
    let matcher = TermAnalyzingGlossaryMatcher::new(&[entry("AB", "alpha beta")], digit_stripping).unwrap();

    let matches = matcher.match_text("alpha 123 beta");
    assert_eq!(matches, vec![matched(0, 14, "AB", "alpha beta")]);
}

// ============================================================================
// CompositeGlossaryMatcherTest & CompositeGlossaryOracleTest
// ============================================================================

#[test]
fn higher_priority_exact_match_wins_over_lower_priority_stemmed_match_at_same_span() {
    let exact: Arc<dyn GlossaryMatcher> =
        Arc::new(AhoCorasickGlossaryMatcher::new(&[entry("EXACT-NN", "neural networks")], false).unwrap());
    let stemmed_analyzer = TermAnalyzer::builder()
        .case_fold()
        .unwrap()
        .stem_algorithm(Algorithm::Snowball(SnowballStemmer::new(SnowballAlgorithm::English)))
        .build();
    let fallback: Arc<dyn GlossaryMatcher> = Arc::new(
        TermAnalyzingGlossaryMatcher::new(
            &[entry("STEM-NN", "neural network"), entry("STEM-GD", "gradient descent")],
            stemmed_analyzer,
        )
        .unwrap(),
    );

    let composite = CompositeGlossaryMatcher::new(vec![exact, fallback]).unwrap();

    let text = "We trained neural networks using gradient descents.";
    let matches = composite.match_text(text);

    assert_eq!(
        matches,
        vec![matched(11, 26, "EXACT-NN", "neural networks"), matched(33, 50, "STEM-GD", "gradient descent"),]
    );
}

#[test]
fn higher_priority_shorter_match_blocks_overlapping_lower_priority_longer_match() {
    let high_priority: Arc<dyn GlossaryMatcher> =
        Arc::new(AhoCorasickGlossaryMatcher::new(&[entry("HP", "new york")], true).unwrap());
    let low_priority: Arc<dyn GlossaryMatcher> = Arc::new(
        AhoCorasickGlossaryMatcher::new(&[entry("LP-LONG", "new york city"), entry("LP-OTHER", "boston")], true)
            .unwrap(),
    );

    let composite = CompositeGlossaryMatcher::new(vec![high_priority, low_priority]).unwrap();

    let text = "Flights from new york city to boston.";
    let matches = composite.match_text(text);

    assert_eq!(matches, vec![matched(13, 21, "HP", "new york"), matched(30, 36, "LP-OTHER", "boston"),]);
}

struct FixedMatcher(Vec<GlossaryMatch>);

impl GlossaryMatcher for FixedMatcher {
    fn match_text(&self, _text: &str) -> Vec<GlossaryMatch> {
        self.0.clone()
    }
}

#[test]
fn enforces_non_overlapping_contract_even_when_delegate_emits_overlapping_or_unsorted_matches() {
    let unsorted_overlapping: Arc<dyn GlossaryMatcher> = Arc::new(FixedMatcher(vec![
        matched(10, 15, "LATE", "world"),
        matched(0, 5, "FIRST", "hello"),
        matched(2, 8, "OVERLAP", "llo wo"),
    ]));

    let composite = CompositeGlossaryMatcher::new(vec![unsorted_overlapping]).unwrap();
    let matches = composite.match_text("hello world!");

    assert_eq!(matches, vec![matched(0, 5, "FIRST", "hello"), matched(10, 15, "LATE", "world"),]);
}

#[test]
fn composite_validation_and_empty_text() {
    assert!(CompositeGlossaryMatcher::new(Vec::new()).is_err());

    let delegate: Arc<dyn GlossaryMatcher> =
        Arc::new(AhoCorasickGlossaryMatcher::new(&[entry("C1", "cat")], false).unwrap());
    let composite = CompositeGlossaryMatcher::new(vec![delegate]).unwrap();
    assert!(composite.match_text("").is_empty());
}

fn naive_composite_oracle(delegates: &[Arc<dyn GlossaryMatcher>], text: &str) -> Vec<GlossaryMatch> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut accepted: Vec<(GlossaryMatch, usize)> = Vec::new();
    for (d, delegate) in delegates.iter().enumerate() {
        for candidate in delegate.match_text(text) {
            if accepted.iter().all(|(ex, _)| !spans_intersect(ex.span(), candidate.span())) {
                accepted.push((candidate, d));
            }
        }
    }
    accepted.sort_by(|a, b| {
        a.0.span()
            .start
            .cmp(&b.0.span().start)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| b.0.span().end.cmp(&a.0.span().end))
    });
    accepted.into_iter().map(|(m, _)| m).collect()
}

#[test]
fn composite_matches_naive_priority_oracle_across_random_delegate_chains() {
    let mut rng = Lcg::new(0xCAFE_BABE_DEAD_BEEF);
    for trial in 0..100 {
        let num_delegates = 1 + rng.next_u32(4) as usize;
        let mut delegates: Vec<Arc<dyn GlossaryMatcher>> = Vec::with_capacity(num_delegates);
        let mut all_entries = Vec::new();
        for _ in 0..num_delegates {
            let entries = random_glossary(&mut rng, None);
            all_entries.extend(entries.iter().cloned());
            if rng.next_bool() {
                delegates.push(Arc::new(AhoCorasickGlossaryMatcher::new(&entries, rng.next_bool()).unwrap()));
            } else {
                let mut b = TermAnalyzer::builder();
                if rng.next_bool() {
                    b = b.case_fold().unwrap();
                }
                if rng.next_bool() {
                    b = b.accent_fold();
                }
                if rng.next_bool() {
                    b = b.stem(PorterStemmer);
                }
                delegates.push(Arc::new(TermAnalyzingGlossaryMatcher::new(&entries, b.build()).unwrap()));
            }
        }
        let composite = CompositeGlossaryMatcher::new(delegates.clone()).unwrap();
        for sample in 0..8 {
            let text = random_text(&mut rng, &all_entries);
            let expected = naive_composite_oracle(&delegates, &text);
            let actual = composite.match_text(&text);
            assert_eq!(actual, expected, "Composite mismatch on trial={trial} sample={sample}");
        }
    }
}

// ============================================================================
// GlossaryInsertionTest
// ============================================================================

#[derive(Copy, Clone, PartialEq, Eq)]
enum InsertionNormalizer {
    Prefix,
    Suffix,
}

impl Normalizer for InsertionNormalizer {
    const ID: &'static str = "test:insertion";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        // Leave non-blank text (including registered glossary terms) unchanged,
        // and insert "term" at the start or end of blank input.
        if !turbo_glossary::is_blank(text) {
            return;
        }
        match self {
            InsertionNormalizer::Prefix => {
                out.replace(0, "term");
            }
            InsertionNormalizer::Suffix => {
                out.keep(text.len());
                out.replace(text.len(), "term");
            }
        }
    }
}

struct PrefixOnFooNormalizer;

impl Normalizer for PrefixOnFooNormalizer {
    const ID: &'static str = "test:prefix-on-foo";

    fn normalize(&self, text: &str, out: &mut ViewBuilder<'_>) {
        if text.starts_with("foo") {
            out.replace(0, "prefix ");
        }
    }
}

#[test]
fn matcher_with_inserting_normalizer_handles_zero_width_original_spans() {
    let entries = vec![entry("FIRST", "term"), entry("SECOND", "term")];
    for mode in [InsertionNormalizer::Prefix, InsertionNormalizer::Suffix] {
        for text in ["", " ", "\t"] {
            for token_matching in [false, true] {
                let norm: Arc<dyn turbo_normalize::DynNormalizer> = Arc::new(mode);
                let actual = if token_matching {
                    let analyzer = TermAnalyzer::builder().case_fold().unwrap().build();
                    let matcher = TermAnalyzingGlossaryMatcher::with_normalizer(&entries, analyzer, norm).unwrap();
                    matcher.match_text(text)
                } else {
                    let matcher = AhoCorasickGlossaryMatcher::with_normalizer(&entries, false, norm).unwrap();
                    matcher.match_text(text)
                };
                let offset = if mode == InsertionNormalizer::Prefix { 0 } else { text.len() as u32 };
                assert_eq!(actual, vec![matched(offset, offset, "FIRST", "term")]);
            }
        }
    }

    let inserting = Arc::new(PrefixOnFooNormalizer);
    let entries2 = vec![entry("P", "prefix"), entry("F", "foo")];
    let matcher = AhoCorasickGlossaryMatcher::with_normalizer(&entries2, false, inserting.clone()).unwrap();

    // "foo" at [0, 3) and "prefix" at [0, 0) both start at 0; "foo" is longer so it wins.
    let matches = matcher.match_text("foo bar");
    assert_eq!(matches, vec![matched(0, 3, "F", "foo")]);

    // When only "prefix" is in the glossary, its zero-width original span [0, 0) is returned.
    let prefix_only = AhoCorasickGlossaryMatcher::with_normalizer(&[entry("P", "prefix")], false, inserting).unwrap();
    assert_eq!(prefix_only.match_text("foo bar"), vec![matched(0, 0, "P", "prefix")]);
}

#[test]
fn composite_matcher_suppresses_zero_width_match_intersecting_accepted_span() {
    let non_empty: Arc<dyn GlossaryMatcher> = Arc::new(FixedMatcher(vec![matched(0, 3, "A", "foo")]));
    let zero_width_at_start: Arc<dyn GlossaryMatcher> = Arc::new(FixedMatcher(vec![matched(0, 0, "B", "inserted")]));
    let zero_width_at_end: Arc<dyn GlossaryMatcher> = Arc::new(FixedMatcher(vec![matched(3, 3, "C", "inserted")]));

    // High-priority non-empty [0, 3) suppresses zero-width at 0 and 3.
    let c1 = CompositeGlossaryMatcher::new(vec![
        Arc::clone(&non_empty),
        Arc::clone(&zero_width_at_start),
        Arc::clone(&zero_width_at_end),
    ])
    .unwrap();
    assert_eq!(c1.match_text("foo bar"), vec![matched(0, 3, "A", "foo")]);

    // High-priority zero-width at 0 suppresses [0, 3).
    let c2 = CompositeGlossaryMatcher::new(vec![Arc::clone(&zero_width_at_start), Arc::clone(&non_empty)]).unwrap();
    assert_eq!(c2.match_text("foo bar"), vec![matched(0, 0, "B", "inserted")]);

    // High-priority zero-width at 3 suppresses [0, 3).
    let c3 = CompositeGlossaryMatcher::new(vec![zero_width_at_end, non_empty]).unwrap();
    assert_eq!(c3.match_text("foo bar"), vec![matched(3, 3, "C", "inserted")]);
}

// ============================================================================
// GlossaryNormalizationContractTest
// ============================================================================

#[test]
fn aho_corasick_normalizer_alignments_produce_valid_word_bounded_original_spans() {
    let normalizers: Vec<Arc<dyn turbo_normalize::DynNormalizer>> = vec![
        Arc::new(UnicodeForm::nfc()),
        Arc::new(UnicodeForm::nfkc()),
        Arc::new(Whitespace::collapse()),
        Arc::new(Dashes::new()),
        Arc::new(CaseFold::lowercase()),
        Arc::new(CaseFold::full()),
        Arc::new(AccentFold::new()),
        Arc::new(Invisible),
        Arc::new(Shrink),
        Arc::new(
            Chain::new()
                .then(Invisible)
                .then(UnicodeForm::nfkc())
                .then(Whitespace::collapse())
                .then(Dashes::new())
                .then(CaseFold::full())
                .then(AccentFold::new()),
        ),
    ];

    let entries = vec![
        entry("E1", "co-operate"),
        entry("E2", "New York"),
        entry("E3", "cafe"),
        entry("E4", "strasse"),
        entry("E5", "fi"),
    ];

    let corpus = [
        "Visit  New\t\nYork and co\u{2014}operate at the caf\u{00E9}!",
        "Die Stra\u{00DF}e in New\u{00A0}York hat ein Caf\u{0065}\u{0301}.",
        "co\u{00AD}operate with \u{FB01} friends in  New   York.",
        "\u{200B}cafe\u{200B} and co\u{2013}operate",
    ];

    for normalizer in &normalizers {
        let matcher = AhoCorasickGlossaryMatcher::with_normalizer(&entries, true, Arc::clone(normalizer)).unwrap();
        for text in corpus {
            let mut boundaries = Vec::new();
            word_boundaries(text, &mut boundaries);
            let matches = matcher.match_text(text);
            for i in 0..matches.len() {
                let s = matches[i].span();
                assert!(s.end as usize <= text.len());
                assert!(boundaries.binary_search(&s.start).is_ok());
                assert!(boundaries.binary_search(&s.end).is_ok());
                assert!(!turbo_glossary::is_blank(&text[s.range()]));
                if i > 0 {
                    let prev = matches[i - 1].span();
                    assert!(prev.end <= s.start);
                    assert!(!spans_intersect(prev, s));
                }
            }
        }
    }
}

// ============================================================================
// GlossaryAnnotatorTest & TokenGlossaryAnnotator Pipeline Tests
// ============================================================================

#[test]
fn glossary_annotator_writes_glossary_terms_layer_in_analyzer_pipeline() {
    let entries = vec![entry("NLP-1", "named entity recognition"), entry("NLP-2", "entity")];
    let matcher = AhoCorasickGlossaryMatcher::new(&entries, true).unwrap();
    let annotator = GlossaryAnnotator::new(matcher);

    let analyzer = Analyzer::builder().add(annotator).build().unwrap();

    let text = "Named Entity Recognition finds each entity.";
    let mut doc = Document::new(text);
    for res in analyzer.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let layer = doc.get::<GlossaryTerms>().unwrap();
    assert_eq!(layer.len(), 2);
    assert_eq!(layer.get(0), (Span::new(0, 24), "NLP-1", "named entity recognition"));
    assert_eq!(layer.get(1), (Span::new(36, 42), "NLP-2", "entity"));
}

#[test]
fn token_glossary_annotator_matches_over_tokens_layer() {
    let entries = vec![entry("NLP-1", "named entity recognition"), entry("NLP-2", "entity")];
    let tok_glossary = TokenGlossaryAnnotator::new(&entries, true).unwrap();

    let analyzer = Analyzer::builder().add(UntypedWordTokenizer::default()).add(tok_glossary).build().unwrap();

    let text = "Named   Entity  Recognition finds each entity.";
    let mut doc = Document::new(text);
    for res in analyzer.run(std::slice::from_mut(&mut doc)) {
        res.unwrap();
    }

    let layer = doc.get::<GlossaryTerms>().unwrap();
    assert_eq!(layer.len(), 2);
    let (s0, id0, term0) = layer.get(0);
    assert_eq!(&text[s0.range()], "Named   Entity  Recognition");
    assert_eq!(id0, "NLP-1");
    assert_eq!(term0, "named entity recognition");

    let (s1, id1, term1) = layer.get(1);
    assert_eq!(&text[s1.range()], "entity");
    assert_eq!(id1, "NLP-2");
    assert_eq!(term1, "entity");

    // Also verify with WhitespaceTokenizer.
    let analyzer2 = Analyzer::builder()
        .add(WhitespaceTokenizer::default())
        .step(Bound::new(TokenGlossaryAnnotator::new(&[entry("E1", "each entity.")], false).unwrap()))
        .build()
        .unwrap();
    let mut doc2 = Document::new(text);
    for res in analyzer2.run(std::slice::from_mut(&mut doc2)) {
        res.unwrap();
    }
    let layer2 = doc2.get::<GlossaryTerms>().unwrap();
    assert_eq!(layer2.len(), 1);
    assert_eq!(layer2.get(0).1, "E1");
}
