//! The annotator with placeholders, the layer's own checks, the differences
//! from OpenNLP, document statistics, and property loops.

use super::*;

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
