//! The unigram segmenter against what OpenNLP's `UnigramSegmenter`
//! returned on the JVM for the same lexicons and inputs
//! (analysis/goldens/data/unigram). Spans compare exactly, in UTF-8
//! offsets, and the UTF-16 offsets of the JVM's spans are checked to be
//! the same characters.

mod common;

use common::{Case, compare_layers, fixtures_dir, run};
use turbo_analysis::{Annotator, Document, View};
use turbo_jvm::{math, strict_math};
use turbo_tokenize::{UnigramLexicon, UnigramSegmenter, WhitespaceMode};
use turbo_unicode::utf16::byte_to_utf16;

const CLASS: &str = "opennlp.tools.tokenize.lattice.UnigramSegmenter";

/// The run's goldens, checked to be the segmenter's with `lexicon` in
/// `mode`, without an error and without a span that cuts a character.
fn cases(run_name: &str, lexicon: &str, mode: WhitespaceMode) -> Vec<Case> {
    let cases = run("unigram", run_name);
    for c in &cases {
        c.assert_annotator(CLASS, "tokenizePos");
        c.assert_models(&[fixtures_dir().join(lexicon)]);
        assert_eq!(c.whitespace(), mode, "{}", c.path.display());
        assert_eq!(c.golden.header.error, None, "{}", c.path.display());
        assert!(!c.has_cut(), "{}", c.path.display());
    }
    cases
}

/// Segments every input of the run and compares with the JVM; returns the
/// number of goldens and of tokens.
fn matches(run_name: &str, lexicon: &str, mode: WhitespaceMode) -> (usize, usize) {
    let lex = UnigramLexicon::read(fixtures_dir().join(lexicon), mode).unwrap();
    let segmenter = UnigramSegmenter::new(lex, mode);
    let cases = cases(run_name, lexicon, mode);
    let mut tokens = 0;
    for case in &cases {
        let text = &case.text;
        let mut out = Vec::new();
        segmenter.segment(text, &mut out);
        let got: Vec<(u32, u32)> = out.iter().map(|s| (s.start, s.end)).collect();
        let jvm = case.jvm_spans();
        let want: Vec<(u32, u32)> = jvm.iter().map(|s| s.utf8).collect();
        assert_eq!(got, want, "{}", case.path.display());
        // The JVM's UTF-16 spans are the same characters.
        for (s, j) in out.iter().zip(&jvm) {
            let unit = |b: u32| byte_to_utf16(text, b as usize).unwrap() as u32;
            assert_eq!((unit(s.start), unit(s.end)), j.utf16, "{}", case.path.display());
        }
        assert!(compare_layers(case, &got, None));
        // The annotator writes the same tokens.
        let doc = Document::new(text.as_str());
        let (layer,) = segmenter.annotate((doc.get::<View>().unwrap(), None)).unwrap();
        let annotated: Vec<(u32, u32)> = layer.spans.iter().map(|(_, s)| (s.start, s.end)).collect();
        assert_eq!(annotated, got, "{}", case.path.display());
        tokens += got.len();
    }
    (cases.len(), tokens)
}

#[test]
fn fixture_lexicon() {
    assert_eq!(matches("zh-unigram-fixture", "zh-unigram-fixture.txt", WhitespaceMode::Unicode), (63, 4741));
}

#[test]
fn fixture_lexicon_legacy() {
    assert_eq!(matches("zh-unigram-fixture+legacy", "zh-unigram-fixture.txt", WhitespaceMode::Legacy), (63, 4738));
}

#[test]
fn modes_lexicon() {
    assert_eq!(matches("zh-unigram-modes", "zh-unigram-modes.txt", WhitespaceMode::Unicode), (12, 952));
}

#[test]
fn modes_lexicon_legacy() {
    assert_eq!(matches("zh-unigram-modes+legacy", "zh-unigram-modes.txt", WhitespaceMode::Legacy), (12, 951));
}

/// The lexicon whose words depend on the whitespace mode reads as the
/// Java read it in each mode: U+001C ends a word in legacy mode only, and
/// U+0085 in Unicode mode only.
#[test]
fn the_modes_lexicon_has_other_words_in_each_mode() {
    let path = fixtures_dir().join("zh-unigram-modes.txt");
    let unicode = UnigramLexicon::read(&path, WhitespaceMode::Unicode).unwrap();
    assert_eq!((unicode.len(), unicode.total()), (4, 13));
    assert!(unicode.log_probability("\u{1c}5").is_some() && unicode.log_probability("x").is_some());
    let legacy = UnigramLexicon::read(&path, WhitespaceMode::Legacy).unwrap();
    assert_eq!((legacy.len(), legacy.total()), (4, 12));
    assert!(legacy.log_probability("x\u{85}5").is_some());
}

#[test]
fn jieba_subset() {
    assert_eq!(matches("jieba-dict-subset", "jieba-dict-subset.txt", WhitespaceMode::Unicode), (57, 4427));
}

#[test]
fn jieba_subset_legacy() {
    assert_eq!(matches("jieba-dict-subset+legacy", "jieba-dict-subset.txt", WhitespaceMode::Legacy), (57, 4424));
}

/// The fixture lexicon has pairs of paths whose counts multiply to the
/// same number, so the exact sums of log probabilities are equal and the
/// rounded scores decide (zh/fixture-words):
///
/// - 研究 6 生命 4 against 研究生 8 命 3: the sums are equal to the bit,
///   so the path whose last word starts first, 研究|生命, is kept, as on
///   the JVM.
/// - 研究 6 生物 12 against 研究生 8 物 9: the later path is higher in the
///   last bit and wins, 研究生|物, as on the JVM.
/// - 人民 1 大会 148 against 人民大 2 会 74: with a correctly rounded
///   logarithm the later path wins, 人民大|会, which is what the JVM
///   returned; with fdlibm's logarithm (`StrictMath.log`) the earlier one
///   would.
#[test]
fn near_ties_follow_the_rounding_of_the_logarithm() {
    let path = fixtures_dir().join("zh-unigram-fixture.txt");
    let lexicon = UnigramLexicon::read(&path, WhitespaceMode::Unicode).unwrap();
    let p = |w: &str| lexicon.log_probability(w).unwrap();
    assert_eq!(p("研究生") + p("命"), p("研究") + p("生命"));
    assert!(p("研究生") + p("物") > p("研究") + p("生物"));
    assert!(p("人民大") + p("会") > p("人民") + p("大会"));
    let total = lexicon.total() as f64;
    let s = UnigramSegmenter::new(lexicon, WhitespaceMode::Unicode);
    let words = |text: &str| {
        let mut out = Vec::new();
        s.segment(text, &mut out);
        out.iter().map(|span| text[span.range()].to_string()).collect::<Vec<_>>()
    };
    assert_eq!(words("研究生命"), ["研究", "生命"]);
    assert_eq!(words("研究生物"), ["研究生", "物"]);
    assert_eq!(words("人民大会"), ["人民大", "会"]);

    // The last comparison with the scores computed by each logarithm.
    let later_wins = |log: fn(f64) -> f64| {
        let score = |c: f64| log(c) - log(total);
        (score(2.0) + score(74.0)) > (score(1.0) + score(148.0))
    };
    assert!(later_wins(math::log));
    assert!(!later_wins(strict_math::log));
}
