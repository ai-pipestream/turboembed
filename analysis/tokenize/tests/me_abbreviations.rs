//! The abbreviation branch of TokenizerME.tokenizePos, which none of the
//! UD models exercise (they ship no abbreviation dictionary), against what
//! the JVM returned for the same model and dictionary: the runs
//! `tokenize/tokenizer-abbreviations` and
//! `tokenize/opennlp-en-ud-ewt-tokens-1.3-2.5.4+english-dictionary` and
//! their dumps, whose decision points show the jump over the dots. The
//! English UD model is downloaded by `turbo-goldens fetch-models`
//! (`common::published_model`); without it the tests over it say so and
//! pass.

mod common;

use common::*;
use turbo_maxent::Dictionary;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{Features, TokenizerMe, UnitFacts, token_context};
use turbo_unicode::{Utf16Cursor, WhitespaceMode};

const EN: &str = "opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin";

/// The English UD model with the generator's English dictionary as its
/// abbreviation dictionary, as the JVM ran it with the dictionary given
/// to `TokenizerFactory`, and the model's path; `None` when the model is
/// not downloaded.
fn english_with_dictionary() -> Option<(TokenizerMe, std::path::PathBuf)> {
    let path = published_model(EN)?;
    let bytes = std::fs::read(fixtures_dir().join("english-dictionary.xml")).unwrap();
    let dictionary = Dictionary::read(&bytes).unwrap();
    let model = read_model(&path).with_abbreviations(Some(dictionary));
    Some((TokenizerMe::new(model, TokenizerOptions::default()), path))
}

/// The fixture model, which holds its own case-insensitive dictionary
/// `Dr.`, `U.S.`, `e.g.`: OpenNLP trained it on six sentences.
fn fixture() -> TokenizerMe {
    TokenizerMe::new(fixture_model("tokenizer-abbreviations.bin"), TokenizerOptions::default())
}

/// The UTF-8 spans and probabilities the port gives for one input.
type Spans = Vec<(u32, u32, f64)>;

/// The spans of each input of one run, by id, and the number of decision
/// points of its dumps.
struct Run {
    spans: Vec<(String, Spans)>,
    points: usize,
}

/// Checks every golden and dump of `tokenize/<name>`, which was made with
/// the model files `models`.
fn check_run(name: &str, t: &TokenizerMe, models: &[std::path::PathBuf]) -> Run {
    let mut out = Run { spans: Vec::new(), points: 0 };
    for case in run("tokenize", name) {
        case.assert_annotator("opennlp.tools.tokenize.TokenizerME", "tokenizePos");
        assert!(case.golden.header.config.is_empty(), "{}", case.path.display());
        case.assert_models(models);
        let expected = expected_me_spans(&case.text, &case.jvm_spans());
        let mut got = Vec::new();
        t.tokenize(&case.text, &mut got);
        let got_spans: Vec<(u32, u32)> = got.iter().map(|(s, _)| (s.start, s.end)).collect();
        assert_eq!(got_spans, expected.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>(), "{}", case.path.display());
        for ((s, p), &(_, _, want)) in got.iter().zip(&expected) {
            assert_close(*p, want, &format_args!("{} at {}..{}", case.path.display(), s.start, s.end));
        }
        compare_layers(&case, &got_spans, Some(got.iter().map(|(_, p)| *p).collect()));
        out.points += check_dump(&case, t.tokenizer_model());
        out.spans.push((case.golden.header.input.id.clone(), expected));
    }
    out
}

/// The text of each span of input `id` of a run.
fn words<'a>(run: &Run, id: &str, input: &'a str) -> Vec<&'a str> {
    let spans = &run.spans.iter().find(|(n, _)| n == id).unwrap().1;
    spans.iter().map(|&(a, b, _)| &input[a as usize..b as usize]).collect()
}

/// The English run, or `None` when the model is not downloaded.
fn english_run() -> Option<Run> {
    let (t, path) = english_with_dictionary()?;
    Some(check_run(
        "opennlp-en-ud-ewt-tokens-1.3-2.5.4+english-dictionary",
        &t,
        &[path, fixtures_dir().join("english-dictionary.xml")],
    ))
}

#[test]
fn the_dictionary_in_the_model_file() {
    let t = fixture();
    let run = check_run("tokenizer-abbreviations", &t, &[fixtures_dir().join("tokenizer-abbreviations.bin")]);
    assert_eq!(run.spans.len(), 38);
    // "Dr.", "U.S.", "e.g." and "DR." (the dictionary ignores case) stay whole.
    let input = "Dr. Smith went to the U.S. office, e.g. today. DR. e.g., (Dr.)";
    let got = words(&run, "fixture-sentence", input);
    assert_eq!(got[..6], ["Dr.", "Smith", "went", "to", "the", "U.S."]);
    assert!(got.contains(&"e.g.") && got.contains(&"DR."));
    assert_eq!(run.points, 2125, "every decision of every dump");
}

#[test]
fn known_abbreviations_stay_whole() {
    let Some(run) = english_run() else { return };
    assert_eq!(run.spans.len(), 38);
    let input = "Dr. Smith met e.g. U.S. officials at 9 a.m. etc. DR. dr.x e.g.,";
    assert_eq!(
        words(&run, "known-abbreviations", input),
        ["Dr.", "Smith", "met", "e.g.", "U.S.", "officials", "at", "9", "a.m.", "etc.", "DR.", "dr.x", "e.g.", ","]
    );
    assert_eq!(run.points, 2120, "every decision of every dump");
}

#[test]
fn the_dot_count_jump_and_its_overlapping_spans() {
    // A listed unit the model splits early: the Java emits the unit from
    // the token start to its end, jumps by the number of dots, and goes on
    // splitting from there, so the spans overlap. `check_run` has also
    // checked the dump: the JVM asked the model about no position the
    // jump passed over.
    let Some(run) = english_run() else { return };
    let input = "(c)Co.ltd xyz.,\"abc A.M.";
    assert_eq!(words(&run, "dot-count-jump", input), ["(c)Co.ltd", "Co.ltd", ".ltd", "xyz.,\"abc", "abc", "A.M."]);
}

#[test]
fn the_pabb_feature() {
    let Some((t, _)) = english_with_dictionary() else { return };
    let maxent = t.tokenizer_model().maxent();
    let t_index = maxent.outcome_index("T").unwrap();
    let mut features = Features::new();
    let mut probs = vec![0.0; 2];
    // (unit, index, features joined by |, P(T)) as the JVM built them.
    let cases = [
        (
            "Dr.",
            2,
            "p=Dr|s=.|p1=r|p1_alpha|p2=D|p2_alpha|p2_caps|p21=Dr|p1f1=r.|f1=.|f1_eos|f2=bok|pabb",
            0.06021829646213318,
        ),
        (
            "A.M.",
            3,
            "p=A.M|s=.|p1=M|p1_alpha|p1_caps|p2=.|p2_eos|p21=.M|p1f1=M.|f1=.|f1_eos|f2=bok|pabb",
            0.15534646060703203,
        ),
        (
            "(c)Co.ltd",
            8,
            "p=(c)Co.lt|s=d|p1=t|p1_alpha|p2=l|p2_alpha|p21=lt|p1f1=td|f1=d|f1_alpha|f2=bok|pabb",
            3.842550867671099E-4,
        ),
    ];
    let dictionary = t.tokenizer_model().abbreviations().unwrap();
    for (unit, index, want, p_t) in cases {
        let mut at = Utf16Cursor::new(unit);
        for _ in 0..index {
            at.step_forward();
        }
        token_context(&at, UnitFacts::of(unit, dictionary.contains(unit)), WhitespaceMode::Unicode, &mut features);
        let got: Vec<&str> = features.iter().map(|f| std::str::from_utf8(f).unwrap()).collect();
        assert_eq!(got.join("|"), want);
        maxent.eval_features(features.iter(), &mut probs);
        assert_close(probs[t_index], p_t, &unit);
    }
}
