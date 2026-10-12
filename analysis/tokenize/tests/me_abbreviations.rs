//! The abbreviation branch of TokenizerME.tokenizePos, which none of the
//! UD models exercise (they ship no abbreviation dictionary), against the
//! files the JVM wrote for the same model and dictionary: the variant runs
//! `abbreviations-fixture` and `abbreviations-en` and their contexts.

mod me_common;

use me_common::*;
use turbo_maxent::Dictionary;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{Features, TokenizerMe, UnitFacts, token_context};
use turbo_unicode::{Utf16Cursor, WhitespaceMode};

/// The English UD model with the generator's English dictionary as its
/// abbreviation dictionary, as the JVM ran it with the dictionary given
/// to `TokenizerFactory`.
fn english_with_dictionary() -> TokenizerMe {
    let bytes = std::fs::read(jvm_variants_dir().join("abbreviations/english-dictionary.xml")).unwrap();
    assert_eq!(turbo_maxent::sha256(&bytes).to_hex(), ENGLISH_DICTIONARY_SHA256);
    let dictionary = Dictionary::read(&bytes).unwrap();
    let model = model("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin").with_abbreviations(Some(dictionary));
    TokenizerMe::new(model, TokenizerOptions::default())
}

const ENGLISH_DICTIONARY_SHA256: &str = "95c6e0579cdb82676807b64a362c68b69b87004a00fff42f44356d10cb7618e9";

/// The fixture model, which holds its own case-insensitive dictionary
/// `Dr.`, `U.S.`, `e.g.`: OpenNLP trained it on six sentences.
fn fixture() -> TokenizerMe {
    let t = TokenizerMe::new(model("tokenizer-abbreviations.bin"), TokenizerOptions::default());
    let generators = std::fs::read(jvm_variants_dir().join("abbreviations/tokenizer-abbreviations.bin")).unwrap();
    assert_eq!(t.tokenizer_model().sha256(), turbo_maxent::sha256(&generators), "the generator's copy");
    t
}

/// The spans of each file of one variant run, by file name.
type Run = Vec<(String, Vec<(u32, u32, f64)>)>;

/// Checks every file of `variants/<run>` and returns the spans.
fn check_run(run: &str, t: &TokenizerMe, abbreviations: &str) -> Run {
    let mut out = Vec::new();
    for path in golden_files(&golden_dir().join("variants").join(run)) {
        let g = Golden::read(&path);
        assert_eq!(g.header("component"), format!("tokenizer-me-{run}"));
        assert_eq!(g.header("whitespace"), "unicode");
        assert_eq!(g.header("abbreviations"), abbreviations);
        check_model_headers(&|k| g.header(k).to_string(), t.tokenizer_model());
        let expected = expected_spans(&g.input, &g.spans);
        check(t, &g.input, &expected, &path.display().to_string());
        out.push((path.file_name().unwrap().to_str().unwrap().to_string(), expected));
    }
    out
}

/// The text of each span of file `name` of a run.
fn words<'a>(run: &Run, name: &str, input: &'a str) -> Vec<&'a str> {
    let spans = &run.iter().find(|(n, _)| n == name).unwrap().1;
    spans.iter().map(|&(a, b, _)| &input[a as usize..b as usize]).collect()
}

#[test]
fn the_dictionary_in_the_model_file() {
    let t = fixture();
    let run = check_run("abbreviations-fixture", &t, "in the model file");
    assert_eq!(run.len(), 38);
    // "Dr.", "U.S.", "e.g." and "DR." (the dictionary ignores case) stay whole.
    let input = "Dr. Smith went to the U.S. office, e.g. today. DR. e.g., (Dr.)";
    let got = words(&run, "fixture-sentence.txt", input);
    assert_eq!(got[..6], ["Dr.", "Smith", "went", "to", "the", "U.S."]);
    assert!(got.contains(&"e.g.") && got.contains(&"DR."));
    let contexts = golden_dir().join("variants/contexts/tokenizer-me-abbreviations-fixture.txt");
    assert!(check_contexts(&contexts, t.tokenizer_model()) > 0);
}

#[test]
fn known_abbreviations_stay_whole() {
    let t = english_with_dictionary();
    let run = check_run("abbreviations-en", &t, &format!("english-dictionary.xml sha256 {ENGLISH_DICTIONARY_SHA256}"));
    assert_eq!(run.len(), 38);
    let input = "Dr. Smith met e.g. U.S. officials at 9 a.m. etc. DR. dr.x e.g.,";
    assert_eq!(
        words(&run, "known-abbreviations.txt", input),
        ["Dr.", "Smith", "met", "e.g.", "U.S.", "officials", "at", "9", "a.m.", "etc.", "DR.", "dr.x", "e.g.", ","]
    );
    let contexts = golden_dir().join("variants/contexts/tokenizer-me-abbreviations-en.txt");
    assert!(check_contexts(&contexts, t.tokenizer_model()) > 0);
}

#[test]
fn the_dot_count_jump_and_its_overlapping_spans() {
    // A listed unit the model splits early: the Java emits the unit from
    // the token start to its end, jumps by the number of dots, and goes on
    // splitting from there, so the spans overlap.
    let run = check_run(
        "abbreviations-en",
        &english_with_dictionary(),
        &format!("english-dictionary.xml sha256 {ENGLISH_DICTIONARY_SHA256}"),
    );
    let input = "(c)Co.ltd xyz.,\"abc A.M.";
    assert_eq!(words(&run, "dot-count-jump.txt", input), ["(c)Co.ltd", "Co.ltd", ".ltd", "xyz.,\"abc", "abc", "A.M."]);
}

#[test]
fn the_pabb_feature() {
    let t = english_with_dictionary();
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
