//! The abbreviation branch of TokenizerME.tokenizePos, which none of the
//! UD models exercise (they ship no abbreviation dictionary). The
//! expected values are what TokenizerME returned on the JVM for the same
//! model and dictionary.

mod me_common;

use me_common::*;
use turbo_maxent::Dictionary;
use turbo_tokenize::TokenizerOptions;
use turbo_tokenize::me::{Features, TokenizerMe, UnitFacts, token_context};
use turbo_unicode::{Utf16Cursor, WhitespaceMode};

fn tokens(t: &TokenizerMe, text: &str) -> Vec<(u32, u32, f64)> {
    let mut out = Vec::new();
    t.tokenize(text, &mut out);
    out.iter().map(|(s, p)| (s.start, s.end, *p)).collect()
}

fn check(t: &TokenizerMe, text: &str, want: &[(u32, u32, f64)]) {
    let got = tokens(t, text);
    let spans = |v: &[(u32, u32, f64)]| v.iter().map(|&(a, b, _)| (a, b)).collect::<Vec<_>>();
    assert_eq!(spans(&got), spans(want), "{text:?}");
    for (g, w) in got.iter().zip(want) {
        assert_close(g.2, w.2, &format_args!("{text:?} at {}..{}", g.0, g.1));
    }
}

#[test]
fn the_dictionary_in_the_model_file() {
    // A model OpenNLP trained on six sentences, with the case-insensitive
    // dictionary Dr., U.S., e.g. (the turbo-maxent fixture tokenizer.bin).
    let t = TokenizerMe::new(model("tokenizer-abbreviations.bin"), TokenizerOptions::default());
    check(
        &t,
        "Dr. Smith went to the U.S. office, e.g. today. DR. e.g., (Dr.)",
        &[
            (0, 3, 0.9360501154079915),
            (4, 9, 0.8850511270839005),
            (10, 14, 0.8410706052224243),
            (15, 17, 0.9502832118985761),
            (18, 21, 0.8917870219991878),
            (22, 26, 0.9158707213396288),
            (27, 33, 0.5970131809433474),
            (33, 34, 1.0),
            (35, 39, 0.8841759635675212),
            (40, 46, 0.39459039439376004),
            (47, 50, 0.7260878077716413),
            (51, 55, 0.5967696608435947),
            (55, 56, 1.0),
            (57, 61, 0.4416594544185194),
            (61, 62, 1.0),
        ],
    );
}

const DICTIONARY: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<dictionary case_sensitive="false">
<entry><token>Dr.</token></entry>
<entry><token>e.g.</token></entry>
<entry><token>U.S.</token></entry>
<entry><token>etc.</token></entry>
<entry><token>a.m.</token></entry>
<entry><token>(c)Co.ltd</token></entry>
<entry><token>xyz.,"abc</token></entry>
</dictionary>
"#;

/// The English UD model with `DICTIONARY` as its abbreviation dictionary,
/// as the JVM ran it with the dictionary added to the model file.
fn english_with_dictionary() -> TokenizerMe {
    let dictionary = Dictionary::read(DICTIONARY.as_bytes()).unwrap();
    let model = model("opennlp-en-ud-ewt-tokens-1.3-2.5.4.bin").with_abbreviations(Some(dictionary));
    TokenizerMe::new(model, TokenizerOptions::default())
}

#[test]
fn known_abbreviations_stay_whole() {
    let t = english_with_dictionary();
    check(
        &t,
        "Dr. Smith met e.g. U.S. officials at 9 a.m. etc. DR. dr.x e.g.,",
        &[
            (0, 3, 0.939533219482152),
            (4, 9, 0.9983146815103471),
            (10, 13, 0.9998544359514855),
            (14, 18, 0.4401265153031162),
            (19, 23, 0.9705450944147819),
            (24, 33, 0.9967136285422301),
            (34, 36, 0.999981475284386),
            (37, 38, 1.0),
            (39, 43, 0.503155428083612),
            (44, 48, 0.7121521983360753),
            (49, 52, 0.6018191824756183),
            (53, 57, 0.8305987012235541),
            (58, 62, 0.874241616212028),
            (62, 63, 1.0),
        ],
    );
}

#[test]
fn the_dot_count_jump_and_its_overlapping_spans() {
    // A listed unit the model splits early: the Java emits the unit from
    // the token start to its end, jumps by the number of dots, and goes on
    // splitting from there, so the spans overlap.
    let t = english_with_dictionary();
    check(
        &t,
        "(c)Co.ltd xyz.,\"abc A.M.",
        &[
            (0, 9, 0.9998072366452877),
            (3, 9, 0.9663181866303656),
            (5, 9, 0.9312317262126846),
            (10, 19, 0.7296261896339622),
            (16, 19, 0.754285095852527),
            (20, 24, 0.6175127491901472),
        ],
    );
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
