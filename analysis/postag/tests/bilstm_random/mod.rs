//! A BiLSTM-CRF model of the trainer's default sizes with random weights
//! (word 100, char 25, char hidden 50, word hidden 128, 17 tags), and
//! sentences of 1 to 40 words of 1 to 12 characters, for the tests that
//! run the packed tagger on its real shapes without a trained model.

use turbo_postag::neural::JavaRng;
use turbo_postag::neural::bilstm_model::{BilstmPosModel, UNKNOWN, char_vocabulary_from_strs};
use turbo_postag::neural::lstm::LstmLayer;

const WORD: usize = 100;
const CHAR: usize = 25;
const CHAR_HIDDEN: usize = 50;
const HIDDEN: usize = 128;
const TAGS: usize = 17;
const LETTERS: &str = "abcdefghijklmnopqrstuvwxyz";

fn matrix(rng: &mut JavaRng, rows: usize, cols: usize, scale: f64) -> Vec<Vec<f64>> {
    (0..rows).map(|_| (0..cols).map(|_| (rng.next_double() * 2.0 - 1.0) * scale).collect()).collect()
}

/// Up to 199 known words (id 0 is unknown); `stacked` adds the second
/// sentence layer, `crf` the CRF.
pub fn model(seed: i64, stacked: bool, crf: bool) -> BilstmPosModel {
    let rng = &mut JavaRng::new(seed);
    let mut words = vec![(UNKNOWN.to_owned(), 0)];
    for w in (1..200).map(word) {
        if !words.iter().any(|(k, _)| *k == w) {
            words.push((w, words.len()));
        }
    }
    let vocab = words.len();
    let mut chars = vec![(UNKNOWN.to_owned(), 0)];
    chars.extend(LETTERS.chars().enumerate().map(|(i, c)| (c.to_string(), i + 1)));
    let lstm = |rng: &mut JavaRng, input| LstmLayer::new(input, HIDDEN, rng).unwrap();
    let rep = WORD + 2 * CHAR_HIDDEN;
    let (cf, cb) = (LstmLayer::new(CHAR, CHAR_HIDDEN, rng).unwrap(), LstmLayer::new(CHAR, CHAR_HIDDEN, rng).unwrap());
    let (wf, wb) = (lstm(rng, rep), lstm(rng, rep));
    let (wf2, wb2) = if stacked { (Some(lstm(rng, 2 * HIDDEN)), Some(lstm(rng, 2 * HIDDEN))) } else { (None, None) };
    let crf_part = |rng: &mut JavaRng| crf.then(|| matrix(rng, 1, TAGS, 0.5).remove(0));
    BilstmPosModel::new(
        words,
        char_vocabulary_from_strs(chars.iter().map(|(c, i)| (c.as_str(), *i))),
        (0..TAGS).map(|t| format!("T{t}")).collect(),
        matrix(rng, vocab, WORD, 0.5),
        matrix(rng, LETTERS.len() + 1, CHAR, 0.5),
        cf,
        cb,
        wf,
        wb,
        wf2,
        wb2,
        matrix(rng, TAGS, 2 * HIDDEN, 0.2),
        matrix(rng, 1, TAGS, 0.1).remove(0),
        40,
        None,
        None,
        crf.then(|| matrix(rng, TAGS, TAGS, 0.5)),
        crf_part(rng),
        crf_part(rng),
        None,
        None,
    )
    .unwrap()
}

/// The known word `i`, spelled in letters so its characters are known.
fn word(i: usize) -> String {
    let b = LETTERS.as_bytes();
    let len = 1 + i % 12;
    (0..len).map(|k| b[(i * 7 + k * 3) % 26] as char).collect()
}

/// `n` sentences of 1 to 40 words: known words, and unknown ones of the
/// same letters (an unknown word still runs the char LSTM).
pub fn sentences(seed: i64, n: usize) -> Vec<Vec<String>> {
    let rng = &mut JavaRng::new(seed);
    (0..n)
        .map(|_| {
            let len = 1 + rng.next_int_bounded(40) as usize;
            (0..len)
                .map(|_| {
                    let w = word(1 + rng.next_int_bounded(199) as usize);
                    if rng.next_int_bounded(4) == 0 { format!("{w}x") } else { w }
                })
                .collect()
        })
        .collect()
}
