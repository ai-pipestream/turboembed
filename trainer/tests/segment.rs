//! The trainer trains a sentence boundary head and a token split head as
//! it trains a token tagger (their tensors are a tagger's), and the
//! trained weights load into the core's own heads, which decode them.

use turbo::head::tagger::Work as CoreWork;
use turbo::head::{SentenceBoundary, TokenSplitter};
use turbo::manifest::{Decode, HeadKind, HeadSpec, Scheme, WordPooling};
use turbo_trainer::fit::{score, score_boundary, score_split};
use turbo_trainer::rng::Rng;
use turbo_trainer::{Config, Example, Tagger, fit};

const HIDDEN: usize = 4;
const DEL: &str = "(m 0 1 () (s 'l' '')) (m 1 0 (s 'd' '') ())";

fn spec(kind: HeadKind, labels: &[&str], decode: Decode) -> HeadSpec {
    HeadSpec {
        name: "seg".into(),
        layer: "turboembed:seg".into(),
        kind,
        labels: labels.iter().map(|s| s.to_string()).collect(),
        scheme: Scheme::Tags,
        word_pooling: WordPooling::First,
        adapter: None,
        decode,
        biaffine: None,
        pos_input: None,
        hash_embed: None,
        span: None,
        coref: None,
        license_class: None,
        artifact: "weights".into(),
        tensor_prefix: "seg".into(),
    }
}

/// Sequences of words of class 0 (a word) or 1 (a full stop), each a
/// noisy unit vector; a full stop ends its sentence, so its label is 1.
fn stops(seed: u64, n: usize) -> Vec<Example> {
    let mut rng = Rng::new(seed);
    (0..n)
        .map(|_| {
            let (mut features, mut labels) = (Vec::new(), Vec::new());
            for _ in 0..4 + rng.below(10) {
                let c = usize::from(rng.below(4) == 0);
                for j in 0..HIDDEN {
                    features.push((if j == c { 1.0 } else { 0.0 } + rng.uniform(0.3)) as f32);
                }
                labels.push(c as u32);
            }
            Example { features, labels, forms: Vec::new() }
        })
        .collect()
}

fn config() -> Config {
    Config { epochs: 8, batch: 8, lr: 1e-2, patience: 0, threads: 1, ..Config::default() }
}

fn getter(t: &Tagger<f32>) -> impl Fn(&str) -> Option<Vec<f32>> {
    let w = t.tensors();
    move |s: &str| w.iter().find(|(n, _)| n == s).map(|(_, v)| v.clone())
}

#[test]
fn a_trained_boundary_head_ends_sentences_at_full_stops() {
    let s = spec(HeadKind::SentenceBoundary, &["inside", "last"], Decode::Argmax);
    // Each sequence ends with a full stop, as a document's last sentence
    // does: the core ends every sequence at its last word.
    let ended = |seed, n| {
        let mut data = stops(seed, n);
        for e in &mut data {
            e.features.extend([0.0, 1.0, 0.0, 0.0]);
            e.labels.push(1);
        }
        data
    };
    let (trained, report) = fit(&s, HIDDEN, &ended(1, 200), &ended(2, 40), &config()).unwrap();
    assert!(report.dev[report.best_epoch] > 0.95, "{report:?}");
    let head = SentenceBoundary::from_tensors(&s, HIDDEN, &getter(&trained)).unwrap();
    let (mut scores, mut ends, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    for e in stops(3, 20) {
        let n = e.labels.len();
        let words: Vec<u32> = (0..n as u32).collect();
        head.scores(&e.features, &words, None, &mut scores, &mut work).unwrap();
        head.decode(&scores, n, &mut ends).unwrap();
        for (i, (&end, &gold)) in ends.iter().zip(&e.labels).enumerate() {
            // The last word ends a sentence whatever it is.
            assert_eq!(end, gold == 1 || i + 1 == n, "word {i} of {n}");
        }
    }
}

#[test]
fn a_trained_split_head_splits_the_tokens_it_learned() {
    let s = spec(HeadKind::TokenSplit, &["_", DEL], Decode::EditTree);
    // Class 1 features are "del", split; class 0 are other tokens, whole.
    let mut data = stops(4, 200);
    for e in &mut data {
        e.forms = e.labels.iter().map(|&l| if l == 1 { "del" } else { "casa" }.to_string()).collect();
    }
    let (trained, report) = fit(&s, HIDDEN, &data[..160], &data[160..], &config()).unwrap();
    assert!(report.dev[report.best_epoch] > 0.95, "{report:?}");
    let head = TokenSplitter::from_tensors(&s, HIDDEN, &getter(&trained)).unwrap();
    let e = &data[0];
    let n = e.labels.len();
    let words: Vec<u32> = (0..n as u32).collect();
    let (mut scores, mut labels, mut work) = (Vec::new(), Vec::new(), CoreWork::default());
    head.scores(&e.features, &words, None, &mut scores, &mut work).unwrap();
    let forms: Vec<&str> = e.forms.iter().map(String::as_str).collect();
    head.decode(&scores, &forms, &mut labels).unwrap();
    assert_eq!(labels, e.labels);
    let split = head.words(&labels, &forms).unwrap();
    for (w, &l) in split.iter().zip(&e.labels) {
        assert_eq!(w.as_slice(), if l == 1 { &["de", "el"][..] } else { &["casa"][..] });
    }
}

#[test]
fn a_span_classifier_is_not_a_word_tagger() {
    let mut s = spec(HeadKind::SpanClassifier, &["O", "X"], Decode::SpansFlat);
    s.span = Some(turbo::manifest::SpanShape { max_words: 2, width_dim: 2 });
    assert!(Tagger::<f32>::new(&s, HIDDEN, &mut Rng::new(1)).is_err());
}

/// `n` sequences of one word whose feature is the unit vector 0, each
/// labelled `label` with form `form`.
fn ones(n: usize, label: u32, form: &str) -> Vec<Example> {
    let mut features = vec![0.0f32; HIDDEN];
    features[0] = 1.0;
    (0..n).map(|_| Example { features: features.clone(), labels: vec![label], forms: vec![form.into()] }).collect()
}

#[test]
fn boundary_dev_scores_are_the_deployed_decode_s() {
    // Trained to call the word `inside`; the core still ends the sequence
    // at its last word, so a dev set of one-word sentences scores 1.
    let s = spec(HeadKind::SentenceBoundary, &["inside", "last"], Decode::Argmax);
    let dev = ones(4, 1, "");
    let dev: Vec<Example> = dev.into_iter().map(|e| Example { forms: Vec::new(), ..e }).collect();
    let train: Vec<Example> = ones(40, 0, "").into_iter().map(|e| Example { forms: Vec::new(), ..e }).collect();
    let (trained, report) = fit(&s, HIDDEN, &train, &dev, &config()).unwrap();
    assert!(report.dev.iter().all(|&d| d == 1.0), "{report:?}");
    assert_eq!(score_boundary(&trained.to_boundary().unwrap(), &dev).unwrap(), 1.0);
    assert_eq!(score(&trained.to_token_tagger().unwrap(), &dev).unwrap(), 0.0, "argmax alone says inside");
}

#[test]
fn split_dev_scores_skip_a_split_the_form_does_not_take() {
    // Trained to split "del"; the same features on "casa" decode `_`,
    // since the split does not apply to it, so the dev set scores 1
    // where an argmax alone, splitting both, scores 0.5.
    let s = spec(HeadKind::TokenSplit, &["_", DEL], Decode::EditTree);
    let dev: Vec<Example> = ones(2, 1, "del").into_iter().chain(ones(2, 0, "casa")).collect();
    let (trained, report) = fit(&s, HIDDEN, &ones(40, 1, "del"), &dev, &config()).unwrap();
    assert_eq!(report.dev[report.best_epoch], 1.0, "{report:?}");
    let head = trained.to_splitter().unwrap();
    assert_eq!(score_split(&head, &dev).unwrap(), 1.0);
    assert_eq!(score(&trained.to_token_tagger().unwrap(), &dev).unwrap(), 0.5);
    let formless: Vec<Example> = dev.into_iter().map(|e| Example { forms: Vec::new(), ..e }).collect();
    assert!(score_split(&head, &formless).is_err(), "a split head's dev examples need their forms");
}
