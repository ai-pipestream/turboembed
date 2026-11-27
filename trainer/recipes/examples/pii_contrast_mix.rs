//! Exact per-batch loss-denominator accounting for the #2596 pilot.
//! This inspects train rows and replays the shuffle, but does not fit.

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::json;
use turbo::bundle::sha256_hex;
use turbo::manifest::HeadSpec;
use turbo_trainer::Tagger;
use turbo_trainer::rng::Rng;
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::nemotron_pii::{context_contrasts, context_overlay, read_pinned_train_dev, shared_names};

#[derive(Clone, Copy)]
struct Row {
    words: usize,
    labels: usize,
    synthetic: bool,
}

fn row(s: &Sentence, synthetic: bool) -> Row {
    Row { words: s.words.len(), labels: s.labels.iter().filter(|l| *l != "O").count(), synthetic }
}

fn tally(base: &[Row], new: &[Sentence], spec: &HeadSpec) -> serde_json::Value {
    let train: Vec<_> = base.iter().copied().chain(new.iter().map(|r| row(r, true))).collect();
    let mut rng = Rng::new(1);
    // The optimizer shuffles after drawing the exact seeded initial head.
    let _ = Tagger::<f32>::new(spec, 768, &mut rng).unwrap();
    let mut order: Vec<_> = (0..train.len()).collect();
    let mut epoch = Vec::new();
    for _ in 0..6 {
        rng.shuffle(&mut order);
        let mut steps = 0;
        let (mut share, mut labelled_share) = (0.0, 0.0);
        for step in order.chunks(128) {
            let mut words = 0;
            let mut synthetic = 0;
            let mut labelled = 0;
            for &i in step {
                let r = train[i];
                words += r.words;
                if r.synthetic {
                    synthetic += r.words;
                    labelled += r.labels;
                }
            }
            share += synthetic as f64 / words as f64;
            labelled_share += labelled as f64 / words as f64;
            steps += 1;
        }
        assert_eq!(steps, 223);
        epoch.push(json!({"updates": steps, "synthetic_gradient_coefficient_share": share / steps as f64,
            "synthetic_labelled_coefficient_share": labelled_share / steps as f64}));
    }
    let mut roles = BTreeMap::<&str, usize>::new();
    for row in new {
        for kind in row.labels.iter().filter_map(|l| l.strip_prefix("B-")) {
            *roles.entry(kind).or_default() += 1;
        }
    }
    json!({
        "span_counts": roles,
        "words": train.iter().map(|r| r.words).sum::<usize>(),
        "synthetic_words": train.iter().filter(|r| r.synthetic).map(|r| r.words).sum::<usize>(),
        "synthetic_labelled_words": train.iter().filter(|r| r.synthetic).map(|r| r.labels).sum::<usize>(),
        "batch_coefficient_means": epoch,
    })
}

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 2, "pii_contrast_mix <publisher-corpus> <original-head.json>");
    let spec = std::fs::read(&args[1]).unwrap();
    assert_eq!(sha256_hex(&spec), "144017d505e46ed88b51b25443c992518a119d63608563a1ca01adc3e1e48696");
    let spec = serde_json::from_slice(&spec).unwrap();
    let (train, _) = read_pinned_train_dev(Path::new(&args[0])).unwrap();
    let base: Vec<_> = train.iter().step_by(4).map(|r| row(r, false)).collect();
    assert_eq!(base.len(), 22_500);
    let (legacy, _) = shared_names::documents().unwrap();
    let mut overlay = legacy.clone();
    for (i, edited) in context_overlay::edits(&legacy).unwrap() {
        overlay[i] = edited;
    }
    let report = json!({
        "control": tally(&base, &legacy, &spec),
        "overlay": tally(&base, &overlay, &spec),
        "candidate": tally(&base, &context_contrasts::documents(false).unwrap(), &spec),
    });
    println!("{}", serde_json::to_string_pretty(&report).unwrap());
}
