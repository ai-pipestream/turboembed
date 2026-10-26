//! Accuracy and throughput of the BiLSTM-CRF tagger's two numerics on UD
//! English EWT, which is CC BY-SA and not in the repository. Ignored unless
//! run with the treebank files named (see the README, "BiLSTM-CRF on the CPU
//! backend"):
//!
//! - `TURBO_UD_EWT_TRAIN`, `TURBO_UD_EWT_DEV`, `TURBO_UD_EWT_TEST`: the
//!   `.conllu` splits, read as syntactic words (a multiword token's range line
//!   is skipped and its parts kept, empty nodes are dropped), UPOS tags.
//! - `TURBO_BILSTM_MODEL`: an `ONLP-BLPT` file, read if it exists, otherwise
//!   trained here on the train split (dev for model selection) and written.
//! - `TURBO_BILSTM_EPOCHS` (5): training, with the trainer's defaults otherwise
//!   and a CRF.
//! - `TURBO_BILSTM_REPS` (5), `TURBO_BILSTM_BATCH` (32 sentences per call):
//!   timing, run in the order packed, reference, reference, packed.

use std::time::Instant;

use turbo_postag::neural::{BilstmPosModel, BilstmPosTagger, BilstmPosTrainer, BilstmSettings, Numerics, PosSample};

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
}

fn number(name: &str, default: usize) -> usize {
    std::env::var(name).ok().map_or(default, |v| v.parse().unwrap_or_else(|_| panic!("{name}: {v}")))
}

/// The syntactic words of a CoNLL-U file: (form, UPOS) per sentence.
fn conllu(path: &str) -> Vec<PosSample> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let (mut out, mut words, mut tags) = (Vec::new(), Vec::new(), Vec::new());
    for line in text.lines().chain(std::iter::once("")) {
        if line.is_empty() {
            if !words.is_empty() {
                out.push(PosSample::new(std::mem::take(&mut words), std::mem::take(&mut tags)).unwrap());
            }
            continue;
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if line.starts_with('#') || cols.len() < 4 || cols[0].contains(['-', '.']) {
            continue;
        }
        words.push(cols[1].to_owned());
        tags.push(cols[3].to_owned());
    }
    out
}

fn model() -> BilstmPosModel {
    let path = env("TURBO_BILSTM_MODEL");
    if std::path::Path::new(&path).exists() {
        return BilstmPosModel::load_from_path(&path).unwrap();
    }
    let (train, dev) = (conllu(&env("TURBO_UD_EWT_TRAIN")), conllu(&env("TURBO_UD_EWT_DEV")));
    let mut settings = BilstmSettings::defaults();
    settings.crf = true;
    settings.epochs = number("TURBO_BILSTM_EPOCHS", 5);
    let start = Instant::now();
    let none = None::<fn(&str) -> Option<Vec<f32>>>;
    let model = BilstmPosTrainer::train_with_vectors(&train, settings, none, None, Some(&dev)).unwrap();
    eprintln!("trained on {} sentences in {:.0} s", train.len(), start.elapsed().as_secs_f64());
    model.serialize_to_path(&path).unwrap();
    model
}

/// Tokens per second of tagging `test` in batches, a fresh model and
/// tagger each repetition (cold caches) or one tagger throughout (warm).
fn rate(bytes: &[u8], numerics: Numerics, test: &[PosSample], batch: usize, warm: Option<&BilstmPosTagger>) -> f64 {
    let fresh;
    let tagger = match warm {
        Some(t) => t,
        None => {
            fresh = BilstmPosTagger::with_numerics(BilstmPosModel::load(bytes).unwrap(), numerics).unwrap();
            &fresh
        }
    };
    let tokens: usize = test.iter().map(|s| s.sentence.len()).sum();
    let start = Instant::now();
    for chunk in test.chunks(batch) {
        let sentences: Vec<&[String]> = chunk.iter().map(|s| s.sentence.as_slice()).collect();
        std::hint::black_box(tagger.best_sequences(&sentences).unwrap());
    }
    tokens as f64 / start.elapsed().as_secs_f64()
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
#[ignore = "needs UD English EWT; see the README"]
fn bilstm_crf_on_ud_english_ewt() {
    let model = model();
    let mut bytes = Vec::new();
    model.serialize(&mut bytes).unwrap();
    eprintln!(
        "model {} {} ({} bytes), {} tags",
        model.model_ref().revision,
        model.model_ref().sha256,
        bytes.len(),
        model.tags().len()
    );
    let test = conllu(&env("TURBO_UD_EWT_TEST"));
    let tokens: usize = test.iter().map(|s| s.sentence.len()).sum();
    let batch = number("TURBO_BILSTM_BATCH", 32);

    let packed = BilstmPosTagger::new(BilstmPosModel::load(bytes.as_slice()).unwrap());
    let parity = BilstmPosTagger::jvm_parity(BilstmPosModel::load(bytes.as_slice()).unwrap());
    let (mut right_packed, mut right_parity, mut agree) = (0, 0, 0);
    for s in &test {
        let (p, r) = (packed.tag(&s.sentence).unwrap(), parity.tag(&s.sentence).unwrap());
        for ((p, r), gold) in p.iter().zip(&r).zip(&s.tags) {
            right_packed += usize::from(p == gold);
            right_parity += usize::from(r == gold);
            agree += usize::from(p == r);
        }
    }
    let pct = |n: usize| 100.0 * n as f64 / tokens as f64;
    eprintln!("{} sentences, {tokens} tokens", test.len());
    eprintln!("packed (default): {right_packed} correct, {:.4}%", pct(right_packed));
    eprintln!("jvm parity:       {right_parity} correct, {:.4}%", pct(right_parity));
    eprintln!("tags equal between the two: {agree} of {tokens}");

    let reps = number("TURBO_BILSTM_REPS", 5);
    // The encoder alone, a sentence per call, warm.
    let encode = |t: &BilstmPosTagger| {
        let start = Instant::now();
        for s in &test {
            std::hint::black_box(t.emissions(&s.sentence).unwrap());
        }
        tokens as f64 / start.elapsed().as_secs_f64()
    };
    let (p, r) = (encode(&packed), encode(&parity));
    eprintln!("emissions alone, warm: packed {p:.0} tokens/s, jvm parity {r:.0} tokens/s, {:.2}x", p / r);
    for (label, warm) in [("cold", false), ("warm", true)] {
        let (mut p, mut r) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            let run = |n, w: &BilstmPosTagger| rate(&bytes, n, &test, batch, warm.then_some(w));
            p.push(run(Numerics::Packed, &packed));
            r.push(run(Numerics::JvmParity, &parity));
            r.push(run(Numerics::JvmParity, &parity));
            p.push(run(Numerics::Packed, &packed));
        }
        let (p, r) = (median(p), median(r));
        eprintln!(
            "{label}, {batch} sentences per call: packed {p:.0} tokens/s, jvm parity {r:.0} tokens/s, {:.2}x",
            p / r
        );
    }
}

/// The device tiers on the test split (#860): MODEL on TURBO_TEST_DEVICE
/// (cuda or levelzero) tags as the CPU packed path does, word for word;
/// FASTEST (F16 weights) is within 0.1 accuracy points of MODEL and gives
/// MODEL's tag on at least 99.9% of the words.
#[test]
#[ignore = "needs UD English EWT and TURBO_TEST_DEVICE; see the README"]
fn device_tiers_on_ud_english_ewt() {
    use turbo::device::{Device, Precision};
    let device = Device::parse(&env("TURBO_TEST_DEVICE")).unwrap();
    let model = model();
    let test = conllu(&env("TURBO_UD_EWT_TEST"));
    let cpu = BilstmPosTagger::new(model.clone());
    let on = |p| BilstmPosTagger::on_device_at(model.clone(), device, p).unwrap();
    let (gpu, fast) = (on(Precision::Model), on(Precision::Fastest));
    let batch = number("TURBO_BILSTM_BATCH", 32);
    let tag = |t: &BilstmPosTagger| -> Vec<String> {
        test.chunks(batch)
            .flat_map(|c| {
                let words: Vec<&[String]> = c.iter().map(|s| s.sentence.as_slice()).collect();
                t.best_sequences(&words).unwrap().into_iter().flat_map(|s| s.outcomes().to_vec())
            })
            .collect()
    };
    let gold: Vec<&String> = test.iter().flat_map(|s| &s.tags).collect();
    let (c, g, f) = (tag(&cpu), tag(&gpu), tag(&fast));
    let right = |v: &[String]| v.iter().zip(&gold).filter(|(a, b)| a == *b).count();
    let pct = |n: usize| 100.0 * n as f64 / gold.len() as f64;
    let (rc, rg, rf) = (right(&c), right(&g), right(&f));
    let same = g.iter().zip(&f).filter(|(a, b)| a == b).count();
    eprintln!(
        "{} tokens; cpu packed {:.4}%, {device:?} MODEL {:.4}%, FASTEST {:.4}%",
        gold.len(),
        pct(rc),
        pct(rg),
        pct(rf)
    );
    eprintln!("FASTEST gives MODEL's tag on {same} of {} words ({:.4}%)", gold.len(), pct(same));
    let differ = c.iter().zip(&g).filter(|(a, b)| a != b).count();
    assert_eq!(differ, 0, "MODEL on {device:?} differs from the CPU packed path on {differ} words");
    assert!((pct(rg) - pct(rf)).abs() <= 0.1, "FASTEST {:.4}% against MODEL {:.4}%", pct(rf), pct(rg));
    assert!(pct(same) >= 99.9, "FASTEST gives MODEL's tag on {:.4}% of the words", pct(same));
}
