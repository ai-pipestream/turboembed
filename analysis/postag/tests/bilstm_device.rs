//! The packed tagger with its BiLSTMs on a GPU (#860), at the trainer's
//! default sizes with random weights: its emissions are the CPU packed
//! path's within 1e-5, a sentence tags the same alone as in a batch, and
//! a release timing interleaved against the CPU path.
//!
//! The device is TURBO_TEST_DEVICE (cuda or cuda:N), else cuda with
//! TURBO_TEST_REQUIRE_CUDA=1; without either the tests say they were
//! skipped. A named device the tagger cannot be made on fails them.

mod bilstm_random;

use std::time::Instant;

use bilstm_random::{model, sentences};
use turbo::device::Device;
use turbo_postag::neural::BilstmPosTagger;

fn device() -> Option<Device> {
    let device = match std::env::var("TURBO_TEST_DEVICE") {
        Ok(name) => Device::parse(&name).unwrap_or_else(|e| panic!("TURBO_TEST_DEVICE={name}: {}", e.message)),
        Err(_) if std::env::var("TURBO_TEST_REQUIRE_CUDA").is_ok_and(|v| v == "1") => Device::Cuda(0),
        Err(_) => return None,
    };
    (device != Device::Cpu).then_some(device)
}

fn on(device: Device, m: &turbo_postag::neural::BilstmPosModel) -> BilstmPosTagger {
    BilstmPosTagger::on_device(m.clone(), device).unwrap_or_else(|e| panic!("the tagger on {device:?}: {e}"))
}

#[test]
fn device_emissions_match_the_cpu() {
    let Some(device) = device() else {
        println!("skipped: neither TURBO_TEST_DEVICE nor TURBO_TEST_REQUIRE_CUDA=1 is set");
        return;
    };
    for (seed, stacked, crf) in [(1, false, true), (2, true, true), (3, false, false)] {
        let m = model(seed, stacked, crf);
        let (cpu, gpu) = (BilstmPosTagger::new(m.clone()), on(device, &m));
        let mut worst = 0f64;
        for sentence in sentences(seed, 24) {
            let (a, b) = (cpu.emissions(&sentence).unwrap(), gpu.emissions(&sentence).unwrap());
            let d = a.iter().zip(&b).flat_map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()));
            worst = d.fold(worst, f64::max);
        }
        println!("{device:?} stacked {stacked}, crf {crf}: max abs emission diff from the CPU {worst:e}");
        assert!(worst < 1e-5, "{device:?} stacked {stacked}: {worst:e}");
    }
}

#[test]
fn a_sentence_tags_the_same_alone_and_in_a_batch_on_the_device() {
    let Some(device) = device() else {
        println!("skipped: neither TURBO_TEST_DEVICE nor TURBO_TEST_REQUIRE_CUDA=1 is set");
        return;
    };
    let m = model(4, true, true);
    let batch = sentences(4, 16);
    let together = on(device, &m).best_sequences(&batch).unwrap();
    for (sentence, seq) in batch.iter().zip(&together) {
        let alone = on(device, &m).best_sequences(&[sentence]).unwrap().remove(0);
        assert_eq!(alone.outcomes(), seq.outcomes(), "{sentence:?}");
    }
}

/// min, p50, mean and p99 of `v`, in ms.
fn summary(v: &mut [f64]) -> String {
    v.sort_by(f64::total_cmp);
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    format!("min {:.3} p50 {:.3} mean {:.3} p99 {:.3}", v[0], p(0.5), mean, p(0.99))
}

/// `n` sentences per call on the CPU packed path and on the device,
/// interleaved: `cold` the first call on a fresh tagger whose workspaces a
/// call of other words of the same shape has made (every word through the
/// char LSTM), `warm` the same call again with the representations cached
/// (the sentence layers, the output layer and the CRF). BILSTM_REPS calls
/// of each (default 30).
#[test]
#[ignore = "timing; run in release on an idle machine with a quiet GPU"]
fn device_tagger_timing() {
    let device = device().expect("set TURBO_TEST_DEVICE or TURBO_TEST_REQUIRE_CUDA=1");
    let reps: usize = std::env::var("BILSTM_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(30);
    for stacked in [false, true] {
        let m = model(5, stacked, true);
        for n in [1, 32, 256] {
            let batch = sentences(5, n);
            // The same shape in words the batch does not hold.
            let other: Vec<Vec<String>> = batch.iter().map(|s| s.iter().map(|w| format!("{w}q")).collect()).collect();
            let words: usize = batch.iter().map(Vec::len).sum();
            let mut times = [[Vec::new(), Vec::new()], [Vec::new(), Vec::new()]];
            for _ in 0..reps {
                for (k, tagger) in [BilstmPosTagger::new(m.clone()), on(device, &m)].into_iter().enumerate() {
                    std::hint::black_box(tagger.best_sequences(&other).unwrap());
                    for call in &mut times[k] {
                        let t = Instant::now();
                        std::hint::black_box(tagger.best_sequences(&batch).unwrap());
                        call.push(t.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            let layers = 1 + stacked as usize;
            for (k, name) in ["cpu", "device"].iter().enumerate() {
                let [cold, warm] = &mut times[k];
                println!("layers {layers}, {n} sentences ({words} words), {name} cold: {}", summary(cold));
                println!("layers {layers}, {n} sentences ({words} words), {name} warm: {}", summary(warm));
            }
        }
    }
}
