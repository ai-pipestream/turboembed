//! Release timing of `score_packed` on the converted checkpoint. Run pinned
//! to one core, for example:
//! `GLINER_PACKAGE=pkg taskset -c 2 cargo test --release -p turbo-gliner --test timing -- --ignored --nocapture`
use std::time::Instant;
use turbo_gliner::{Decode, ExtractionSchema, package::Manifest};

const LABELS: [&str; 32] = [
    "person",
    "organization",
    "location",
    "date",
    "city",
    "country",
    "company",
    "product",
    "event",
    "time",
    "money",
    "law",
    "language",
    "group",
    "animal",
    "food",
    "color",
    "sport",
    "book",
    "film",
    "song",
    "game",
    "drug",
    "disease",
    "school",
    "river",
    "planet",
    "ship",
    "car",
    "phone",
    "street",
    "award",
];
const TEXT: [&str; 8] = ["the", "cat", "sat", "on", "a", "mat", "and", "slept"];

#[test]
#[ignore = "requires converted GLiNER checkpoint; see docs/gliner.md"]
fn score_packed_timing() {
    let root = std::env::var("GLINER_PACKAGE").expect("GLINER_PACKAGE");
    let reps: usize = std::env::var("GLINER_REPS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let model = Manifest::load(std::path::Path::new(&root)).unwrap();
    for words in [64, 128, 384] {
        for labels in [4, 32] {
            let schema =
                ExtractionSchema::new(LABELS[..labels].iter().map(|s| s.to_string()).collect(), 0.5, Decode::Flat)
                    .unwrap();
            let text: Vec<_> = (0..words).map(|i| TEXT[i % TEXT.len()]).collect();
            let p = model.pack(&text, &schema).unwrap();
            let mut checksum = model.score_packed(&p, &schema).unwrap().iter().map(|c| c.logit as f64).sum::<f64>();
            let mut times: Vec<f64> = (0..reps)
                .map(|_| {
                    let t = Instant::now();
                    let out = model.score_packed(&p, &schema).unwrap();
                    checksum += out[0].logit as f64;
                    t.elapsed().as_secs_f64() * 1e3
                })
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "timing words={words} labels={labels} ids={} median_ms={:.2} checksum={checksum:.3}",
                p.ids.len(),
                times[reps / 2]
            );
        }
    }
}
