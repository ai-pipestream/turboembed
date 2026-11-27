//! #2596: CPU-only synthetic feature tails for a matched cached PII fit.

use std::path::Path;
use std::time::Instant;

use turbo::manifest::WordPooling;
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cache;
use turbo_trainer_recipes::nemotron_pii::{context_contrasts, shared_names};

const ENCODER_SHA: &str = "65a0720298fc5fe7f229ee57ffbe1ccf0eb3836a91526840ca23fa30a5937ac8";

fn main() {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    assert!((3..=4).contains(&args.len()), "pii_contrast_cache <encoder> <out> <legacy|contrast> [rows-per-split]");
    let bundle = turbo_pins::require(&args[0]);
    let precision = EncoderOptions { device: Device::Cpu, cpu_threads: 6, max_batch: 8, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(&bundle, &precision).unwrap();
    let record = cache::encoder_record(&bundle, &encoder).unwrap();
    assert_eq!(record.manifest_sha256, ENCODER_SHA);
    assert_eq!(record.layers, Some(22));
    let (train, dev) = match args[2].as_str() {
        "legacy" => shared_names::documents().unwrap(),
        "contrast" => (context_contrasts::documents(false).unwrap(), context_contrasts::documents(true).unwrap()),
        other => panic!("unknown example family {other}"),
    };
    let limit: usize = args.get(3).map_or(usize::MAX, |n| n.parse().unwrap());
    let splits = [("train", &train[..limit.min(train.len())]), ("dev", &dev[..limit.min(dev.len())])];
    let out = Path::new(&args[1]);
    std::fs::create_dir(out).unwrap();
    let start = Instant::now();
    for split in splits {
        let (_, counts) = cache::write(&mut encoder, &[split], WordPooling::First, 1, 22, out).unwrap();
        let f = out.join(format!("{}.json", split.0));
        std::fs::write(&f, serde_json::to_vec_pretty(&(record.clone(), &counts[0])).unwrap()).unwrap();
        println!("{} {:?}: {} words after {:.1}s", args[2], split.0, counts[0].words, start.elapsed().as_secs_f64());
    }
}
