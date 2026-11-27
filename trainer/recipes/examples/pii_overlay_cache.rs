//! #2596: encode just the row-matched intervention, reusing the rest verbatim.

use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Seek, Write};
use std::path::Path;

use turbo::bundle::sha256_hex;
use turbo::manifest::WordPooling;
use turbo_analysis_embed::{Device, EncoderOptions};
use turbo_tagger::WordEncoder;
use turbo_trainer_recipes::cache::{self, SplitRecord};
use turbo_trainer_recipes::nemotron_pii::{context_overlay, shared_names};
use turbo_trainer_recipes::receipt::EncoderRecord;

const LEGACY_SHA: &str = "701a72403f0f140eb8957be830f4b9ea7b997efc1cc884446278a16a28be8aa5";

fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    assert_eq!(args.len(), 3, "pii_overlay_cache <encoder> <legacy-tail> <overlay-tail>");
    let bundle = turbo_pins::require(&args[0]);
    let legacy = Path::new(&args[1]);
    let output = Path::new(&args[2]);
    let (old_enc, old): (EncoderRecord, SplitRecord) =
        serde_json::from_slice(&fs::read(legacy.join("train.json")).unwrap()).unwrap();
    assert_eq!(old.name, "train");
    assert_eq!(old_enc.backend, "cpu");
    let old_file = legacy.join(&old.files[0]);
    assert_eq!(sha256_hex(&fs::read(&old_file).unwrap()), LEGACY_SHA);
    let (original, _) = shared_names::documents().unwrap();
    let (indices, replacements): (Vec<_>, Vec<_>) = context_overlay::edits(&original).unwrap().into_iter().unzip();
    let options = EncoderOptions { device: Device::Cpu, cpu_threads: 6, max_batch: 8, ..EncoderOptions::default() };
    let mut encoder = WordEncoder::open(&bundle, &options).unwrap();
    let enc = cache::encoder_record(&bundle, &encoder).unwrap();
    assert_eq!(enc.manifest_sha256, old_enc.manifest_sha256);
    assert_eq!(enc.layers, Some(22));
    assert_eq!(enc.hidden, old_enc.hidden);
    assert_eq!(enc.precision, old_enc.precision);
    fs::create_dir(output).unwrap();
    let edits = output.join("edited");
    fs::create_dir(&edits).unwrap();
    let (_, meta) = cache::write(&mut encoder, &[("train", &replacements)], WordPooling::First, 1, 22, &edits).unwrap();
    let mut patches = BufReader::new(File::open(edits.join(&meta[0].files[0])).unwrap());
    let mut source = BufReader::new(File::open(old_file).unwrap());
    let mut destination = BufWriter::new(File::create(output.join(&old.files[0])).unwrap());
    let mut select = indices.iter().copied().peekable();
    let mut buffer = Vec::new();
    for (i, s) in original.iter().enumerate() {
        buffer.resize(s.words.len() * enc.hidden * 4, 0);
        if select.peek() == Some(&i) {
            select.next();
            source.seek_relative(buffer.len() as i64).unwrap();
            patches.read_exact(&mut buffer).unwrap();
        } else {
            source.read_exact(&mut buffer).unwrap();
        }
        destination.write_all(&buffer).unwrap();
    }
    assert!(select.next().is_none());
    assert_eq!(source.stream_position().unwrap(), old.bytes_per_file);
    assert_eq!(patches.stream_position().unwrap(), meta[0].bytes_per_file);
    let out_file = destination.into_inner().unwrap();
    out_file.sync_all().unwrap();
    assert_eq!(out_file.metadata().unwrap().len(), old.bytes_per_file);
    fs::write(output.join("train.json"), serde_json::to_vec_pretty(&(enc.clone(), &old)).unwrap()).unwrap();
    fs::write(output.join("row-indices.json"), serde_json::to_vec(&indices).unwrap()).unwrap();
    let neutral = output.join("negative");
    fs::create_dir(&neutral).unwrap();
    let words: Vec<_> = context_overlay::particle_negatives().unwrap().into_iter().map(|(s, _)| s).collect();
    let (_, count) = cache::write(&mut encoder, &[("dev", &words)], WordPooling::First, 1, 22, &neutral).unwrap();
    fs::write(neutral.join("dev.json"), serde_json::to_vec_pretty(&(enc, &count[0])).unwrap()).unwrap();
    println!("{} rows replaced; exact word count {} unchanged", indices.len(), old.words);
}
