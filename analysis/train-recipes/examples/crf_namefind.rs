//! Full MASC CRF recipes with the standard split and paired gates.
//! Usage: crf_namefind MASC_DIRECTORY REFERENCES OUTPUT [person|location|organization|date]
//! Local outputs include models and evaluation inputs; publish only after all gates pass.
#[path = "crf_support/full.rs"]
mod full;
use std::path::Path;
use std::time::Instant;
use turbo_eval::model_gate::{Counts, HeldOutSet, ModelGateRecord, ModelId, Task};
use turbo_formats::masc::MascCorpus;
use turbo_train::crf::CrfTrainer;
use turbo_train_components::Stamp;
use turbo_train_components::namefind::{TokenNameFinderFactory, train_crf_name_model};
use turbo_train_recipes::{masc, namefind, spans};
use turbo_zoo::Sha256;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: crf_namefind MASC_DIRECTORY REFERENCES OUTPUT [entity-type]".into());
    }
    let (root, references, out) = (Path::new(&args[0]), Path::new(&args[1]), Path::new(&args[2]));
    let selected = args.get(3).map(|s| s.to_str().ok_or("entity type is not UTF-8")).transpose()?;
    let recipes: Vec<_> = namefind::RECIPES.iter().filter(|r| selected.is_none_or(|s| s == r.kind())).collect();
    if recipes.is_empty() {
        return Err("entity type selected no recipes".into());
    }
    let digest = masc::corpus_digest(root)?;
    if !masc::RELEASE.is(&digest) {
        return Err("MASC corpus digest differs from the pinned release".into());
    }
    let split = masc::SPLIT.apply(MascCorpus::read(root)?, root)?;
    // Validate both data and references before creating any model output.
    for r in &recipes {
        r.check_training(&split.train)?;
        r.held_out(&split.held_out)?;
        r.reference.check(&std::fs::read(references.join(r.reference.file()?))?)?;
        if out.join(r.kind()).exists() {
            return Err(format!("output already exists for {}", r.kind()).into());
        }
    }
    std::fs::create_dir_all(out)?;
    let mut all_crf = Vec::new();
    let mut all_maxent = Vec::new();
    let mut all_published = Vec::new();
    let mut crf_hashes = Vec::new();
    let mut maxent_hashes = Vec::new();
    let mut published_hashes = Vec::new();
    let mut sample_hashes = Vec::new();
    let mut failures = Vec::new();
    let mut token_digest = None;
    let mut sets = Vec::new();
    for r in recipes {
        let dir = out.join(r.kind());
        std::fs::create_dir(&dir)?;
        let mut train = Vec::new();
        let (train_counts, skipped) = namefind::read_samples(&split.train, &r.types, |s| {
            train.push(s);
            Ok(())
        })?;
        let held = r.held_out(&split.held_out)?;
        let tokens: Vec<Vec<_>> =
            held.samples().iter().map(|s| s.tokens().iter().map(|t| t.as_str()).collect()).collect();
        let current_digest = Sha256::of(&serde_json::to_vec(&tokens)?).to_string();
        if token_digest.as_ref().is_some_and(|d| d != &current_digest) {
            return Err("entity recipes have different held-out tokens".into());
        }
        token_digest = Some(current_digest);
        full::write_inputs(&dir.join("held-out.json"), held.samples())?;
        let factory = TokenNameFinderFactory::new("en", None, namefind::CODEC)?;
        println!(
            "{}: train {} samples, {} names; held out {} samples, {} names",
            r.kind(),
            train.len(),
            train_counts.names,
            held.len(),
            held.counts().names
        );
        let trainer = CrfTrainer { codec: namefind::CODEC, cutoff: 2, l2: 1.0, iterations: 100 };
        let started = Instant::now();
        let crf = train_crf_name_model(&factory, &train, &trainer, Stamp::at(0))?;
        let seconds = started.elapsed().as_secs_f64();
        let crf_bytes = crf.package.to_bytes();
        std::fs::write(dir.join("crf.bin"), &crf_bytes)?;
        let candidate = full::score_crf(&crf.model, &crf_bytes, held.samples())?;
        println!("{}: CRF trained in {seconds:.3} seconds", r.kind());
        let baseline = r.train(&split.train)?;
        std::fs::write(dir.join("maxent.bin"), &baseline.bytes)?;
        let maxent = r.score(&baseline.bytes, &held)?;
        let published_bytes = std::fs::read(references.join(r.reference.file()?))?;
        let published = r.score(&published_bytes, &held)?;
        let model_id = |id: String, bytes: &[u8]| ModelId { id, sha256: Sha256::of(bytes).to_string() };
        let crf_id = model_id(format!("turbo-en-masc-3.0.0-crf-{}-1", r.kind()), &crf_bytes);
        let maxent_id = model_id(r.model_id.to_owned(), &baseline.bytes);
        let published_id = model_id(r.reference.id().to_owned(), &published_bytes);
        let set = HeldOutSet {
            name: format!("MASC 3.0.0 standard held-out split, {}", r.kind()),
            corpus_sha256: digest.sha256.clone(),
            samples_sha256: held.sha256(&r.types),
            skipped_sentences: held.skipped() as u64,
            scoring: r.types.scoring(held.counts().names, held.counts().other_types, Vec::new()),
        };
        for (label, id, units) in [("maxent", &maxent_id, &maxent), ("published", &published_id, &published)] {
            let gate = ModelGateRecord::new(
                Task::NameFinder,
                set.clone(),
                crf_id.clone(),
                id.clone(),
                spans::pair(candidate.clone(), units.clone()),
            )?;
            gate.write(&dir.join(format!("gate-{label}.json")))?;
            println!("{gate}");
            if !gate.result.pass {
                failures.push(format!("{} vs {label}", r.kind()));
            }
        }
        let card = serde_json::json!({
            "corpus_sha256": digest.sha256, "candidate": crf_id, "maxent": maxent_id, "published": published_id,
            "train_samples": train.len(), "train_tokens": train.iter().map(|s| s.tokens().len()).sum::<usize>(),
            "train_names": train_counts, "train_skipped": skipped, "held_out_samples": held.len(),
            "held_out_names": held.counts(), "held_out_skipped": held.skipped(), "samples_sha256": set.samples_sha256,
            "held_out_documents": split.held_out_paths, "excluded_documents": split.excluded,
            "crf": {"codec": "BIO", "cutoff": 2, "l2": 1.0, "iterations": 100, "seconds": seconds},
            "maxent_parameters": {"algorithm": "MAXENT", "cutoff": 0, "iterations": 300, "threads": 1},
            "scores": {"crf": full::totals(&candidate), "maxent": full::totals(&maxent), "published": full::totals(&published)}
        });
        std::fs::write(dir.join("card.json"), serde_json::to_vec_pretty(&card)?)?;
        full::add(&mut all_crf, &candidate)?;
        full::add(&mut all_maxent, &maxent)?;
        full::add(&mut all_published, &published)?;
        crf_hashes.push(crf_id.sha256);
        maxent_hashes.push(maxent_id.sha256);
        published_hashes.push(published_id.sha256);
        sample_hashes.push(set.samples_sha256.clone());
        sets.push(set);
    }
    if selected.is_none() {
        let types = spans::SpanTypes {
            types: &["person", "location", "organization", "date"],
            renamed: &[("org", "organization")],
        };
        let set = HeldOutSet {
            name: "MASC 3.0.0 standard held-out split, all entity types".into(),
            corpus_sha256: digest.sha256,
            samples_sha256: Sha256::of(sample_hashes.join("\n").as_bytes()).to_string(),
            skipped_sentences: sets[0].skipped_sentences,
            scoring: types.scoring(all_crf.iter().map(|c: &Counts| c.target).sum(), 0, Vec::new()),
        };
        for (label, hashes, units) in
            [("maxent", maxent_hashes, all_maxent), ("published", published_hashes, all_published)]
        {
            let id = |name: &str, hashes: &[String]| ModelId {
                id: format!("masc-{name}-collection"),
                sha256: Sha256::of(hashes.join("\n").as_bytes()).to_string(),
            };
            let gate = ModelGateRecord::new(
                Task::NameFinder,
                set.clone(),
                id("crf", &crf_hashes),
                id(label, &hashes),
                spans::pair(all_crf.clone(), units),
            )?;
            gate.write(&out.join(format!("overall-{label}.json")))?;
            println!("{gate}");
            if !gate.result.pass {
                failures.push(format!("overall vs {label}"));
            }
        }
    }
    if failures.is_empty() { Ok(()) } else { Err(format!("CRF gates failed: {failures:?}").into()) }
}
