//! Compare saved full-corpus CRF gates with real transformer predictions.
//! Re-reads the pinned MASC split, verifies input hashes and reports only shared types.
#[path = "crf_support/full.rs"]
pub mod full;
use std::error::Error;
use std::path::Path;
use turbo_eval::model_gate::{Counts, ModelGateRecord, ModelId, Task};
use turbo_formats::{
    evaluate::{FMeasure, LabeledSpan},
    masc::MascCorpus,
};
use turbo_train_recipes::{masc, namefind, spans};
use turbo_zoo::Sha256;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 3 {
        return Err("usage: crf_compare_dl MASC_DIRECTORY CRF_OUTPUT PREDICTIONS_JSON".into());
    }
    let root = Path::new(&args[0]);
    let out = Path::new(&args[1]);
    let digest = masc::corpus_digest(root)?;
    if !masc::RELEASE.is(&digest) {
        return Err("MASC corpus digest differs from the pinned release".into());
    }
    let split = masc::SPLIT.apply(MascCorpus::read(root)?, root)?;
    let record: serde_json::Value = serde_json::from_slice(&std::fs::read(&args[2])?)?;
    if record["version"] != 1 || record["backend"] != "cpu" || record["precision"] != "Exact" {
        return Err("unsupported prediction record".into());
    }
    let expected = Sha256::of(&full::input_bytes(namefind::PERSON.held_out(&split.held_out)?.samples())?).to_string();
    if record["input_sha256"] != expected {
        return Err("prediction inputs differ from the pinned held-out samples".into());
    }
    let predictions: Vec<Vec<(u32, u32, String)>> = serde_json::from_value(record["predicted"].clone())?;
    let model = ModelId {
        id: record["model"]["id"].as_str().ok_or("model id missing")?.to_owned(),
        sha256: record["bundle_sha256"].as_str().ok_or("bundle hash missing")?.to_owned(),
    };
    let mut combined_crf = Vec::new();
    let mut combined_maxent = Vec::new();
    let mut combined_dl = Vec::new();
    let mut hashes = Vec::new();
    let mut maxent_hashes = Vec::new();
    let mut sample_hashes = Vec::new();
    let mut last_set = None;
    for r in [namefind::PERSON, namefind::LOCATION, namefind::ORGANIZATION] {
        let held = r.held_out(&split.held_out)?;
        let old = ModelGateRecord::read(&out.join(r.kind()).join("gate-maxent.json"))?;
        old.check()?;
        if old.set.samples_sha256 != held.sha256(&r.types)
            || old.set.corpus_sha256 != digest.sha256
            || predictions.len() != held.len()
        {
            return Err("saved gate and predictions do not use the same held-out split".into());
        }
        let mut units = Vec::new();
        for (sample, names) in held.samples().iter().zip(&predictions) {
            let gold: Vec<_> = sample.names().iter().map(|n| LabeledSpan::new(n.start, n.end, n.kind)).collect();
            let mut selected = Vec::new();
            for (a, b, kind) in names {
                if a >= b || *b as usize > sample.tokens().len() {
                    return Err("predicted token span is out of bounds".into());
                }
                if !["person", "location", "organization", "misc"].contains(&kind.as_str()) {
                    return Err("unexpected prediction type".into());
                }
                if kind == r.kind() {
                    let span = LabeledSpan::new(*a, *b, Some(kind.as_str()));
                    if selected.contains(&span) {
                        return Err("duplicate predicted span".into());
                    }
                    selected.push(span);
                }
            }
            let mut score = FMeasure::default();
            score.update(&gold, &selected);
            units.push(spans::counts(&score));
        }
        let crf: Vec<_> = old.units.iter().map(|u| u.candidate).collect();
        let maxent: Vec<_> = old.units.iter().map(|u| u.reference).collect();
        for (label, id, counts) in
            [("crf", old.candidate.clone(), crf.clone()), ("maxent", old.reference.clone(), maxent.clone())]
        {
            let gate = ModelGateRecord::new(
                Task::NameFinder,
                old.set.clone(),
                id,
                model.clone(),
                spans::pair(counts, units.clone()),
            )?;
            gate.write(&out.join(r.kind()).join(format!("{label}-vs-dl.json")))?;
            println!("{gate}");
        }
        let scores = serde_json::json!({"crf":full::totals(&crf),"maxent":full::totals(&maxent),"namefind_dl":full::totals(&units),"prediction_file_sha256":Sha256::of(&std::fs::read(&args[2])?).to_string()});
        std::fs::write(out.join(r.kind()).join("dl-scores.json"), serde_json::to_vec_pretty(&scores)?)?;
        full::add(&mut combined_crf, &crf)?;
        full::add(&mut combined_maxent, &maxent)?;
        full::add(&mut combined_dl, &units)?;
        hashes.push(old.candidate.sha256);
        maxent_hashes.push(old.reference.sha256);
        sample_hashes.push(old.set.samples_sha256.clone());
        last_set = Some(old.set);
    }
    let mut set = last_set.ok_or("no shared entity types")?;
    set.name = "MASC 3.0.0 held-out split, shared person/location/organization types".into();
    set.samples_sha256 = Sha256::of(sample_hashes.join("\n").as_bytes()).to_string();
    let types =
        spans::SpanTypes { types: &["person", "location", "organization"], renamed: &[("org", "organization")] };
    let dates = namefind::DATE.held_out(&split.held_out)?;
    set.scoring = types.scoring(combined_dl.iter().map(|c: &Counts| c.target).sum(), dates.counts().names, Vec::new());
    for (label, hashes, counts) in [("crf", hashes, &combined_crf), ("maxent", maxent_hashes, &combined_maxent)] {
        let id = ModelId {
            id: format!("masc-{label}-shared-collection"),
            sha256: Sha256::of(hashes.join("\n").as_bytes()).to_string(),
        };
        let gate = ModelGateRecord::new(
            Task::NameFinder,
            set.clone(),
            id,
            model.clone(),
            spans::pair(counts.clone(), combined_dl.clone()),
        )?;
        gate.write(&out.join(format!("shared-{label}-vs-dl.json")))?;
        println!("{gate}");
    }
    let summary = serde_json::json!({"crf":full::totals(&combined_crf),"maxent":full::totals(&combined_maxent),"namefind_dl":full::totals(&combined_dl),"unsupported":"date"});
    std::fs::write(out.join("shared-dl-scores.json"), serde_json::to_vec_pretty(&summary)?)?;
    Ok(())
}
