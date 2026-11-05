//! Native NER commands. Bundle assembly and transport remain shared services.
use std::path::{Path, PathBuf};

use serde::Serialize;
use turbo_analysis_embed::EncoderOptions;

use crate::cli::{Args, OutDir, commit, encoder_args};
use crate::corpus::read_file;
use crate::ner_compare::{self, PredictionFile, Request};
use crate::{Error, Result, ner_predict, peer, read_corpus, run};

pub fn predict(corpus: &str, dir: &Path, unrecorded: bool, mut args: Args) -> Result<()> {
    let bundle = PathBuf::from(args.required("--bundle")?);
    let head = args.required("--head")?;
    let destination = args.required("--out")?;
    let commit = commit(args.value("--commit")?);
    if args.value("--pooling")?.is_some() {
        return Err(Error("predict-ner uses the trained head's pooling; --pooling cannot override it".into()));
    }
    let (device, precision, _) = encoder_args(&mut args)?;
    args.done()?;
    let corpus = read_corpus(corpus, dir, unrecorded)?;
    let prediction = ner_predict::predict(
        &corpus,
        &bundle,
        &head,
        &EncoderOptions { device, precision, ..EncoderOptions::default() },
        &commit,
    )?;
    let mut out = OutDir::claim(&destination)?;
    run::write(&out.path, "predictions.bio", prediction.bio.as_bytes())?;
    run::write(&out.path, "prediction.json", &serde_json::to_vec_pretty(&prediction.record).map_err(json_error)?)?;
    out.keep = true;
    println!("{}: {} test sentences predicted", corpus.record.id, corpus.splits.test.len());
    Ok(())
}

#[derive(Serialize)]
struct ComparisonRecord {
    schema: &'static str,
    commit: String,
    peer: String,
    candidate: ner_predict::PredictionRecord,
    comparison: ner_compare::Report,
}

pub fn compare(corpus: &str, dir: &Path, unrecorded: bool, mut args: Args) -> Result<()> {
    let candidate = PathBuf::from(args.required("--predictions")?);
    let peer_file = PathBuf::from(args.required("--peer-predictions")?);
    let candidate_record = PathBuf::from(args.required("--prediction-record")?);
    let name = args.required("--peer")?;
    if name.trim().is_empty() {
        return Err(Error("--peer requires a system/version name".into()));
    }
    let types: Vec<String> = args.required("--types")?.split(',').map(str::to_owned).collect();
    let mapping = args.value("--map")?.map(|s| peer::parse_map(&s)).transpose()?.unwrap_or_default();
    let destination = args.required("--out")?;
    let commit = commit(args.value("--commit")?);
    let seed = args.parsed("--seed", turbo_trainer::Config::default().seed)?;
    let resamples = args.parsed("--resamples", 1000usize)?;
    args.done()?;
    let corpus = read_corpus(corpus, dir, unrecorded)?;
    let (candidate_name, candidate_bytes) = prediction_file(&candidate)?;
    let (peer_name, peer_bytes) = prediction_file(&peer_file)?;
    let record_bytes = std::fs::read(&candidate_record).map_err(|e| Error(format!("prediction record: {e}")))?;
    let candidate_record: ner_predict::PredictionRecord = serde_json::from_slice(&record_bytes).map_err(json_error)?;
    if candidate_record.schema != "turboembed.ner.prediction/1"
        || candidate_record.corpus != corpus.record
        || candidate_record.predictions.sha256 != turbo::bundle::sha256_hex(&candidate_bytes)
        || candidate_record.predictions.size != candidate_bytes.len() as u64
        || candidate_record.head.labels != corpus.labels()
        || candidate_record.head.scheme != turbo::manifest::Scheme::Bio
        || candidate_record.commit.trim().is_empty()
    {
        return Err(Error("candidate prediction receipt does not match corpus, schema or bytes".into()));
    }
    candidate_record.head.validate("prediction head").map_err(|e| Error(e.message))?;
    let comparison = ner_compare::compare(Request {
        corpus: &corpus.record,
        gold: &corpus.splits.test,
        train: &corpus.splits.train,
        candidate: PredictionFile { name: &candidate_name, bytes: &candidate_bytes },
        peer: PredictionFile { name: &peer_name, bytes: &peer_bytes },
        selected_types: &types,
        peer_mapping: &mapping,
        seed,
        resamples,
    })?;
    let record = ComparisonRecord {
        schema: "turboembed.ner.comparison/1",
        commit,
        peer: name,
        candidate: candidate_record,
        comparison,
    };
    let mut out = OutDir::claim(&destination)?;
    run::write(&out.path, "comparison.json", &serde_json::to_vec_pretty(&record).map_err(json_error)?)?;
    out.keep = true;
    println!("{}: paired selected-type comparison written", corpus.record.id);
    Ok(())
}

fn json_error(error: serde_json::Error) -> Error {
    Error(format!("receipt: {error}"))
}

fn prediction_file(path: &Path) -> Result<(String, Vec<u8>)> {
    let name =
        path.file_name().and_then(|s| s.to_str()).ok_or_else(|| Error("prediction file needs a UTF-8 name".into()))?;
    let (_, bytes) = read_file(path.parent().unwrap_or(Path::new(".")), name)?;
    Ok((name.into(), bytes))
}
