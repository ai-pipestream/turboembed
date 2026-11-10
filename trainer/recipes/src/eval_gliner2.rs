//! `eval-gliner2`: native GLiNER2 extraction on the shared open-span recipe.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_gliner::{Decode, ExtractionSchema};
use turbo_schema_extract::model::Gliner2Extractor;
use turbo_trainer_recipes::corpus::Sentence;
use turbo_trainer_recipes::eval::{Majority, Seen, evaluate};
use turbo_trainer_recipes::open_span::{dictionary_baseline, filter_sentences};
use turbo_trainer_recipes::open_span_mix::{combine, evaluate_predictions, read_corpora, type_schema};
use turbo_trainer_recipes::peer::{Column, PeerRecord};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::run::replace;
use turbo_trainer_recipes::{Error, Result, gliner_calibrate::Calibration, gliner_eval, gliner2_eval};

use super::{Args, OutDir, RESAMPLES, commit, finish};

const ID: &str = "fastino/gliner2-base-v1";
const REVISION: &str = "f9634218e53580c56edf0de97ca1a7d3f1c2354e";

pub(super) fn evaluate_command(
    argv: &[String],
    corpus_id: &str,
    dir: &Path,
    unrecorded: bool,
    mut a: Args,
) -> Result<()> {
    let package = PathBuf::from(a.required("--package")?);
    let hold_out_arg = a.required("--hold-out")?;
    let threshold: f32 = a.parsed("--threshold", 0.5)?;
    let calibrate = a.flag("--calibrate");
    let calibration_path = a.value("--thresholds")?.map(PathBuf::from);
    let guidelines = a
        .value("--guidelines")?
        .map(|path| {
            std::fs::read(&path).map_err(|e| Error(format!("{path}: {e}"))).and_then(|bytes| {
                serde_json::from_slice::<BTreeMap<String, String>>(&bytes).map_err(|e| Error(format!("{path}: {e}")))
            })
        })
        .transpose()?;
    let split = a.value("--split")?.unwrap_or_else(|| "test".into());
    let seed: u64 = a.parsed("--seed", 1)?;
    let threads: usize = a.parsed("--threads", 1)?;
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    ExtractionSchema::new(vec!["entity".into()], threshold, Decode::Flat).map_err(Error)?;
    if threads == 0 {
        return Err(Error("--threads must be at least 1".into()));
    }
    if !matches!(split.as_str(), "dev" | "test") {
        return Err(Error("--split must be dev or test".into()));
    }
    if calibrate && split != "dev" {
        return Err(Error("--calibrate is restricted to --split dev".into()));
    }
    if calibrate && calibration_path.is_some() {
        return Err(Error("--calibrate and --thresholds are mutually exclusive".into()));
    }

    let t = Instant::now();
    let corpora = read_corpora(corpus_id, dir, unrecorded)?;
    let (all_types, _) = type_schema(&corpora)?;
    if let Some(guidelines) = &guidelines {
        let unknown: Vec<&str> =
            guidelines.keys().filter(|label| !all_types.contains(*label)).map(String::as_str).collect();
        if !unknown.is_empty() {
            return Err(Error(format!("--guidelines names labels outside the corpus: {unknown:?}")));
        }
    }
    let schema_digest = gliner_eval::digest(&all_types, threshold, guidelines.as_ref());
    let hold_out: Vec<String> = hold_out_arg.split(',').map(|x| x.trim().to_owned()).collect();
    if hold_out.iter().any(|h| h.is_empty() || !all_types.contains(h)) {
        return Err(Error(format!("--hold-out must name corpus entity types, got {hold_out:?}")));
    }
    let held_schema_digest = gliner_eval::digest(&hold_out, threshold, guidelines.as_ref());
    let seen_types: Vec<String> = all_types.iter().filter(|kind| !hold_out.contains(*kind)).cloned().collect();
    if seen_types.is_empty() {
        return Err(Error("--hold-out leaves no seen entity types".into()));
    }
    let combined = combine(&corpora)?;
    let evaluation = match split.as_str() {
        "dev" => &combined.splits.dev,
        "test" => &combined.splits.test,
        _ => unreachable!(),
    };
    let read = t.elapsed().as_secs_f64();

    let t = Instant::now();
    let model = open_model(&package)?;
    let (calibration, raw) = if calibrate {
        let (all, held) = std::thread::scope(|scope| {
            let all_task =
                scope.spawn(|| gliner2_eval::candidates(&model, evaluation, &all_types, guidelines.as_ref()));
            let held = gliner2_eval::candidates(&model, evaluation, &hold_out, guidelines.as_ref())?;
            let all = all_task.join().map_err(|_| Error("a GLiNER2 calibration worker panicked".into()))??;
            Ok::<_, Error>((all, held))
        })?;
        let mut calibration = Calibration::fit(evaluation, &all_types, &all, threshold, schema_digest.clone())?;
        calibration.held_thresholds =
            Some(Calibration::fit(evaluation, &hold_out, &held, threshold, held_schema_digest.clone())?.thresholds);
        calibration.held_schema = Some(held_schema_digest.clone());
        (calibration, Some(Reuse { all, held }))
    } else if let Some(path) = calibration_path {
        let calibration = Calibration::read(
            &std::fs::read(&path).map_err(|e| Error(format!("{}: {e}", path.display())))?,
            &all_types,
            &hold_out,
            &schema_digest,
            &held_schema_digest,
        )?;
        (calibration, None)
    } else {
        (Calibration::uniform(&all_types, schema_digest.clone(), threshold), None)
    };
    let calibration_bytes = calibration.bytes()?;
    let calibration_digest = turbo::bundle::sha256_hex(&calibration_bytes);
    let identity = model.model();
    let peer_name = format!(
        "gliner2:{}@{}#{};threshold={threshold};decode=flat;schema={schema_digest};calibration={calibration_digest}",
        identity.id, identity.revision, identity.sha256
    );

    let predictions = predict_partitioned(
        &package,
        model,
        evaluation,
        &all_types,
        &hold_out,
        &seen_types,
        threshold,
        threads,
        guidelines.as_ref(),
        &calibration,
        raw,
    )?;
    let Predictions { all: predicted, held, seen: predicted_seen } = predictions;
    let overall = evaluate_predictions(
        &combined,
        &all_types,
        &seen_types,
        &hold_out,
        &predicted,
        Some(&held),
        &split,
        RESAMPLES,
        seed,
    )?;
    let mut datasets = Vec::with_capacity(corpora.len());
    let mut at = 0;
    for corpus in &corpora {
        let corpus_evaluation = match split.as_str() {
            "dev" => &corpus.splits.dev,
            "test" => &corpus.splits.test,
            _ => unreachable!(),
        };
        let end = at + corpus_evaluation.len();
        datasets.push(evaluate_predictions(
            corpus,
            &all_types,
            &seen_types,
            &hold_out,
            &predicted[at..end],
            Some(&held[at..end]),
            &split,
            RESAMPLES,
            seed,
        )?);
        at = end;
    }

    let train_filtered = filter_sentences(&combined.splits.train, &seen_types);
    let majority = Majority::train(&train_filtered);
    let base: Vec<Vec<String>> = evaluation.iter().map(|s| majority.predict(&s.words)).collect();
    let evaluation_seen = filter_sentences(evaluation, &seen_types);
    let seen = Seen::of(&train_filtered);
    let dict_seen = dictionary_baseline(&evaluation_seen, &seen_types);
    let seen_model = evaluate(combined.record.task, &evaluation_seen, &predicted_seen, &seen, None);
    let seen_baseline = evaluate(combined.record.task, &evaluation_seen, &base, &seen, None);
    let seen_dictionary = evaluate(combined.record.task, &evaluation_seen, &dict_seen, &seen, None);
    let eval = EvalRecord {
        split: split.clone(),
        model: Some(overall.model.clone()),
        baseline: overall.baseline.clone(),
        dictionary: Some(overall.dictionary.clone()),
        open_span: Some(OpenSpanEvalRecord {
            seen_types,
            held_out_types: hold_out,
            seen_model,
            seen_baseline,
            seen_dictionary,
            held_out_model: overall.held_out_model.clone(),
            held_out_baseline: overall.held_out_baseline.clone(),
            held_out_dictionary: overall.held_out_dictionary.clone(),
            dictionary_comparison: Some(overall.dictionary_comparison.clone()),
            held_out_comparison: overall.held_out_comparison.clone(),
            held_out_dictionary_comparison: overall.held_out_dictionary_comparison.clone(),
            datasets,
        }),
        comparison: Some(overall.comparison.clone()),
        uas: None,
        rules: None,
        against: None,
        by_language: Vec::new(),
    };

    let prediction_bytes = super::eval_gliner::prediction_text(evaluation, &predicted)?.into_bytes();
    let checked = turbo_trainer_recipes::ner_peer::read("predictions.bio", &prediction_bytes, evaluation)?;
    assert_eq!(checked, predicted);
    let prediction_record = replace(&out.path, "predictions.bio", &prediction_bytes)?;
    let calibration_record = replace(&out.path, "thresholds.json", &calibration_bytes)?;
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: combined.record.clone(),
        encoder: None,
        head: None,
        peer: Some(PeerRecord {
            name: peer_name,
            column: Column::Bio,
            predictions: prediction_record.clone(),
            map: BTreeMap::new(),
        }),
        config: None,
        training: None,
        eval,
        timing_s: Timing { read, eval: t.elapsed().as_secs_f64(), ..Timing::default() },
        outputs: vec![prediction_record, calibration_record],
    };
    finish(out, receipt)
}

#[derive(Default)]
struct Predictions {
    all: Vec<Vec<String>>,
    held: Vec<Vec<String>>,
    seen: Vec<Vec<String>>,
}

struct Reuse {
    all: Vec<Vec<turbo_gliner::Candidate>>,
    held: Vec<Vec<turbo_gliner::Candidate>>,
}

fn open_model(package: &Path) -> Result<Gliner2Extractor> {
    Gliner2Extractor::load(&package.join("encoder"), package, ID, REVISION).map_err(Error)
}

#[allow(clippy::too_many_arguments)]
fn predict_three(
    model: &Gliner2Extractor,
    sentences: &[Sentence],
    all_types: &[String],
    hold_out: &[String],
    seen_types: &[String],
    threshold: f32,
    guidelines: Option<&BTreeMap<String, String>>,
    calibration: &Calibration,
    include_all: bool,
    include_held: bool,
) -> Result<Predictions> {
    let predict = |labels: &[String], held: bool| {
        let raw = gliner2_eval::candidates(model, sentences, labels, guidelines)?;
        let thresholds = if held { calibration.held_values(labels)? } else { calibration.values(labels)? };
        gliner_eval::decode_candidates(sentences, labels, &raw, &thresholds)
    };
    let _ = threshold;
    Ok(Predictions {
        all: if include_all { predict(all_types, false)? } else { Vec::new() },
        held: if include_held { predict(hold_out, true)? } else { Vec::new() },
        seen: predict(seen_types, false)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn predict_partitioned(
    package: &Path,
    model: Gliner2Extractor,
    sentences: &[Sentence],
    all_types: &[String],
    hold_out: &[String],
    seen_types: &[String],
    threshold: f32,
    threads: usize,
    guidelines: Option<&BTreeMap<String, String>>,
    calibration: &Calibration,
    raw: Option<Reuse>,
) -> Result<Predictions> {
    if sentences.is_empty() {
        return Ok(Predictions::default());
    }
    let (include_all, include_held) = (raw.is_none(), raw.is_none());
    let chunk_size = sentences.len().div_ceil(threads);
    let mut chunks = sentences.chunks(chunk_size);
    let first = chunks.next().expect("a nonempty split has one chunk");
    let mut out = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in chunks {
            handles.push(scope.spawn(move || {
                let worker = open_model(package)?;
                predict_three(
                    &worker,
                    chunk,
                    all_types,
                    hold_out,
                    seen_types,
                    threshold,
                    guidelines,
                    calibration,
                    include_all,
                    include_held,
                )
            }));
        }
        let mut parts = vec![predict_three(
            &model,
            first,
            all_types,
            hold_out,
            seen_types,
            threshold,
            guidelines,
            calibration,
            include_all,
            include_held,
        )?];
        for handle in handles {
            parts.push(handle.join().map_err(|_| Error("a GLiNER2 evaluation worker panicked".into()))??);
        }
        let mut out = Predictions::default();
        for part in parts {
            out.all.extend(part.all);
            out.held.extend(part.held);
            out.seen.extend(part.seen);
        }
        Ok::<_, Error>(out)
    })?;
    if let Some(raw) = raw {
        out.all = gliner_eval::decode_candidates(sentences, all_types, &raw.all, &calibration.values(all_types)?)?;
        out.held = gliner_eval::decode_candidates(sentences, hold_out, &raw.held, &calibration.held_values(hold_out)?)?;
    }
    Ok(out)
}
