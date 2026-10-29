//! `export` and `peer`: a tag corpus's test words out as CoNLL-U for a
//! peer system to tag, and the peer's tags back in, scored as `train`
//! scores a head.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_trainer_recipes::corpus::{Task, read_file};
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::peer::{Column, PeerRecord, apply, conllu, parse_map, read};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::{Error, Result, read_corpus};

use super::{Args, OutDir, RESAMPLES, baseline, commit, finish};

pub(super) fn export(corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let out = PathBuf::from(a.required("--out")?);
    a.done()?;
    let c = read_corpus(corpus_id, dir, unrecorded)?;
    if c.record.task != Task::Tags {
        return Err(Error(format!("{corpus_id}: export takes a tag corpus")));
    }
    let text = conllu(&c.splits.test);
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out)
        .and_then(|mut f| std::io::Write::write_all(&mut f, text.as_bytes()))
        .map_err(|e| Error(format!("{}: {e}", out.display())))?;
    println!("{corpus_id}: {} test sentences to {}", c.splits.test.len(), out.display());
    Ok(())
}

pub(super) fn peer(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let name = a.required("--peer")?;
    let pred = PathBuf::from(a.required("--predictions")?);
    let column: Column = a.parsed("--column", Column::Xpos)?;
    let map = a.value("--map")?.map(|m| parse_map(&m)).transpose()?.unwrap_or_default();
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    let t = Instant::now();
    let c = read_corpus(corpus_id, dir, unrecorded)?;
    if c.record.task != Task::Tags {
        return Err(Error(format!("{corpus_id}: a peer is scored on a tag corpus")));
    }
    let parent = pred.parent().unwrap_or(Path::new("."));
    let file =
        pred.file_name().and_then(|f| f.to_str()).ok_or_else(|| Error(format!("{}: no file name", pred.display())))?;
    let (record, bytes) = read_file(parent, file)?;
    let read_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let test = &c.splits.test;
    let mut predicted = read(file, &bytes, column, test)?;
    apply(&map, &mut predicted);
    let base = baseline(&c.splits.train, test);
    let seen = Seen::of(&c.splits.train);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(Task::Tags, test, &predicted, &seen, None)),
        baseline: evaluate(Task::Tags, test, &base, &seen, None),
        comparison: Some(bootstrap(
            Task::Tags,
            test,
            &predicted,
            &base,
            RESAMPLES,
            turbo_trainer::Config::default().seed,
        )),
        uas: None,
    };
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        host: host(),
        unix_time: unix_time(),
        corpus: c.record.clone(),
        encoder: None,
        head: None,
        peer: Some(PeerRecord { name, column, predictions: record, map }),
        config: None,
        training: None,
        eval,
        timing_s: Timing { read: read_s, eval: t.elapsed().as_secs_f64(), ..Timing::default() },
        outputs: Vec::new(),
    };
    finish(out, receipt)
}
