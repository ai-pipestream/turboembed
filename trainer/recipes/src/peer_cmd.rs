//! `export` and `peer`: a tag corpus's or treebank's test words out as
//! CoNLL-U for a peer system to tag, and the peer's tags, lemmas or
//! parses back in, scored as `train`, `train-lemma` and `train-parse`
//! score a head.

use std::path::{Path, PathBuf};
use std::time::Instant;

use turbo_trainer_recipes::corpus::{Corpus, Splits, Task, read_file};
use turbo_trainer_recipes::eval::{Seen, bootstrap, evaluate};
use turbo_trainer_recipes::peer::{Column, PeerRecord, apply, conllu, parse_map, read};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::ud::{TREEBANKS, read_treebank};
use turbo_trainer_recipes::ud_heads::{UdHead, deps_baseline, lemma_baseline, record, uas};
use turbo_trainer_recipes::{Error, Result, read_corpus};

use super::{Args, OutDir, RESAMPLES, baseline, commit, finish};

pub(super) fn export(corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    let out = PathBuf::from(a.required("--out")?);
    a.done()?;
    let c = if TREEBANKS.iter().any(|b| b.id == corpus_id) {
        treebank(corpus_id, dir, unrecorded, UdHead::Lemma)?.0
    } else {
        read_corpus(corpus_id, dir, unrecorded)?
    };
    let text = match c.record.task {
        Task::Tags | Task::Coref => conllu(&c.splits.test),
        Task::Bio => turbo_trainer_recipes::ner_peer::export(&c.splits.test)?,
        Task::Relations => return Err(Error(format!("{corpus_id}: a relation corpus has no words to export"))),
    };
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
    let (c, base) = match column {
        Column::Lemma => treebank(corpus_id, dir, unrecorded, UdHead::Lemma)?,
        Column::Deps => treebank(corpus_id, dir, unrecorded, UdHead::Deps)?,
        Column::Bio => {
            let c = read_corpus(corpus_id, dir, unrecorded)?;
            let base = baseline(&c.splits.train, &c.splits.test);
            (c, base)
        }
        Column::Upos | Column::Xpos | Column::Feats => {
            if TREEBANKS.iter().any(|b| b.id == corpus_id) {
                let tag = format!("{column:?}").to_lowercase();
                return Err(Error(format!("{corpus_id}: --column {tag} reads the tag corpus {corpus_id}-{tag}")));
            }
            let c = read_corpus(corpus_id, dir, unrecorded)?;
            let base = baseline(&c.splits.train, &c.splits.test);
            (c, base)
        }
    };
    if (c.record.task == Task::Bio) != (column == Column::Bio) {
        return Err(Error(format!(
            "{corpus_id}: BIO corpora require --column bio; tag corpora require upos, xpos or feats"
        )));
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
    if c.record.task == Task::Bio {
        turbo_trainer_recipes::ner_peer::validate(corpus_id, &predicted)?;
    }
    let seen = Seen::of(&c.splits.train);
    let eval = EvalRecord {
        split: "test".into(),
        model: Some(evaluate(c.record.task, test, &predicted, &seen, turbo_trainer_recipes::coarse(corpus_id))),
        baseline: evaluate(c.record.task, test, &base, &seen, turbo_trainer_recipes::coarse(corpus_id)),
        comparison: Some(bootstrap(
            c.record.task,
            test,
            &predicted,
            &base,
            RESAMPLES,
            turbo_trainer::Config::default().seed,
        )),
        uas: (column == Column::Deps).then(|| Uas { model: uas(test, &predicted), baseline: uas(test, &base) }),
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

/// A treebank read for `head`: its sentences labelled as the head scores
/// them, and the head's baseline on test. Refuses an id that names no
/// treebank, as a column of `peer` it is the treebank's (`ud-es-ancora`).
fn treebank(id: &str, dir: &Path, unrecorded: bool, head: UdHead) -> Result<(Corpus, Vec<Vec<String>>)> {
    let bank = TREEBANKS.iter().find(|b| b.id == id).ok_or_else(|| {
        Error(format!(
            "{id}: --column lemma and deps take a treebank id, one of {}",
            TREEBANKS.map(|b| b.id).join(", ")
        ))
    })?;
    let (ud, files, recorded) = read_treebank(bank, dir, unrecorded)?;
    let base = match head {
        UdHead::Lemma => lemma_baseline(&ud.train, &ud.test),
        UdHead::Deps => deps_baseline(&ud.train, &ud.test),
    };
    let splits =
        Splits { train: head.sentences(&ud.train), dev: head.sentences(&ud.dev), test: head.sentences(&ud.test) };
    Ok((Corpus { record: record(bank, head, &ud, files, recorded), splits }, base))
}
