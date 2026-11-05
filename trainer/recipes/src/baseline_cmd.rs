//! `baseline`: a corpus's majority baseline on its test split as a
//! receipt (LitBank coref's is the singleton one), and re-docred's
//! `--dependency-path` rule row (`train_relation::dependency_path`).

use std::path::Path;
use std::time::Instant;

use turbo_trainer_recipes::cli::{Args, OutDir, commit};
use turbo_trainer_recipes::eval::{Seen, evaluate};
use turbo_trainer_recipes::receipt::*;
use turbo_trainer_recipes::ud_heads::{UdHead, uas};
use turbo_trainer_recipes::{Result, coarse, read_corpus};

use super::{baseline, finish, peer_cmd, train_relation};

pub(super) fn baseline_cmd(argv: &[String], corpus_id: &str, dir: &Path, unrecorded: bool, mut a: Args) -> Result<()> {
    if a.flag("--dependency-path") {
        return train_relation::dependency_path(argv, corpus_id, dir, unrecorded, a);
    }
    let out = OutDir::claim(&a.required("--out")?)?;
    let commit = commit(a.value("--commit")?);
    a.done()?;
    let t = Instant::now();
    let (record, eval, read) = if corpus_id == turbo_trainer_recipes::litbank::ID {
        let coref = turbo_trainer_recipes::litbank::read_coref(dir, unrecorded)?;
        let read = t.elapsed().as_secs_f64();
        let base = turbo_trainer_recipes::litbank_eval::baseline(&coref.test);
        let baseline = turbo_trainer_recipes::litbank_eval::evaluate_coref(&coref.test, &base)?;
        (coref.record, EvalRecord::baseline_only(baseline), read)
    } else if let Some((bank, head)) = peer_cmd::ud_head_corpus(corpus_id) {
        let (c, base) = peer_cmd::treebank(bank, dir, unrecorded, head)?;
        let read = t.elapsed().as_secs_f64();
        let test = &c.splits.test;
        let baseline = evaluate(c.record.task, test, &base, &Seen::of(&c.splits.train), None);
        let mut eval = EvalRecord::baseline_only(baseline);
        eval.uas = (head == UdHead::Deps).then(|| Uas { model: 0.0, baseline: uas(test, &base) });
        (c.record, eval, read)
    } else {
        let c = read_corpus(corpus_id, dir, unrecorded)?;
        let read = t.elapsed().as_secs_f64();
        let base = baseline(&c.splits.train, &c.splits.test);
        let baseline = evaluate(c.record.task, &c.splits.test, &base, &Seen::of(&c.splits.train), coarse(corpus_id));
        (c.record, EvalRecord::baseline_only(baseline), read)
    };
    let t = Instant::now();
    let receipt = Receipt {
        schema: SCHEMA.into(),
        command: argv.to_vec(),
        commit,
        unix_time: unix_time(),
        corpus: record,
        encoder: None,
        head: None,
        peer: None,
        config: None,
        training: None,
        eval,
        timing_s: Timing { read, eval: t.elapsed().as_secs_f64(), ..Timing::default() },
        outputs: Vec::new(),
    };
    finish(out, receipt)
}
