//! A sweep compares only compatible peer scores on the same gold test data.
use std::path::Path;

use crate::cached_ud::{CachedParams, resolve_bank};
use crate::corpus::{CorpusRecord, Task};
use crate::eval::Metrics;
use crate::fit_cache::CacheTask;
use crate::peer::Column;
use crate::receipt::{Receipt, SCHEMA};
use crate::{Error, Result};

use super::SweepPeer;

pub(super) fn read(path: &Path, p: &CachedParams, task: CacheTask) -> Result<SweepPeer> {
    let fail = |why: &str| Error(format!("{}: peer receipt {why}", path.display()));
    let bytes = std::fs::read(path).map_err(|e| fail(&e.to_string()))?;
    let r: Receipt = serde_json::from_slice(&bytes).map_err(|e| fail(&e.to_string()))?;
    if r.schema != SCHEMA || r.eval.split != "test" {
        return Err(fail("requires the current schema and test split"));
    }
    // A limited training prefix may change train counts, but never the
    // gold test corpus, column, release pins, split policy or test counts.
    let expected = gold_record(p, task)?;
    let mut corpus = r.corpus.clone();
    corpus.train = expected.train;
    if corpus != expected {
        return Err(fail("corpus, task or release differs from the sweep"));
    }
    let peer = r.peer.as_ref().ok_or_else(|| fail("has no peer identity"))?;
    if peer.name.trim().is_empty()
        || peer.predictions.path.is_empty()
        || peer.predictions.size == 0
        || peer.predictions.sha256.len() != 64
        || !peer.predictions.sha256.bytes().all(|b| b.is_ascii_hexdigit())
        || r.head.is_some()
        || r.encoder.is_some()
        || r.training.is_some()
        || r.config.is_some()
    {
        return Err(fail("requires an identified peer and pinned predictions, not a trained-head receipt"));
    }
    let column = expected_column(&expected, task).ok_or_else(|| fail("unsupported metric for this head"))?;
    if peer.column != column {
        return Err(fail("column/metric differs from the sweep"));
    }
    let metric = r.eval.model.as_ref().ok_or_else(|| fail("has no eval.model score"))?;
    let compatible =
        matches!((expected.task, metric), (Task::Tags, Metrics::Tags { .. }) | (Task::Bio, Metrics::Spans { .. }));
    if !compatible || !valid_score(metric.score()) {
        return Err(fail("model metric differs from the sweep or is outside [0, 1]"));
    }
    if let Metrics::Tags { words, correct, accuracy, .. } = metric {
        let score = if *words == 0 { 0.0 } else { *correct as f64 / *words as f64 };
        if *words != expected.test.words || correct > words || (accuracy - score).abs() > 1e-12 {
            return Err(fail("model score/counts do not describe the gold test split"));
        }
    }
    let uas = r.eval.uas.as_ref().map(|u| u.model);
    if uas.is_some() != (column == Column::Deps) || uas.is_some_and(|v| !valid_score(v)) {
        return Err(fail("UAS metric is missing, unexpected or outside [0, 1]"));
    }
    Ok(SweepPeer { name: peer.name.clone(), test: metric.score(), uas })
}

fn valid_score(v: f64) -> bool {
    v.is_finite() && (0.0..=1.0).contains(&v)
}

fn expected_column(corpus: &CorpusRecord, task: CacheTask) -> Option<Column> {
    match task {
        CacheTask::Lemma => Some(Column::Lemma),
        CacheTask::Parse => Some(Column::Deps),
        CacheTask::Tag if corpus.task == Task::Bio => Some(Column::Bio),
        CacheTask::Tag => {
            if corpus.id == crate::masc_pos::ID {
                Some(Column::Xpos)
            } else {
                crate::ud::find(&corpus.id).map(|(_, c)| match c {
                    crate::ud::Column::Upos => Column::Upos,
                    crate::ud::Column::Xpos => Column::Xpos,
                    crate::ud::Column::Feats => Column::Feats,
                })
            }
        }
        CacheTask::Split => None,
    }
}

// Read the gold corpus to establish its identity. Reject a mismatched peer
// before spending time fitting the first layer of a potentially long sweep.
fn gold_record(p: &CachedParams, task: CacheTask) -> Result<CorpusRecord> {
    match task {
        CacheTask::Tag => Ok(crate::read_corpus(&p.corpus_id, &p.corpus_dir, p.unrecorded)?.record),
        CacheTask::Lemma | CacheTask::Parse => {
            let bank = resolve_bank(&p.corpus_id)?;
            let (splits, files, recorded) = crate::ud::read_treebank(&bank, &p.corpus_dir, p.unrecorded)?;
            let head =
                if task == CacheTask::Lemma { crate::ud_heads::UdHead::Lemma } else { crate::ud_heads::UdHead::Deps };
            Ok(crate::ud_heads::record(&bank, head, &splits, files, recorded))
        }
        CacheTask::Split => Err(Error("peer receipt: surface-token split peers are not supported".into())),
    }
}
