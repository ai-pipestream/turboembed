//! Cut dumps: where a run of the turbo-chunker cut each document of a
//! set, written beside its record so two runs (two backends, or two
//! precision tiers) can be checked for the same cuts afterwards with
//! `turbo_chunker::compare` (#172, #856).
//!
//! Version 2 has a bounded binary header, framed document graphs and a
//! required document-count/SHA-256 footer. Version 1 must be regenerated.
//! Graphs retain source text and non-vector layers; this is not encrypted
//! storage (security design is tracked separately in #1490).
//! Comparison means the chunker's cut-start identity, not complete range
//! coverage or retrieval-quality equivalence.

use std::fmt;
#[cfg(test)]
use std::fs;
use std::path::{Path, PathBuf};

use turbo_analysis::export::{Codecs, DocumentGraph, export};
use turbo_analysis::{Document, Execution, Layer};
use turbo_analysis_layers::{Boundaries, Chunks, Sentences, Subwords, TokenVectorWindows, Vectors};
use turbo_chunker::Budget;
use turbo_chunker::compare::{Pricing, Report, Run, compare};

use crate::Result;
use crate::record::Setting;

mod codec;
use crate::artifact::{ArtifactKey, EvaluationArtifactStore, TemporaryFileStore};
pub use codec::Writer;

pub struct DumpConfig {
    pub store: Box<dyn EvaluationArtifactStore>,
    pub precision: turbo_analysis_embed::Precision,
}

fn key(set: &str) -> Result<ArtifactKey> {
    ArtifactKey::new(format!("{set}.cuts")).map_err(|e| e.to_string())
}

/// The file a set's cuts are written to.
pub fn cuts_path(dir: &Path, set: &str) -> PathBuf {
    dir.join(format!("{set}.cuts"))
}

/// What a dump says about the run that made it.
#[derive(Debug, Clone, PartialEq)]
pub struct Header {
    pub set: String,
    pub sha256: String,
    /// Where the embedder ran: the model behind every vector the
    /// chunker's costs read.
    pub backend: String,
    pub device: String,
    pub precision: String,
    pub max: u32,
    pub target: u32,
    pub size_weight: u32,
    pub overhead: u32,
    pub chunker: Vec<Setting>,
}

impl Header {
    pub fn budget(&self) -> Budget {
        Budget { max: self.max, target: self.target, size_weight: self.size_weight, overhead: self.overhead }
    }

    fn execution(&self) -> Execution {
        Execution { backend: self.backend.clone(), device: self.device.clone(), precision: self.precision.clone() }
    }
}

fn codecs() -> Codecs {
    let mut c = Codecs::new();
    turbo_analysis_layers::export::register_codecs(&mut c);
    // The clause cuts a `--clauses` run splits over-budget sentences by.
    turbo_chunker::export::register_codecs(&mut c);
    c
}

/// The document graph without its vector layers: the sentence vectors
/// drift and cohesion read are most of a document's bytes, and the
/// comparison reads only `Sentences`, `Boundaries` and `Chunks` (and
/// the layers their codecs index, such as the structure). What the
/// remaining layers' provenance says they read from the vectors is
/// dropped with them.
fn graph(doc: &Document, codecs: &Codecs) -> Result<DocumentGraph> {
    let mut g = export(doc, codecs).map_err(|e| format!("cut dump: {e}"))?;
    let vectors = [<Vectors<Sentences> as Layer>::ID, <TokenVectorWindows as Layer>::ID, <Subwords as Layer>::ID];
    let dropped = |layer: &str| vectors.iter().any(|v| layer.starts_with(v));
    g.layers.retain(|l| !dropped(&l.layer));
    for l in &mut g.layers {
        l.provenance.reads.retain(|r| !dropped(&r.layer));
        l.provenance.on.retain(|r| !dropped(&r.layer));
    }
    Ok(g)
}

/// A set's dump, read back whole.
pub struct Dump {
    pub header: Header,
    /// Each document's id and its cut document, in the set's order.
    pub docs: Vec<(String, Document)>,
}

/// One document whose cuts differ, with the comparison's own report.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Differing {
    pub id: String,
    pub matched: usize,
    pub disagreements: Vec<turbo_chunker::compare::Disagreement>,
}

/// Two dumps of one set compared document by document.
#[derive(Debug, Clone, PartialEq)]
pub struct SetCuts {
    pub set: String,
    pub a: Execution,
    pub b: Execution,
    pub documents: usize,
    /// Cuts both runs made.
    pub matched: usize,
    pub differing: Vec<Differing>,
}

impl SetCuts {
    pub fn is_identical(&self) -> bool {
        self.differing.is_empty()
    }
}

/// Compares two dumps of the same set: the same bytes (SHA-256), the
/// same documents in the same order and the same budget, each run's
/// cuts priced with its own boundary costs.
pub fn compare_dumps(a: &Dump, b: &Dump) -> Result<SetCuts> {
    let (ha, hb) = (&a.header, &b.header);
    if ha.set != hb.set || ha.sha256 != hb.sha256 {
        return Err(format!("set {} ({}) against set {} ({})", ha.set, ha.sha256, hb.set, hb.sha256));
    }
    if ha.budget() != hb.budget() {
        return Err(format!("set {}: the runs were cut with different budgets", ha.set));
    }
    if a.docs.len() != b.docs.len() {
        return Err(format!("set {}: {} documents against {}", ha.set, a.docs.len(), b.docs.len()));
    }
    let (ea, eb) = (ha.execution(), hb.execution());
    let mut out = SetCuts {
        set: ha.set.clone(),
        a: ea.clone(),
        b: eb.clone(),
        documents: a.docs.len(),
        matched: 0,
        differing: Vec::new(),
    };
    for ((ida, da), (idb, db)) in a.docs.iter().zip(&b.docs) {
        if ida != idb {
            return Err(format!("set {}: document {ida} against {idb}", ha.set));
        }
        if da.source() != db.source() {
            return Err(format!("set {}: document {ida}: the texts differ", ha.set));
        }
        let (ra, rb) = (run(da, &ea, ha)?, run(db, &eb, hb)?);
        if da.text(ra.chunks.spans.view()) != db.text(rb.chunks.spans.view()) {
            return Err(format!("set {}: document {ida}: the anchored view texts differ", ha.set));
        }
        let report: Report<'_> = compare(ra, rb).map_err(|e| format!("set {}: document {ida}: {e}", ha.set))?;
        out.matched += report.matched;
        if !report.is_identical() {
            out.differing.push(Differing {
                id: ida.clone(),
                matched: report.matched,
                disagreements: report.disagreements,
            });
        }
    }
    Ok(out)
}

fn run<'a>(d: &'a Document, execution: &'a Execution, h: &Header) -> Result<Run<'a>> {
    let pricing = Pricing {
        sentences: d.get::<Sentences>().ok_or("no sentences")?,
        boundaries: d.get::<Boundaries>().ok_or("no boundaries")?,
        budget: h.budget(),
    };
    let chunks = d.get::<Chunks>().ok_or("no chunks")?;
    Ok(Run { chunks, execution: Some(execution), pricing: Some(pricing) })
}

/// The dumps of every set in both directories, by set name; a set in one
/// directory only is an error.
pub fn compare_dirs(a: &Path, b: &Path) -> Result<Vec<SetCuts>> {
    compare_stores(&TemporaryFileStore::new(a), &TemporaryFileStore::new(b))
}

pub fn compare_stores(a: &dyn EvaluationArtifactStore, b: &dyn EvaluationArtifactStore) -> Result<Vec<SetCuts>> {
    let names = |store: &dyn EvaluationArtifactStore| -> Result<Vec<ArtifactKey>> {
        let mut names: Vec<_> =
            store.list().map_err(|e| e.to_string())?.into_iter().filter(|k| k.as_str().ends_with(".cuts")).collect();
        names.sort();
        Ok(names)
    };
    let (na, nb) = (names(a)?, names(b)?);
    if na != nb {
        return Err(format!("stores holds sets {na:?} against {nb:?}"));
    }
    if na.is_empty() {
        return Err("no cut dumps (*.cuts)".into());
    }
    na.iter().map(|key| compare_dumps(&Dump::read_from(a, key)?, &Dump::read_from(b, key)?)).collect()
}

fn margin(m: Option<i128>) -> String {
    m.map_or_else(|| "-".into(), |m| m.to_string())
}

impl fmt::Display for SetCuts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{}: {} documents, {} shared cuts, {} documents differ ({}/{}/{} against {}/{}/{})",
            self.set,
            self.documents,
            self.matched,
            self.differing.len(),
            self.a.backend,
            self.a.device,
            self.a.precision,
            self.b.backend,
            self.b.device,
            self.b.precision
        )?;
        for d in &self.differing {
            for x in &d.disagreements {
                let starts = |c: &[turbo_chunker::compare::CutAt]| {
                    c.iter().map(|c| c.start.to_string()).collect::<Vec<_>>().join(",")
                };
                writeln!(
                    f,
                    "  {} level {} after {:?} before {:?}: A cuts at [{}], B at [{}], margin A {} B {}",
                    d.id,
                    x.level,
                    x.after,
                    x.before,
                    starts(&x.a),
                    starts(&x.b),
                    margin(x.margin_a),
                    margin(x.margin_b)
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "cuts_tests.rs"]
mod tests;
