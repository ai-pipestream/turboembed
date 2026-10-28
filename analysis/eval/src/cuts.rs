//! Cut dumps: where a run of the turbo-chunker cut each document of a
//! set, written beside its record so two runs (two backends, or two
//! precision tiers) can be checked for the same cuts afterwards with
//! `turbo_chunker::compare` (#172, #856).
//!
//! A dump is one file per set, `<dir>/<set>.cuts`: the magic line, a
//! JSON header (the set, the backend the run's model ran on, the
//! budget and the chunker's settings), then one entry per document in
//! the set's order: its id and a document graph holding only the
//! source, `Sentences`, `Boundaries` and `Chunks`, through the shared
//! layers' codecs. Every length is a little-endian `u32`.

use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use turbo_analysis::export::{Codecs, DocumentGraph, export, import};
use turbo_analysis::{Document, Execution, Layer};
use turbo_analysis_layers::{Boundaries, Chunks, Sentences, TokenVectorWindows, Vectors};
use turbo_chunker::Budget;
use turbo_chunker::compare::{Pricing, Report, Run, compare};

use crate::Result;
use crate::record::Setting;

const MAGIC: &[u8] = b"turbo-eval cuts 1\n";

/// The file a set's cuts are written to.
pub fn cuts_path(dir: &Path, set: &str) -> PathBuf {
    dir.join(format!("{set}.cuts"))
}

/// What a dump says about the run that made it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
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
    c
}

fn u32_len(n: usize, what: &str) -> Result<[u8; 4]> {
    u32::try_from(n).map(u32::to_le_bytes).map_err(|_| format!("{what} of {n} bytes: over a u32"))
}

/// Writes a set's dump, one document at a time. The file is made new:
/// one already there is never replaced.
pub struct Writer {
    path: PathBuf,
    out: std::io::BufWriter<fs::File>,
    codecs: Codecs,
    docs: u32,
}

impl Writer {
    pub fn create(dir: &Path, header: &Header) -> Result<Writer> {
        fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let path = cuts_path(dir, &header.set);
        let at = |e: std::io::Error| format!("{}: {e}", path.display());
        let file = fs::OpenOptions::new().write(true).create_new(true).open(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::AlreadyExists {
                format!("{}: a cut dump is never replaced", path.display())
            } else {
                at(e)
            }
        })?;
        let mut out = std::io::BufWriter::new(file);
        let json = serde_json::to_vec(header).map_err(|e| format!("cut dump header: {e}"))?;
        out.write_all(MAGIC).map_err(at)?;
        out.write_all(&u32_len(json.len(), "a header")?).map_err(at)?;
        out.write_all(&json).map_err(at)?;
        Ok(Writer { path, out, codecs: codecs(), docs: 0 })
    }

    /// Writes one document's cuts: `doc` is the document the chunker
    /// cut, holding at least `Sentences`, `Boundaries` and `Chunks`.
    pub fn document(&mut self, id: &str, doc: &Document) -> Result<()> {
        let bytes = graph(doc, &self.codecs)?.to_bytes().map_err(|e| format!("document {id}: {e}"))?;
        let at = |e: std::io::Error| format!("{}: {e}", self.path.display());
        self.out.write_all(&u32_len(id.len(), "a document id")?).map_err(at)?;
        self.out.write_all(id.as_bytes()).map_err(at)?;
        self.out.write_all(&u32_len(bytes.len(), "a document graph")?).map_err(at)?;
        self.out.write_all(&bytes).map_err(at)?;
        self.docs += 1;
        Ok(())
    }

    /// Flushes the file; returns its path and the documents written.
    pub fn finish(mut self) -> Result<(PathBuf, u32)> {
        self.out.flush().map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok((self.path, self.docs))
    }
}

/// The document graph without its vector layers: the sentence vectors
/// drift and cohesion read are most of a document's bytes, and the
/// comparison reads only `Sentences`, `Boundaries` and `Chunks` (and
/// the layers their codecs index, such as the structure). What the
/// remaining layers' provenance says they read from the vectors is
/// dropped with them.
fn graph(doc: &Document, codecs: &Codecs) -> Result<DocumentGraph> {
    let mut g = export(doc, codecs).map_err(|e| format!("cut dump: {e}"))?;
    let vectors = [<Vectors<Sentences> as Layer>::ID, <TokenVectorWindows as Layer>::ID];
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

struct Bytes<'a>(&'a [u8]);

impl<'a> Bytes<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.0.len() < n {
            return Err("cut dump: cut short".into());
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn chunk(&mut self) -> Result<&'a [u8]> {
        let n = u32::from_le_bytes(self.take(4)?.try_into().expect("four bytes"));
        self.take(n as usize)
    }
}

impl Dump {
    pub fn read(path: &Path) -> Result<Dump> {
        let at = |e: String| format!("{}: {e}", path.display());
        let all = fs::read(path).map_err(|e| at(e.to_string()))?;
        let mut b = Bytes(&all);
        if b.take(MAGIC.len()).map_err(at)? != MAGIC {
            return Err(at("not a cut dump".into()));
        }
        let header: Header = serde_json::from_slice(b.chunk().map_err(at)?).map_err(|e| at(e.to_string()))?;
        let codecs = codecs();
        let mut docs = Vec::new();
        while !b.0.is_empty() {
            let id = std::str::from_utf8(b.chunk().map_err(at)?).map_err(|e| at(e.to_string()))?.to_owned();
            let graph = DocumentGraph::from_bytes(b.chunk().map_err(at)?).map_err(|e| at(format!("{id}: {e}")))?;
            let doc = import(&graph, &codecs).map_err(|e| at(format!("{id}: {e}")))?;
            for (what, there) in [
                ("sentences", doc.get::<Sentences>().is_some()),
                ("boundaries", doc.get::<Boundaries>().is_some()),
                ("chunks", doc.get::<Chunks>().is_some()),
            ] {
                if !there {
                    return Err(at(format!("{id}: no {what} layer")));
                }
            }
            docs.push((id, doc));
        }
        Ok(Dump { header, docs })
    }
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
    let names = |dir: &Path| -> Result<Vec<String>> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "cuts"))
            .filter_map(|p| p.file_stem().and_then(|s| s.to_str()).map(str::to_owned))
            .collect();
        v.sort();
        Ok(v)
    };
    let (na, nb) = (names(a)?, names(b)?);
    if na != nb {
        return Err(format!("{} holds sets {na:?}, {} holds {nb:?}", a.display(), b.display()));
    }
    if na.is_empty() {
        return Err(format!("{}: no cut dumps (*.cuts)", a.display()));
    }
    na.iter().map(|s| compare_dumps(&Dump::read(&cuts_path(a, s))?, &Dump::read(&cuts_path(b, s))?)).collect()
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
