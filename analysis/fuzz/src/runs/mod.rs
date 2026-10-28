//! The Rust side of every golden run the differential covers: for a run
//! name as the generator writes it (`<annotator>/<model>`), the annotators
//! that reproduce it, how a text is annotated with them, and how the
//! result is compared with the golden the JVM wrote for the same text.
//!
//! The comparison is the one the crates' golden tests make:
//! `turbo_goldens::matches_golden` on the layers, and the rules of
//! `turbo_goldens::expect` where the port differs from the JVM by design
//! (a tokenizer split inside a surrogate pair, the keep new lines defects
//! of `SimpleTokenizer`, WordPiece spans over stripped marks, edits
//! reconstructed by a diff). Each difference comes back as an `Outcome`:
//! a match, a documented difference, a divergence to look into, or a
//! golden that cannot be compared.

mod catalog;
mod confusables_17;
mod language;
mod normalize;
mod numeric;
mod sentences;
mod subwords;
mod tagging;
mod temporal;
mod tokens;

use std::fmt;

use turbo_analysis::{Analyzer, Document};
use turbo_analysis_layers::Tokens;
use turbo_goldens::{Golden, Mismatch};

pub use catalog::{NOT_COVERED, RUNS, open};

/// What comparing one golden with the Rust side found.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The layers are the same, after the rules the golden tests apply.
    Match,
    /// A difference the crate's documentation records as deliberate, with
    /// which rule or section explains it.
    Documented(String),
    /// A difference nothing explains: a bug in the port, or a JVM
    /// behaviour the port does not reproduce yet.
    Divergence(String),
    /// A golden that cannot be compared: a span of it cuts a surrogate
    /// pair where the Rust layer has no counterpart, a file that does not
    /// read, a configuration the run does not take.
    Skipped(String),
}

impl Outcome {
    pub fn is_divergence(&self) -> bool {
        matches!(self, Outcome::Divergence(_))
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::Match => write!(f, "match"),
            Outcome::Documented(what) => write!(f, "documented: {what}"),
            Outcome::Divergence(what) => write!(f, "divergence: {what}"),
            Outcome::Skipped(what) => write!(f, "skipped: {what}"),
        }
    }
}

/// The Rust side failing on a text: the message and, for the lattice
/// tokenizer's "no candidate", the byte offset it failed at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RustError {
    pub message: String,
    pub offset: Option<usize>,
}

impl fmt::Display for RustError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

/// The result of checking one text against its golden: the document the
/// Rust side made, when it made one, and the outcome.
pub struct Checked {
    pub doc: Option<Document>,
    pub outcome: Outcome,
}

/// One run: what the JVM ran as `<annotator>/<model>`, on the Rust side.
pub trait Run: Send + Sync {
    /// Annotates `text`. The runs that take the JVM's own input layers
    /// (the taggers, the lemmatizers, the chunker) load them from
    /// `golden` when it is given, so the comparison is of their decisions
    /// alone, and make them with the Rust pipeline otherwise.
    fn annotate(&self, text: &str, golden: Option<&Golden>) -> Result<Document, RustError>;

    /// Compares the document `annotate` made of the golden's input with
    /// the golden.
    fn compare(&self, doc: &Document, golden: &Golden) -> Outcome;

    /// Annotates and compares. A run whose JVM side can throw overrides
    /// this to compare the error with the Rust side's.
    fn check(&self, text: &str, golden: &Golden) -> Checked {
        match self.annotate(text, Some(golden)) {
            Ok(doc) => {
                let outcome = match jvm_threw(golden) {
                    Some(what) => what,
                    None => self.compare(&doc, golden),
                };
                Checked { doc: Some(doc), outcome }
            }
            Err(e) => Checked { doc: None, outcome: Outcome::Divergence(format!("the port failed: {e}")) },
        }
    }
}

/// A divergence when the golden records an exception on the JVM, since
/// the Rust side returned a result.
pub(crate) fn jvm_threw(golden: &Golden) -> Option<Outcome> {
    let e = golden.header.error.as_ref()?;
    Some(Outcome::Divergence(format!("the JVM threw {}: {}; the port returned a result", e.class, e.message)))
}

/// Runs `analyzer` over a document of `text`.
pub(crate) fn analyze(analyzer: &Analyzer, text: &str) -> Result<Document, RustError> {
    let mut docs = vec![Document::new(text)];
    for result in analyzer.run(&mut docs) {
        result.map_err(|e| RustError { message: e.to_string(), offset: None })?;
    }
    Ok(docs.remove(0))
}

/// Runs `analyzer` over a document that holds layers already.
pub(crate) fn analyze_document(analyzer: &Analyzer, doc: Document) -> Result<Document, RustError> {
    let mut docs = vec![doc];
    for result in analyzer.run(&mut docs) {
        result.map_err(|e| RustError { message: e.to_string(), offset: None })?;
    }
    Ok(docs.remove(0))
}

/// The outcome of a `matches_golden` result.
pub(crate) fn outcome(result: Result<(), Mismatch>) -> Outcome {
    match result {
        Ok(()) => Outcome::Match,
        Err(m) => Outcome::Divergence(m.to_string()),
    }
}

/// The default token spans of `doc` as byte pairs.
pub(crate) fn token_spans(doc: &Document) -> Vec<(u32, u32)> {
    doc.get::<Tokens>().map_or_else(Vec::new, |t| t.spans.iter().map(|(_, s)| (s.start, s.end)).collect())
}

/// Whether the golden's spans of `layer` include one that cuts a
/// surrogate pair, which its `turbo:utf16<layer>` block says.
pub(crate) fn has_cut(golden: &Golden, layer: &str) -> bool {
    golden.layer(&format!("turbo:utf16<{layer}>")).is_some()
}

/// The value of the golden's `config` record `key`.
pub(crate) fn config<'g>(golden: &'g Golden, key: &str) -> Option<&'g str> {
    golden.header.config.iter().find(|c| c.key == key).map(|c| c.value.as_str())
}

/// A run name split at its `+` options: `opennlp-en-...+beam1` gives the
/// model and `["beam1"]`.
pub(crate) fn split_options(run: &str) -> (&str, Vec<&str>) {
    let mut parts = run.split('+');
    let base = parts.next().unwrap_or("");
    (base, parts.collect())
}
