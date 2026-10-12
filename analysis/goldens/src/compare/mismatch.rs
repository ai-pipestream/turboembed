//! What a comparison found: the first difference between a Rust layer and
//! its golden, typed, with a `Display` written for the person who reads a
//! failed test.

use std::fmt;

use turbo_analysis::Span;

use super::Tolerance;
use crate::format::GoldenAnchor;
use crate::hex::to_hex;

/// The first difference between a layer on a document and its golden.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
    /// The document's text is not the golden's input.
    Input { layer: String, expected_len: u32, found_len: usize, expected_sha256: [u8; 32], found_sha256: [u8; 32] },
    /// The golden file has no layer with this id.
    MissingGolden { layer: String, available: Vec<String> },
    /// The document has no data for the layer; `status` says why when it
    /// was skipped or failed.
    MissingLayer { layer: String, status: Option<String> },
    /// The golden layer is anchored differently from the layer type.
    Anchor { layer: String, expected: GoldenAnchor, found: GoldenAnchor },
    /// The golden layer's columns are not the layer type's, by name and
    /// type, both as `name:type`.
    Columns { layer: String, expected: Vec<String>, found: Vec<String> },
    /// The golden has a value column the Rust layer holds no values for,
    /// such as POS tags made without probabilities.
    NoValues { layer: String, column: &'static str },
    /// One side has more items. `extra` is the first item past the
    /// shorter side, from the longer one.
    Count { layer: String, rust: usize, golden: usize, extra: Box<Item> },
    /// Item `index` differs in `what`: `span`, `tokens`, or the name of a
    /// label or value column. Both items are reported whole.
    Item { layer: String, index: usize, what: &'static str, tolerance: Tolerance, rust: Box<Item>, golden: Box<Item> },
}

/// One item as a report shows it: what it covers in the input text and
/// what it holds. A field the layer does not have is `None`.
#[derive(Clone, Debug, PartialEq)]
pub struct Item {
    /// Its span in source bytes: its own for a layer on the text, its
    /// anchor item's for an aligned layer.
    pub span: Option<Span>,
    /// The text the span covers, when the span is inside the text and on
    /// character boundaries.
    pub text: Option<String>,
    /// For an entity, its tokens `[start, end)`.
    pub tokens: Option<(u32, u32)>,
    /// The label column's name and value: a tag, a lemma, a kind, a code.
    pub label: Option<(&'static str, String)>,
    /// The value column's name and value: a probability or confidence.
    pub value: Option<(&'static str, f64)>,
}

impl Mismatch {
    /// The layer, as the document names it (`opennlp:tokens`, or
    /// `opennlp:tokens@gold` for a named instance).
    pub fn layer(&self) -> &str {
        use Mismatch::*;
        match self {
            Input { layer, .. }
            | MissingGolden { layer, .. }
            | MissingLayer { layer, .. }
            | Anchor { layer, .. }
            | Columns { layer, .. }
            | NoValues { layer, .. }
            | Count { layer, .. }
            | Item { layer, .. } => layer,
        }
    }
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use Mismatch::*;
        match self {
            Input { layer, expected_len, found_len, expected_sha256, found_sha256 } => write!(
                f,
                "{layer}: the document's text is not the golden's input: {found_len} bytes with SHA-256 {}, \
                 expected {expected_len} bytes with SHA-256 {}",
                to_hex(found_sha256),
                to_hex(expected_sha256)
            ),
            MissingGolden { layer, available } => {
                write!(f, "{layer}: the golden has no such layer; it has [{}]", available.join(", "))
            }
            MissingLayer { layer, status: None } => write!(f, "{layer}: the document has no such layer"),
            MissingLayer { layer, status: Some(s) } => write!(f, "{layer}: the document's layer is {s}"),
            Anchor { layer, expected, found } => {
                write!(f, "{layer}: the golden layer is anchored to {found}, expected {expected}")
            }
            Columns { layer, expected, found } => write!(
                f,
                "{layer}: the golden layer has the columns [{}], expected [{}]",
                found.join(", "),
                expected.join(", ")
            ),
            NoValues { layer, column } => write!(
                f,
                "{layer}: the Rust layer has no probabilities, and the golden has a {column} column to compare them with"
            ),
            Count { layer, rust, golden, extra } => {
                let (longer, n) = if rust > golden { ("Rust", golden) } else { ("golden", rust) };
                write!(
                    f,
                    "{layer}: {rust} items in Rust, {golden} in the golden; the first past the shorter side, \
                     {longer} item {n}:\n  {longer}: {extra}"
                )
            }
            Item { layer, index, what, tolerance, rust, golden } => {
                write!(f, "{layer} item {index}: the {what} differs")?;
                if let (Some((name, a)), Some((_, b))) = (rust.value, golden.value)
                    && name == *what
                {
                    write!(f, " (by {:e}, tolerance {tolerance})", (a - b).abs())?;
                }
                write!(f, "\n  rust:   {rust}\n  golden: {golden}")
            }
        }
    }
}

impl fmt::Display for Item {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut sep = "";
        if let Some(s) = self.span {
            write!(f, "bytes {}..{}", s.start, s.end)?;
            match &self.text {
                Some(t) => write!(f, " {t:?}")?,
                None => f.write_str(" (outside the text or not on character boundaries)")?,
            }
            sep = ", ";
        }
        if let Some((a, b)) = self.tokens {
            write!(f, "{sep}tokens {a}..{b}")?;
            sep = ", ";
        }
        if let Some((name, l)) = &self.label {
            write!(f, "{sep}{name} {l:?}")?;
            sep = ", ";
        }
        if let Some((name, v)) = self.value {
            write!(f, "{sep}{name} {:016x} ({v:?})", v.to_bits())?;
        }
        Ok(())
    }
}
