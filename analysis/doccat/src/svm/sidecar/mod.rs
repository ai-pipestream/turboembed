//! The sidecar: what an OpenNLP `SvmDoccatModel` holds besides its LIBSVM
//! model, in a line based text format.
//!
//! OpenNLP saves the model with Java serialization, which this port does
//! not read. The golden generator writes the LIBSVM part with
//! `svm_save_model` and the rest as this sidecar. One record per line,
//! fields separated by one tab:
//!
//! ```text
//! svm-doccat 1
//! language    eng
//! weighting   TF_IDF
//! selection   INFORMATION_GAIN    40
//! scaling     0.0 1.0
//! generator   bag-of-words
//! generator   ngram   2   3
//! category    0   sports
//! feature     1   1.791759469228055   0.0 2.0 bow=goal
//! ```
//!
//! - The first line is `svm-doccat 1`, the format and its version.
//! - `language` (at most once): the model's language code.
//! - `weighting`: the `TermWeightingStrategy` constant.
//! - `selection`: the `FeatureSelectionStrategy` constant and the maximum
//!   number of features; it records how the vocabulary was chosen and
//!   changes nothing at prediction.
//! - `scaling`: `none`, or the lower and upper end of the range.
//! - `generator` (one or more, in the order the categorizer runs them):
//!   `bag-of-words` (`BagOfWordsFeatureGenerator()`) or `ngram` with its
//!   minimum and maximum (`NGramFeatureGenerator`).
//! - `category`: the category index, from 0 in order, and its name.
//! - `feature`: the feature index, from 1 in order, its inverse document
//!   frequency, its minimum and maximum weighted value in training, and
//!   the feature string.
//!
//! Numbers are written as `Double.toString` and `Integer.toString` write
//! them and read with Java's grammar, so each reads back to the bits that
//! were written. A name is the last field and is escaped as golden files
//! escape `str` fields: `\t`, `\n`, `\r`, `\\`, and `\uXXXX` for any other
//! character below U+0020.

use std::collections::HashMap;
use std::fmt::Write as _;

use turbo_jvm::text::{double_to_string, parse_double, parse_int};

use super::error::{File, LineError, ModelError};
use super::text::lines;
use super::weighting::{FeatureSelection, Scaling, TermWeighting};
use crate::features::FeatureGenerator;
use crate::hash::BuildFoldHasher;

/// The first line of a sidecar.
pub const HEADER: &str = "svm-doccat 1";

/// The selected features: each string's index, from 1, with its inverse
/// document frequency and its training range.
#[derive(Clone, Debug, Default)]
pub struct Vocabulary {
    index: HashMap<Box<str>, u32, BuildFoldHasher>,
    /// By index minus one.
    idf: Vec<f64>,
    min: Vec<f64>,
    max: Vec<f64>,
}

impl Vocabulary {
    /// The number of features; their indexes are `1..=len`.
    pub fn len(&self) -> usize {
        self.idf.len()
    }

    pub fn is_empty(&self) -> bool {
        self.idf.is_empty()
    }

    /// The index of a feature string, from 1.
    #[inline]
    pub fn index(&self, feature: &str) -> Option<u32> {
        self.index.get(feature).copied()
    }

    /// The inverse document frequency of feature `index`.
    #[inline]
    pub fn idf(&self, index: u32) -> f64 {
        self.idf[index as usize - 1]
    }

    /// The smallest and largest weighted value of feature `index` in
    /// training, 0 included.
    #[inline]
    pub fn range(&self, index: u32) -> (f64, f64) {
        let i = index as usize - 1;
        (self.min[i], self.max[i])
    }

    /// The feature strings by index, for writing.
    fn names(&self) -> Vec<&str> {
        let mut names = vec![""; self.len()];
        for (name, &i) in &self.index {
            names[i as usize - 1] = name;
        }
        names
    }

    /// Adds the next feature; `false` when the string is already there.
    pub fn push(&mut self, name: &str, idf: f64, min: f64, max: f64) -> bool {
        if self.index.contains_key(name) {
            return false;
        }
        self.idf.push(idf);
        self.min.push(min);
        self.max.push(max);
        self.index.insert(name.into(), self.idf.len() as u32);
        true
    }
}

/// Everything of an `SvmDoccatModel` besides the LIBSVM model.
#[derive(Clone, Debug)]
pub struct Sidecar {
    pub language: Option<String>,
    pub weighting: TermWeighting,
    pub selection: FeatureSelection,
    pub max_features: i32,
    pub scaling: Scaling,
    pub generators: Vec<FeatureGenerator>,
    /// By category index; the LIBSVM labels are these indexes.
    pub categories: Vec<String>,
    pub vocabulary: Vocabulary,
}

fn line_error(line: usize, error: LineError) -> ModelError {
    ModelError::Line { file: File::Sidecar, line, error }
}

fn count(field: &'static str, fields: &[&str], expected: usize) -> Result<(), LineError> {
    if fields.len() != expected {
        return Err(LineError::Count { field, expected, found: fields.len() });
    }
    Ok(())
}

fn double(field: &'static str, text: &str) -> Result<f64, LineError> {
    parse_double(text).map_err(|_| LineError::BadValue { field, text: text.into() })
}

fn int(field: &'static str, text: &str) -> Result<i32, LineError> {
    parse_int(text).map_err(|_| LineError::BadValue { field, text: text.into() })
}

fn name(field: &'static str, text: &str) -> Result<String, LineError> {
    unescape(text).ok_or_else(|| LineError::BadValue { field, text: text.into() })
}

/// The sidecar's records as read, before they are checked for presence.
#[derive(Default)]
struct Records {
    language: Option<String>,
    weighting: Option<TermWeighting>,
    selection: Option<(FeatureSelection, i32)>,
    scaling: Option<Scaling>,
    generators: Vec<FeatureGenerator>,
    categories: Vec<String>,
    vocabulary: Vocabulary,
}

fn once<T>(slot: &mut Option<T>, field: &'static str, value: T) -> Result<(), LineError> {
    if slot.is_some() {
        return Err(LineError::Repeated(field));
    }
    *slot = Some(value);
    Ok(())
}

impl Records {
    fn record(&mut self, f: &[&str]) -> Result<(), LineError> {
        match f[0] {
            "language" => {
                count("language", f, 2)?;
                once(&mut self.language, "language", name("language", f[1])?)
            }
            "weighting" => {
                count("weighting", f, 2)?;
                let w = TermWeighting::from_name(f[1])
                    .ok_or(LineError::BadValue { field: "weighting", text: f[1].into() })?;
                once(&mut self.weighting, "weighting", w)
            }
            "selection" => {
                count("selection", f, 3)?;
                let s = FeatureSelection::from_name(f[1])
                    .ok_or(LineError::BadValue { field: "selection", text: f[1].into() })?;
                once(&mut self.selection, "selection", (s, int("selection", f[2])?))
            }
            "scaling" => {
                let s = if f.len() == 2 && f[1] == "none" {
                    Scaling::None
                } else {
                    count("scaling", f, 3)?;
                    let (lower, upper) = (double("scaling", f[1])?, double("scaling", f[2])?);
                    // SvmDoccatConfiguration.Builder.setScaleRange refuses
                    // an upper end not above the lower.
                    if upper <= lower || upper.is_nan() || lower.is_nan() {
                        return Err(LineError::BadValue { field: "scaling", text: format!("{}\t{}", f[1], f[2]) });
                    }
                    Scaling::Range { lower, upper }
                };
                once(&mut self.scaling, "scaling", s)
            }
            "generator" => {
                let g = match f.get(1).copied() {
                    Some("bag-of-words") if f.len() == 2 => FeatureGenerator::BagOfWords,
                    Some("ngram") => {
                        count("generator", f, 4)?;
                        let (min, max) = (int("generator", f[2])?, int("generator", f[3])?);
                        FeatureGenerator::ngram(i64::from(min), i64::from(max))
                            .map_err(|_| LineError::BadValue { field: "generator", text: f[2..].join("\t") })?
                    }
                    _ => return Err(LineError::BadValue { field: "generator", text: f[1..].join("\t") }),
                };
                self.generators.push(g);
                Ok(())
            }
            "category" => {
                count("category", f, 3)?;
                let i = int("category", f[1])?;
                if i64::from(i) != self.categories.len() as i64 {
                    return Err(LineError::Unordered { field: "category", index: i64::from(i) });
                }
                self.categories.push(name("category", f[2])?);
                Ok(())
            }
            "feature" => {
                count("feature", f, 6)?;
                let i = int("feature", f[1])?;
                if i64::from(i) != self.vocabulary.len() as i64 + 1 {
                    return Err(LineError::Unordered { field: "feature", index: i64::from(i) });
                }
                let (idf, min, max) = (double("feature", f[2])?, double("feature", f[3])?, double("feature", f[4])?);
                let n = name("feature", f[5])?;
                if !self.vocabulary.push(&n, idf, min, max) {
                    return Err(LineError::BadValue { field: "feature", text: f[5].into() });
                }
                Ok(())
            }
            other => Err(LineError::UnknownKeyword(other.into())),
        }
    }
}

impl Sidecar {
    /// Reads a sidecar from its text.
    pub fn read(text: &str) -> Result<Sidecar, ModelError> {
        let mut lines = lines(text);
        match lines.next() {
            Some((_, HEADER)) => {}
            Some((n, other)) => return Err(line_error(n, LineError::Header(other.into()))),
            None => return Err(line_error(1, LineError::Header(String::new()))),
        }
        let mut r = Records::default();
        let mut fields = Vec::new();
        for (n, line) in lines {
            fields.clear();
            fields.extend(line.split('\t'));
            r.record(&fields).map_err(|e| line_error(n, e))?;
        }
        let missing = |field| ModelError::Missing { file: File::Sidecar, field };
        let (selection, max_features) = r.selection.ok_or(missing("selection"))?;
        if r.generators.is_empty() {
            return Err(missing("generator"));
        }
        if r.categories.is_empty() {
            return Err(missing("category"));
        }
        let mut seen = std::collections::HashSet::new();
        if let Some(c) = r.categories.iter().find(|c| !seen.insert(c.as_str())) {
            return Err(ModelError::Invalid {
                file: File::Sidecar,
                detail: format!("the category {c:?} is given twice"),
            });
        }
        Ok(Sidecar {
            language: r.language,
            weighting: r.weighting.ok_or(missing("weighting"))?,
            selection,
            max_features,
            scaling: r.scaling.ok_or(missing("scaling"))?,
            generators: r.generators,
            categories: r.categories,
            vocabulary: r.vocabulary,
        })
    }

    /// The sidecar's text, as `read` reads it back.
    pub fn write(&self) -> String {
        let mut out = String::new();
        out.push_str(HEADER);
        out.push('\n');
        if let Some(l) = &self.language {
            let _ = writeln!(out, "language\t{}", escape(l));
        }
        let _ = writeln!(out, "weighting\t{}", self.weighting.name());
        let _ = writeln!(out, "selection\t{}\t{}", self.selection.name(), self.max_features);
        match self.scaling {
            Scaling::None => out.push_str("scaling\tnone\n"),
            Scaling::Range { lower, upper } => {
                let _ = writeln!(out, "scaling\t{}\t{}", double_to_string(lower), double_to_string(upper));
            }
        }
        for g in &self.generators {
            match g {
                FeatureGenerator::BagOfWords => out.push_str("generator\tbag-of-words\n"),
                FeatureGenerator::NGram { min, max } => {
                    let _ = writeln!(out, "generator\tngram\t{min}\t{max}");
                }
            }
        }
        for (i, c) in self.categories.iter().enumerate() {
            let _ = writeln!(out, "category\t{i}\t{}", escape(c));
        }
        for (i, name) in self.vocabulary.names().into_iter().enumerate() {
            let index = i as u32 + 1;
            let (min, max) = self.vocabulary.range(index);
            let _ = writeln!(
                out,
                "feature\t{index}\t{}\t{}\t{}\t{}",
                double_to_string(self.vocabulary.idf(index)),
                double_to_string(min),
                double_to_string(max),
                escape(name)
            );
        }
        out
    }
}

/// A name as a field: `\t`, `\n`, `\r` and `\\` written so, and any other
/// character below U+0020 as `\uXXXX`.
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// The reverse of [`escape`]; also takes a character outside the Basic
/// Multilingual Plane as two `\u` units. `None` for a lone surrogate, a
/// raw tab or line end, or an escape the format does not have.
pub fn unescape(s: &str) -> Option<String> {
    if !s.contains('\\') {
        return if s.contains(['\t', '\n', '\r']) { None } else { Some(s.to_string()) };
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    let mut high: Option<u32> = None;
    while let Some(c) = chars.next() {
        if c != '\\' {
            if high.is_some() || matches!(c, '\t' | '\n' | '\r') {
                return None;
            }
            out.push(c);
            continue;
        }
        let unit = match chars.next()? {
            't' => '\t' as u32,
            'n' => '\n' as u32,
            'r' => '\r' as u32,
            '\\' => '\\' as u32,
            'u' => {
                let hex: String = chars.by_ref().take(4).collect();
                if hex.len() != 4 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
                    return None;
                }
                u32::from_str_radix(&hex, 16).ok()?
            }
            _ => return None,
        };
        match (high.take(), unit) {
            (None, 0xd800..=0xdbff) => high = Some(unit),
            (Some(h), 0xdc00..=0xdfff) => out.push(char::from_u32(0x10000 + ((h - 0xd800) << 10) + (unit - 0xdc00))?),
            (Some(_), _) => return None,
            (None, u) => out.push(char::from_u32(u)?),
        }
    }
    if high.is_some() { None } else { Some(out) }
}

#[cfg(test)]
mod tests;
