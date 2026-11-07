//! Redaction with a caller format template ([`FormatRedactor`]) and fixed
//! static replacement text ([`StaticReplacer`]).
//!
//! [`FormatRedactor`] expands a template string for each mention:
//! - `%t`: the kind in uppercase ASCII (`EMAIL`, `US-SSN`, `AWS-ACCESS-KEY`),
//!   matching [`super::Pseudonymizer`] and [`super::HmacTokenizer`].
//! - `%k`: the lowercase kind name (`email`, `us-ssn`, `aws-access-key`), as
//!   returned by [`PiiKind::name`].
//! - `%v`: the 1-based pseudonym number for `(kind, normalized)` within the
//!   document, reusing the same number when the same kind and normalized
//!   value appear again in the same rewrite.
//! - `%%`: a literal `%`.
//!
//! [`StaticReplacer`] replaces every mention with a fixed string (such as
//! `"[REDACTED]"`), optionally overridden per [`PiiKind`], producing the
//! same replacement within a document and across documents.

use std::collections::HashMap;
use std::fmt::{self, Write};

use turbo_analysis_layers::{Pii, PiiKind};

use super::{PiiRewrite, RewriteError, kind_name};

/// A format template [`FormatRedactor`] cannot accept.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FormatError {
    /// The template is empty and would produce an empty replacement.
    EmptyTemplate,
    /// The template ends with an unescaped `%`.
    TrailingPercent,
    /// An unrecognized `%` specifier; only `%t`, `%k`, `%v` and `%%` are valid.
    UnknownSpecifier(char),
}

impl fmt::Display for FormatError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FormatError::EmptyTemplate => f.write_str("the format template must not be empty"),
            FormatError::TrailingPercent => f.write_str("the format template ends with an unescaped '%'"),
            FormatError::UnknownSpecifier(c) => {
                write!(f, "unknown format specifier '%{c}'; expected '%t', '%k', '%v' or '%%'")
            }
        }
    }
}

impl std::error::Error for FormatError {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Segment {
    Literal(String),
    KindUpper,
    KindLower,
    ValueNumber,
}

/// Replaces mentions using a caller format template such as
/// `{{{REDACTED-%t}}}` or `[%t-%v]`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FormatRedactor {
    template: String,
    segments: Vec<Segment>,
    uses_number: bool,
    refuse_overlaps: bool,
}

impl FormatRedactor {
    /// Parses and validates `template`.
    ///
    /// Supported specifiers are `%t` (uppercase kind name), `%k` (lowercase
    /// kind name), `%v` (1-based per-kind pseudonym number), and `%%`
    /// (literal `%`).
    pub fn new(template: impl Into<String>) -> Result<FormatRedactor, FormatError> {
        let template = template.into();
        if template.is_empty() {
            return Err(FormatError::EmptyTemplate);
        }
        let mut segments = Vec::new();
        let mut literal = String::new();
        let mut uses_number = false;
        let mut chars = template.chars();
        while let Some(c) = chars.next() {
            if c != '%' {
                literal.push(c);
                continue;
            }
            let Some(spec) = chars.next() else {
                return Err(FormatError::TrailingPercent);
            };
            match spec {
                '%' => literal.push('%'),
                't' => {
                    if !literal.is_empty() {
                        segments.push(Segment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(Segment::KindUpper);
                }
                'k' => {
                    if !literal.is_empty() {
                        segments.push(Segment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(Segment::KindLower);
                }
                'v' => {
                    if !literal.is_empty() {
                        segments.push(Segment::Literal(std::mem::take(&mut literal)));
                    }
                    segments.push(Segment::ValueNumber);
                    uses_number = true;
                }
                other => return Err(FormatError::UnknownSpecifier(other)),
            }
        }
        if !literal.is_empty() {
            segments.push(Segment::Literal(literal));
        }
        Ok(FormatRedactor { template, segments, uses_number, refuse_overlaps: false })
    }

    /// The template string this redactor was built from.
    pub fn template(&self) -> &str {
        &self.template
    }

    /// Whether the template includes `%v` (the per-kind pseudonym number).
    pub const fn uses_number(&self) -> bool {
        self.uses_number
    }

    /// This redactor, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> FormatRedactor {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Appends the formatted label for `kind` and 1-based `number` to `out`.
    pub fn write_label(&self, kind: &str, number: u32, out: &mut String) {
        for seg in &self.segments {
            match seg {
                Segment::Literal(s) => out.push_str(s),
                Segment::KindUpper => out.extend(kind.chars().map(|c| c.to_ascii_uppercase())),
                Segment::KindLower => out.push_str(kind),
                Segment::ValueNumber => {
                    let _ = write!(out, "{number}");
                }
            }
        }
    }

    /// Formats the label for the kind named `kind` and 1-based `number`.
    pub fn format_label(&self, kind: &str, number: u32) -> String {
        let mut out = String::new();
        self.write_label(kind, number, &mut out);
        out
    }

    /// Replaces every mention of `pii`, a layer over `text`, with its
    /// formatted label.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut assigned: HashMap<(PiiKind, &str), u32> = HashMap::new();
        let mut counters: HashMap<PiiKind, u32> = HashMap::new();
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, &pii.custom_kinds, pii.mentions())?
        };
        PiiRewrite::replace(text, &pii.custom_kinds, mentions, |m, out| {
            let n = if self.uses_number {
                *assigned.entry((m.kind, m.normalized)).or_insert_with(|| {
                    let c = counters.entry(m.kind).or_insert(0);
                    *c += 1;
                    *c
                })
            } else {
                1
            };
            self.write_label(kind_name(&pii.custom_kinds, m.kind), n, out);
        })
    }
}

/// Replaces mentions with a fixed string (such as `"[REDACTED]"`), with
/// optional per-[`PiiKind`] overrides. Replacement is stateless and
/// identical within a document and across documents.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticReplacer {
    default_replacement: String,
    overrides: Vec<(PiiKind, String)>,
    refuse_overlaps: bool,
}

impl Default for StaticReplacer {
    fn default() -> StaticReplacer {
        StaticReplacer::new(StaticReplacer::DEFAULT_REPLACEMENT)
    }
}

impl StaticReplacer {
    /// The default static replacement string.
    pub const DEFAULT_REPLACEMENT: &'static str = "[REDACTED]";

    /// A static replacer writing `default_replacement` for every kind.
    pub fn new(default_replacement: impl Into<String>) -> StaticReplacer {
        StaticReplacer {
            default_replacement: default_replacement.into(),
            overrides: Vec::new(),
            refuse_overlaps: false,
        }
    }

    /// Overrides the static replacement text for `kind`.
    pub fn with(mut self, kind: PiiKind, replacement: impl Into<String>) -> StaticReplacer {
        let replacement = replacement.into();
        match self.overrides.binary_search_by_key(&kind.code(), |(k, _)| k.code()) {
            Ok(at) => self.overrides[at].1 = replacement,
            Err(at) => self.overrides.insert(at, (kind, replacement)),
        }
        self
    }

    /// The replacement text configured for `kind`.
    pub fn replacement(&self, kind: PiiKind) -> &str {
        self.overrides
            .binary_search_by_key(&kind.code(), |(k, _)| k.code())
            .map_or(self.default_replacement.as_str(), |at| self.overrides[at].1.as_str())
    }

    /// This replacer, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> StaticReplacer {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Replaces every mention of `pii`, a layer over `text`, with its fixed
    /// replacement string.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, &pii.custom_kinds, pii.mentions())?
        };
        PiiRewrite::replace(text, &pii.custom_kinds, mentions, |m, out| {
            out.push_str(self.replacement(m.kind));
        })
    }
}
