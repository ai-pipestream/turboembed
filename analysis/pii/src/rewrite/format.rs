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

use turbo_analysis::View;
use turbo_analysis_layers::{CustomKinds, KindError, Pii, PiiKind, PiiStrategyKind};

use super::replace::EditOrigin;
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
    /// Folds every parameter that changes what this strategy writes into
    /// `h`, for the step's config hash.
    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.str(&self.template).u8(u8::from(self.refuse_overlaps));
    }

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

    /// Replaces every mention of `pii`, a layer over `src`, with its
    /// formatted label, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut assigned: HashMap<(PiiKind, &str), u32> = HashMap::new();
        let mut counters: HashMap<PiiKind, u32> = HashMap::new();
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                let m = c.merged;
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
                Ok(EditOrigin::only(PiiStrategyKind::Format))
            },
        )
    }
}

/// Replaces mentions with a fixed string (such as `"[REDACTED]"`), with
/// optional per-[`PiiKind`] overrides. Replacement is stateless and
/// identical within a document and across documents.
///
/// The replacer owns a custom-kind table: a custom kind given to
/// [`StaticReplacer::with`] indexes it, and a layer's custom kind is
/// matched to it by name, so two tables that number their kinds in another
/// order never share an override.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StaticReplacer {
    default_replacement: String,
    custom_kinds: CustomKinds,
    overrides: Vec<(PiiKind, String)>,
    refuse_overlaps: bool,
}

impl Default for StaticReplacer {
    fn default() -> StaticReplacer {
        StaticReplacer::new(StaticReplacer::DEFAULT_REPLACEMENT)
    }
}

impl StaticReplacer {
    /// Folds every parameter that changes what this strategy writes into
    /// `h`, for the step's config hash.
    /// A custom kind's override is folded by the kind's name, so two
    /// tables that number their kinds apart hash alike.
    pub(super) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.str(&self.default_replacement).usize(self.overrides.len());
        for (kind, rep) in &self.overrides {
            match kind.builtin_id() {
                Some(id) => h.u8(0).u16(id),
                None => h.u8(1).str(self.custom_kinds.name(*kind).unwrap_or("")),
            };
            h.str(rep);
        }
        h.u8(u8::from(self.refuse_overlaps));
    }

    /// The default static replacement string.
    pub const DEFAULT_REPLACEMENT: &'static str = "[REDACTED]";

    /// A static replacer writing `default_replacement` for every kind.
    pub fn new(default_replacement: impl Into<String>) -> StaticReplacer {
        StaticReplacer {
            default_replacement: default_replacement.into(),
            custom_kinds: CustomKinds::new(),
            overrides: Vec::new(),
            refuse_overlaps: false,
        }
    }

    /// This replacer with `kinds` as its custom-kind table, which every
    /// custom kind given to [`StaticReplacer::with`] afterwards indexes.
    /// Drops overrides of custom kinds made under an earlier table.
    pub fn with_custom_kinds(mut self, kinds: CustomKinds) -> StaticReplacer {
        self.overrides.retain(|(k, _)| !k.is_custom());
        self.custom_kinds = kinds;
        self
    }

    /// The table every custom kind of this replacer indexes.
    pub fn custom_kinds(&self) -> &CustomKinds {
        &self.custom_kinds
    }

    /// Overrides the static replacement text for `kind`.
    ///
    /// # Panics
    ///
    /// When `kind` is a custom kind outside the replacer's table; see
    /// [`StaticReplacer::try_with`].
    pub fn with(self, kind: PiiKind, replacement: impl Into<String>) -> StaticReplacer {
        self.try_with(kind, replacement).expect("a kind of the replacer's custom-kind table")
    }

    /// Overrides the static replacement text for `kind`, refused when
    /// `kind` is a custom kind outside [`StaticReplacer::custom_kinds`].
    pub fn try_with(mut self, kind: PiiKind, replacement: impl Into<String>) -> Result<StaticReplacer, KindError> {
        self.custom_kinds.check(kind)?;
        let replacement = replacement.into();
        match self.overrides.binary_search_by_key(&kind.code(), |(k, _)| k.code()) {
            Ok(at) => self.overrides[at].1 = replacement,
            Err(at) => self.overrides.insert(at, (kind, replacement)),
        }
        Ok(self)
    }

    /// The replacement text configured for `kind` of this replacer's table.
    pub fn replacement(&self, kind: PiiKind) -> &str {
        self.overrides
            .binary_search_by_key(&kind.code(), |(k, _)| k.code())
            .map_or(self.default_replacement.as_str(), |at| self.overrides[at].1.as_str())
    }

    /// The replacement text for `kind` of a layer whose custom kinds index
    /// `kinds`: a custom kind is matched to this replacer's by name, and one
    /// the replacer does not list takes the default replacement.
    pub fn replacement_in(&self, kinds: &CustomKinds, kind: PiiKind) -> &str {
        self.custom_kinds.translate(kinds, kind).map_or(self.default_replacement.as_str(), |k| self.replacement(k))
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

    /// Replaces every mention of `pii`, a layer over `src`, with its fixed
    /// replacement string, as a view derived from `src`.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        PiiRewrite::derive(
            src,
            pii,
            |_| self.refuse_overlaps,
            0,
            |c, out| {
                out.push_str(self.replacement_in(&pii.custom_kinds, c.merged.kind));
                Ok(EditOrigin::only(PiiStrategyKind::Static))
            },
        )
    }
}
