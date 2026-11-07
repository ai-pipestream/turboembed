//! Caller lookup-table replacement ([`MapReplacer`], [`MapTarget`],
//! [`MapError`]) with case sensitivity control and a fallback
//! [`RewriteStrategy`] for unmapped mentions.
//!
//! Lookup tables are always provided by the caller (inline from key/value
//! pairs or loaded from a caller TSV or delimited file) and never bundled.

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use turbo_analysis_layers::{Pii, PiiMention};

use super::policy::RewriteState;
use super::{PiiRewrite, Pseudonymizer, RewriteError, RewriteStrategy};

/// The default field delimiter for TSV lookup files.
pub const TSV_DELIMITER: char = '\t';

/// Prefix marking a comment line in a delimited lookup table.
const COMMENT_PREFIX: char = '#';

/// Errors returned when constructing or loading a [`MapReplacer`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MapError {
    /// A lookup key is empty after trimming.
    EmptyKey,
    /// A replacement value is empty after trimming.
    EmptyReplacement,
    /// A non-empty, non-comment line in delimited input lacks the delimiter
    /// (`line` is 1-based).
    MissingDelimiter { line: usize },
    /// Reading a lookup table file failed.
    Io(String),
}

impl fmt::Display for MapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapError::EmptyKey => f.write_str("lookup table key must not be empty"),
            MapError::EmptyReplacement => f.write_str("lookup table replacement must not be empty"),
            MapError::MissingDelimiter { line } => {
                write!(f, "lookup table line {line} is missing the field delimiter")
            }
            MapError::Io(msg) => write!(f, "failed to read lookup table: {msg}"),
        }
    }
}

impl std::error::Error for MapError {}

/// Which mention text [`MapReplacer`] probes in its lookup table.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum MapTarget {
    /// Check the mention's normalized value first, then its surface text in
    /// the source view.
    #[default]
    NormalizedOrSurface,
    /// Check only the mention's normalized value.
    Normalized,
    /// Check only the mention's surface text in the source view.
    Surface,
}

/// Replaces mentions found in a caller lookup table and delegates unmapped
/// mentions to a configurable fallback [`RewriteStrategy`] (defaulting to
/// [`Pseudonymizer::default`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MapReplacer {
    table: HashMap<String, String>,
    case_sensitive: bool,
    target: MapTarget,
    fallback: Box<RewriteStrategy>,
    refuse_overlaps: bool,
}

impl MapReplacer {
    /// Builds a lookup table from `(key, replacement)` pairs.
    ///
    /// Keys and replacements are trimmed of surrounding whitespace; empty
    /// keys or empty replacements are refused with [`MapError`]. When
    /// `case_sensitive` is `false`, keys are folded with
    /// [`str::to_lowercase`].
    pub fn new<I, K, V>(entries: I, case_sensitive: bool) -> Result<MapReplacer, MapError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        let mut table = HashMap::new();
        for (raw_k, raw_v) in entries {
            let k_owned = raw_k.into();
            let v_owned = raw_v.into();
            let k = k_owned.trim();
            let v = v_owned.trim();
            if k.is_empty() {
                return Err(MapError::EmptyKey);
            }
            if v.is_empty() {
                return Err(MapError::EmptyReplacement);
            }
            let norm_k = if case_sensitive { k.to_owned() } else { k.to_lowercase() };
            table.insert(norm_k, v.to_owned());
        }
        Ok(MapReplacer {
            table,
            case_sensitive,
            target: MapTarget::NormalizedOrSurface,
            fallback: Box::new(RewriteStrategy::Pseudonym(Pseudonymizer::default())),
            refuse_overlaps: false,
        })
    }

    /// Parses a tab-separated (`\t`) lookup table from `input`, skipping
    /// blank lines and lines whose first non-whitespace character is `#`.
    pub fn from_tsv(input: &str, case_sensitive: bool) -> Result<MapReplacer, MapError> {
        MapReplacer::from_delimited(input, TSV_DELIMITER, case_sensitive)
    }

    /// Parses a `delimiter`-separated lookup table from `input`, skipping
    /// blank lines and lines whose first non-whitespace character is `#`.
    pub fn from_delimited(input: &str, delimiter: char, case_sensitive: bool) -> Result<MapReplacer, MapError> {
        let mut pairs = Vec::new();
        for (idx, raw_line) in input.lines().enumerate() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || trimmed.starts_with(COMMENT_PREFIX) {
                continue;
            }
            let line = raw_line.trim_end_matches('\r');
            let Some((k, v)) = line.split_once(delimiter) else {
                return Err(MapError::MissingDelimiter { line: idx + 1 });
            };
            pairs.push((k.trim().to_owned(), v.trim().to_owned()));
        }
        MapReplacer::new(pairs, case_sensitive)
    }

    /// Loads a tab-separated (`\t`) lookup table from `path`.
    pub fn from_tsv_file(path: impl AsRef<Path>, case_sensitive: bool) -> Result<MapReplacer, MapError> {
        MapReplacer::from_delimited_file(path, TSV_DELIMITER, case_sensitive)
    }

    /// Loads a `delimiter`-separated lookup table from `path`.
    pub fn from_delimited_file(
        path: impl AsRef<Path>,
        delimiter: char,
        case_sensitive: bool,
    ) -> Result<MapReplacer, MapError> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| MapError::Io(e.to_string()))?;
        MapReplacer::from_delimited(&content, delimiter, case_sensitive)
    }

    /// Sets the fallback strategy applied when a mention's value is not in
    /// the lookup table.
    pub fn with_fallback(mut self, fallback: impl Into<RewriteStrategy>) -> MapReplacer {
        self.fallback = Box::new(fallback.into());
        self
    }

    /// Sets whether lookup checks the normalized value, surface text, or
    /// both.
    pub fn with_target(mut self, target: MapTarget) -> MapReplacer {
        self.target = target;
        self
    }

    /// Whether lookups are case-sensitive.
    pub const fn is_case_sensitive(&self) -> bool {
        self.case_sensitive
    }

    /// Which mention text is probed during lookup.
    pub const fn target(&self) -> MapTarget {
        self.target
    }

    /// The fallback strategy used for unmapped mentions.
    pub fn fallback(&self) -> &RewriteStrategy {
        &self.fallback
    }

    /// Number of entries in the lookup table.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    /// Whether the lookup table has no entries.
    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Looks up `key` in the table according to [`MapReplacer::is_case_sensitive`].
    pub fn get(&self, key: &str) -> Option<&str> {
        let key = key.trim();
        if self.case_sensitive {
            self.table.get(key).map(String::as_str)
        } else {
            let folded = key.to_lowercase();
            self.table.get(&folded).map(String::as_str)
        }
    }

    /// This replacer, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> MapReplacer {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced
    /// (either because this replacer or its fallback strategy refuses
    /// overlaps).
    pub fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps || self.fallback.refuses_overlaps()
    }

    pub(crate) fn lookup_mention(&self, text: &str, m: &PiiMention<'_>) -> Option<&str> {
        let surface = &text[m.span.range()];
        match self.target {
            MapTarget::NormalizedOrSurface => self.get(m.normalized).or_else(|| self.get(surface)),
            MapTarget::Normalized => self.get(m.normalized),
            MapTarget::Surface => self.get(surface),
        }
    }

    pub(crate) fn write_mention<'a>(
        &self,
        state: &mut RewriteState<'a>,
        text: &str,
        m: &PiiMention<'a>,
        out: &mut String,
    ) -> Result<(), RewriteError> {
        if let Some(replacement) = self.lookup_mention(text, m) {
            out.push_str(replacement);
            Ok(())
        } else {
            state.write_strategy(&self.fallback, text, m, out)
        }
    }

    /// Replaces every mention of `pii`, a layer over `text`, using the
    /// lookup table or the configured fallback strategy when unmapped.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut state = RewriteState::new(&pii.custom_kinds);
        if self.refuses_overlaps() {
            return PiiRewrite::try_replace(text, pii, pii.mentions(), |m, out| {
                self.write_mention(&mut state, text, m, out)
            });
        }
        let clusters = PiiRewrite::group_clusters(text, &pii.custom_kinds, pii.mentions())?;
        let strategy = RewriteStrategy::Map(Box::new(self.clone()));
        let mut idx = 0;
        PiiRewrite::try_replace(text, pii, clusters.iter().map(|c| c.merged), |_, out| {
            let cluster = &clusters[idx];
            idx += 1;
            state.write_cluster(|_| &strategy, text, cluster, out)
        })
    }
}
