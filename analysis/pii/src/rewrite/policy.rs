//! Composable per-[`PiiKind`] rewrite policy ([`RewritePolicy`],
//! [`RewriteStrategy`]).
//!
//! Callers configure a default [`RewriteStrategy`] and override individual
//! built-in or [`PiiKind::Custom`] kinds with `.with(kind, strategy)`.

use std::collections::HashMap;

use turbo_analysis_layers::{Pii, PiiKind, PiiMention};

use super::{
    Abbreviator, FormatRedactor, HmacTokenizer, MapReplacer, MaskPolicy, PiiRewrite, Pseudonymizer, RewriteError,
    Sha256Tokenizer, StaticReplacer, Truncate,
};

/// Any mention replacement strategy supported by [`PiiRewrite`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RewriteStrategy {
    /// Byte-length-preserving character masking ([`MaskPolicy`]).
    Mask(MaskPolicy),
    /// Per-document numbered pseudonyms ([`Pseudonymizer`]).
    Pseudonym(Pseudonymizer),
    /// Keyed HMAC-SHA-256 tokens ([`HmacTokenizer`]).
    Hmac(HmacTokenizer),
    /// Caller template formatting ([`FormatRedactor`]).
    Format(FormatRedactor),
    /// Fixed static replacement text ([`StaticReplacer`]).
    Static(StaticReplacer),
    /// Unkeyed SHA-256 hex digest ([`Sha256Tokenizer`]).
    Sha256(Sha256Tokenizer),
    /// Caller lookup table with fallback strategy ([`MapReplacer`]).
    Map(Box<MapReplacer>),
    /// Keep `N` characters at start or end and mask the rest ([`Truncate`]).
    Truncate(Truncate),
    /// Initials abbreviation ([`Abbreviator`]).
    Abbreviate(Abbreviator),
}

impl Default for RewriteStrategy {
    fn default() -> RewriteStrategy {
        RewriteStrategy::Pseudonym(Pseudonymizer::default())
    }
}

impl From<MaskPolicy> for RewriteStrategy {
    fn from(v: MaskPolicy) -> RewriteStrategy {
        RewriteStrategy::Mask(v)
    }
}

impl From<Pseudonymizer> for RewriteStrategy {
    fn from(v: Pseudonymizer) -> RewriteStrategy {
        RewriteStrategy::Pseudonym(v)
    }
}

impl From<HmacTokenizer> for RewriteStrategy {
    fn from(v: HmacTokenizer) -> RewriteStrategy {
        RewriteStrategy::Hmac(v)
    }
}

impl From<FormatRedactor> for RewriteStrategy {
    fn from(v: FormatRedactor) -> RewriteStrategy {
        RewriteStrategy::Format(v)
    }
}

impl From<StaticReplacer> for RewriteStrategy {
    fn from(v: StaticReplacer) -> RewriteStrategy {
        RewriteStrategy::Static(v)
    }
}

impl From<Sha256Tokenizer> for RewriteStrategy {
    fn from(v: Sha256Tokenizer) -> RewriteStrategy {
        RewriteStrategy::Sha256(v)
    }
}

impl From<MapReplacer> for RewriteStrategy {
    fn from(v: MapReplacer) -> RewriteStrategy {
        RewriteStrategy::Map(Box::new(v))
    }
}

impl From<Truncate> for RewriteStrategy {
    fn from(v: Truncate) -> RewriteStrategy {
        RewriteStrategy::Truncate(v)
    }
}

impl From<Abbreviator> for RewriteStrategy {
    fn from(v: Abbreviator) -> RewriteStrategy {
        RewriteStrategy::Abbreviate(v)
    }
}

impl RewriteStrategy {
    /// Replaces every mention of `pii`, a layer over `text`, with this
    /// strategy.
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        match self {
            RewriteStrategy::Mask(s) => s.rewrite(text, pii),
            RewriteStrategy::Pseudonym(s) => s.rewrite(text, pii),
            RewriteStrategy::Hmac(s) => s.rewrite(text, pii),
            RewriteStrategy::Format(s) => s.rewrite(text, pii),
            RewriteStrategy::Static(s) => s.rewrite(text, pii),
            RewriteStrategy::Sha256(s) => s.rewrite(text, pii),
            RewriteStrategy::Map(s) => s.rewrite(text, pii),
            RewriteStrategy::Truncate(s) => s.rewrite(text, pii),
            RewriteStrategy::Abbreviate(s) => s.rewrite(text, pii),
        }
    }
}

#[derive(Default)]
pub(crate) struct RewriteState<'a> {
    assigned: HashMap<(PiiKind, &'a str), u32>,
    counters: HashMap<PiiKind, u32>,
}

impl<'a> RewriteState<'a> {
    pub(crate) fn number(&mut self, kind: PiiKind, normalized: &'a str) -> u32 {
        *self.assigned.entry((kind, normalized)).or_insert_with(|| {
            let n = self.counters.entry(kind).or_insert(0);
            *n += 1;
            *n
        })
    }

    pub(crate) fn write_strategy(
        &mut self,
        strategy: &RewriteStrategy,
        text: &str,
        m: &PiiMention<'a>,
        out: &mut String,
    ) {
        match strategy {
            RewriteStrategy::Mask(policy) => {
                out.push_str(&policy.masked(&text[m.span.range()]));
            }
            RewriteStrategy::Pseudonym(pseudo) => {
                let n = self.number(m.kind, m.normalized);
                pseudo.write_label(m.kind, n, out);
            }
            RewriteStrategy::Hmac(hmac) => {
                let _ = hmac.write_token(m.kind, m.normalized, out);
            }
            RewriteStrategy::Format(fmt) => {
                let n = if fmt.uses_number() { self.number(m.kind, m.normalized) } else { 1 };
                fmt.write_label(m.kind, n, out);
            }
            RewriteStrategy::Static(stat) => {
                out.push_str(stat.replacement(m.kind));
            }
            RewriteStrategy::Sha256(sha) => {
                let _ = sha.write_digest(m.kind, m.normalized, out);
            }
            RewriteStrategy::Map(map) => {
                map.write_mention(self, text, m, out);
            }
            RewriteStrategy::Truncate(trunc) => {
                trunc.write_truncated(&text[m.span.range()], out);
            }
            RewriteStrategy::Abbreviate(abbr) => {
                abbr.write_abbreviated(&text[m.span.range()], out);
            }
        }
    }
}

/// Dispatches each mention to the [`RewriteStrategy`] configured for its
/// [`PiiKind`], or to a default fallback strategy.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RewritePolicy {
    overrides: Vec<(PiiKind, RewriteStrategy)>,
    default_strategy: RewriteStrategy,
    refuse_overlaps: bool,
}

impl RewritePolicy {
    /// A policy applying `default_strategy` to every kind not explicitly
    /// overridden with [`RewritePolicy::with`].
    pub fn new(default_strategy: impl Into<RewriteStrategy>) -> RewritePolicy {
        RewritePolicy { overrides: Vec::new(), default_strategy: default_strategy.into(), refuse_overlaps: false }
    }

    /// Assigns `strategy` to mentions of `kind` (built-in or
    /// [`PiiKind::Custom`]).
    pub fn with(mut self, kind: PiiKind, strategy: impl Into<RewriteStrategy>) -> RewritePolicy {
        let strategy = strategy.into();
        PiiKind::remember_custom(kind);
        match self.overrides.binary_search_by_key(&kind, |(k, _)| *k) {
            Ok(at) => self.overrides[at].1 = strategy,
            Err(at) => self.overrides.insert(at, (kind, strategy)),
        }
        self
    }

    /// The strategy configured for `kind`.
    pub fn strategy(&self, kind: PiiKind) -> &RewriteStrategy {
        self.overrides
            .binary_search_by_key(&kind, |(k, _)| *k)
            .map_or(&self.default_strategy, |at| &self.overrides[at].1)
    }

    /// The default fallback strategy for kinds without an explicit override.
    pub fn default_strategy(&self) -> &RewriteStrategy {
        &self.default_strategy
    }

    /// This policy, refusing overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refusing_overlaps(mut self) -> RewritePolicy {
        self.refuse_overlaps = true;
        self
    }

    /// Whether overlapping mentions are refused rather than coalesced.
    pub const fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
    }

    /// Replaces every mention of `pii`, a layer over `text`, with the
    /// strategy configured for its [`PiiKind`].
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mentions: Vec<_> = if self.refuse_overlaps {
            pii.mentions().collect()
        } else {
            PiiRewrite::coalesce_clusters(text, pii.mentions())?
        };
        let mut state = RewriteState::default();
        PiiRewrite::replace(text, mentions, |m, out| {
            let strategy = self.strategy(m.kind);
            state.write_strategy(strategy, text, m, out);
        })
    }
}
