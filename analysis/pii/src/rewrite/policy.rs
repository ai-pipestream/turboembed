//! Composable per-[`PiiKind`] rewrite policy ([`RewritePolicy`],
//! [`RewriteStrategy`]).
//!
//! Callers configure a default [`RewriteStrategy`] and override individual
//! built-in or [`PiiKind::Custom`] kinds with `.with(kind, strategy)`.

use turbo_analysis::View;
use turbo_analysis_layers::{CustomKinds, KindError, Pii, PiiKind, PiiMention, PiiRule, PiiStrategyKind};

use super::mask::CharRetention;
use super::replace::{EditOrigin, MentionCluster};
use super::state::MemberInitial;
pub(crate) use super::state::RewriteState;
use super::{
    Abbreviator, AgeSafeHarbor, DateRewrite, FormatRedactor, HmacTokenizer, MapReplacer, MaskPolicy, PiiRewrite,
    Pseudonymizer, RewriteError, Sha256Tokenizer, StaticReplacer, Truncate, ZipSafeHarbor,
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
    /// HIPAA Safe Harbor ZIP generalization ([`ZipSafeHarbor`]).
    ZipSafeHarbor(ZipSafeHarbor),
    /// HIPAA Safe Harbor age aggregation ([`AgeSafeHarbor`]).
    AgeSafeHarbor(AgeSafeHarbor),
    /// Date truncation, shifting and relative dates ([`DateRewrite`]).
    Date(DateRewrite),
}

/// A whole-value `'*'` mask. Never a per-document pseudonym (`EMAIL-1`
/// names different values in different documents) or an unkeyed SHA-256
/// (issue #1882).
impl Default for RewriteStrategy {
    fn default() -> RewriteStrategy {
        RewriteStrategy::Mask(MaskPolicy::new(MaskPolicy::DEFAULT_MASK).expect("'*' is ASCII"))
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

impl From<ZipSafeHarbor> for RewriteStrategy {
    fn from(v: ZipSafeHarbor) -> RewriteStrategy {
        RewriteStrategy::ZipSafeHarbor(v)
    }
}

impl From<AgeSafeHarbor> for RewriteStrategy {
    fn from(v: AgeSafeHarbor) -> RewriteStrategy {
        RewriteStrategy::AgeSafeHarbor(v)
    }
}

impl From<DateRewrite> for RewriteStrategy {
    fn from(v: DateRewrite) -> RewriteStrategy {
        RewriteStrategy::Date(v)
    }
}

impl RewriteStrategy {
    /// Folds the strategy's kind and every parameter that changes what it
    /// writes into `h`.
    pub(crate) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u8(self.kind().id());
        match self {
            RewriteStrategy::Mask(s) => s.fold_config(h),
            RewriteStrategy::Pseudonym(s) => s.fold_config(h),
            RewriteStrategy::Hmac(s) => s.fold_config(h),
            RewriteStrategy::Format(s) => s.fold_config(h),
            RewriteStrategy::Static(s) => s.fold_config(h),
            RewriteStrategy::Sha256(s) => s.fold_config(h),
            RewriteStrategy::Map(s) => s.fold_config(h),
            RewriteStrategy::Truncate(s) => s.fold_config(h),
            RewriteStrategy::Abbreviate(s) => s.fold_config(h),
            RewriteStrategy::ZipSafeHarbor(s) => s.fold_config(h),
            RewriteStrategy::AgeSafeHarbor(s) => s.fold_config(h),
            RewriteStrategy::Date(s) => s.fold_config(h),
        }
    }

    /// Whether this strategy refuses overlapping mentions with
    /// [`RewriteError::Overlap`].
    pub fn refuses_overlaps(&self) -> bool {
        match self {
            RewriteStrategy::Mask(_) => false,
            RewriteStrategy::Pseudonym(s) => s.refuses_overlaps(),
            RewriteStrategy::Hmac(s) => s.refuses_overlaps(),
            RewriteStrategy::Format(s) => s.refuses_overlaps(),
            RewriteStrategy::Static(s) => s.refuses_overlaps(),
            RewriteStrategy::Sha256(s) => s.refuses_overlaps(),
            RewriteStrategy::Map(s) => s.refuses_overlaps(),
            RewriteStrategy::Truncate(s) => s.refuses_overlaps(),
            RewriteStrategy::Abbreviate(s) => s.refuses_overlaps(),
            RewriteStrategy::ZipSafeHarbor(s) => s.refuses_overlaps(),
            RewriteStrategy::AgeSafeHarbor(s) => s.refuses_overlaps(),
            RewriteStrategy::Date(s) => s.refuses_overlaps(),
        }
    }

    /// Replaces every mention of `pii`, a layer over `text`, with this
    /// strategy.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        match self {
            RewriteStrategy::Mask(s) => s.rewrite(src, pii),
            RewriteStrategy::Pseudonym(s) => s.rewrite(src, pii),
            RewriteStrategy::Hmac(s) => s.rewrite(src, pii),
            RewriteStrategy::Format(s) => s.rewrite(src, pii),
            RewriteStrategy::Static(s) => s.rewrite(src, pii),
            RewriteStrategy::Sha256(s) => s.rewrite(src, pii),
            RewriteStrategy::Map(s) => s.rewrite(src, pii),
            RewriteStrategy::Truncate(s) => s.rewrite(src, pii),
            RewriteStrategy::Abbreviate(s) => s.rewrite(src, pii),
            RewriteStrategy::ZipSafeHarbor(s) => s.rewrite(src, pii),
            RewriteStrategy::AgeSafeHarbor(s) => s.rewrite(src, pii),
            RewriteStrategy::Date(s) => s.rewrite(src, pii),
        }
    }

    /// The kind of this strategy, as the edits layer records it.
    pub const fn kind(&self) -> PiiStrategyKind {
        match self {
            RewriteStrategy::Mask(_) => PiiStrategyKind::Mask,
            RewriteStrategy::Pseudonym(_) => PiiStrategyKind::Pseudonym,
            RewriteStrategy::Hmac(_) => PiiStrategyKind::Hmac,
            RewriteStrategy::Format(_) => PiiStrategyKind::Format,
            RewriteStrategy::Static(_) => PiiStrategyKind::Static,
            RewriteStrategy::Sha256(_) => PiiStrategyKind::Sha256,
            RewriteStrategy::Map(_) => PiiStrategyKind::Map,
            RewriteStrategy::Truncate(_) => PiiStrategyKind::Truncate,
            RewriteStrategy::Abbreviate(_) => PiiStrategyKind::Abbreviate,
            RewriteStrategy::ZipSafeHarbor(_) => PiiStrategyKind::ZipSafeHarbor,
            RewriteStrategy::AgeSafeHarbor(_) => PiiStrategyKind::AgeSafeHarbor,
            RewriteStrategy::Date(s) => s.kind(),
        }
    }

    pub(crate) fn requires_normalized(&self, text: &str, m: &PiiMention<'_>) -> bool {
        match self {
            RewriteStrategy::Hmac(_) | RewriteStrategy::Sha256(_) => true,
            RewriteStrategy::Map(map) => {
                map.lookup_mention(text, m).is_none() && map.fallback().requires_normalized(text, m)
            }
            _ => false,
        }
    }

    pub(super) fn retention(&self, text: &str, m: &PiiMention<'_>) -> CharRetention {
        match self {
            RewriteStrategy::Mask(policy) => policy.retention(&text[m.span.range()]),
            RewriteStrategy::Truncate(trunc) => trunc.retention(&text[m.span.range()]),
            RewriteStrategy::Map(map) => {
                if map.lookup_mention(text, m).is_none() {
                    map.fallback().retention(text, m)
                } else {
                    CharRetention::NONE
                }
            }
            RewriteStrategy::Pseudonym(_)
            | RewriteStrategy::Hmac(_)
            | RewriteStrategy::Format(_)
            | RewriteStrategy::Static(_)
            | RewriteStrategy::Sha256(_)
            | RewriteStrategy::Abbreviate(_)
            | RewriteStrategy::ZipSafeHarbor(_)
            | RewriteStrategy::AgeSafeHarbor(_)
            | RewriteStrategy::Date(_) => CharRetention::NONE,
        }
    }

    pub(super) fn member_initial(&self, text: &str, m: &PiiMention<'_>) -> MemberInitial {
        match self {
            RewriteStrategy::Abbreviate(_) => MemberInitial::Abbreviate,
            RewriteStrategy::Map(map) => {
                if map.lookup_mention(text, m).is_none() {
                    map.fallback().member_initial(text, m)
                } else {
                    MemberInitial::Retention(CharRetention::NONE)
                }
            }
            other => MemberInitial::Retention(other.retention(text, m)),
        }
    }
}

/// Dispatches each mention to the [`RewriteStrategy`] configured for its
/// [`PiiKind`], or to a default fallback strategy.
///
/// The policy owns a custom-kind table: a custom kind given to
/// [`RewritePolicy::with`] indexes it, and a layer's custom kind is matched
/// to the policy's by name, so one policy serves layers whose tables list
/// their kinds in another order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RewritePolicy {
    custom_kinds: CustomKinds,
    overrides: Vec<(PiiKind, RewriteStrategy)>,
    default_strategy: RewriteStrategy,
    refuse_overlaps: bool,
    revision: u64,
}

impl RewritePolicy {
    /// A policy applying `default_strategy` to every kind not explicitly
    /// overridden with [`RewritePolicy::with`].
    pub fn new(default_strategy: impl Into<RewriteStrategy>) -> RewritePolicy {
        RewritePolicy {
            custom_kinds: CustomKinds::new(),
            overrides: Vec::new(),
            default_strategy: default_strategy.into(),
            refuse_overlaps: false,
            revision: 0,
        }
    }

    /// This policy with `kinds` as its custom-kind table, which every custom
    /// kind given to [`RewritePolicy::with`] afterwards indexes. Drops the
    /// overrides of custom kinds made under an earlier table.
    pub fn with_custom_kinds(mut self, kinds: CustomKinds) -> RewritePolicy {
        self.overrides.retain(|(k, _)| !k.is_custom());
        self.custom_kinds = kinds;
        self
    }

    /// The table every custom kind of this policy indexes.
    pub fn custom_kinds(&self) -> &CustomKinds {
        &self.custom_kinds
    }

    /// Assigns `strategy` to mentions of `kind` (built-in, or a custom kind
    /// of [`RewritePolicy::custom_kinds`]).
    ///
    /// # Panics
    ///
    /// When `kind` is a custom kind outside the policy's table; see
    /// [`RewritePolicy::try_with`].
    pub fn with(self, kind: PiiKind, strategy: impl Into<RewriteStrategy>) -> RewritePolicy {
        self.try_with(kind, strategy).expect("a kind of the policy's custom-kind table")
    }

    /// Assigns `strategy` to mentions of `kind`, refused when `kind` is a
    /// custom kind outside [`RewritePolicy::custom_kinds`].
    pub fn try_with(mut self, kind: PiiKind, strategy: impl Into<RewriteStrategy>) -> Result<RewritePolicy, KindError> {
        self.custom_kinds.check(kind)?;
        let strategy = strategy.into();
        match self.overrides.binary_search_by_key(&kind.code(), |(k, _)| k.code()) {
            Ok(at) => self.overrides[at].1 = strategy,
            Err(at) => self.overrides.insert(at, (kind, strategy)),
        }
        Ok(self)
    }

    /// The strategy configured for `kind` of this policy's table.
    pub fn strategy(&self, kind: PiiKind) -> &RewriteStrategy {
        self.overrides
            .binary_search_by_key(&kind.code(), |(k, _)| k.code())
            .map_or(&self.default_strategy, |at| &self.overrides[at].1)
    }

    /// The strategy for `kind` of a layer whose custom kinds index `kinds`:
    /// a custom kind is matched to this policy's by name, and one the
    /// policy does not list takes the default strategy.
    pub fn strategy_in(&self, kinds: &CustomKinds, kind: PiiKind) -> &RewriteStrategy {
        match self.custom_kinds.translate(kinds, kind) {
            Some(k) => self.strategy(k),
            None => &self.default_strategy,
        }
    }

    /// This policy at `revision`, the caller's number for this version of
    /// it, which every edit it makes records. It names what the edits
    /// layer leaves out, such as keys and lookup tables.
    pub fn with_revision(mut self, revision: u64) -> RewritePolicy {
        self.revision = revision;
        self
    }

    /// The caller's revision of this policy; 0 when none was given.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// The rule of this policy for `kind` of a layer whose custom kinds
    /// index `kinds`: its override, named as the layer names the kind, or
    /// the default.
    pub fn rule_in(&self, kinds: &CustomKinds, kind: PiiKind) -> PiiRule {
        match self.custom_kinds.translate(kinds, kind) {
            Some(k) if self.overrides.binary_search_by_key(&k.code(), |(o, _)| o.code()).is_ok() => PiiRule::Kind(kind),
            _ => PiiRule::Default,
        }
    }

    /// Folds the revision, whether the policy refuses overlaps, the
    /// default strategy and each override, a custom kind by its name, with
    /// every parameter of each strategy, into `h`.
    pub(crate) fn fold_config(&self, h: &mut turbo_analysis::Fnv1a) {
        h.u64(self.revision).u8(u8::from(self.refuse_overlaps));
        self.default_strategy.fold_config(h);
        h.usize(self.overrides.len());
        for (kind, strategy) in &self.overrides {
            match kind.builtin_id() {
                Some(id) => h.u8(0).u16(id),
                None => h.u8(1).str(self.custom_kinds.name(*kind).unwrap_or("")),
            };
            strategy.fold_config(h);
        }
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
    pub fn refuses_overlaps(&self) -> bool {
        self.refuse_overlaps
            || self.default_strategy.refuses_overlaps()
            || self.overrides.iter().any(|(_, s)| s.refuses_overlaps())
    }

    /// Replaces every mention of `pii`, a layer over `src`, with the
    /// strategy configured for its [`PiiKind`], as a view derived from
    /// `src`. Each edit records the rule of its cluster's winning kind and
    /// this policy's revision.
    pub fn rewrite(&self, src: &View, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let text = src.text();
        let kinds = &pii.custom_kinds;
        let mut state = RewriteState::new(kinds);
        let refuses = |c: &MentionCluster<'_>| {
            self.refuse_overlaps || c.members.iter().any(|m| self.strategy_in(kinds, m.kind).refuses_overlaps())
        };
        PiiRewrite::derive(src, pii, refuses, self.revision, |cluster, out| {
            let winner = cluster.merged.kind;
            let strategy = self.strategy_in(kinds, winner);
            if self.refuse_overlaps {
                state.write_strategy(strategy, text, &cluster.merged, out)?;
            } else {
                state.write_cluster(|m| self.strategy_in(kinds, m.kind), text, cluster, out)?;
            }
            Ok(EditOrigin { rule: self.rule_in(kinds, winner), strategy: strategy.kind(), rule_id: "" })
        })
    }
}
