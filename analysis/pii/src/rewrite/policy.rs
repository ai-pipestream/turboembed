//! Composable per-[`PiiKind`] rewrite policy ([`RewritePolicy`],
//! [`RewriteStrategy`]).
//!
//! Callers configure a default [`RewriteStrategy`] and override individual
//! built-in or [`PiiKind::Custom`] kinds with `.with(kind, strategy)`.

use std::collections::HashMap;

use turbo_analysis_layers::{CustomKinds, KindError, Pii, PiiKind, PiiMention};

use super::kind_name;
use super::mask::CharRetention;
use super::replace::MentionCluster;
use super::{
    Abbreviator, FormatRedactor, HmacTokenizer, MapReplacer, MaskPolicy, PiiRewrite, Pseudonymizer, RewriteError,
    Sha256Tokenizer, StaticReplacer, Truncate,
};

#[derive(Copy, Clone)]
pub(super) enum MemberInitial {
    Abbreviate,
    Retention(CharRetention),
}

impl MemberInitial {
    #[inline]
    fn keeps(self, piece: &str, rel_at: usize, c: char) -> bool {
        match self {
            MemberInitial::Abbreviate => Abbreviator::is_initial_at(piece, rel_at),
            MemberInitial::Retention(ret) => ret.keeps(rel_at, c),
        }
    }
}

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

impl RewriteStrategy {
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
        }
    }

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
            | RewriteStrategy::Abbreviate(_) => CharRetention::NONE,
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

pub(crate) struct RewriteState<'a> {
    kinds: &'a CustomKinds,
    assigned: HashMap<(PiiKind, &'a str), u32>,
    counters: HashMap<PiiKind, u32>,
}

impl<'a> RewriteState<'a> {
    /// Numbering state for mentions of a layer whose custom kinds index
    /// `kinds`.
    pub(crate) fn new(kinds: &'a CustomKinds) -> RewriteState<'a> {
        RewriteState { kinds, assigned: HashMap::new(), counters: HashMap::new() }
    }

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
    ) -> Result<(), RewriteError> {
        match strategy {
            RewriteStrategy::Mask(policy) => {
                out.push_str(&policy.masked(&text[m.span.range()]));
                Ok(())
            }
            RewriteStrategy::Pseudonym(pseudo) => {
                let n = self.number(m.kind, m.normalized);
                pseudo.write_label(kind_name(self.kinds, m.kind), n, out);
                Ok(())
            }
            RewriteStrategy::Hmac(hmac) => hmac
                .write_token(kind_name(self.kinds, m.kind), m.normalized, out)
                .map_err(|_| RewriteError::EmptyValue(m.span)),
            RewriteStrategy::Format(fmt) => {
                let n = if fmt.uses_number() { self.number(m.kind, m.normalized) } else { 1 };
                fmt.write_label(kind_name(self.kinds, m.kind), n, out);
                Ok(())
            }
            RewriteStrategy::Static(stat) => {
                out.push_str(stat.replacement_in(self.kinds, m.kind));
                Ok(())
            }
            RewriteStrategy::Sha256(sha) => sha
                .write_digest(kind_name(self.kinds, m.kind), m.normalized, out)
                .map_err(|_| RewriteError::EmptyValue(m.span)),
            RewriteStrategy::Map(map) => map.write_mention(self, text, m, out),
            RewriteStrategy::Truncate(trunc) => {
                trunc.write_truncated(&text[m.span.range()], out);
                Ok(())
            }
            RewriteStrategy::Abbreviate(abbr) => {
                abbr.write_abbreviated(&text[m.span.range()], out);
                Ok(())
            }
        }
    }

    pub(crate) fn write_cluster<'b>(
        &mut self,
        strategy_for: impl Fn(PiiKind) -> &'b RewriteStrategy,
        text: &str,
        cluster: &MentionCluster<'a>,
        out: &mut String,
    ) -> Result<(), RewriteError> {
        for mem in &cluster.members {
            if mem.normalized.is_empty() && strategy_for(mem.kind).requires_normalized(text, mem) {
                return Err(RewriteError::EmptyValue(mem.span));
            }
        }
        let winner = &cluster.merged;
        if cluster.members.len() == 1 {
            return self.write_strategy(strategy_for(winner.kind), text, winner, out);
        }
        let mut strategy = strategy_for(winner.kind);
        while let RewriteStrategy::Map(map) = strategy {
            if let Some(rep) = map.lookup_mention(text, winner) {
                out.push_str(rep);
                return Ok(());
            }
            strategy = map.fallback();
        }
        let piece = &text[winner.span.range()];
        match strategy {
            RewriteStrategy::Truncate(trunc) => {
                let winner_ret = trunc.retention(piece);
                let member_rets: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, strategy_for(mem.kind).retention(text, mem))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = winner.span.start + rel_at as u32;
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    out.push(if keep { c } else { trunc.mask() });
                }
                Ok(())
            }
            RewriteStrategy::Mask(policy) => {
                let winner_ret = policy.retention(piece);
                let member_rets: Vec<_> =
                    cluster.members.iter().map(|mem| (mem.span, strategy_for(mem.kind).retention(text, mem))).collect();
                for (rel_at, c) in piece.char_indices() {
                    let abs_at = winner.span.start + rel_at as u32;
                    let keep = winner_ret.keeps(rel_at, c)
                        && member_rets
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, ret)| ret.keeps((abs_at - span.start) as usize, c));
                    if keep {
                        out.push(c);
                    } else {
                        for _ in 0..c.len_utf8() {
                            out.push(policy.mask());
                        }
                    }
                }
                Ok(())
            }
            RewriteStrategy::Abbreviate(abbr) => {
                let member_inits: Vec<_> = cluster
                    .members
                    .iter()
                    .map(|mem| (mem.span, strategy_for(mem.kind).member_initial(text, mem)))
                    .collect();
                abbr.write_abbreviated_filtered(
                    piece,
                    |rel_at, c| {
                        let abs_at = winner.span.start + rel_at as u32;
                        member_inits
                            .iter()
                            .filter(|(span, _)| span.start <= abs_at && abs_at < span.end)
                            .all(|(span, init)| init.keeps(&text[span.range()], (abs_at - span.start) as usize, c))
                    },
                    out,
                );
                Ok(())
            }
            other => self.write_strategy(other, text, winner, out),
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

    /// Replaces every mention of `pii`, a layer over `text`, with the
    /// strategy configured for its [`PiiKind`].
    pub fn rewrite(&self, text: &str, pii: &Pii) -> Result<PiiRewrite, RewriteError> {
        let mut state = RewriteState::new(&pii.custom_kinds);
        if self.refuse_overlaps {
            return PiiRewrite::try_replace(text, pii, pii.mentions(), |m, out| {
                state.write_strategy(self.strategy_in(&pii.custom_kinds, m.kind), text, m, out)
            });
        }
        let clusters = PiiRewrite::group_clusters(text, &pii.custom_kinds, pii.mentions())?;
        for cluster in &clusters {
            if cluster.members.len() > 1
                && cluster.members.iter().any(|m| self.strategy_in(&pii.custom_kinds, m.kind).refuses_overlaps())
            {
                return Err(RewriteError::Overlap(cluster.members[0].span, cluster.members[1].span));
            }
        }
        let mut idx = 0;
        PiiRewrite::try_replace(text, pii, clusters.iter().map(|c| c.merged), |_, out| {
            let cluster = &clusters[idx];
            idx += 1;
            state.write_cluster(|k| self.strategy_in(&pii.custom_kinds, k), text, cluster, out)
        })
    }
}
