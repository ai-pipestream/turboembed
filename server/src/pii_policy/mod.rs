//! The typed PII redaction policy (issue #1882 part A): the native form of
//! a `turbo.result.v1.RedactionPolicy` message, its conversion from the
//! wire with typed errors, and its canonical [`PolicyDigest`].
//!
//! - [`RedactionPolicy::from_proto`] converts a message into #1902's
//!   rewrite types ([`RewriteStrategy`]) and the detector's list types,
//!   resolving key and dictionary references through a
//!   [`PolicyResolver`]. Every enum's UNSPECIFIED value is refused; an
//!   unknown kind id or strategy arm is refused unless the policy opts
//!   into redact-and-flag, which masks and records a [`PolicyFlag`] and
//!   never passes a mention through.
//! - A kind with no rule, or whose rule has no strategy that applies, is
//!   rewritten with the policy's default: a mask or a keyed token, never a
//!   per-document pseudonym or an unkeyed SHA-256.
//! - [`canonical::encode`] is the policy's canonical encoding and
//!   [`PolicyDigest::of_canonical`] its identity; see that module for the
//!   exact rules.

pub mod canonical;
mod convert;
mod detect;
mod error;
mod named;
mod redact;
mod registry;
mod resolve;
mod strategy;

pub use detect::IDENTIFIERS_DETECTOR;
pub use error::{At, PolicyError};
pub use named::{NamedPolicies, PolicyFiles, Redaction, SelectError, select};
pub use redact::{REDACT_STEP, RedactPlanStep, RedactStep};
pub use registry::{MAP_FORMAT, ServerResolver, TERMS_FORMAT};
pub use resolve::{KEY_MATERIAL_HEX_LEN, MAX_REF_ID_BYTES, NoRegistry, PolicyResolver, ResolveError};
pub use strategy::{edit_strategy, strategy_to_proto, whole_mask};
pub use turbo_analysis::PolicyDigest;

use turbo_pii::rewrite::RewriteStrategy;
use turbo_pii::{AllowList, ChecksumAlgorithm, CustomKinds, DenyList, LabelPolicy, Pattern, PiiEvidence, PiiKind};
use turbo_pii::{PiiScore, PiiScoreKind};

/// A converted redaction policy. Its `Debug` names the policy by id,
/// revision and digest only: terms, map tables and keys are never printed.
#[derive(Clone)]
pub struct RedactionPolicy {
    pub id: String,
    pub revision: u64,
    /// The SHA-256 of the message's canonical encoding.
    pub digest: PolicyDigest,
    /// The policy's custom kinds; every custom [`PiiKind`] below indexes it.
    pub custom_kinds: CustomKinds,
    /// The kind rules in message order, at most one per kind.
    pub rules: Vec<KindRule>,
    /// What a kind with no applicable strategy is rewritten with.
    pub default_strategy: RewriteStrategy,
    pub allow: AllowList,
    pub deny: Vec<DenyRule>,
    pub custom_identifiers: Vec<CustomIdentifier>,
    /// What redact-and-flag replaced, in message order; empty unless the
    /// policy opted in.
    pub flags: Vec<PolicyFlag>,
}

/// What a policy does with one kind.
#[derive(Clone, Debug)]
pub struct KindRule {
    pub rule_id: String,
    pub kind: PiiKind,
    /// False: the kind is not detected.
    pub enabled: bool,
    pub strategies: Vec<StrategyEntry>,
    pub allow: AllowList,
    pub labels: Option<LabelOverride>,
}

/// One strategy of a rule and the conditions under which it applies.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyEntry {
    /// All must hold; empty means always.
    pub when: Vec<Condition>,
    pub strategy: RewriteStrategy,
}

/// A test over a mention's recorded evidence facts.
#[derive(Clone, Debug, PartialEq)]
pub enum Condition {
    /// A checksum fact passed, with one of these algorithms (any when
    /// empty).
    ChecksumPassed(Vec<ChecksumAlgorithm>),
    /// A label fact is recorded.
    LabelPresent,
    /// The detector has this id and, when given, this version.
    Detector { id: String, version: Option<String> },
    /// A score of `kind` at least `min`, calibrated when `calibrated`.
    ScoreAtLeast { kind: PiiScoreKind, min: f32, calibrated: bool },
}

/// The facts of one mention a [`Condition`] reads.
#[derive(Copy, Clone, Debug)]
pub struct MentionFacts<'a> {
    pub evidence: &'a [PiiEvidence],
    pub detector_id: &'a str,
    pub detector_version: &'a str,
    pub score: Option<&'a PiiScore>,
}

impl Condition {
    /// Whether the condition holds for `facts`.
    pub fn holds(&self, facts: &MentionFacts<'_>) -> bool {
        match self {
            Condition::ChecksumPassed(algorithms) => facts.evidence.iter().any(|e| match e {
                PiiEvidence::Checksum { algorithm, passed: true } => {
                    algorithms.is_empty() || algorithms.contains(algorithm)
                }
                _ => false,
            }),
            Condition::LabelPresent => facts.evidence.iter().any(|e| e.is_label()),
            Condition::Detector { id, version } => {
                facts.detector_id == id && version.as_deref().is_none_or(|v| v == facts.detector_version)
            }
            Condition::ScoreAtLeast { kind, min, calibrated } => {
                facts.score.is_some_and(|s| s.kind == *kind && s.value >= *min && (!calibrated || s.calibrated))
            }
        }
    }
}

/// A kind's context label settings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LabelOverride {
    pub policy: LabelPolicy,
    /// `None` keeps the detector's window.
    pub window_words: Option<usize>,
    pub extra_labels: Vec<String>,
}

/// Terms that are always PII of a custom kind.
#[derive(Clone, Debug)]
pub struct DenyRule {
    pub rule_id: String,
    pub list: DenyList,
}

/// A shape pattern that finds a custom kind.
#[derive(Clone, Debug)]
pub struct CustomIdentifier {
    pub rule_id: String,
    pub kind: PiiKind,
    pub pattern: Pattern,
}

/// What redact-and-flag replaced with a whole-value `'*'` mask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PolicyFlag {
    /// A rule for a built-in kind id this server does not know was
    /// dropped: no mention of it can reach the policy.
    UnknownKind { rule_id: String, id: i32 },
    /// A strategy entry whose strategy or one of whose conditions this
    /// server does not know became an unconditional whole-value mask.
    UnknownStrategy { rule_id: String, entry: usize },
}

impl std::fmt::Debug for RedactionPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RedactionPolicy")
            .field("id", &self.id)
            .field("revision", &self.revision)
            .field("digest", &self.digest.to_hex())
            .field("rules", &self.rules.len())
            .field("flags", &self.flags.len())
            .finish_non_exhaustive()
    }
}

impl RedactionPolicy {
    /// The rule for `kind`, a kind of [`RedactionPolicy::custom_kinds`].
    pub fn rule(&self, kind: PiiKind) -> Option<&KindRule> {
        self.rules.iter().find(|r| r.kind == kind)
    }

    /// The strategy a mention of `kind` with `facts` is rewritten with, and
    /// the id of the rule that chose it (`None`: the policy default).
    pub fn strategy_for(&self, kind: PiiKind, facts: &MentionFacts<'_>) -> (&RewriteStrategy, Option<&str>) {
        if let Some(rule) = self.rule(kind)
            && let Some(entry) = rule.strategies.iter().find(|e| e.when.iter().all(|c| c.holds(facts)))
        {
            return (&entry.strategy, Some(&rule.rule_id));
        }
        (&self.default_strategy, None)
    }
}
