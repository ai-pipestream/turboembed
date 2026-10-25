//! `ModalityAnnotator`: deterministic negation scope, epistemic
//! factuality and deontic modality over the dependency tree.
//!
//! Cues (`not`, `might`, `allegedly`, `if`, `realized that`, `must`,
//! `permitted to`, `prohibited from`, `may`) are found with one
//! `turbo-matcher` phrase automaton over lowercase tokens. Each cue is
//! scoped to its governing predicate: the nearest clausal node above it
//! in the dependency tree, whose subtree is the scope minus coordinated
//! (`conj`), adverbial (`advcl`) and paratactic clauses. Without
//! dependencies, a cue scopes forward to the next punctuation mark or
//! clause conjunction, at most `max_window_tokens` tokens. Cues that
//! share a scope compose into one [`ModalityTag`]: an odd number of
//! negations is `Negated` (`not impossible` is affirmed and hedged), the
//! epistemic status is the strongest of conditional, reported, hedged
//! and presupposed, and a negation after an obligation is a prohibition
//! (`must not`) while one before it removes it (`not required to`).
//! `may` is a permission when its subject is one of `deontic_subjects`,
//! else a hedge.

mod cues;
mod scope;

use std::collections::BTreeMap;

use turbo_analysis::{Annotator, Document, Error, Fnv1a, Key, Refs, Span, View};
use turbo_analysis_layers::modality::{DeonticKind, EpistemicStatus, Modality, ModalityTag, Polarity};
use turbo_analysis_layers::{Dependencies, Tokens};

use cues::{Class, Cue, Lexicon};
use scope::{Input, Scope};

/// How [`ModalityAnnotator`] scopes and resolves cues.
#[derive(Debug, Clone, PartialEq)]
pub struct ModalityConfig {
    /// The longest window scope, in tokens past the cue, when there are
    /// no dependencies.
    pub max_window_tokens: usize,
    /// Lower-case subjects that make `may` a permission.
    pub deontic_subjects: Vec<String>,
}

impl Default for ModalityConfig {
    fn default() -> Self {
        let parties = [
            "user",
            "users",
            "customer",
            "customers",
            "vendor",
            "vendors",
            "supplier",
            "suppliers",
            "employee",
            "employees",
            "tenant",
            "tenants",
            "landlord",
            "licensee",
            "licensees",
            "licensor",
            "party",
            "parties",
            "contractor",
            "contractors",
            "member",
            "members",
            "you",
            "buyer",
            "seller",
            "subscriber",
            "subscribers",
            "staff",
            "visitors",
            "applicants",
        ];
        ModalityConfig { max_window_tokens: 8, deontic_subjects: parties.iter().map(|s| s.to_string()).collect() }
    }
}

/// Writes a [`Modality`] layer from tokens and, when present,
/// dependencies.
pub struct ModalityAnnotator {
    pub config: ModalityConfig,
    lexicon: Lexicon,
}

impl Default for ModalityAnnotator {
    fn default() -> Self {
        Self::with_config(ModalityConfig::default())
    }
}

impl std::fmt::Debug for ModalityAnnotator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModalityAnnotator").field("config", &self.config).finish_non_exhaustive()
    }
}

impl ModalityAnnotator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_config(config: ModalityConfig) -> Self {
        ModalityAnnotator { config, lexicon: Lexicon::new() }
    }

    /// The modality of the default layers of `doc`, without adding it.
    pub fn modality(&self, doc: &Document) -> Result<Modality, Error> {
        let (m,) = self.annotate((
            doc.require(&Key::<View>::DEFAULT)?,
            doc.require(&Key::<Tokens>::DEFAULT)?,
            doc.get_key(&Key::<Dependencies>::DEFAULT),
        ))?;
        Ok(m)
    }

    fn fail(detail: String) -> Error {
        Error::Annotator { annotator: <Self as Annotator>::ID, detail }
    }
}

/// The cues sharing one scope, in text order.
struct Group {
    lo: usize,
    hi: usize,
    cues: Vec<(Cue, Class)>,
}

impl Annotator for ModalityAnnotator {
    const ID: &'static str = "turbo:modality-annotator";
    const VERSION: &'static str = "1";
    type Reads = (View, Tokens, Option<Dependencies>);
    type Writes = (Modality,);

    fn annotate(&self, (view, tokens, deps): Refs<'_, Self>) -> Result<Self::Writes, Error> {
        let n = tokens.spans.len();
        if deps.is_some_and(|d| d.head.len() != n) {
            return Err(Self::fail(format!("dependencies not aligned to the {n} tokens")));
        }
        // The scope walks climb heads and descend children; a cycle would never end.
        if let Some(d) = deps {
            d.check_forest().map_err(Self::fail)?;
        }
        let mut out = Modality::new(view.id());
        if n == 0 {
            return Ok((out,));
        }
        let found = self.lexicon.find(view.text(), tokens).map_err(|e| Self::fail(e.to_string()))?;
        let input = Input::new(view.text(), tokens.spans.iter().map(|(_, s)| s).collect(), deps);
        let mut groups: BTreeMap<usize, Group> = BTreeMap::new();
        for cue in found {
            let Scope { key, lo, hi, predicate } = input.scope(&cue, self.config.max_window_tokens);
            let class = match cue.class {
                Class::May if input.party_subject(cue.start, &self.config.deontic_subjects, predicate) => Class::Permit,
                Class::May => Class::Hedge,
                c => c,
            };
            let g = groups.entry(key).or_insert(Group { lo, hi, cues: Vec::new() });
            g.lo = g.lo.min(lo);
            g.hi = g.hi.max(hi);
            g.cues.push((cue, class));
        }
        let mut rows: Vec<(Span, ModalityTag)> =
            groups.into_values().map(|g| (input.bytes(g.lo, g.hi), compose(&g.cues, &input, input.parsed()))).collect();
        rows.sort_by_key(|(s, _)| (s.start, std::cmp::Reverse(s.end)));
        for (scope, tag) in rows {
            out.push(scope, tag);
        }
        Ok((out,))
    }

    fn config_hash(&self) -> u64 {
        let mut h = Fnv1a::new();
        h.bytes(Self::VERSION.as_bytes())
            .bytes(&(self.config.max_window_tokens as u64).to_le_bytes())
            .bytes(&(self.lexicon.phrases() as u64).to_le_bytes());
        for s in &self.config.deontic_subjects {
            h.bytes(s.as_bytes()).bytes(&[0]);
        }
        h.finish()
    }
}

/// One tag from the cues sharing a scope.
fn compose(cues: &[(Cue, Class)], input: &Input<'_>, parsed: bool) -> ModalityTag {
    let cue_span = |c: &Cue| input.bytes(c.start, c.end - 1);
    let negs: Vec<&Cue> =
        cues.iter().filter(|(_, k)| matches!(k, Class::Neg | Class::NegHedge)).map(|(c, _)| c).collect();
    let first = |k: Class| cues.iter().find(|(_, c)| *c == k).map(|(c, _)| c);
    let polarity = if negs.len() % 2 == 1 { Polarity::Negated } else { Polarity::Affirmed };
    let negated = polarity == Polarity::Negated;

    let hedge = first(Class::Hedge).or_else(|| first(Class::NegHedge));
    let (epistemic, epistemic_cue) = if let Some(c) = first(Class::Cond) {
        (EpistemicStatus::Conditional, Some(c))
    } else if let Some(c) = first(Class::Reported) {
        (EpistemicStatus::Reported, Some(c))
    } else if let Some(c) = hedge {
        (EpistemicStatus::Hedged, Some(c))
    } else if let Some(c) = first(Class::Factive) {
        (EpistemicStatus::Presupposed, Some(c))
    } else {
        (EpistemicStatus::Asserted, None)
    };

    let (deontic, deontic_cue) = if let Some(c) = first(Class::Prohib) {
        (if negated { DeonticKind::Permission } else { DeonticKind::Prohibition }, Some(c))
    } else if let Some(c) = first(Class::Oblig) {
        if negs.iter().any(|n| n.start > c.start) {
            (DeonticKind::Prohibition, Some(c))
        } else if negs.iter().any(|n| n.start < c.start) {
            (DeonticKind::None, None)
        } else {
            (DeonticKind::Obligation, Some(c))
        }
    } else if let Some(c) = first(Class::Permit) {
        (if negated { DeonticKind::Prohibition } else { DeonticKind::Permission }, Some(c))
    } else {
        (DeonticKind::None, None)
    };

    let cue = deontic_cue
        .or(if negated { negs.first().copied() } else { None })
        .or(epistemic_cue)
        .or(cues.first().map(|(c, _)| c))
        .map(cue_span);
    let soft = cues.iter().any(|(c, _)| matches!(input.lower(c.start).as_str(), "may" | "should"));
    let confidence = match (parsed, soft) {
        (_, true) => 0.6,
        (true, false) => 0.9,
        (false, false) => 0.7,
    };
    ModalityTag { polarity, epistemic, deontic, cue, confidence }
}
