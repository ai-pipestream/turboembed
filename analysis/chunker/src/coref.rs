//! Coreference chain continuity boundary costs (`CorefCohesion`,
//! `CorefBoundaries`) and per-chunk antecedent context headers
//! (`ChunkAntecedents`, `ChunkAntecedentBuilder`).
//!
//! Reads coreference clusters (`Groups<Mentions>` / `Groups<Entities>`) and/or
//! directed antecedent links (`Edges<Mentions, Mentions>` /
//! `Edges<Entities, Entities>`) alongside [`Sentences`], penalizing boundary
//! cuts that sever active coreference chains across adjacent sentences
//! (especially unresolved pronominal mentions whose antecedent is in an
//! earlier sentence) while rewarding clean boundaries where no active
//! coreference chain crosses.

mod extract;
mod mentions;

pub use extract::extract_chains;
pub use mentions::{
    Agreement, Animacy, AntecedentLink, ChunkAntecedentBuilder, ChunkAntecedents, Gender, MentionKind, MentionOrigin,
    Mentions, Number, Person, chunk_antecedents,
};

use std::num::NonZeroU32;

use turbo_analysis::{Annotator, Bound, Edges, Error, Groups, Refs, Span, View};
use turbo_analysis_layers::{Boundaries, Cut, Entities, Sentences};

use crate::TERM_MAX;
use crate::weigh::COREF;

pub(crate) const MILLI: i64 = 1_000;

/// Parameters for [`coref_boundaries`] and [`CorefBoundaries`].
///
/// Costs are expressed in thousandths of a [`Boundaries::UNIT`], so every
/// calculation is deterministic integer arithmetic.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct CorefCohesion {
    /// Penalty added to a gap for each active coreference chain link that
    /// crosses the gap within `max_sentence_distance`. Default `600`.
    pub chain_penalty: u32,
    /// Additional penalty added when a coreference link directly connects
    /// immediately adjacent sentences `g` and `g + 1`. Default `400`.
    pub adjacent_penalty: u32,
    /// Additional penalty added when a pronominal or demonstrative mention in
    /// `[g + 1, g + pronoun_window]` resolves to an antecedent in sentence
    /// `<= g` across gap `g`. Default `1000` (one [`Boundaries::UNIT`]).
    pub pronoun_penalty: u32,
    /// Cost subtracted (`cost < 0`) at a gap when mentions exist in the
    /// surrounding window and zero active coreference chains cross the gap.
    /// Default `300`.
    pub break_reward: u32,
    /// Maximum sentence distance (`s_hi - s_lo`) for a coreference link to
    /// count as an active continuity bridge across intervening gaps. Default `3`.
    pub max_sentence_distance: NonZeroU32,
    /// Sentences after gap `g` checked for pronominal mentions whose
    /// antecedent lies at or before `g`. Default `1`.
    pub pronoun_window: NonZeroU32,
    /// Minimum `break_reward` to label a clean entity-turnover boundary
    /// [`Cut::TopicShift`] when both sides hold mentions. Default `300`.
    pub shift_threshold: u32,
}

impl Default for CorefCohesion {
    fn default() -> Self {
        Self {
            chain_penalty: 600,
            adjacent_penalty: 400,
            pronoun_penalty: Boundaries::UNIT as u32,
            break_reward: 300,
            max_sentence_distance: NonZeroU32::new(3).expect("3 > 0"),
            pronoun_window: NonZeroU32::new(1).expect("1 > 0"),
            shift_threshold: 300,
        }
    }
}

impl CorefCohesion {
    /// Whole-chunk term for sentences `[i, j)`: penalizes chunks that contain
    /// mentions (especially pronominal mentions) whose antecedent lies before
    /// `i` or whose chain continues past `j` within `max_sentence_distance`.
    pub fn term(&self, chains: &CorefChains, i: usize, j: usize) -> i64 {
        if i >= j {
            return 0;
        }
        let max_dist = self.max_sentence_distance.get() as usize;
        let mut total: i64 = 0;
        for link in &chains.links {
            if link.s_hi - link.s_lo > max_dist {
                continue;
            }
            let severed_left = link.s_lo < i && (i..j).contains(&link.s_hi);
            let severed_right = (i..j).contains(&link.s_lo) && link.s_hi >= j;
            if severed_left {
                let mut p = i64::from(self.chain_penalty);
                if link.s_hi == i && link.s_lo + 1 == i {
                    p += i64::from(self.adjacent_penalty);
                }
                if link.target_is_pronoun {
                    p += i64::from(self.pronoun_penalty);
                }
                total = total.saturating_add((p * link.weight_milli) / MILLI);
            } else if severed_right {
                let mut p = i64::from(self.chain_penalty);
                if link.s_lo + 1 == j && link.s_hi == j {
                    p += i64::from(self.adjacent_penalty);
                }
                total = total.saturating_add((p * link.weight_milli) / MILLI);
            }
        }
        total.clamp(-TERM_MAX, TERM_MAX)
    }
}

/// One resolved mention anchored to its sentence index and byte span.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedMention {
    pub sentence: usize,
    pub span: Span,
    pub is_pronoun: bool,
}

/// One directed coreference continuity link from sentence `s_lo` to `s_hi` (`s_lo < s_hi`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorefLink {
    pub s_lo: usize,
    pub s_hi: usize,
    pub span_lo: Span,
    pub span_hi: Span,
    pub target_is_pronoun: bool,
    /// Confidence weight in thousandths `[1, 1000]`.
    pub weight_milli: i64,
}

/// Extracted coreference clusters and cross-sentence continuity links over a
/// document's sentences.
#[derive(Clone, Debug, Default)]
pub struct CorefChains {
    pub sentence_count: usize,
    pub sentence_has_mention: Vec<bool>,
    pub clusters: Vec<Vec<ResolvedMention>>,
    pub links: Vec<CorefLink>,
}

pub(crate) fn is_pronoun_text(text: &str) -> bool {
    matches!(
        text.trim().to_ascii_lowercase().as_str(),
        "i" | "me"
            | "my"
            | "mine"
            | "myself"
            | "we"
            | "us"
            | "our"
            | "ours"
            | "ourselves"
            | "you"
            | "your"
            | "yours"
            | "yourself"
            | "yourselves"
            | "he"
            | "him"
            | "his"
            | "himself"
            | "she"
            | "her"
            | "hers"
            | "herself"
            | "it"
            | "its"
            | "itself"
            | "they"
            | "them"
            | "their"
            | "theirs"
            | "themselves"
            | "this"
            | "that"
            | "these"
            | "those"
            | "who"
            | "whom"
            | "whose"
            | "which"
    )
}

/// Scores every sentence boundary in `sentences` from coreference chains and
/// antecedent links.
#[allow(clippy::too_many_arguments)]
pub fn coref_boundaries(
    sentences: &Sentences,
    view: Option<&View>,
    mentions: Option<&Mentions>,
    mention_groups: Option<&Groups<Mentions>>,
    mention_edges: Option<&Edges<Mentions, Mentions>>,
    entities: Option<&Entities>,
    entity_groups: Option<&Groups<Entities>>,
    entity_edges: Option<&Edges<Entities, Entities>>,
    cohesion: CorefCohesion,
) -> Boundaries {
    let chains =
        extract_chains(sentences, view, mentions, mention_groups, mention_edges, entities, entity_groups, entity_edges);
    coref_boundaries_from_chains(&chains, cohesion)
}

/// Computes [`Boundaries`] directly from extracted [`CorefChains`].
pub fn coref_boundaries_from_chains(chains: &CorefChains, cfg: CorefCohesion) -> Boundaries {
    let n = chains.sentence_count;
    if n == 0 {
        return Boundaries { cost: Vec::new(), reason: Vec::new() };
    }
    let mut cost = vec![0i32; n];
    let mut reason = vec![Cut::Sentence; n];
    let max_dist = cfg.max_sentence_distance.get() as usize;
    let pron_win = cfg.pronoun_window.get() as usize;

    for g in 0..n.saturating_sub(1) {
        let mut penalty: i64 = 0;
        for link in &chains.links {
            if link.s_lo <= g && g < link.s_hi && link.s_hi - link.s_lo <= max_dist {
                penalty = penalty.saturating_add((i64::from(cfg.chain_penalty) * link.weight_milli) / MILLI);
                if link.s_lo == g && link.s_hi == g + 1 {
                    penalty = penalty.saturating_add((i64::from(cfg.adjacent_penalty) * link.weight_milli) / MILLI);
                }
                if link.target_is_pronoun && link.s_hi <= g + pron_win {
                    penalty = penalty.saturating_add((i64::from(cfg.pronoun_penalty) * link.weight_milli) / MILLI);
                }
            }
        }
        if penalty > 0 {
            cost[g] = penalty.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        } else {
            let l_lo = (g + 1).saturating_sub(max_dist);
            let r_hi = (g + 1 + max_dist).min(n);
            let left_has = chains.sentence_has_mention[l_lo..=g].iter().any(|&b| b);
            let right_has = chains.sentence_has_mention[g + 1..r_hi].iter().any(|&b| b);
            if left_has && right_has && cfg.break_reward > 0 {
                let reward = -(i64::from(cfg.break_reward));
                cost[g] = reward.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
                if cfg.break_reward >= cfg.shift_threshold {
                    reason[g] = Cut::TopicShift;
                }
            }
        }
    }

    Boundaries { cost, reason }
}

/// Annotator that writes coreference continuity boundary costs under [`COREF`]
/// (`"coref"`).
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct CorefBoundaries {
    pub cohesion: CorefCohesion,
}

impl CorefBoundaries {
    pub const fn new(cohesion: CorefCohesion) -> Self {
        Self { cohesion }
    }

    /// Binds the output [`Boundaries`] to [`COREF`].
    pub fn bound(self) -> Bound<Self> {
        Bound::new(self).write(0, COREF)
    }
}

impl Annotator for CorefBoundaries {
    const ID: &'static str = "turbo:coref-boundaries";
    type Reads = (
        Sentences,
        Option<Mentions>,
        Option<Groups<Mentions>>,
        Option<Edges<Mentions, Mentions>>,
        Option<Entities>,
        Option<Groups<Entities>>,
        Option<Edges<Entities, Entities>>,
        Option<View>,
    );
    type Writes = (Boundaries,);

    fn annotate(
        &self,
        (sents, mentions, m_groups, m_edges, entities, e_groups, e_edges, view): Refs<'_, Self>,
    ) -> Result<(Boundaries,), Error> {
        let v = sents.spans.view();
        if mentions.is_some_and(|m| m.spans.view() != v)
            || entities.is_some_and(|e| e.spans.view() != v)
            || view.is_some_and(|vw| vw.id() != v)
        {
            return Err(Error::Annotator { annotator: Self::ID, detail: "its inputs are in different views".into() });
        }
        Ok((coref_boundaries(sents, view, mentions, m_groups, m_edges, entities, e_groups, e_edges, self.cohesion),))
    }

    fn config_hash(&self) -> u64 {
        let c = self.cohesion;
        [
            u64::from(c.chain_penalty),
            u64::from(c.adjacent_penalty),
            u64::from(c.pronoun_penalty),
            u64::from(c.break_reward),
            u64::from(c.max_sentence_distance.get()),
            u64::from(c.pronoun_window.get()),
            u64::from(c.shift_threshold),
        ]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325, |h, &x| (h ^ x).wrapping_mul(0x0100_0000_01b3))
    }
}
