//! Identity-term bias metrics for [`Incivility`] scores, after Borkan et
//! al., "Nuanced Metrics for Measuring Unintended Bias with Real Data for
//! Text Classification" (WWW 2019 Companion).
//!
//! For each label and each identity subgroup:
//! - subgroup AUC: the ROC AUC over the units that mention the subgroup;
//! - BPSN AUC (background positive, subgroup negative): positives that do
//!   not mention the subgroup against negatives that do. A low value means
//!   neutral mentions of the subgroup score above real positives
//!   elsewhere;
//! - BNSP AUC (background negative, subgroup positive): positives that
//!   mention the subgroup against negatives that do not. A low value means
//!   real positives that mention the subgroup score below neutral text
//!   elsewhere;
//! - the false positive rate on neutral mentions: of the units that
//!   mention the subgroup and are negative for the label, the share that
//!   scores at or above the threshold.
//!
//! An AUC with no positive or no negative unit is `None`. Ties count one
//! half, as in the Mann-Whitney U statistic.
//!
//! [`IdentityTerms`] finds which subgroups a unit's tokens mention, with
//! the shared token key (typographic apostrophes, Unicode punctuation and
//! full case folding), so "Muslim", "MUSLIM" and "muslim," match alike.

use super::*;
use crate::sentiment::token_key::token_key;
use turbo_unicode::category::general_category;

/// Identity terms, each a word or a phrase of several words, grouped by
/// the subgroup they name.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentityTerms {
    /// `(term words as token keys, subgroup index)`, longest term first.
    terms: Vec<(Vec<String>, usize)>,
    subgroups: Vec<String>,
}

impl IdentityTerms {
    /// The terms from `(subgroup, term)` pairs; a term of several words is
    /// split at whitespace and matches that run of tokens. An empty
    /// subgroup or term, or a word of punctuation only, is an error.
    pub fn new<'a>(entries: impl IntoIterator<Item = (&'a str, &'a str)>) -> Result<Self, String> {
        let mut subgroups: Vec<String> = Vec::new();
        let mut terms = Vec::new();
        for (subgroup, term) in entries {
            if subgroup.trim().is_empty() {
                return Err(format!("an empty subgroup for the term {term:?}"));
            }
            if term.split_whitespace().any(|w| w.chars().all(|c| general_category(c).is_punctuation())) {
                return Err(format!("the term {term:?} in the subgroup {subgroup:?} has a word of punctuation only"));
            }
            let words: Vec<String> = term.split_whitespace().map(token_key).collect();
            if words.is_empty() {
                return Err(format!("an empty term in the subgroup {subgroup:?}"));
            }
            let s = match subgroups.iter().position(|g| g == subgroup) {
                Some(s) => s,
                None => {
                    subgroups.push(subgroup.to_owned());
                    subgroups.len() - 1
                }
            };
            terms.push((words, s));
        }
        terms.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then_with(|| a.0.cmp(&b.0)));
        terms.dedup();
        Ok(IdentityTerms { terms, subgroups })
    }

    /// The subgroup names, in the order they first appeared.
    pub fn subgroups(&self) -> &[String] {
        &self.subgroups
    }

    /// The subgroups `tokens` mention, sorted and without repeats.
    pub fn find<S: AsRef<str>>(&self, tokens: &[S]) -> Vec<&str> {
        let keys: Vec<String> = tokens.iter().map(|t| token_key(t.as_ref())).collect();
        let mut found = vec![false; self.subgroups.len()];
        for (words, s) in &self.terms {
            if !found[*s] && keys.windows(words.len()).any(|w| w == words.as_slice()) {
                found[*s] = true;
            }
        }
        let mut out: Vec<&str> =
            found.iter().enumerate().filter(|&(_, &f)| f).map(|(s, _)| self.subgroups[s].as_str()).collect();
        out.sort_unstable();
        out
    }
}

/// Gold labels and identity subgroups for one unit in
/// [`evaluate_incivility_bias`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldBiasUnit {
    pub positive_labels: Vec<IncivilityLabel>,
    /// The subgroups the unit mentions, such as from [`IdentityTerms::find`].
    pub subgroups: Vec<String>,
}

/// The bias metrics of one subgroup for one label.
#[derive(Clone, Debug, PartialEq)]
pub struct SubgroupBias {
    pub subgroup: String,
    /// Units that mention the subgroup.
    pub units: usize,
    /// Of those, units positive for the label.
    pub positives: usize,
    pub subgroup_auc: Option<f64>,
    pub bpsn_auc: Option<f64>,
    pub bnsp_auc: Option<f64>,
    /// Units that mention the subgroup and are negative for the label.
    pub neutral_mentions: usize,
    /// Of those, units scored at or above the threshold.
    pub neutral_false_positives: usize,
    /// `neutral_false_positives / neutral_mentions`; `None` with no
    /// neutral mention.
    pub neutral_fpr: Option<f64>,
}

/// The bias metrics of one label.
#[derive(Clone, Debug, PartialEq)]
pub struct LabelBias {
    pub label: IncivilityLabel,
    /// The ROC AUC over all units.
    pub overall_auc: Option<f64>,
    /// One entry per subgroup in the gold units, sorted by name.
    pub subgroups: Vec<SubgroupBias>,
}

/// [`evaluate_incivility_bias`]'s result, one entry per label in
/// [`IncivilityLabel::ALL`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct IncivilityBiasMetrics {
    pub per_label: Vec<LabelBias>,
}

/// The ROC AUC of `positives` against `negatives`: the chance that a
/// positive scores above a negative, ties counting one half. `None` when
/// either side is empty or a score is not finite.
pub fn roc_auc(positives: &[f64], negatives: &[f64]) -> Option<f64> {
    if positives.is_empty() || negatives.is_empty() || positives.iter().chain(negatives).any(|s| !s.is_finite()) {
        return None;
    }
    let mut all: Vec<(f64, bool)> =
        positives.iter().map(|&s| (s, true)).chain(negatives.iter().map(|&s| (s, false))).collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    // Sum of the positives' ranks (1-based), tied runs taking their mean rank.
    let mut rank_sum = 0.0;
    let mut i = 0;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        rank_sum += mean_rank * all[i..j].iter().filter(|e| e.1).count() as f64;
        i = j;
    }
    let (p, n) = (positives.len() as f64, negatives.len() as f64);
    Some((rank_sum - p * (p + 1.0) / 2.0) / (p * n))
}

/// The bias metrics of `predicted` against `gold` at `threshold`. The
/// units must match one for one, every score must be finite, and the
/// threshold must be in `0.0..=1.0`.
pub fn evaluate_incivility_bias(
    gold: &[GoldBiasUnit],
    predicted: &Incivility,
    threshold: f64,
) -> Result<IncivilityBiasMetrics, String> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(format!("a threshold of {threshold}, outside 0..=1"));
    }
    if gold.len() != predicted.len() {
        return Err(format!("{} gold units and {} predicted units", gold.len(), predicted.len()));
    }
    let scores: Vec<IncivilityScores> = (0..gold.len()).map(|u| predicted.scores(Ix::new(u as u32))).collect();
    if let Some(u) = scores.iter().position(|s| s.iter().iter().any(|(_, v)| !v.is_finite())) {
        return Err(format!("unit {u} has a score that is not finite"));
    }
    let mut names: Vec<&str> = gold.iter().flat_map(|g| g.subgroups.iter().map(String::as_str)).collect();
    names.sort_unstable();
    names.dedup();

    let per_label = IncivilityLabel::ALL
        .into_iter()
        .map(|label| {
            let pos = |u: usize| gold[u].positive_labels.contains(&label);
            let score = |u: usize| scores[u].get(label);
            let pick = |keep: &dyn Fn(usize) -> bool| -> Vec<f64> {
                (0..gold.len()).filter(|&u| keep(u)).map(score).collect()
            };
            let subgroups = names
                .iter()
                .map(|&name| {
                    let has = |u: usize| gold[u].subgroups.iter().any(|g| g == name);
                    let sub_pos = pick(&|u| has(u) && pos(u));
                    let sub_neg = pick(&|u| has(u) && !pos(u));
                    let back_pos = pick(&|u| !has(u) && pos(u));
                    let back_neg = pick(&|u| !has(u) && !pos(u));
                    let flagged = sub_neg.iter().filter(|&&s| s >= threshold).count();
                    SubgroupBias {
                        subgroup: name.to_owned(),
                        units: sub_pos.len() + sub_neg.len(),
                        positives: sub_pos.len(),
                        subgroup_auc: roc_auc(&sub_pos, &sub_neg),
                        bpsn_auc: roc_auc(&back_pos, &sub_neg),
                        bnsp_auc: roc_auc(&sub_pos, &back_neg),
                        neutral_mentions: sub_neg.len(),
                        neutral_false_positives: flagged,
                        neutral_fpr: (!sub_neg.is_empty()).then(|| flagged as f64 / sub_neg.len() as f64),
                    }
                })
                .collect();
            LabelBias { label, overall_auc: roc_auc(&pick(&pos), &pick(&|u| !pos(u))), subgroups }
        })
        .collect();
    Ok(IncivilityBiasMetrics { per_label })
}
