//! A rewrite whose strategy is chosen mention by mention
//! ([`StrategyChooser`]), for a policy whose rules test each mention's
//! facts as well as its kind. Each edit records the rule that chose its
//! strategy and the id the policy gives that rule.

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiKind, PiiMention, PiiRule};

use super::replace::{EditOrigin, MentionCluster};
use super::state::RewriteState;
use super::{PiiRewrite, RewriteError, RewriteStrategy};

/// What a [`StrategyChooser`] chose for one mention: the strategy, the
/// rule that chose it, and that rule's id in its policy (empty for the
/// default).
#[derive(Copy, Clone, Debug)]
pub struct Choice<'r> {
    pub strategy: &'r RewriteStrategy,
    pub rule: PiiRule,
    pub rule_id: &'r str,
}

/// Chooses the strategy of each mention of a pii layer.
pub trait StrategyChooser {
    /// The choice for `m`, a mention of `pii`. A [`PiiRule::Kind`] names
    /// the kind as `pii`'s custom-kind table does.
    fn choose<'r>(&'r self, pii: &Pii, m: &PiiMention<'_>) -> Choice<'r>;
}

/// The strategy chosen for each kind of a cluster: the winner's choice for
/// the winner's kind, and the first member's for each other kind.
fn cluster_choices<'r>(
    chooser: &'r impl StrategyChooser,
    pii: &Pii,
    c: &MentionCluster<'_>,
) -> (Choice<'r>, Vec<(PiiKind, &'r RewriteStrategy)>) {
    let winner = chooser.choose(pii, &c.merged);
    let mut by_kind = vec![(c.merged.kind, winner.strategy)];
    for m in &c.members {
        if !by_kind.iter().any(|(k, _)| *k == m.kind) {
            by_kind.push((m.kind, chooser.choose(pii, m).strategy));
        }
    }
    (winner, by_kind)
}

/// Replaces every mention of `pii`, a layer over `src`, with the strategy
/// `chooser` chooses, as a view derived from `src`. Overlapping mentions
/// are coalesced as [`super::RewritePolicy::rewrite`] does; a cluster any
/// of whose chosen strategies refuses overlaps is refused. Each edit
/// records its winner's rule and rule id, and `revision`.
pub fn rewrite_chosen(
    src: &View,
    pii: &Pii,
    chooser: &impl StrategyChooser,
    revision: u64,
) -> Result<PiiRewrite, RewriteError> {
    let text = src.text();
    let mut state = RewriteState::new(&pii.custom_kinds);
    let refuses = |c: &MentionCluster<'_>| c.members.iter().any(|m| chooser.choose(pii, m).strategy.refuses_overlaps());
    PiiRewrite::derive(src, pii, refuses, revision, |cluster, out| {
        let (winner, by_kind) = cluster_choices(chooser, pii, cluster);
        let strategy_for =
            |k: PiiKind| by_kind.iter().find(|(kind, _)| *kind == k).map_or(winner.strategy, |(_, s)| *s);
        state.write_cluster(strategy_for, text, cluster, out)?;
        Ok(EditOrigin { rule: winner.rule, strategy: winner.strategy.kind(), rule_id: winner.rule_id })
    })
}
