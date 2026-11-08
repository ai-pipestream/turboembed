//! A rewrite whose strategy is chosen mention by mention
//! ([`StrategyChooser`]), for a policy whose rules test each mention's
//! facts as well as its kind. Each edit records the rule that chose its
//! strategy and the id the policy gives that rule.

use turbo_analysis::View;
use turbo_analysis_layers::{Pii, PiiMention, PiiRule};

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

/// Replaces every mention of `pii`, a layer over `src`, with the strategy
/// `chooser` chooses, as a view derived from `src`. Overlapping mentions
/// are coalesced as [`super::RewritePolicy::rewrite`] does; the merged
/// mention takes the winner's choice and each member keeps its own, so
/// a member's stricter retention is never replaced by another member's
/// of the same kind. A cluster any of whose chosen strategies refuses
/// overlaps is refused. Each edit
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
        // The winner's choice for the merged mention, and each member's own
        // choice for its retention: two members of one kind can take
        // different strategies, and the stricter one must hold.
        // write_cluster asks for the merged mention's strategy, which is the
        // winner's choice.
        let winner = chooser.choose(pii, &cluster.merged);
        state.write_cluster(|m| chooser.choose(pii, m).strategy, text, cluster, out)?;
        Ok(EditOrigin { rule: winner.rule, strategy: winner.strategy.kind(), rule_id: winner.rule_id })
    })
}
