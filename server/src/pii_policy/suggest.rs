//! The redaction policy a `turbo:pii-risk` layer suggests (#1889): one
//! enabled rule per kind found, each with the layer's suggested rewrite,
//! as a `RedactionPolicy` message a caller can review, change and send
//! back as `pii_redaction`. Every suggested strategy needs no key, no
//! dictionary and no per-document numbering, and the default is a mask.

use turbo_analysis::Layer;
use turbo_pii::{PiiKind, PiiRisk};

use crate::proto::result as pb;
use pb::pii_rewrite_strategy::Strategy;

/// The id of every suggested policy.
pub const SUGGESTED_POLICY_ID: &str = "turbo-pii:risk-suggested";

fn mask(keep_format: bool) -> Strategy {
    Strategy::Mask(pb::PiiMaskStrategy { mask_char: "*".into(), keep_format })
}

/// The strategy of one suggested rewrite id, or `None` for an unknown id.
fn strategy(suggested: i32) -> Option<Strategy> {
    Some(match suggested {
        1 => mask(false),
        2 => mask(true),
        3 => Strategy::Format(pb::PiiFormatStrategy { template: "[%t]".into() }),
        4 => Strategy::LastN(pb::PiiLastNStrategy { count: 4, mask_char: "*".into(), keep_format: true }),
        _ => return None,
    })
}

/// The rule id of a kind: `suggested:` and its built-in name, or
/// `suggested:custom-` and its index, so no caller text is copied in.
fn rule_id(kind: Option<&pb::PiiKindRef>) -> Option<String> {
    match kind?.kind? {
        pb::pii_kind_ref::Kind::Builtin(id) => {
            let name = u16::try_from(id).ok().and_then(PiiKind::from_id)?.builtin_name()?;
            Some(format!("suggested:{name}"))
        }
        pb::pii_kind_ref::Kind::Custom(i) => Some(format!("suggested:custom-{i}")),
    }
}

/// The policy `layer`, a projected `turbo:pii-risk` layer, suggests:
/// `None` when it is another layer or holds no risk annotation. A kind
/// whose reference or suggestion this server does not know is left to the
/// policy's default mask.
pub fn suggested_policy(layer: &pb::TypedLayer) -> Option<pb::RedactionPolicy> {
    if layer.id != PiiRisk::ID {
        return None;
    }
    let risk = layer.annotations.iter().find_map(|a| match &a.value {
        Some(pb::typed_annotation::Value::PiiRisk(r)) => Some(r),
        _ => None,
    })?;
    let kind_rules = risk
        .kinds
        .iter()
        .filter_map(|k| {
            Some(pb::PiiKindRule {
                rule_id: rule_id(k.kind.as_ref())?,
                kind: k.kind,
                state: pb::PiiRuleState::Enabled as i32,
                strategies: vec![pb::PiiStrategyEntry {
                    when: Vec::new(),
                    strategy: Some(pb::PiiRewriteStrategy { strategy: Some(strategy(k.suggested)?) }),
                }],
                allow: None,
                labels: None,
                release_ceiling: None,
            })
        })
        .collect();
    Some(pb::RedactionPolicy {
        id: SUGGESTED_POLICY_ID.into(),
        revision: 1,
        custom_kinds: layer.custom_pii_kinds.clone(),
        kind_rules,
        default_strategy: Some(pb::PiiDefaultStrategy {
            strategy: Some(pb::pii_default_strategy::Strategy::Mask(pb::PiiMaskStrategy::default())),
        }),
        ..Default::default()
    })
}
