//! The typed projection of `turbo:pii-edits`: one `PiiEditAnnotation` an
//! edit, its spans the replacement on the requested view, its source span
//! in canonical UTF-8 bytes of the native view's parent, its covered
//! mention rows, rule, strategy and policy revision. No value is carried.

use std::any::TypeId;

use turbo_analysis::{Ix, Key, KeyId};
use turbo_analysis_layers::{Pii, PiiEdits, PiiRule};

use super::layers_pii::pii_kind_ref;
use super::{ProjectionCx, build_layer};
use crate::proto::result::typed_annotation::Value;
use crate::proto::result::{PiiEditAnnotation, PiiEditRule, SpanFragment, TypedAnnotation, TypedLayer, pii_edit_rule};

fn rule(r: PiiRule) -> PiiEditRule {
    let rule = match r {
        PiiRule::Default => pii_edit_rule::Rule::DefaultRule(true),
        PiiRule::Kind(k) => pii_edit_rule::Rule::Kind(pii_kind_ref(k)),
    };
    PiiEditRule { rule: Some(rule) }
}

pub(crate) fn project_pii_edits_layer(cx: &mut ProjectionCx<'_>, key: KeyId) -> Option<TypedLayer> {
    if key.ty != TypeId::of::<PiiEdits>() {
        return None;
    }
    let mut layer = build_layer::<PiiEdits>(cx, key, |cx, l, mref| {
        let mref = mref.map(ToOwned::to_owned);
        l.edits()
            .enumerate()
            .map(|(i, e)| TypedAnnotation {
                spans: cx.project_spans(&l.spans, Ix::new(u32::try_from(i).expect("fewer than 2^32 edits"))),
                probability: None,
                model_ref: mref.clone(),
                value: Some(Value::PiiEdit(PiiEditAnnotation {
                    source: Some(SpanFragment::exact(e.source.start, e.source.end)),
                    members: e.members.iter().map(|m| u32::try_from(m.get()).expect("a u32 row")).collect(),
                    rule: Some(rule(e.rule)),
                    strategy: i32::from(e.strategy.id()),
                    policy_revision: l.policy_revision,
                })),
            })
            .collect()
    });
    // The covered pii layer's custom-kind table, which a custom rule kind
    // indexes: the instance the edits were made on.
    let covered = cx.doc.made_on(key).iter().find(|k| k.ty == TypeId::of::<Pii>()).map(|k| k.name);
    if let Some(pii) = covered.and_then(|name| cx.doc.get_key(&Key::<Pii>::named(name))) {
        layer.custom_pii_kinds = pii.custom_kinds.names().map(str::to_owned).collect();
    }
    Some(layer)
}
