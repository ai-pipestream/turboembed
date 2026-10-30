//! What the pipeline bears out of each step's declared locality.

use std::any::TypeId;
use std::collections::HashMap;

use crate::annotate::{Locality, Step};
use crate::doc::View;
use crate::layer::{Key, KeyId};

/// Each step's effective locality, in step order. A step is
/// `Partition(P)` only when it declares that, writes no view, and every
/// layer it reads is the source text, `P` or another output of `P`'s
/// producer, or the output of a step that is itself `Partition(P)`.
/// Anything else, a given layer included, makes it `Document`. Steps are
/// visited in stage order, so a producer's effective locality is settled
/// before its readers are.
pub(super) fn effective(given: &[KeyId], steps: &[Box<dyn Step>], stages: &[Vec<usize>]) -> Vec<Locality> {
    let mut producer: HashMap<KeyId, Option<usize>> = given.iter().map(|&k| (k, None)).collect();
    for (i, s) in steps.iter().enumerate() {
        for w in s.writes() {
            producer.insert(w, Some(i));
        }
    }
    let source = Key::<View>::DEFAULT.id();
    let view_ty = TypeId::of::<View>();
    let mut effective = vec![Locality::Document; steps.len()];
    for &i in stages.iter().flatten() {
        let Locality::Partition(p) = steps[i].locality() else { continue };
        if steps[i].writes().iter().any(|w| w.ty == view_ty) {
            continue;
        }
        let Some(&p_producer) = producer.get(&p) else { continue };
        let local = steps[i].reads().into_iter().all(|r| {
            if r == p || r == source {
                return true;
            }
            match producer.get(&r) {
                None => true,
                Some(None) => false,
                Some(&Some(k)) => Some(k) == p_producer || effective[k] == Locality::Partition(p),
            }
        });
        if local {
            effective[i] = Locality::Partition(p);
        }
    }
    effective
}

/// Why a step's declaration was not borne out, for the report.
pub(super) fn downgrade_reason(
    i: usize,
    given: &[KeyId],
    steps: &[Box<dyn Step>],
    effective: &[Locality],
) -> Option<String> {
    let Locality::Partition(p) = steps[i].locality() else { return None };
    if effective[i] == Locality::Partition(p) {
        return None;
    }
    if steps[i].writes().iter().any(|w| w.ty == TypeId::of::<View>()) {
        return Some("it writes a view".into());
    }
    let producer_of = |r: KeyId| steps.iter().position(|s| s.writes().contains(&r));
    let Some(p_producer) = producer_of(p) else {
        return Some(if given.contains(&p) {
            format!("{p} is given, not made")
        } else {
            format!("nothing writes {p}")
        });
    };
    let source = Key::<View>::DEFAULT.id();
    for r in steps[i].reads() {
        if r == p || r == source {
            continue;
        }
        match producer_of(r) {
            Some(k) if k == p_producer || effective[k] == Locality::Partition(p) => {}
            Some(k) => return Some(format!("it reads {r} from {}, which is document-wide", steps[k].id())),
            None if given.contains(&r) => return Some(format!("it reads {r}, which is given")),
            None => {}
        }
    }
    Some("the pipeline does not bear it out".into())
}
