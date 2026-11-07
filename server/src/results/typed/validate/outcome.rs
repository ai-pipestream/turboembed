//! The outcome rules on every `SpanFragment` (issue #1907, item 7): an
//! outcome is present, deleted only on an empty fragment, members only
//! when coalesced and then ascending with the fragment's outcome merged
//! from theirs, and everything exact on the result's own view.

use super::*;
use crate::proto::result::SpanOutcome;
use std::collections::HashSet;

fn parse(lid: &str, what: &str, raw: i32) -> Result<SpanOutcome, Failure> {
    match SpanOutcome::try_from(raw) {
        Ok(SpanOutcome::Unspecified) => Err(bad(format!("{lid} {what} has no outcome"))),
        Ok(o) => Ok(o),
        Err(_) => Err(bad(format!("{lid} {what} has unknown outcome {raw}"))),
    }
}

/// The outcome of a coalesced fragment from its members': widened when
/// any member is, else exact when any member is, else deleted, as
/// `Document::project_fragments` merges them.
fn merged(members: &[SpanOutcome]) -> SpanOutcome {
    if members.contains(&SpanOutcome::Widened) {
        SpanOutcome::Widened
    } else if members.contains(&SpanOutcome::Exact) {
        SpanOutcome::Exact
    } else {
        SpanOutcome::Deleted
    }
}

/// Checks `frag`'s outcome and members; `own_view` when the fragment is
/// on the view its layer's spans are natively in, where projection
/// changes nothing.
pub(super) fn check_fragment(
    lid: &str,
    what: &str,
    frag: &SpanFragment,
    own_view: bool,
) -> Result<SpanOutcome, Failure> {
    let outcome = parse(lid, what, frag.outcome)?;
    if outcome == SpanOutcome::Deleted && frag.start != frag.end {
        return Err(bad(format!("{lid} {what} is deleted and not empty")));
    }
    if own_view && outcome != SpanOutcome::Exact {
        return Err(bad(format!("{lid} {what} is on its own view and not exact")));
    }
    if frag.members.len() == 1 {
        return Err(bad(format!("{lid} {what} lists one member; a fragment that is not coalesced lists none")));
    }
    if frag.members.is_empty() {
        return Ok(outcome);
    }
    let mut outcomes = Vec::with_capacity(frag.members.len());
    for (k, m) in frag.members.iter().enumerate() {
        let o = parse(lid, &format!("{what} member {k}"), m.outcome)?;
        if own_view && o != SpanOutcome::Exact {
            return Err(bad(format!("{lid} {what} member {k} is on its own view and not exact")));
        }
        if k > 0 && m.index <= frag.members[k - 1].index {
            return Err(bad(format!("{lid} {what} members are not in ascending native order")));
        }
        outcomes.push(o);
    }
    if merged(&outcomes) != outcome {
        return Err(bad(format!("{lid} {what} outcome is not the one its members merge to")));
    }
    Ok(outcome)
}

/// One projected span (a nested payload span, a pii edit's source):
/// `project_wire_span` maps a single span, which never coalesces, so
/// members are refused before anything walks them.
pub(super) fn check_single(lid: &str, what: &str, frag: &SpanFragment, own_view: bool) -> Result<SpanOutcome, Failure> {
    if !frag.members.is_empty() {
        return Err(bad(format!("{lid} {what} lists members; a single projected span is never coalesced")));
    }
    check_fragment(lid, what, frag, own_view)
}

/// The fragments of one annotation, each checked, with an empty one in
/// a multi-fragment span allowed only as a deletion anchor, and no
/// native fragment a member of two of them. Linear in the members: the
/// message's member total is already held under `MAX_FRAGMENTS` by
/// `validate_typed_document` before any fragment is read.
pub(super) fn check_fragments(lid: &str, i: usize, frags: &[SpanFragment], own_view: bool) -> Result<(), Failure> {
    let total: usize = frags.iter().map(|f| f.members.len()).sum();
    let mut seen: HashSet<u32> = HashSet::with_capacity(total);
    for (f_ix, frag) in frags.iter().enumerate() {
        let what = format!("[{i}] fragment {f_ix}");
        let outcome = check_fragment(lid, &what, frag, own_view)?;
        if frags.len() > 1 && frag.start == frag.end && outcome != SpanOutcome::Deleted {
            return Err(bad(format!("{lid}[{i}] fragment {f_ix} is empty in a multi-fragment span")));
        }
        for m in &frag.members {
            if !seen.insert(m.index) {
                return Err(bad(format!("{lid}[{i}] native fragment {} is a member of two fragments", m.index)));
            }
        }
    }
    Ok(())
}
