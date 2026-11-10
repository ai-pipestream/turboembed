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
        read_member();
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
            read_member();
            if !seen.insert(m.index) {
                return Err(bad(format!("{lid}[{i}] native fragment {} is a member of two fragments", m.index)));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
thread_local! {
    /// Member reads: once when its fragment is checked and once when it is
    /// claimed against the other fragments, each a constant-time step.
    static MEMBER_READS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// One member read.
#[inline]
fn read_member() {
    #[cfg(test)]
    MEMBER_READS.with(|n| n.set(n.get() + 1));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::result::FragmentMember;

    /// Members read to check `fragments` coalesced fragments of `per` members each.
    fn reads(fragments: u32, per: u32) -> u64 {
        let frags: Vec<SpanFragment> = (0..fragments)
            .map(|f| SpanFragment {
                start: f * 2,
                end: f * 2 + 1,
                outcome: SpanOutcome::Exact as i32,
                members: (0..per)
                    .map(|k| FragmentMember { index: f * per + k, outcome: SpanOutcome::Exact as i32 })
                    .collect(),
            })
            .collect();
        MEMBER_READS.with(|n| n.set(0));
        check_fragments("l", 0, &frags, false).unwrap();
        MEMBER_READS.with(std::cell::Cell::get)
    }

    /// Codex on #1921: reuse across fragments was a `Vec::contains`, so a
    /// long list cost its square. Each member is read twice, in one long
    /// fragment or spread across many, and the claim against the others
    /// is one set probe.
    #[test]
    fn each_member_is_read_twice() {
        for (fragments, per) in [(1, 50_000), (1, 200_000), (500, 400)] {
            assert_eq!(reads(fragments, per), 2 * u64::from(fragments * per), "{fragments} x {per}");
        }
    }
}
