//! Normalization x-ray alignment diff runs (`NormalizationXrayProto`,
//! issue #1220): composes view alignments between any two views in a
//! [`DocumentGraph`], emits contiguous `Equal`, `Replaced`, `Inserted`, and
//! `Deleted` runs partitioning both views, and projects all span offsets into
//! the caller's [`OffsetMode`].

use turbo_analysis::export::{DocumentGraph, StatusRecord};
use turbo_analysis::{Alignment, Span};

use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{NormalizationXrayProto, XrayEditKindProto, XrayEditRunProto, XrayStepSummaryProto};
use crate::results::{MAX_FRAGMENTS, over};

pub(crate) struct ResolvedView<'a> {
    id: u32,
    parent: Option<u32>,
    instance: &'a str,
    annotator: &'a str,
    version: &'a str,
    config_hash: u64,
    text: &'a str,
    alignment: Alignment,
}

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

/// Resolves all complete views from `graph` in their assigned `ViewId` order.
pub(crate) fn resolve_views(graph: &DocumentGraph) -> Result<Vec<ResolvedView<'_>>, Failure> {
    let mut out: Vec<ResolvedView<'_>> = Vec::new();
    for record in &graph.views {
        if !matches!(record.status, StatusRecord::Complete) {
            continue;
        }
        let id = out.len() as u32;
        let text = record.text.as_deref().ok_or_else(|| bad(format!("view {id} is complete but carries no text")))?;
        let parent = match record.parent.as_deref() {
            None => None,
            Some(p_name) => {
                let p_idx =
                    out.iter().find(|v| v.instance == p_name).map(|v| v.id).ok_or_else(|| {
                        bad(format!("view '{}' references unknown parent '{p_name}'", record.instance))
                    })?;
                Some(p_idx)
            }
        };
        let alignment =
            Alignment::from_edits(record.edits.clone()).map_err(|e| bad(format!("view {id} alignment: {e}")))?;
        out.push(ResolvedView {
            id,
            parent,
            instance: &record.instance,
            annotator: &record.provenance.annotator,
            version: &record.provenance.version,
            config_hash: record.provenance.config,
            text,
            alignment,
        });
    }
    Ok(out)
}

fn ancestor_chain(views: &[ResolvedView<'_>], mut v: u32) -> Vec<u32> {
    let mut chain = vec![v];
    while let Some(p) = views[v as usize].parent {
        chain.push(p);
        v = p;
    }
    chain.reverse();
    chain
}

fn step_summary(views: &[ResolvedView<'_>], v: &ResolvedView<'_>) -> XrayStepSummaryProto {
    let parent_bytes = v.parent.map_or(v.text.len() as u32, |p| views[p as usize].text.len() as u32);
    XrayStepSummaryProto {
        view: v.id,
        parent_view: v.parent,
        instance: v.instance.to_owned(),
        annotator: v.annotator.to_owned(),
        version: v.version.to_owned(),
        config_hash: v.config_hash,
        edit_count: v.alignment.edits().len() as u32,
        parent_bytes,
        child_bytes: v.text.len() as u32,
    }
}

/// Computes the composed [`NormalizationXrayProto`] from `from_view` to
/// `to_view` in `offset_mode`.
pub(crate) fn compute_xray(
    graph: &DocumentGraph,
    from_view: u32,
    to_view: u32,
    offset_mode: OffsetMode,
) -> Result<NormalizationXrayProto, Failure> {
    let views = resolve_views(graph)?;
    let count = views.len() as u32;
    if from_view >= count {
        return Err(bad(format!("view {from_view} is out of range for document with {count} view(s)")));
    }
    if to_view >= count {
        return Err(bad(format!("xray_target_view {to_view} is out of range for document with {count} view(s)")));
    }

    let from_chain = ancestor_chain(&views, from_view);
    let to_chain = ancestor_chain(&views, to_view);
    let common_len = from_chain.iter().zip(&to_chain).take_while(|(a, b)| a == b).count();
    let mut net = Alignment::identity();
    let mut steps = Vec::new();

    // Climb from `from_view` up to the lowest common ancestor, inverting each step.
    for &v_id in from_chain[common_len..].iter().rev() {
        let v = &views[v_id as usize];
        net = net.compose(&v.alignment.invert());
        steps.push(step_summary(&views, v));
    }
    // Descend from the lowest common ancestor down to `to_view`.
    for &v_id in &to_chain[common_len..] {
        let v = &views[v_id as usize];
        net = net.compose(&v.alignment);
        steps.push(step_summary(&views, v));
    }

    let from_text = views[from_view as usize].text;
    let to_text = views[to_view as usize].text;
    let from_off = Offsetter::new(from_text, offset_mode);
    let to_off = Offsetter::new(to_text, offset_mode);

    let mut runs = Vec::new();
    let (mut p_cur, mut c_cur) = (0u32, 0u32);
    for edit in net.edits() {
        if edit.parent.start > p_cur || edit.child.start > c_cur {
            let p_span = Span::new(p_cur, edit.parent.start);
            let c_span = Span::new(c_cur, edit.child.start);
            push_run(&mut runs, from_text, &from_off, to_text, &to_off, p_span, c_span, XrayEditKindProto::Equal);
        }
        let kind = if edit.parent.is_empty() {
            XrayEditKindProto::Inserted
        } else if edit.child.is_empty() {
            XrayEditKindProto::Deleted
        } else {
            XrayEditKindProto::Replaced
        };
        push_run(&mut runs, from_text, &from_off, to_text, &to_off, edit.parent, edit.child, kind);
        p_cur = edit.parent.end;
        c_cur = edit.child.end;
    }
    let from_len = from_text.len() as u32;
    let to_len = to_text.len() as u32;
    if p_cur < from_len || c_cur < to_len {
        let p_span = Span::new(p_cur, from_len);
        let c_span = Span::new(c_cur, to_len);
        push_run(&mut runs, from_text, &from_off, to_text, &to_off, p_span, c_span, XrayEditKindProto::Equal);
    }

    if runs.len() as u64 > MAX_FRAGMENTS {
        return Err(over("max_fragments", "xray edit runs", runs.len() as u64));
    }

    Ok(NormalizationXrayProto { from_view, to_view, offsets: offset_mode.to_proto(), runs, steps })
}

#[allow(clippy::too_many_arguments)]
fn push_run(
    runs: &mut Vec<XrayEditRunProto>,
    from_text: &str,
    from_off: &Offsetter,
    to_text: &str,
    to_off: &Offsetter,
    p_span: Span,
    c_span: Span,
    kind: XrayEditKindProto,
) {
    let (parent_start, parent_end) = from_off.to_wire(from_text, p_span);
    let (child_start, child_end) = to_off.to_wire(to_text, c_span);
    runs.push(XrayEditRunProto {
        parent_start,
        parent_end,
        child_start,
        child_end,
        kind: kind as i32,
        parent_text: from_text[p_span.range()].to_owned(),
        child_text: to_text[c_span.range()].to_owned(),
    });
}
