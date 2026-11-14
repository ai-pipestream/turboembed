//! Validation of `turbo:token-graph` (`TokenArcAnnotation`, issue #2340).
use super::*;
use crate::proto::result::TokenArcAnnotation;
use turbo_analysis::Layer;
use turbo_analysis_layers::{TokenArcKind, TokenGraph};

pub(super) fn layer(layer: &TypedLayer) -> Result<(), Failure> {
    if layer.id != TokenGraph::ID || (layer.status != LayerWireStatus::Ok as i32 && layer.annotations.is_empty()) {
        return Ok(());
    }
    if layer.scope != AnchorKind::Text as i32
        || layer.annotations.iter().any(|a| !matches!(&a.value, Some(Value::TokenArc(_))))
    {
        return Err(bad("token graph needs text scope and matching token_arc payloads"));
    }
    let mut last_pos = 0u32;
    let mut expected_num: Option<u32> = None;
    for (i, ann) in layer.annotations.iter().enumerate() {
        let Some(Value::TokenArc(arc)) = &ann.value else { continue };
        if let Some(exp) = expected_num {
            if arc.num_positions != exp {
                return Err(bad(format!("{}[{i}] inconsistent num_positions", layer.id)));
            }
        } else {
            expected_num = Some(arc.num_positions);
        }
        if i > 0 && arc.position < last_pos {
            return Err(bad(format!("{}[{i}] position {} decreases after {last_pos}", layer.id, arc.position)));
        }
        last_pos = arc.position;
    }
    Ok(())
}

pub(super) fn record(
    ctx: &LayerCtx<'_>,
    i: usize,
    ann: &TypedAnnotation,
    arc: &TokenArcAnnotation,
) -> Result<(), Failure> {
    if ctx.lid != TokenGraph::ID || ann.spans.is_empty() {
        return Err(bad("token_arc payload needs turbo:token-graph layer and projected span"));
    }
    if arc.position_length == 0 {
        return Err(bad(format!("{}[{i}] token_arc position_length must be >= 1", ctx.lid)));
    }
    let end = arc
        .position
        .checked_add(arc.position_length)
        .ok_or_else(|| bad(format!("{}[{i}] token_arc position + position_length overflows", ctx.lid)))?;
    if end > arc.num_positions {
        return Err(bad(format!("{}[{i}] token_arc end {end} exceeds num_positions {}", ctx.lid, arc.num_positions)));
    }
    if u8::try_from(arc.kind).ok().and_then(TokenArcKind::from_byte).is_none() {
        return Err(bad(format!("{}[{i}] unknown token_arc kind {}", ctx.lid, arc.kind)));
    }
    Ok(())
}
