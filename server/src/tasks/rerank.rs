//! Serving `TASK_RERANK` bundles over KServe v2.

use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use turbo::{
    TURBO_PLACE_HOST, TURBO_PLACE_PINNED, TURBO_PLACE_SHARED, turbo_rerank_options, turbo_text, turbo_text_pair,
};

use crate::api::{Failure, Result, TURBO_E_CAPACITY, TURBO_E_INTERNAL, TURBO_E_INVALID_ARGUMENT};
use crate::deadline::Deadline;
use crate::infer::{EMPTY, bad, bad_shape, has_contents, kind, only, shape, split_bytes, view};
use crate::kserve::{Data, Vectors, datatype, result_parameters};
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::{InferParameter, ModelInferRequest, ModelInferResponse};
use crate::state::{Loaded, Served};
use crate::status::RERANK_OPTIONS;

/// Executes a `TASK_RERANK` `ModelInfer` call.
pub(crate) fn infer_rerank(
    served: &Arc<Served>,
    loaded: &Arc<Loaded>,
    req: ModelInferRequest,
    deadline: &Deadline,
) -> Result<ModelInferResponse> {
    for o in &req.outputs {
        if o.name != "scores" {
            return Err(bad(format!("requested output `{}`: the rerank model's output is `scores`", o.name)));
        }
        if !o.parameters.is_empty() {
            return Err(bad(format!("requested output `{}` has parameters; none are defined", o.name)));
        }
    }
    if req.outputs.len() > 1 {
        return Err(bad("requested output is named more than once"));
    }

    let pairs = rerank_pairs(&req)?;
    let opts = rerank_options(&req.parameters)?;
    let max_batch = loaded.slots[0].info.max_batch as usize;
    if pairs.len() > max_batch {
        return Err(Failure::new(
            TURBO_E_CAPACITY,
            0,
            format!("batch of {} exceeds max_batch {max_batch}", pairs.len()),
        ));
    }

    let _docs = loaded.metrics.track_docs(pairs.len().max(1), Some(loaded.device));
    deadline.check_alive()?;
    let held = loaded.take()?;
    deadline.check_alive()?;
    let slot = held.slot();
    slot.session.write_pairs(&pairs, &opts)?;
    deadline.check_alive()?;
    loaded.metrics.record_batch_size(pairs.len());
    let result = slot.session.run()?;
    deadline.check_alive()?;
    let first = result.info()?;
    let data = match first.placement {
        TURBO_PLACE_HOST | TURBO_PLACE_PINNED | TURBO_PLACE_SHARED => {
            let buffer = result.buffer()?;
            let ptr = buffer.host_ptr()?;
            Data::Mapped { ptr, len: first.bytes as usize, _buffer: buffer }
        }
        _ => Data::Read(result.read(first.bytes)?),
    };
    let info = result.info()?;
    let dt = datatype(info.dtype).ok_or_else(|| {
        Failure::new(TURBO_E_INTERNAL, 0, format!("the result's dtype {} has no datatype in the protocol", info.dtype))
    })?;
    let vectors = Vectors { data, _result: result, _held: held };

    Ok(ModelInferResponse {
        model_name: served.name.clone(),
        model_version: loaded.revision.clone(),
        id: req.id,
        parameters: result_parameters(&info),
        outputs: vec![InferOutputTensor {
            name: "scores".to_string(),
            datatype: dt.to_string(),
            shape: vec![info.batch as i64, info.dim as i64],
            parameters: HashMap::new(),
            contents: None,
        }],
        raw_output_contents: vec![Bytes::from_owner(vectors)],
    })
}

fn rerank_pairs(req: &ModelInferRequest) -> Result<Vec<turbo_text_pair>> {
    for i in &req.inputs {
        if !i.parameters.is_empty() {
            return Err(bad(format!("input `{}` has parameters; none are defined", i.name)));
        }
        if i.datatype != "BYTES" {
            return Err(bad(format!("input `{}` is {}; it takes BYTES", i.name, i.datatype)));
        }
    }
    let find = |name: &str| -> Result<Option<usize>> {
        let mut at = req.inputs.iter().enumerate().filter(|(_, i)| i.name == name).map(|(k, _)| k);
        let first = at.next();
        if at.next().is_some() {
            return Err(bad(format!("input `{name}` is given more than once")));
        }
        Ok(first)
    };
    for i in &req.inputs {
        if !matches!(i.name.as_str(), "query" | "passages" | "passage" | "pairs") {
            return Err(bad(format!(
                "input `{}`: rerank inputs are `query` + `passages` (or `passage`), or `pairs`",
                i.name
            )));
        }
    }
    let (query, passages, passage, pairs) = (find("query")?, find("passages")?, find("passage")?, find("pairs")?);
    let pass_idx = match (passages, passage) {
        (Some(_), Some(_)) => return Err(bad("give `passages` or `passage`, not both")),
        (Some(k), None) | (None, Some(k)) => Some(k),
        (None, None) => None,
    };

    let raw = !req.raw_input_contents.is_empty();
    if raw && req.inputs.iter().filter_map(|i| i.contents.as_ref()).any(has_contents) {
        return Err(bad("the request has both raw_input_contents and typed contents"));
    }
    if raw && req.raw_input_contents.len() != req.inputs.len() {
        return Err(bad(format!(
            "{} raw_input_contents entries for {} inputs",
            req.raw_input_contents.len(),
            req.inputs.len()
        )));
    }
    let extract = |k: usize| -> Result<Vec<turbo_text>> {
        if raw {
            split_bytes(&req.raw_input_contents[k])
        } else {
            let c = req.inputs[k].contents.as_ref().unwrap_or(&EMPTY);
            only(&req.inputs[k], c, "bytes_contents")?;
            Ok(c.bytes_contents.iter().map(|b| view(b)).collect())
        }
    };

    match (query, pass_idx, pairs) {
        (None, None, Some(k)) => {
            let ext = shape(&req.inputs[k], 2)?;
            if ext[1] != 2 {
                return Err(bad_shape(format!("input `pairs` has shape {:?}; second dimension must be 2", ext)));
            }
            let batch = ext[0] as usize;
            let views = extract(k)?;
            if views.len() != batch * 2 {
                return Err(bad_shape(format!(
                    "input `pairs` is shaped [{batch}, 2] and has {} elements",
                    views.len()
                )));
            }
            Ok(views.as_chunks::<2>().0.iter().map(|&[query, passage]| turbo_text_pair { query, passage }).collect())
        }
        (Some(qk), Some(pk), None) => {
            let q_ext = shape(&req.inputs[qk], 1)?;
            let p_ext = shape(&req.inputs[pk], 1)?;
            let q_views = extract(qk)?;
            let p_views = extract(pk)?;
            if q_views.len() != q_ext[0] as usize {
                return Err(bad_shape(format!(
                    "input `query` is shaped [{}] and has {} elements",
                    q_ext[0],
                    q_views.len()
                )));
            }
            if p_views.len() != p_ext[0] as usize {
                return Err(bad_shape(format!(
                    "input `{}` is shaped [{}] and has {} elements",
                    req.inputs[pk].name,
                    p_ext[0],
                    p_views.len()
                )));
            }
            if q_views.len() == p_views.len() {
                Ok(q_views
                    .into_iter()
                    .zip(p_views)
                    .map(|(query, passage)| turbo_text_pair { query, passage })
                    .collect())
            } else if q_views.len() == 1 && !p_views.is_empty() {
                let query = q_views[0];
                Ok(p_views.into_iter().map(|passage| turbo_text_pair { query, passage }).collect())
            } else {
                Err(bad_shape(format!(
                    "input `query` has {} rows and `{}` has {}; `query` must have 1 row or match passages",
                    q_views.len(),
                    req.inputs[pk].name,
                    p_views.len()
                )))
            }
        }
        _ => Err(bad("rerank request requires `query` + `passages` (or `passage`), or `pairs` alone")),
    }
}

pub fn rerank_options(params: &HashMap<String, InferParameter>) -> Result<turbo_rerank_options> {
    let mut unknown: Vec<&String> = params.keys().filter(|k| !RERANK_OPTIONS.contains(&k.as_str())).collect();
    unknown.sort();
    if let Some(name) = unknown.first() {
        return Err(bad(format!("parameter `{name}` is not a field of turbo_rerank_options")));
    }
    let mut v = [0u32; 3];
    for (k, name) in RERANK_OPTIONS.iter().enumerate() {
        let Some(p) = params.get(*name) else { continue };
        let field = k as u32 + 1;
        match p.parameter_choice.as_ref() {
            Some(ParameterChoice::Int64Param(n)) => {
                v[k] = u32::try_from(*n).map_err(|_| {
                    Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        field,
                        format!("parameter {name}: {n} is outside 0 to 4294967295"),
                    )
                })?;
            }
            other => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name} takes an int64_param, not {}", kind(other)),
                ));
            }
        }
    }
    Ok(turbo_rerank_options {
        struct_size: size_of::<turbo_rerank_options>() as u32,
        query_tokens: v[0],
        max_tokens: v[1],
        stride: v[2],
    })
}
