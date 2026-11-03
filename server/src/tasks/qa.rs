//! Serving `TASK_EXTRACTIVE_QA` bundles over KServe v2.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;

use crate::api::{Failure, Result, TURBO_E_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8};
use crate::deadline::Deadline;
use crate::infer::{EMPTY, bad, bad_shape, has_contents, kind, only, shape, split_bytes, view};
use crate::kserve::result_parameters;
use crate::proto::infer_parameter::ParameterChoice;
use crate::proto::model_infer_response::InferOutputTensor;
use crate::proto::{InferParameter, ModelInferRequest, ModelInferResponse};
use crate::state::{Loaded, Served};
use crate::status::QA_OPTIONS;

/// Companion state for an extractive-QA bundle whose token-level hidden
/// states come from the C runtime via singleton token ranges.
pub struct QaModel {
    pub tokenizer: turbo::tokenizer::Tokenizer,
    pub head: turbo::head::Head,
    pub config: turbo::manifest::Qa,
    pub family: turbo::manifest::Family,
}

impl QaModel {
    pub fn load(bundle_path: &str) -> Result<Self> {
        let map_err = |e: turbo::status::Error| Failure::new(e.code, e.field, e.message);
        let bundle = turbo::bundle::Bundle::open(Path::new(bundle_path)).map_err(map_err)?;
        let m = &bundle.manifest;
        let config = m.qa.clone().ok_or_else(|| bad("missing QA configuration"))?;
        let family = m.architecture.as_ref().ok_or_else(|| bad("missing encoder architecture"))?.family;
        let tokenizer = turbo::tokenizer::Tokenizer::load(&bundle).map_err(map_err)?;
        let head = turbo::head::Head::load(&bundle).map_err(map_err)?;
        if head.classifier.out_dim != 2 {
            return Err(bad(format!(
                "head.safetensors has {} labels; extractive QA requires 2",
                head.classifier.out_dim
            )));
        }
        Ok(Self { tokenizer, head, config, family })
    }
}

/// Executes a `TASK_EXTRACTIVE_QA` `ModelInfer` call.
pub(crate) fn infer_qa(
    served: &Arc<Served>,
    loaded: &Arc<Loaded>,
    req: ModelInferRequest,
    deadline: &Deadline,
) -> Result<ModelInferResponse> {
    const ALL_OUTPUTS: [&str; 3] = ["spans", "scores", "probabilities"];
    let mut requested = Vec::new();
    for o in &req.outputs {
        if !ALL_OUTPUTS.contains(&o.name.as_str()) {
            return Err(bad(format!(
                "requested output `{}`: extractive QA outputs are `spans`, `scores`, and `probabilities`",
                o.name
            )));
        }
        if !o.parameters.is_empty() {
            return Err(bad(format!("requested output `{}` has parameters; none are defined", o.name)));
        }
        if requested.contains(&o.name.as_str()) {
            return Err(bad(format!("requested output `{}` is named more than once", o.name)));
        }
        requested.push(o.name.as_str());
    }
    let outputs_to_build: Vec<&str> = if requested.is_empty() { ALL_OUTPUTS.to_vec() } else { requested };

    let (question, context) = qa_inputs(&req)?;
    let decode_opts = qa_options(&req.parameters)?;
    let qa = loaded.qa.as_ref().expect("TASK_EXTRACTIVE_QA loaded state has QaModel");
    let map_err = |e: turbo::status::Error| Failure::new(e.code, e.field, e.message);

    let _docs = loaded.metrics.track_docs(1, Some(loaded.device));
    deadline.check_alive()?;
    let held = loaded.take()?;
    deadline.check_alive()?;
    let slot = held.slot();
    let mut cfg = qa.config.clone();
    cfg.max_seq = slot.info.max_seq;
    let windows = turbo::qa::windows(&qa.tokenizer, qa.family, &cfg, question, context).map_err(map_err)?;
    if windows.is_empty() {
        return Err(bad("extractive QA produced no windows"));
    }

    let mut win_logits = Vec::with_capacity(windows.len());
    let mut last_info = None;
    let fill = qa.tokenizer.fill_id();
    for chunk in windows.chunks(slot.info.max_batch as usize) {
        deadline.check_alive()?;
        loaded.metrics.record_batch_size(chunk.len());
        let (batch_logits, info) =
            unsafe { turbo::qa::run_window_batch(slot.session.as_ptr(), fill, &qa.head, chunk) }.map_err(map_err)?;
        win_logits.extend(batch_logits);
        last_info = Some(info);
    }
    drop(held);
    deadline.check_alive()?;

    let decoded = turbo::qa::decode(context, &win_logits, decode_opts).map_err(map_err)?;
    let (answers, abstained, null_score, null_probability) = match decoded {
        turbo::qa::Answer::None { null_score } => (Vec::new(), true, null_score, 1.0),
        turbo::qa::Answer::Spans { answers, null_score, null_probability } => {
            (answers, false, null_score, null_probability)
        }
    };
    let n = answers.len() as i64;
    let mut out_tensors = Vec::with_capacity(outputs_to_build.len());
    let mut raw_outputs = Vec::with_capacity(outputs_to_build.len());

    for name in outputs_to_build {
        match name {
            "spans" => {
                let mut bytes = Vec::with_capacity(answers.len() * 8);
                for a in &answers {
                    bytes.extend((a.bytes.start as i32).to_le_bytes());
                    bytes.extend((a.bytes.end as i32).to_le_bytes());
                }
                out_tensors.push(InferOutputTensor {
                    name: "spans".to_string(),
                    datatype: "INT32".to_string(),
                    shape: vec![n, 2],
                    parameters: HashMap::new(),
                    contents: None,
                });
                raw_outputs.push(Bytes::from(bytes));
            }
            "scores" => {
                let mut bytes = Vec::with_capacity(answers.len() * 8);
                for a in &answers {
                    bytes.extend(a.score.to_le_bytes());
                }
                out_tensors.push(InferOutputTensor {
                    name: "scores".to_string(),
                    datatype: "FP64".to_string(),
                    shape: vec![n],
                    parameters: HashMap::new(),
                    contents: None,
                });
                raw_outputs.push(Bytes::from(bytes));
            }
            "probabilities" => {
                let mut bytes = Vec::with_capacity(answers.len() * 8);
                for a in &answers {
                    bytes.extend(a.probability.to_le_bytes());
                }
                out_tensors.push(InferOutputTensor {
                    name: "probabilities".to_string(),
                    datatype: "FP64".to_string(),
                    shape: vec![n],
                    parameters: HashMap::new(),
                    contents: None,
                });
                raw_outputs.push(Bytes::from(bytes));
            }
            _ => unreachable!(),
        }
    }

    let mut params = result_parameters(&last_info.expect("at least one window"));
    params.insert(
        "abstained".to_string(),
        InferParameter { parameter_choice: Some(ParameterChoice::BoolParam(abstained)) },
    );
    params.insert(
        "null_score".to_string(),
        InferParameter { parameter_choice: Some(ParameterChoice::DoubleParam(null_score)) },
    );
    params.insert(
        "null_probability".to_string(),
        InferParameter { parameter_choice: Some(ParameterChoice::DoubleParam(null_probability)) },
    );

    Ok(ModelInferResponse {
        model_name: served.name.clone(),
        model_version: loaded.revision.clone(),
        id: req.id,
        parameters: params,
        outputs: out_tensors,
        raw_output_contents: raw_outputs,
    })
}

fn qa_inputs<'a>(req: &'a ModelInferRequest) -> Result<(&'a str, &'a str)> {
    for i in &req.inputs {
        if !i.parameters.is_empty() {
            return Err(bad(format!("input `{}` has parameters; none are defined", i.name)));
        }
        if !matches!(i.name.as_str(), "question" | "context") {
            return Err(bad(format!("input `{}`: extractive QA inputs are `question` and `context`", i.name)));
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
    let (Some(qk), Some(ck)) = (find("question")?, find("context")?) else {
        return Err(bad("extractive QA requires both `question` and `context` inputs"));
    };
    let q_ext = shape(&req.inputs[qk], 1)?;
    let c_ext = shape(&req.inputs[ck], 1)?;
    if q_ext != c_ext {
        return Err(bad_shape(format!("input `question` is shaped {:?} and `context` {:?}", q_ext, c_ext)));
    }
    let batch = q_ext[0];
    if batch == 0 {
        return Err(bad("extractive QA requires a non-empty batch"));
    }
    if batch != 1 {
        return Err(bad_shape(format!("extractive QA takes a single [1] question/context pair, not [{batch}]")));
    }

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
    let one_str = |k: usize| -> Result<&'a str> {
        let views = if raw {
            split_bytes(&req.raw_input_contents[k])?
        } else {
            let c = req.inputs[k].contents.as_ref().unwrap_or(&EMPTY);
            only(&req.inputs[k], c, "bytes_contents")?;
            c.bytes_contents.iter().map(|b| view(b)).collect()
        };
        if views.len() != 1 {
            return Err(bad_shape(format!(
                "input `{}` is shaped [1] and has {} elements",
                req.inputs[k].name,
                views.len()
            )));
        }
        let bytes = unsafe { std::slice::from_raw_parts(views[0].ptr as *const u8, views[0].len as usize) };
        std::str::from_utf8(bytes).map_err(|_| {
            Failure::new(TURBO_E_INVALID_UTF8, 0, format!("input `{}` is not valid UTF-8", req.inputs[k].name))
        })
    };
    Ok((one_str(qk)?, one_str(ck)?))
}

pub fn qa_options(params: &HashMap<String, InferParameter>) -> Result<turbo::qa::DecodeOptions> {
    let mut unknown: Vec<&String> = params.keys().filter(|k| !QA_OPTIONS.contains(&k.as_str())).collect();
    unknown.sort();
    if let Some(name) = unknown.first() {
        return Err(bad(format!("parameter `{name}` is not an extractive QA option")));
    }
    let mut opts = turbo::qa::DecodeOptions::default();
    for (k, name) in QA_OPTIONS.iter().enumerate() {
        let Some(p) = params.get(*name) else { continue };
        let field = k as u32 + 1;
        match (*name, p.parameter_choice.as_ref()) {
            ("max_answer_tokens", Some(ParameterChoice::Int64Param(n))) => {
                let v = u32::try_from(*n).map_err(|_| {
                    Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        field,
                        format!("parameter {name}: {n} is outside 0 to 4294967295"),
                    )
                })?;
                if v > 0 {
                    opts.max_answer_tokens = v as usize;
                }
            }
            ("top_k", Some(ParameterChoice::Int64Param(n))) => {
                let v = u32::try_from(*n).map_err(|_| {
                    Failure::new(
                        TURBO_E_INVALID_ARGUMENT,
                        field,
                        format!("parameter {name}: {n} is outside 0 to 4294967295"),
                    )
                })?;
                if v > 0 {
                    opts.top_k = v as usize;
                }
            }
            ("null_threshold", Some(ParameterChoice::DoubleParam(v))) if v.is_finite() => {
                opts.null_threshold = *v;
            }
            ("null_threshold", Some(ParameterChoice::DoubleParam(v))) => {
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name}: {v} must be finite"),
                ));
            }
            ("non_overlapping", Some(ParameterChoice::BoolParam(b))) => {
                opts.non_overlapping = *b;
            }
            (_, other) => {
                let want = match *name {
                    "max_answer_tokens" | "top_k" => "an int64_param",
                    "null_threshold" => "a double_param",
                    "non_overlapping" => "a bool_param",
                    _ => unreachable!(),
                };
                return Err(Failure::new(
                    TURBO_E_INVALID_ARGUMENT,
                    field,
                    format!("parameter {name} takes {want}, not {}", kind(other)),
                ));
            }
        }
    }
    Ok(opts)
}
