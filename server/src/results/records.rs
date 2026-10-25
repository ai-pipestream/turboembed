//! RecordsResult: the structured records of a schema extraction, the
//! schema-extract `turbo:records` layer plus the model identity, over one
//! view's text, every span offset counted in the message's `OffsetMode`.
//! Decoding converts the offsets to UTF-8 bytes through the text's
//! checkpoint table, rebuilds the label sets and the layer, and re-proves
//! the layer's own check, so a malformed message is refused the way the
//! layer refuses a malformed column.

use std::sync::Arc;

use turbo_analysis::{Layer, ModelRef, Tag, TagSet, ViewId};
use turbo_schema_extract::{Limits, Records};

use super::layers::{spans_from_wire, spans_to_wire};
use super::{MAX_RECORD_VALUES, Wire, mismatched, model_from_proto, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{RecordsResult, SchemaLimits};

/// The record layer over one view, the view's text (the offsets' ground
/// truth), the wire unit `to_msg` emits, and the model's identity.
pub struct RecordSet {
    pub model: ModelRef,
    pub text: String,
    pub view: ViewId,
    pub offsets: OffsetMode,
    pub records: Records,
}

/// A schema's limits as the capability listing shows them; a limit past
/// `u32` shows as `u32::MAX`.
pub fn schema_limits(l: &Limits) -> SchemaLimits {
    let n = |v: usize| u32::try_from(v).unwrap_or(u32::MAX);
    SchemaLimits {
        max_entity_types: n(l.max_entity_types),
        max_classification_tasks: n(l.max_classification_tasks),
        max_labels_per_task: n(l.max_labels_per_task),
        max_structures: n(l.max_structures),
        max_fields_per_structure: n(l.max_fields_per_structure),
        max_text_bytes: n(l.max_text_bytes),
        max_prompt_subwords: n(l.max_prompt_subwords),
        max_words: n(l.max_words),
    }
}

fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

fn labels(set: &TagSet) -> Vec<String> {
    (0..set.len()).map(|i| set.label(Tag(i as u16)).to_string()).collect()
}

/// A label set from the wire: at most a `u16` of labels, each once.
fn tag_set(what: &str, name: &str, labels: &[String]) -> Result<Arc<TagSet>, Failure> {
    if labels.len() > usize::from(u16::MAX) {
        return Err(over("max_record_values", what, labels.len() as u64));
    }
    let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
    let set = TagSet::new(name, &refs);
    if let Some(dup) = refs.iter().enumerate().find(|&(i, l)| set.tag(l) != Some(Tag(i as u16))) {
        return Err(bad(format!("{what}: label {:?} is listed twice", dup.1)));
    }
    Ok(set)
}

/// Wire tags into a set of `n` labels.
fn tags(what: &str, col: &[u32], n: usize) -> Result<Vec<Tag>, Failure> {
    col.iter()
        .map(|&t| {
            if (t as usize) < n {
                Ok(Tag(t as u16))
            } else {
                Err(bad(format!("{what}: tag {t} is outside the {n} labels")))
            }
        })
        .collect()
}

fn source_code(what: &str, col: &[u32]) -> Result<Vec<u8>, Failure> {
    col.iter().map(|&s| u8::try_from(s).map_err(|_| bad(format!("{what}: source {s} is not a known source")))).collect()
}

impl Wire for RecordSet {
    type Msg = RecordsResult;

    fn to_msg(&self) -> RecordsResult {
        let r = &self.records;
        let offsetter = Offsetter::new(&self.text, self.offsets);
        let (value_frag_start, value_frag_end, value_frag_at) = spans_to_wire(&r.spans, &self.text, &offsetter);
        let wide = |t: &[Tag]| t.iter().map(|t| u32::from(t.0)).collect::<Vec<u32>>();
        RecordsResult {
            model: Some(model_to_proto(&self.model)),
            text: self.text.clone(),
            view: u32::from(self.view.0),
            offsets: self.offsets.to_proto(),
            types_name: r.types.name.clone(),
            types: labels(&r.types),
            paths_name: r.paths.name.clone(),
            paths: labels(&r.paths),
            choices_name: r.choices.name.clone(),
            choices: labels(&r.choices),
            record_type: wide(&r.record_type),
            record_instance: r.instance.clone(),
            value_frag_start,
            value_frag_end,
            value_frag_at,
            value_record: r.record.clone(),
            value_path: wide(&r.path),
            value_score: r.score.clone(),
            value_logit: r.logit.clone(),
            support_end: r.support_end.clone(),
            support_source: r.support_source.iter().map(|&s| u32::from(s)).collect(),
            support_row: r.support_row.clone(),
            choice_record: r.choice_record.clone(),
            choice_path: wide(&r.choice_path),
            choice: wide(&r.choice),
            choice_score: r.choice_score.clone(),
            choice_logit: r.choice_logit.clone(),
        }
    }

    fn from_msg(msg: &RecordsResult) -> Result<Self, Failure> {
        let offsets = OffsetMode::from_proto(msg.offsets)?;
        if msg.view > u32::from(u16::MAX) {
            return Err(bad(format!("view {} is outside 0 to 65535", msg.view)));
        }
        let view = ViewId(msg.view as u16);
        let values = msg.value_frag_at.len();
        for (what, n) in [
            ("record values", values),
            ("records", msg.record_type.len()),
            ("choice values", msg.choice.len()),
            ("support rows", msg.support_source.len()),
        ] {
            if n as u64 > MAX_RECORD_VALUES {
                return Err(over("max_record_values", what, n as u64));
            }
        }
        for (name, col) in [
            ("value_record", msg.value_record.len()),
            ("value_path", msg.value_path.len()),
            ("value_score", msg.value_score.len()),
            ("value_logit", msg.value_logit.len()),
            ("support_end", msg.support_end.len()),
        ] {
            if col != values {
                return Err(mismatched(name, col, values));
            }
        }
        if msg.record_instance.len() != msg.record_type.len() {
            return Err(mismatched("record_instance", msg.record_instance.len(), msg.record_type.len()));
        }
        if msg.support_row.len() != msg.support_source.len() {
            return Err(mismatched("support_row", msg.support_row.len(), msg.support_source.len()));
        }
        let choices = msg.choice.len();
        for (name, col) in [
            ("choice_record", msg.choice_record.len()),
            ("choice_path", msg.choice_path.len()),
            ("choice_score", msg.choice_score.len()),
            ("choice_logit", msg.choice_logit.len()),
        ] {
            if col != choices {
                return Err(mismatched(name, col, choices));
            }
        }
        let types = tag_set("types", &msg.types_name, &msg.types)?;
        let paths = tag_set("paths", &msg.paths_name, &msg.paths)?;
        let choice_set = tag_set("choices", &msg.choices_name, &msg.choices)?;
        let offsetter = Offsetter::new(&msg.text, offsets);
        let spans = spans_from_wire(
            "record value",
            view,
            &msg.text,
            &offsetter,
            &msg.value_frag_start,
            &msg.value_frag_end,
            &msg.value_frag_at,
        )?;
        let records = Records {
            spans,
            record: msg.value_record.clone(),
            path: tags("value_path", &msg.value_path, paths.len())?,
            score: msg.value_score.clone(),
            logit: msg.value_logit.clone(),
            support_end: msg.support_end.clone(),
            support_source: source_code("support_source", &msg.support_source)?,
            support_row: msg.support_row.clone(),
            record_type: tags("record_type", &msg.record_type, types.len())?,
            instance: msg.record_instance.clone(),
            choice_record: msg.choice_record.clone(),
            choice_path: tags("choice_path", &msg.choice_path, paths.len())?,
            choice: tags("choice", &msg.choice, choice_set.len())?,
            choice_score: msg.choice_score.clone(),
            choice_logit: msg.choice_logit.clone(),
            types,
            paths,
            choices: choice_set,
        };
        records.check().map_err(|e| bad(format!("records: {e}")))?;
        let model = msg.model.as_ref().map(model_from_proto).ok_or_else(|| bad("records carry no model identity"))?;
        Ok(RecordSet { model, text: msg.text.clone(), view, offsets, records })
    }
}
