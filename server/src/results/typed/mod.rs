//! Strongly-typed `TypedDocumentResult` projection (`proto/turbo_typed_annotations.proto`,
//! issue #1219) and validation (`RoundTripTypedDocument`, `InspectDocumentGraph`).

mod layers_context;
mod layers_core;
mod layers_domain;
mod layers_entities;
mod layers_quality;
mod layers_sentiment;
mod layers_shared;
mod layers_structure;
mod validate;

use std::any::TypeId;
use std::collections::HashMap;

use turbo_analysis::export::{DocumentGraph, import};
use turbo_analysis::{
    Anchor, Document, Evidence, Execution, Ix, Key, KeyId, Layer, ModelRef, Provenance, Span, Spans, Status, View,
    ViewId,
};
use turbo_analysis_layers::{Categories, Chunks, Entities, Opinions, Sentences, Structure};

use super::document::codecs;
use super::{MAX_GRAPH_BYTES, Wire, model_to_proto, over};
use crate::api::{Failure, TURBO_E_INVALID_ARGUMENT};
use crate::offsets::{OffsetMode, Offsetter};
use crate::proto::result::{
    AnchorKind, ExecutionInfo, InspectDocumentGraphRequest, LayerIndexBound, LayerWireStatus, ModelProvenance,
    SpanFragment, TypedAnnotation, TypedDocumentResult, TypedLayer, ViewInfo,
};

pub use validate::validate_typed_document;

pub(crate) fn bad(message: impl Into<String>) -> Failure {
    Failure::new(TURBO_E_INVALID_ARGUMENT, 0, message)
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ProvenanceKey {
    annotator: String,
    version: String,
    config_hash: u64,
    model_id: String,
    model_rev: String,
    model_sha: String,
    backend: String,
    device: String,
    precision: String,
}

fn execution_to_proto(exec: &Execution) -> ExecutionInfo {
    ExecutionInfo { backend: exec.backend.clone(), device: exec.device.clone(), precision: exec.precision.clone() }
}

pub(crate) struct ProjectionCx<'a> {
    pub(crate) doc: &'a Document,
    pub(crate) target_view: ViewId,
    pub(crate) target_text: &'a str,
    pub(crate) offsetter: Offsetter,
    pub(crate) models: Vec<ModelProvenance>,
    model_map: HashMap<ProvenanceKey, String>,
}

impl<'a> ProjectionCx<'a> {
    fn new(doc: &'a Document, target_view: ViewId, target_text: &'a str, offsets: OffsetMode) -> Self {
        Self {
            doc,
            target_view,
            target_text,
            offsetter: Offsetter::new(target_text, offsets),
            models: Vec::new(),
            model_map: HashMap::new(),
        }
    }

    pub(crate) fn register_provenance(&mut self, prov: &Provenance) -> String {
        let (model_id, model_rev, model_sha) = match &prov.model {
            Some(m) => (m.id.clone(), m.revision.clone(), m.sha256.clone()),
            None => (String::new(), String::new(), String::new()),
        };
        let (backend, device, precision) = match &prov.execution {
            Some(e) => (e.backend.clone(), e.device.clone(), e.precision.clone()),
            None => (String::new(), String::new(), String::new()),
        };
        let key = ProvenanceKey {
            annotator: prov.annotator.to_owned(),
            version: prov.version.to_owned(),
            config_hash: prov.config,
            model_id,
            model_rev,
            model_sha,
            backend,
            device,
            precision,
        };
        if let Some(existing) = self.model_map.get(&key) {
            return existing.clone();
        }
        let model_ref = format!("m{}", self.models.len());
        self.models.push(ModelProvenance {
            model_ref: model_ref.clone(),
            annotator: prov.annotator.to_owned(),
            version: prov.version.to_owned(),
            config_hash: prov.config,
            bundle: prov.model.as_ref().map(model_to_proto),
            execution: prov.execution.as_ref().map(execution_to_proto),
        });
        self.model_map.insert(key, model_ref.clone());
        model_ref
    }

    pub(crate) fn register_producer(&mut self, annotator: &str, model: Option<&ModelRef>) -> String {
        if let Some(found) = self.models.iter().find(|m| {
            m.annotator == annotator
                && match (&m.bundle, model) {
                    (Some(b), Some(want)) => b.id == want.id && b.revision == want.revision && b.sha256 == want.sha256,
                    (None, None) => true,
                    _ => false,
                }
        }) {
            return found.model_ref.clone();
        }
        let (model_id, model_rev, model_sha) = match model {
            Some(m) => (m.id.clone(), m.revision.clone(), m.sha256.clone()),
            None => (String::new(), String::new(), String::new()),
        };
        let key = ProvenanceKey {
            annotator: annotator.to_owned(),
            version: "0".to_owned(),
            config_hash: 0,
            model_id,
            model_rev,
            model_sha,
            backend: String::new(),
            device: String::new(),
            precision: String::new(),
        };
        if let Some(existing) = self.model_map.get(&key) {
            return existing.clone();
        }
        let model_ref = format!("m{}", self.models.len());
        self.models.push(ModelProvenance {
            model_ref: model_ref.clone(),
            annotator: annotator.to_owned(),
            version: "0".to_owned(),
            config_hash: 0,
            bundle: model.map(model_to_proto),
            execution: None,
        });
        self.model_map.insert(key, model_ref.clone());
        model_ref
    }

    pub(crate) fn project_span(&self, layer_view: ViewId, span: Span) -> Span {
        if layer_view == self.target_view {
            span
        } else {
            let src = self.doc.to_source(layer_view, span);
            self.doc.from_source(self.target_view, src)
        }
    }

    pub(crate) fn project_wire_span(&self, layer_view: ViewId, span: Span) -> SpanFragment {
        let mapped = self.project_span(layer_view, span);
        let (start, end) = self.offsetter.to_wire(self.target_text, mapped);
        SpanFragment { start, end }
    }

    pub(crate) fn project_spans<L>(&self, spans: &Spans<L>, item: Ix<L>) -> Vec<SpanFragment> {
        let view = spans.view();
        spans.fragments(item).map(|frag| self.project_wire_span(view, frag)).collect()
    }

    pub(crate) fn target_slice(&self, layer_view: ViewId, span: Span) -> &'a str {
        let mapped = self.project_span(layer_view, span);
        self.target_text.get(mapped.range()).unwrap_or_default()
    }
}

pub(crate) fn coalesce_wire_spans(mut frags: Vec<SpanFragment>) -> Vec<SpanFragment> {
    if frags.len() <= 1 {
        return frags;
    }
    frags.retain(|f| f.start < f.end);
    frags.sort_by_key(|f| (f.start, f.end));
    let mut out: Vec<SpanFragment> = Vec::with_capacity(frags.len());
    for f in frags {
        if let Some(last) = out.last_mut()
            && f.start < last.end
        {
            last.end = last.end.max(f.end);
        } else {
            out.push(f);
        }
    }
    out
}

pub(crate) fn find_aligned_to<A: Layer>(doc: &Document, anchor_key: KeyId) -> Option<&A> {
    for k in doc.layers() {
        if k.ty != TypeId::of::<A>() {
            continue;
        }
        let made = doc.made_on(k);
        let bound_to_anchor = match made.first() {
            Some(&first) => first == anchor_key,
            None => k.name == anchor_key.name,
        };
        if bound_to_anchor {
            return doc.get_key(&Key::<A>::named(k.name));
        }
    }
    None
}

pub(crate) fn status_to_wire(status: Option<&Status>) -> (LayerWireStatus, String) {
    match status {
        Some(Status::Complete) | None => (LayerWireStatus::Ok, String::new()),
        Some(Status::Skipped(reason)) => (LayerWireStatus::Skipped, reason.clone()),
        Some(Status::Failed(err)) => (LayerWireStatus::Failed, format!("{err:?}")),
    }
}

pub(crate) fn build_layer<L: Layer>(
    cx: &mut ProjectionCx<'_>,
    key: KeyId,
    project: impl FnOnce(&mut ProjectionCx<'_>, &L, Option<&str>) -> Vec<TypedAnnotation>,
) -> TypedLayer {
    let (status, status_detail) = status_to_wire(cx.doc.status(key));
    let model_ref = cx.doc.provenance(key).map(|p| cx.register_provenance(p));
    let mut model_refs: Vec<String> = model_ref.iter().cloned().collect();
    let made_on = cx.doc.made_on(key);

    let (scope, anchor_layer) = match L::anchor() {
        Anchor::Text => (AnchorKind::Text, None),
        Anchor::Aligned { layer, .. } => {
            let resolved = made_on.first().map(ToString::to_string).unwrap_or_else(|| layer.to_string());
            (AnchorKind::Aligned, Some(resolved))
        }
        Anchor::Document => (AnchorKind::Document, None),
    };

    let (annotations, indexes_layer) = match cx.doc.get_key(&Key::<L>::named(key.name)) {
        Some(layer) => {
            let raw_indexes = layer.indexes();
            let mut bounds = Vec::with_capacity(raw_indexes.len());
            for (idx_key, bound) in raw_indexes {
                let resolved = if !idx_key.name.is_empty() {
                    idx_key
                } else {
                    made_on.iter().copied().find(|k| k.ty == idx_key.ty).unwrap_or(idx_key)
                };
                bounds.push(LayerIndexBound {
                    layer: resolved.layer.to_string(),
                    instance: resolved.name.to_owned(),
                    bound: bound as u32,
                });
            }
            let anns = project(cx, layer, model_ref.as_deref());
            for a in &anns {
                if let Some(m) = &a.model_ref
                    && !model_refs.contains(m)
                {
                    model_refs.push(m.clone());
                }
                if let Some(crate::proto::result::typed_annotation::Value::Entity(ent)) = &a.value {
                    for v in &ent.votes {
                        if let Some(m) = &v.model_ref
                            && !model_refs.contains(m)
                        {
                            model_refs.push(m.clone());
                        }
                    }
                }
            }
            (anns, bounds)
        }
        None => (Vec::new(), Vec::new()),
    };

    TypedLayer {
        id: key.layer.to_string(),
        instance: key.name.to_owned(),
        status: status as i32,
        status_detail,
        scope: scope as i32,
        model_refs,
        anchor_layer,
        annotations,
        indexes_layer,
    }
}

fn is_evidence_layer(ty: TypeId) -> bool {
    ty == TypeId::of::<Evidence<Entities>>()
        || ty == TypeId::of::<Evidence<Categories<View>>>()
        || ty == TypeId::of::<Evidence<Categories<Sentences>>>()
        || ty == TypeId::of::<Evidence<Categories<Structure>>>()
        || ty == TypeId::of::<Evidence<Categories<Chunks>>>()
        || ty == TypeId::of::<Evidence<Categories<Opinions>>>()
        || ty == TypeId::of::<Evidence<Categories<turbo_analysis_layers::Aspects>>>()
        || ty == TypeId::of::<Evidence<turbo_relation::Relations>>()
        || ty == TypeId::of::<Evidence<turbo_schema_extract::Records>>()
}

fn collect_views(doc: &Document) -> (Vec<ViewInfo>, Vec<(ViewId, String)>) {
    let mut views = Vec::new();
    let mut id_names = Vec::new();
    for key in doc.layers() {
        if key.ty == TypeId::of::<View>()
            && let Some(v) = doc.get_key(&Key::<View>::named(key.name))
        {
            id_names.push((v.id(), key.name.to_owned()));
        }
    }
    for key in doc.layers() {
        if key.ty != TypeId::of::<View>() {
            continue;
        }
        let (status, status_detail) = status_to_wire(doc.status(key));
        let annotator = doc.provenance(key).map(|p| p.annotator.to_owned()).unwrap_or_default();
        let parent = doc
            .get_key(&Key::<View>::named(key.name))
            .and_then(|v| v.parent())
            .and_then(|pid| id_names.iter().find(|(id, _)| *id == pid).map(|(_, name)| name.clone()))
            .unwrap_or_default();
        views.push(ViewInfo { instance: key.name.to_owned(), status: status as i32, status_detail, annotator, parent });
    }
    (views, id_names)
}

/// Projects all shared layers on `doc` into a strongly-typed [`TypedDocumentResult`]
/// with spans mapped to `target_view` and converted into `offsets`.
pub fn document_to_typed(
    doc_id: &str,
    doc: &Document,
    target_view: ViewId,
    offsets: OffsetMode,
) -> Result<TypedDocumentResult, Failure> {
    let (views, view_ids) = collect_views(doc);
    if !view_ids.iter().any(|(id, _)| *id == target_view) {
        return Err(bad(format!("target view {} is outside the document's {} views", target_view.0, view_ids.len())));
    }
    let target_text = doc.text(target_view);
    let mut cx = ProjectionCx::new(doc, target_view, target_text, offsets);
    let mut layers = Vec::new();

    for key in doc.layers() {
        if key.ty == TypeId::of::<View>() || is_evidence_layer(key.ty) {
            continue;
        }
        if let Some(typed) = layers_core::project_core_layer(&mut cx, key)
            .or_else(|| layers_entities::project_entity_layer(&mut cx, key))
            .or_else(|| layers_domain::project_domain_layer(&mut cx, key))
            .or_else(|| layers_sentiment::project_sentiment_layer(&mut cx, key))
            .or_else(|| layers_quality::project_quality_layer(&mut cx, key))
            .or_else(|| layers_structure::project_structure_layer(&mut cx, key))
            .or_else(|| layers_shared::project_shared_layer(&mut cx, key))
            .or_else(|| layers_context::project_context_layer(&mut cx, key))
        {
            layers.push(typed);
        } else {
            let (status, status_detail) = status_to_wire(cx.doc.status(key));
            let model_refs = cx.doc.provenance(key).map(|p| vec![cx.register_provenance(p)]).unwrap_or_default();
            layers.push(TypedLayer {
                id: key.layer.to_string(),
                instance: key.name.to_owned(),
                status: status as i32,
                status_detail,
                scope: AnchorKind::Document as i32,
                model_refs,
                anchor_layer: None,
                annotations: Vec::new(),
                indexes_layer: Vec::new(),
            });
        }
    }

    Ok(TypedDocumentResult {
        doc_id: doc_id.to_owned(),
        raw_text: target_text.to_owned(),
        view: u32::from(target_view.0),
        offsets: offsets.to_proto(),
        models: cx.models,
        views,
        layers,
    })
}

/// Imports the binary `DocumentGraph` in `req` and projects it into a
/// [`TypedDocumentResult`] in the requested view and [`OffsetMode`].
pub fn inspect_document_graph(req: &InspectDocumentGraphRequest) -> Result<TypedDocumentResult, Failure> {
    if req.graph.len() as u64 > MAX_GRAPH_BYTES {
        return Err(over("max_graph_bytes", "document graph bytes", req.graph.len() as u64));
    }
    let offsets = OffsetMode::from_proto(req.offsets)?;
    let view_u16 = u16::try_from(req.view).map_err(|_| bad(format!("view {} exceeds u16::MAX", req.view)))?;
    let graph = DocumentGraph::from_bytes(&req.graph).map_err(|e| bad(format!("document graph: {e}")))?;
    let doc = import(&graph, codecs()).map_err(|e| bad(format!("document graph import: {e}")))?;
    document_to_typed(&req.doc_id, &doc, ViewId(view_u16), offsets)
}

/// Validates `msg` and returns a clone for `RoundTripTypedDocument`.
pub fn round_trip_typed_document(msg: &TypedDocumentResult) -> Result<TypedDocumentResult, Failure> {
    validate_typed_document(msg)?;
    Ok(msg.clone())
}

impl Wire for TypedDocumentResult {
    type Msg = TypedDocumentResult;

    fn to_msg(&self) -> Self::Msg {
        self.clone()
    }

    fn from_msg(msg: &Self::Msg) -> Result<Self, Failure> {
        round_trip_typed_document(msg)
    }
}
