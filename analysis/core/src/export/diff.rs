//! Structural diff and disjoint merge of two [`DocumentGraph`]s
//! over one source: comparing two annotator versions, or folding
//! the layers of parallel pipeline branches back into one graph.
//!
//! Identity is by name, as everywhere at this boundary: a view by
//! its instance, a layer by its [`LayerRef`] (layer name and
//! instance). The diff is deterministic (both sides read in sorted
//! key order) and shallow by design: it reports which records were
//! added, removed or changed and how their item counts moved, not a
//! per-item delta, which belongs to the layer codecs. Merging is
//! strict about the shared parts (source and any view both graphs
//! hold must be record-equal, whatever `overwrite` says) and about
//! conflicting layers unless `overwrite` is set, matching the
//! boundary's no-silent-loss rule.

use super::*;

/// How one layer instance held by both graphs changed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerDiffSummary {
    pub layer: LayerRef,
    pub provenance_changed: bool,
    pub status_changed: bool,
    /// The codec bodies differ (either side missing a body counts).
    pub data_changed: bool,
    pub schema: i32,
    pub left_len: u32,
    pub right_len: u32,
}

/// The structural difference between two graphs, from the left
/// graph to the right one. Every list is in `(layer, instance)`
/// (or view instance) order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphDiff {
    pub source_matches: bool,
    pub version: (u16, u16),
    pub added_views: Vec<String>,
    pub removed_views: Vec<String>,
    pub modified_views: Vec<String>,
    pub added_layers: Vec<LayerRef>,
    pub removed_layers: Vec<LayerRef>,
    pub modified_layers: Vec<LayerDiffSummary>,
}

fn layer_key(record: &LayerRecord) -> LayerRef {
    LayerRef { layer: record.layer.clone(), instance: record.instance.clone() }
}

impl DocumentGraph {
    /// The structural difference from this graph to `other`.
    pub fn diff(&self, other: &DocumentGraph) -> GraphDiff {
        let mut added_views = Vec::new();
        let mut removed_views = Vec::new();
        let mut modified_views = Vec::new();
        let left_views: Vec<&ViewRecord> = self.views.iter().collect();
        for view in &other.views {
            match left_views.iter().find(|v| v.instance == view.instance) {
                None => added_views.push(view.instance.clone()),
                Some(left) if *left != view => modified_views.push(view.instance.clone()),
                Some(_) => {}
            }
        }
        for view in &self.views {
            if !other.views.iter().any(|v| v.instance == view.instance) {
                removed_views.push(view.instance.clone());
            }
        }
        added_views.sort();
        removed_views.sort();
        modified_views.sort();

        let mut added_layers = Vec::new();
        let mut removed_layers = Vec::new();
        let mut modified_layers = Vec::new();
        for record in &other.layers {
            let key = layer_key(record);
            match self.layers.iter().find(|l| layer_key(l) == key) {
                None => added_layers.push(key),
                Some(left) if left != record => modified_layers.push(LayerDiffSummary {
                    layer: key,
                    provenance_changed: left.provenance != record.provenance,
                    status_changed: left.status != record.status,
                    data_changed: left.body != record.body,
                    schema: i32::from(record.schema) - i32::from(left.schema),
                    left_len: left.len,
                    right_len: record.len,
                }),
                Some(_) => {}
            }
        }
        for record in &self.layers {
            let key = layer_key(record);
            if !other.layers.iter().any(|l| layer_key(l) == key) {
                removed_layers.push(key);
            }
        }
        added_layers.sort();
        removed_layers.sort();
        modified_layers.sort_by(|a, b| a.layer.cmp(&b.layer));

        GraphDiff {
            source_matches: self.source == other.source,
            version: (self.version, other.version),
            added_views,
            removed_views,
            modified_views,
            added_layers,
            removed_layers,
            modified_layers,
        }
    }

    /// Folds `other`'s views and layers into this graph. The format
    /// versions and the source records must match exactly, and a
    /// view both graphs hold must be the same record; those are
    /// refused whatever `overwrite` says, because every layer in
    /// both graphs names its view by position among them. A layer
    /// both graphs hold must be the same record too, unless
    /// `overwrite` is set, in which case `other`'s record replaces
    /// this one's. New views and layers are appended in `other`'s
    /// order, so a disjoint branch merge is deterministic. On any
    /// refusal this graph is left unchanged.
    pub fn merge_layers(&mut self, other: DocumentGraph, overwrite: bool) -> Result<(), ExportError> {
        if self.version != other.version {
            return Err(ExportError::UnsupportedVersion { found: other.version, supported: self.version });
        }
        if self.source != other.source {
            return Err(ExportError::InvalidSource("graphs over different sources cannot merge".to_string()));
        }
        for view in &other.views {
            if let Some(left) = self.views.iter().find(|v| v.instance == view.instance)
                && left != view
            {
                return Err(ExportError::DuplicateLayer {
                    layer: "turbo:view".to_string(),
                    instance: view.instance.clone(),
                });
            }
        }
        for record in &other.layers {
            let key = layer_key(record);
            if let Some(left) = self.layers.iter().find(|l| layer_key(l) == key)
                && left != record
                && !overwrite
            {
                return Err(ExportError::DuplicateLayer { layer: key.layer, instance: key.instance });
            }
        }
        for view in other.views {
            if !self.views.iter().any(|v| v.instance == view.instance) {
                self.views.push(view);
            }
        }
        for record in other.layers {
            let key = layer_key(&record);
            match self.layers.iter_mut().find(|l| layer_key(l) == key) {
                Some(left) => {
                    if overwrite {
                        *left = record;
                    }
                }
                None => self.layers.push(record),
            }
        }
        Ok(())
    }
}
