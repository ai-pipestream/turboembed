use super::*;

/// How one layer type crosses the boundary. The payload is the codec's
/// own, written with [`Writer`] and read with [`Reader`]; span layers
/// and tag columns use [`put_spans`], [`get_spans`], [`put_tagset`] and
/// [`get_tagset`] so those fields agree across codecs. An import must
/// refuse a payload that does not hold together (a view that is not
/// here, an endpoint past its layer) with the typed errors on
/// [`ImportCx`], never by clamping or defaulting. Import also sees the
/// document built so far, for checks that need the actual layers and
/// view texts already restored (a word's subspan against its token's
/// text, for one), which item counts alone cannot prove.
pub trait LayerCodec: Layer {
    fn export(&self, cx: &ExportCx) -> Result<Vec<u8>, ExportError>;
    fn import(data: &[u8], cx: &mut ImportCx, doc: &Document) -> Result<Self, ExportError>;
}

/// What an exporting codec needs beyond its own layer: the instance
/// name of each view, so span columns name their view across the
/// boundary instead of carrying a process-local [`ViewId`], and the
/// instance a layer was actually made on (see [`ExportCx::resolve`]).
pub struct ExportCx {
    pub(super) views: HashMap<ViewId, String>,
    pub(super) on: Vec<KeyId>,
    pub(super) reads: Vec<KeyId>,
}

impl ExportCx {
    pub fn view_instance(&self, view: ViewId) -> Result<&str, ExportError> {
        self.views
            .get(&view)
            .map(String::as_str)
            .ok_or_else(|| ExportError::UnknownView { instance: format!("view {}", view.0) })
    }

    /// The instance a reference through `key` actually names on the
    /// document being exported. Relation layers built with
    /// `Edges::new` or `Groups::new` carry the default key even when
    /// the step that wrote them was bound to a named instance; the
    /// document resolved the empty name through the layer's provenance
    /// when it took the layer, and this applies the same resolution,
    /// so an exported endpoint names the instance it indexes. Codecs
    /// that write layer references (relation endpoints above all) must
    /// resolve them through here.
    pub fn resolve<L: Layer>(&self, key: &Key<L>) -> LayerRef {
        if key.name.is_empty() {
            let ty = TypeId::of::<L>();
            if let Some(found) = self.on.iter().chain(&self.reads).find(|r| r.ty == ty) {
                return ref_of(*found);
            }
        }
        layer_ref(key)
    }
}

/// What an importing codec needs: view instances resolved to the new
/// document's [`ViewId`]s, the item counts of the layers restored so
/// far, and `&'static str` names for keys from a process-wide interner.
pub struct ImportCx {
    pub(super) views: HashMap<String, ViewId>,
    pub(super) lens: HashMap<(String, String), usize>,
    pub(super) known: HashSet<(String, String)>,
    pub(super) made_on: Vec<LayerRef>,
    pub(super) layer: String,
}

impl ImportCx {
    /// The view a record names, in the document being built.
    pub fn view(&self, instance: &str) -> Result<ViewId, ExportError> {
        self.views.get(instance).copied().ok_or_else(|| ExportError::UnknownView { instance: instance.to_owned() })
    }

    /// The layer instance being imported, for error fields that name
    /// the importing layer.
    pub fn layer(&self) -> &str {
        &self.layer
    }

    /// How many items a restored layer instance holds.
    pub fn layer_len(&self, layer: &LayerRef) -> Result<usize, ExportError> {
        self.lens
            .get(&(layer.layer.clone(), layer.instance.clone()))
            .copied()
            .ok_or_else(|| ExportError::DanglingReference { layer: self.layer.clone(), reference: layer.clone() })
    }

    /// The instance of `layer` the document will bind an empty-named
    /// reference to for the record being imported: the first key of
    /// that layer in its provenance `on`, then in its `reads`, else the
    /// default instance. This is the resolution `Document::add`
    /// applies, mirrored so a payload can be held to it.
    pub fn resolved_instance(&self, layer: &str) -> &str {
        self.made_on.iter().find(|r| r.layer == layer).map(|r| r.instance.as_str()).unwrap_or("")
    }

    /// Refuses unless `named` is the target a default-keyed index or
    /// range column into `Target` must name on this record: the right
    /// layer type, the instance `Document::add` will bind it to (see
    /// [`ImportCx::resolved_instance`]), and an instance already
    /// restored. Codecs call this for columns whose layer type indexes
    /// through `Key::DEFAULT`; a payload naming any other instance
    /// would otherwise be bounds-checked against one instance and
    /// silently attached to another.
    pub fn expect_target<Target: Layer>(&self, named: &LayerRef) -> Result<(), ExportError> {
        if named.layer != Target::NAME.to_string() {
            return Err(ExportError::InvalidLayer {
                layer: self.layer.clone(),
                detail: format!("an index column into {}, not {}", named.layer, Target::NAME),
            });
        }
        let resolved = self.resolved_instance(&named.layer);
        if named.instance != resolved {
            return Err(ExportError::InvalidLayer {
                layer: self.layer.clone(),
                detail: format!(
                    "the payload indexes {named} but this layer's provenance binds it to instance {resolved:?}"
                ),
            });
        }
        self.layer_len(named)?;
        Ok(())
    }

    /// Refuses unless `named` can be restored as the stored key of a
    /// keyed reference into `Target` without changing which instance
    /// it names. A non-empty instance is explicit and is preserved
    /// as-is. An empty instance restores as `Key::DEFAULT`, which
    /// `Document::add` and `ExportCx::resolve` reinterpret through
    /// this record's provenance, so it is only accepted when that
    /// provenance also resolves `Target` to the default instance;
    /// otherwise the payload would be validated against the default
    /// instance and silently rebound to the resolved one.
    pub fn want_keyed_target<Target: Layer>(&self, named: &LayerRef) -> Result<(), ExportError> {
        if named.layer != Target::NAME.to_string() {
            return Err(ExportError::InvalidLayer {
                layer: self.layer.clone(),
                detail: format!("an index column into {}, not {}", named.layer, Target::NAME),
            });
        }
        if named.instance.is_empty() {
            let resolved = self.resolved_instance(&named.layer);
            if !resolved.is_empty() {
                return Err(ExportError::InvalidLayer {
                    layer: self.layer.clone(),
                    detail: format!(
                        "the payload indexes the default {named} but this layer's provenance binds an empty key to instance {resolved:?}, so an explicit default cannot be restored"
                    ),
                });
            }
        }
        Ok(())
    }

    /// Refuses unless the node is an item of a restored layer instance.
    pub fn require_node(&self, node: &NodeRef) -> Result<(), ExportError> {
        if node.index as usize >= self.layer_len(&node.layer)? {
            return Err(ExportError::DanglingEndpoint { layer: self.layer.clone(), endpoint: node.clone() });
        }
        Ok(())
    }

    /// An instance name as the `&'static str` a [`Key`] needs. Names
    /// are interned process-wide: one leaked copy per distinct name
    /// ever imported, however many documents cross.
    pub fn intern(&mut self, name: &str) -> &'static str {
        intern_name(name)
    }

    /// The key of an instance of `L`, by the name a record carries.
    pub fn key<L: Layer>(&mut self, instance: &str) -> Key<L> {
        Key::named(self.intern(instance))
    }
}

/// The process-wide name interner behind [`ImportCx::intern`].
pub(super) fn intern_name(name: &str) -> &'static str {
    if name.is_empty() {
        return "";
    }
    static NAMES: OnceLock<Mutex<HashSet<&'static str>>> = OnceLock::new();
    let mut names = NAMES.get_or_init(|| Mutex::new(HashSet::new())).lock().expect("the name interner is not poisoned");
    match names.get(name) {
        Some(&found) => found,
        None => {
            let leaked: &'static str = Box::leak(name.to_owned().into_boxed_str());
            names.insert(leaked);
            leaked
        }
    }
}
