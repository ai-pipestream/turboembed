use super::*;

type Exporter = fn(&dyn Any, &ExportCx) -> Result<Vec<u8>, ExportError>;
type Importer = fn(&mut Document, &LayerRecord, Vec<u8>, &mut ImportCx, Provenance) -> Result<usize, ExportError>;

pub(super) struct Codec {
    pub(super) schema: u16,
    pub(super) export: Exporter,
    pub(super) import: Importer,
    pub(super) key: fn(&'static str) -> KeyId,
}

/// The codecs a caller works with, one per layer type. Export and
/// import refuse a layer with none, so what crosses the boundary is
/// exactly what the caller registered.
#[derive(Default)]
pub struct Codecs {
    by_type: HashMap<TypeId, Codec>,
    by_name: HashMap<String, TypeId>,
}

impl Codecs {
    pub fn new() -> Codecs {
        Codecs::default()
    }

    /// Registers `L`'s codec under its full layer name.
    pub fn register<L: LayerCodec>(&mut self) -> &mut Codecs {
        let codec = Codec {
            schema: L::SCHEMA,
            export: |value, cx| {
                let layer =
                    value.downcast_ref::<L>().ok_or_else(|| malformed("a layer value that is not its key's type"))?;
                L::export(layer, cx)
            },
            import: |doc, record, body, cx, provenance| {
                let layer = L::import(&body, cx, doc)?;
                let len = layer.len();
                let key = Key::<L>::named(cx.intern(&record.instance));
                doc.add(&key, layer, provenance)
                    .map_err(|e| ExportError::InvalidLayer { layer: record.layer.clone(), detail: e.to_string() })?;
                Ok(len)
            },
            key: |instance| Key::<L>::named(instance).id(),
        };
        self.by_name.insert(L::NAME.to_string(), TypeId::of::<L>());
        self.by_type.insert(TypeId::of::<L>(), codec);
        self
    }

    pub(super) fn by_type_id(&self, ty: TypeId) -> Option<&Codec> {
        self.by_type.get(&ty)
    }

    pub(super) fn by_layer_name(&self, name: &str) -> Option<&Codec> {
        self.by_name.get(name).and_then(|ty| self.by_type.get(ty))
    }

    pub(super) fn key_id(&self, layer: &LayerRef, cx: &mut ImportCx) -> Result<KeyId, ExportError> {
        if layer.layer == View::ID {
            return Ok(Key::<View>::named(cx.intern(&layer.instance)).id());
        }
        let codec =
            self.by_layer_name(&layer.layer).ok_or_else(|| ExportError::NoCodec { layer: layer.layer.clone() })?;
        Ok((codec.key)(cx.intern(&layer.instance)))
    }
}
