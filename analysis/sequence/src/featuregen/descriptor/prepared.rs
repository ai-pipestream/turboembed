//! An immutable parsed descriptor and resources, with fresh adaptive generators.
use super::*;

/// A descriptor parsed and validated once. Dictionaries and clusters are
/// shared by Arc; each call to `create` owns its caches and document history.
pub struct Prepared {
    root: Element,
    mode: PatternMode,
    kinds: HashMap<String, Kind>,
    loaded: HashMap<String, Resource>,
}

impl Prepared {
    pub fn new(xml: &[u8], resources: &dyn Resources, mode: PatternMode) -> Result<Self> {
        let root = tree::parse(xml)?;
        if root.name != "featureGenerators" {
            return Err(Error::Descriptor(
                "[OPENNLP-1174] - Classic configuration format is no longer supported!".into(),
            ));
        }
        let mut kinds = HashMap::new();
        resource_kinds(&root, &mut kinds);
        let mut builder =
            Builder { resources: Some(resources), mode, kinds: &kinds, loaded: HashMap::new(), shared: None };
        // Validate all factory parameters now and retain every parsed resource.
        let _ = build_root(&root, &mut builder)?;
        let loaded = builder.loaded;
        Ok(Self { root, mode, kinds, loaded })
    }

    /// Fresh mutable generators without reading the archive, XML or lexicons.
    pub fn create(&self) -> Result<BoxedGenerator> {
        let mut builder = Builder {
            resources: None,
            mode: self.mode,
            kinds: &self.kinds,
            loaded: HashMap::new(),
            shared: Some(&self.loaded),
        };
        build_root(&self.root, &mut builder)
    }
}
