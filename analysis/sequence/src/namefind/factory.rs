//! Shared parsed resources, with adaptive state owned by each document.
use super::*;
use crate::featuregen::descriptor::Resources;

pub struct NameContextFactory {
    descriptor: descriptor::Prepared,
    mode: PatternMode,
}

impl NameContextFactory {
    pub fn for_model(model: &ModelFile, mode: PatternMode) -> Result<Self> {
        let xml = model.resource(descriptor::ENTRY_NAME)?;
        let xml = xml.as_deref().unwrap_or(DEFAULT_DESCRIPTOR.as_bytes());
        Ok(Self { descriptor: descriptor::Prepared::new(xml, model, mode)?, mode })
    }

    /// Independent sentence caches and document history over shared lexicons.
    pub fn create(&self) -> Result<DefaultNameContext> {
        let additional = Box::new(Window::new(Box::new(AdditionalContext::new()), 8, 8));
        Ok(DefaultNameContext::new(vec![self.descriptor.create()?, additional], self.mode))
    }
}
