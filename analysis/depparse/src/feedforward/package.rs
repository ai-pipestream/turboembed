use super::{Activation, FeatureLayout, FeedforwardDependencyModel, wire};
use crate::{Result, error};
use turbo_analysis::ModelRef;
use turbo_train::package::{Manifest, ModelPackage};

impl FeedforwardDependencyModel {
    /// Retired under #1511; replacement dependency heads are tracked in #1509.
    pub fn from_bytes(_bytes: &[u8]) -> Result<Self> {
        Err(error(turbo_maxent::Error::Disabled))
    }
    /// Version 3 preserves UTF-8, beam width, and explicit label encoding.
    /// `to_java_bytes` writes only the legacy ONLP-FFDP-1 stream, not current Apache.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut package = ModelPackage::new(Manifest::new("FeedforwardDependencyParser", "und", 0));
        package.add_entry("format_version", b"3\n".to_vec());
        let layout = match self.layout {
            FeatureLayout::OpenNlp14 => "OpenNlp14",
            FeatureLayout::ChenManning18 => "ChenManning18",
        };
        let activation = match self.activation {
            Activation::Cube => "cube",
            Activation::Relu => "relu",
        };
        package.add_entry(
            "config.properties",
            format!(
                "layout={layout}\nactivation={activation}\nbeam_size={}\nlabel_encoding={}\n",
                self.default_beam_size,
                self.label_encoding.name()
            )
            .into_bytes(),
        );
        let mut vocab = Vec::new();
        wire::vocabulary_utf8(&mut vocab, &self.vocabulary, &self.transitions)?;
        package.add_entry("vocab.bin", vocab);
        let mut weights = Vec::new();
        wire::weights(&mut weights, &self.weights)?;
        package.add_entry("weights.bin", weights);
        Ok(package.to_bytes())
    }
    pub fn identity(&self) -> Result<ModelRef> {
        Ok(ModelRef {
            id: if self.transitions.contains(&crate::Transition::Swap) {
                "turbo:SwapFeedforwardDependencyParser"
            } else {
                "opennlp:FeedforwardDependencyParser"
            }
            .into(),
            revision: "3".into(),
            sha256: turbo_maxent::sha256(&self.to_bytes()?).to_hex(),
        })
    }
}
