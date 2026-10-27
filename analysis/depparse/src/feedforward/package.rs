use super::{Activation, FeatureLayout, FeedforwardDependencyModel, wire};
use crate::{Result, error};
use turbo_analysis::ModelRef;
use turbo_maxent::zip::Zip;
use turbo_train::package::{Manifest, ModelPackage};

impl FeedforwardDependencyModel {
    /// Reads legacy Java streams, current Apache models, or Turbo ZIP versions 1, 2 and 3.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.starts_with(b"FFDP") {
            return Self::from_apache_bytes(bytes);
        }
        if !bytes.starts_with(b"PK") {
            return Self::from_java_bytes(bytes);
        }
        let zip = Zip::new(bytes).map_err(error)?;
        if zip.contains("feedforward.network") {
            return Self::from_apache_bytes(bytes);
        }
        let mut names = std::collections::HashSet::new();
        for name in zip.names() {
            if !names.insert(name) {
                return Err(error("duplicate ZIP entry"));
            }
            let limit = match name {
                b"format_version" => 16,
                b"config.properties" | b"manifest.properties" => 4096,
                b"vocab.bin" => 64_000_000,
                b"weights.bin" => 400_000_128,
                _ => return Err(error("unknown neural ZIP entry")),
            };
            let name = std::str::from_utf8(name).map_err(error)?;
            if zip.size(name).is_none_or(|n| n > limit) {
                return Err(error("neural ZIP entry exceeds size limit"));
            }
        }
        let version = match zip.entry("format_version").map_err(error)?.as_ref() {
            b"1\n" => 1,
            b"2\n" => 2,
            b"3\n" => 3,
            _ => return Err(error("unsupported neural ZIP version")),
        };
        let config = zip.entry("config.properties").map_err(error)?;
        let config = std::str::from_utf8(&config).map_err(error)?;
        let mut layout = None;
        let mut activation = None;
        let mut beam = None;
        let mut encoding = None;
        for line in config.lines() {
            match line {
                "layout=OpenNlp14" if layout.is_none() => layout = Some(FeatureLayout::OpenNlp14),
                "layout=ChenManning18" if layout.is_none() => layout = Some(FeatureLayout::ChenManning18),
                "activation=cube" if activation.is_none() => activation = Some(Activation::Cube),
                "activation=relu" if activation.is_none() => activation = Some(Activation::Relu),
                s if version >= 2 && s.starts_with("beam_size=") && beam.is_none() => {
                    beam = Some(super::apache::beam_size(&s[10..])?);
                }
                s if version == 3 && s.starts_with("label_encoding=") && encoding.is_none() => {
                    encoding = Some(crate::LabelEncoding::read(&s[15..])?);
                }
                _ => return Err(error("invalid or duplicate neural configuration")),
            }
        }
        let layout = layout.ok_or_else(|| error("missing layout"))?;
        let activation = activation.ok_or_else(|| error("missing activation"))?;
        let vocab_bytes = zip.entry("vocab.bin").map_err(error)?;
        let mut r = wire::Reader { data: &vocab_bytes, values: 0, utf8: version >= 2 };
        let vocabulary = r.vocabulary()?;
        let transitions = r.transitions()?;
        r.end()?;
        let weight_bytes = zip.entry("weights.bin").map_err(error)?;
        let mut r = wire::Reader { data: &weight_bytes, values: 0, utf8: version >= 2 };
        let weights = r.weights(vocabulary.len(), transitions.len(), layout)?;
        r.end()?;
        let mut model = Self::new(vocabulary, transitions, layout, activation, weights)?;
        model.default_beam_size = if version == 1 { 1 } else { beam.ok_or_else(|| error("missing beam size"))? };
        model.label_encoding = if version == 3 {
            encoding.ok_or_else(|| error("missing label encoding"))?
        } else {
            crate::LabelEncoding::historical(bytes)
        };
        Ok(model)
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
            id: "opennlp:FeedforwardDependencyParser".into(),
            revision: "3".into(),
            sha256: turbo_maxent::sha256(&self.to_bytes()?).to_hex(),
        })
    }
}
