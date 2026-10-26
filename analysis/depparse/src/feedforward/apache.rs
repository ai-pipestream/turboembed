//! Import current Apache BaseModel/FFDP artifacts into the native representation.
use super::{Activation, FeatureLayout, FeedforwardDependencyModel, wire::Reader};
use crate::{Result, error};
use turbo_maxent::{ModelFile, zip::Zip};

pub(super) fn beam_size(value: &str) -> Result<usize> {
    let beam = value.parse::<i32>().map_err(error)?;
    if beam <= 0 {
        return Err(error("beam size must be positive"));
    }
    Ok(beam as usize)
}

impl FeedforwardDependencyModel {
    /// Imports Apache's FFDP version 1 stream or its BaseModel ZIP container.
    /// No JVM is needed. Legacy ONLP-FFDP-1 remains available through `from_bytes`.
    pub fn from_apache_bytes(bytes: &[u8]) -> Result<Self> {
        if !bytes.starts_with(b"PK") {
            return Self::from_apache_network(bytes);
        }
        let zip = Zip::new(bytes).map_err(error)?;
        let mut names = std::collections::HashSet::new();
        for name in zip.names() {
            if !names.insert(name) {
                return Err(error("duplicate Apache ZIP entry"));
            }
            let limit = match name {
                b"manifest.properties" => 4096,
                b"feedforward.network" => 464_000_128,
                _ => return Err(error("unknown Apache dependency ZIP entry")),
            };
            let name = std::str::from_utf8(name).map_err(error)?;
            if zip.size(name).is_none_or(|size| size > limit) {
                return Err(error("Apache dependency ZIP entry exceeds size limit"));
            }
        }
        let file = ModelFile::open(bytes.to_vec()).map_err(error)?;
        if file.component_name() != "FeedforwardDependencyParser" {
            return Err(error("wrong Apache dependency model component"));
        }
        let beam = file.manifest().get("BeamSize").map(beam_size).transpose()?.unwrap_or(1);
        let mut model = Self::from_apache_network(&file.entry("feedforward.network").map_err(error)?)?;
        model.default_beam_size = beam;
        Ok(model)
    }

    fn from_apache_network(bytes: &[u8]) -> Result<Self> {
        let mut reader = Reader { data: bytes, values: 0, utf8: true };
        if reader.take(4)? != b"FFDP" || reader.int()? != 1 {
            return Err(error("unsupported Apache FFDP format/version"));
        }
        let vocabulary = reader.vocabulary()?;
        let mut offset = 0;
        for (map, root) in [(&vocabulary.words, true), (&vocabulary.pos, true), (&vocabulary.labels, false)] {
            for (index, symbol) in ["*UNK*", "*NULL*", "*ROOT*"].iter().take(if root { 3 } else { 2 }).enumerate() {
                if map.get(*symbol) != Some(&(offset + index)) {
                    return Err(error("Apache reserved vocabulary row mismatch"));
                }
            }
            if map.values().any(|&row| row < offset || row >= offset + map.len()) {
                return Err(error("Apache vocabulary blocks overlap"));
            }
            offset += map.len();
        }
        let transitions = reader.transitions()?;
        let weights = reader.weights(vocabulary.len(), transitions.len(), FeatureLayout::OpenNlp14)?;
        reader.end()?;
        Self::new(vocabulary, transitions, FeatureLayout::OpenNlp14, Activation::Cube, weights)
    }
}
