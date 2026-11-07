//! The processing configuration identity of a document (issue #1908): what
//! was asked of it, apart from what it was asked of. A per-layer
//! [`ProvenanceDigest`](super::ProvenanceDigest) chains the source and view
//! text digests, so it changes with the input; this digest does not, so a
//! caller can tell one operation key reused under a different configuration.

use turbo_sha256::Sha256;

use super::{Document, ProcessingConfigDigest};

/// The domain tag the digest starts with.
pub const PROCESSING_CONFIG_DIGEST_RULE: &str = "turbo.processing-config.v1";

impl Document {
    /// SHA-256 over the domain tag [`PROCESSING_CONFIG_DIGEST_RULE`], then
    /// the entry count and, for every view and layer entry sorted by (layer
    /// id, instance): the layer id, the instance, the annotator and its
    /// version (each a u64 little-endian length then the UTF-8 bytes), the
    /// config hash as u64 little-endian, the model identity (byte 0 when
    /// absent, else byte 1 then its id, revision and sha256) and the read
    /// keys sorted by (layer id, instance), their count then each layer id
    /// and instance. No text, view, source or content digest takes part:
    /// the same configuration over different text has the same digest, and
    /// the order layers were added in does not matter.
    pub fn processing_config_digest(&self) -> ProcessingConfigDigest {
        let mut entries: Vec<(String, &str, &crate::Provenance)> =
            self.entries.iter().map(|e| (e.key.layer.to_string(), e.key.name, &*e.provenance)).collect();
        entries.sort_unstable_by(|a, b| (a.0.as_str(), a.1).cmp(&(b.0.as_str(), b.1)));
        let mut h = Sha256::new();
        h.str(PROCESSING_CONFIG_DIGEST_RULE);
        h.usize(entries.len());
        for (layer, instance, p) in &entries {
            h.str(layer).str(instance).str(p.annotator).str(p.version).u64(p.config);
            match &p.model {
                None => h.u8(0),
                Some(m) => h.u8(1).str(&m.id).str(&m.revision).str(&m.sha256),
            };
            let mut reads: Vec<(String, &str)> = p.reads.iter().map(|k| (k.layer.to_string(), k.name)).collect();
            reads.sort_unstable();
            h.usize(reads.len());
            for (layer, instance) in &reads {
                h.str(layer).str(instance);
            }
        }
        ProcessingConfigDigest(h.finish())
    }
}
