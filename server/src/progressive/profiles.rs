//! Named, versioned analysis profiles (`default-v1`, `retrieval-v1`,
//! `minimal-v1`, issue `#1302`) for `AnalyzeDocumentProgressive` (`#1218`).
//!
//! Every profile defines its immutable step list, the exact set of standard
//! layer IDs it produces, and the bundle capabilities it requires. A profile's
//! output set never changes within a version.

use crate::proto::result::ProfileCapabilities;

/// Static definition of a named, versioned analysis profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnalysisProfile {
    /// Canonical profile name (`"default-v1"`, `"retrieval-v1"`, `"minimal-v1"`).
    pub name: &'static str,
    /// Immutable profile version (`"1"`).
    pub version: &'static str,
    /// Ordered step IDs executed by this profile.
    pub steps: &'static [&'static str],
    /// Standard layer IDs produced by this profile.
    pub layers: &'static [&'static str],
    /// Bundle capability identifiers required to run this profile.
    pub required_bundles: &'static [&'static str],
}

/// `default-v1`: structure, sentences, tokens, entities, subwords, structure
/// boundaries, chunks, levels, and chunk vectors.
pub const DEFAULT_V1: AnalysisProfile = AnalysisProfile {
    name: "default-v1",
    version: "1",
    steps: &[
        "turbo-structure:blocks",
        "turbo-sentdetect:rules",
        "turbo-tokenize:uax29-untyped",
        "turbo:structure-boundaries",
        "turbo-geo:names",
        "turbo:subwords",
        "turbo:chunker",
        "turbo:level-builder",
        "turbo:chunk-vectors",
    ],
    layers: &[
        "turbo:structure",
        "opennlp:sentences",
        "opennlp:tokens",
        "turbo:boundaries",
        "opennlp:entities",
        "opennlp:subwords",
        "turbo:chunks",
        "turbo:levels",
        "opennlp:embeddings<turbo:chunks>",
    ],
    required_bundles: &["TASK_EMBED"],
};

/// `retrieval-v1`: `default-v1` plus learned sparse terms (`turbo:sparse`).
pub const RETRIEVAL_V1: AnalysisProfile = AnalysisProfile {
    name: "retrieval-v1",
    version: "1",
    steps: &[
        "turbo-structure:blocks",
        "turbo-sentdetect:rules",
        "turbo-tokenize:uax29-untyped",
        "turbo:structure-boundaries",
        "turbo-geo:names",
        "turbo:subwords",
        "turbo:chunker",
        "turbo:sparse",
        "turbo:level-builder",
        "turbo:chunk-vectors",
    ],
    layers: &[
        "turbo:structure",
        "opennlp:sentences",
        "opennlp:tokens",
        "turbo:boundaries",
        "opennlp:entities",
        "opennlp:subwords",
        "turbo:chunks",
        "turbo:sparse-terms<turbo:view>",
        "turbo:levels",
        "opennlp:embeddings<turbo:chunks>",
    ],
    required_bundles: &["TASK_EMBED", "embed.sparse"],
};

/// `minimal-v1`: structure, sentences, and tokens (requires no model bundles).
pub const MINIMAL_V1: AnalysisProfile = AnalysisProfile {
    name: "minimal-v1",
    version: "1",
    steps: &["turbo-structure:blocks", "turbo-sentdetect:rules", "turbo-tokenize:uax29-untyped"],
    layers: &["turbo:structure", "opennlp:sentences", "opennlp:tokens"],
    required_bundles: &[],
};

/// All versioned profiles served by `AnalysisService`.
pub const PROFILES: &[AnalysisProfile] = &[DEFAULT_V1, RETRIEVAL_V1, MINIMAL_V1];

/// Looks up a named, versioned profile by its canonical identifier.
pub fn find_profile(name: &str) -> Option<&'static AnalysisProfile> {
    PROFILES.iter().find(|p| p.name == name)
}

/// Converts [`PROFILES`] into wire [`ProfileCapabilities`] entries for
/// `AnalysisService::Capabilities`.
pub fn profile_capabilities() -> Vec<ProfileCapabilities> {
    PROFILES
        .iter()
        .map(|p| ProfileCapabilities {
            name: p.name.to_owned(),
            version: p.version.to_owned(),
            steps: p.steps.iter().map(|&s| s.to_owned()).collect(),
            layers: p.layers.iter().map(|&l| l.to_owned()).collect(),
            required_bundles: p.required_bundles.iter().map(|&b| b.to_owned()).collect(),
        })
        .collect()
}
