use super::*;

/// The largest relative geometry the DeBERTa-v2 encoders take, as
/// turbo::deberta::Config::check bounds it.
const MAX_POSITION_BUCKETS: u32 = 4096;
const MAX_RELATIVE_POSITIONS: u32 = 65536;
/// The longest sequence the DeBERTa-v2 encoders take.
const MAX_SEQUENCE: u32 = 4096;

impl Manifest {
    /// How the architecture places tokens: absolute positions and token
    /// types for BERT and RoBERTa, relative buckets and neither for
    /// DeBERTa-v2.
    pub(super) fn validate_positions(&self, a: &Architecture) -> Result<()> {
        if a.family != Family::DebertaV2 {
            if a.position_embedding != PositionEmbedding::Absolute || a.relative.is_some() {
                return Err(invalid(
                    "manifest.json: architecture.position_embedding: FAMILY_BERT and FAMILY_ROBERTA take POSITION_ABSOLUTE and no relative",
                ));
            }
            return positive("architecture.token_types", a.token_types);
        }
        // Only the backends' raw-weights path holds the family: no compiled
        // graph format takes relative attention.
        if let Some(art) = self.artifacts.iter().find(|art| art.format != Format::Safetensors) {
            return Err(invalid(format!(
                "manifest.json: artifacts {:?}: FAMILY_DEBERTA_V2 is loaded from FORMAT_SAFETENSORS only",
                art.name
            )));
        }
        // A table the encoder never reads is a manifest that says
        // something the model does not do: refused, not ignored.
        for art in &self.artifacts {
            for role in [TensorRole::PositionEmbeddings, TensorRole::TokenTypeEmbeddings] {
                if art.tensor_names.contains_key(&role) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts {:?}.tensor_names: FAMILY_DEBERTA_V2 has no {}",
                        art.name,
                        role_name(role)
                    )));
                }
            }
        }
        if a.max_positions == 0 || a.max_positions > MAX_SEQUENCE {
            return Err(invalid(format!(
                "manifest.json: architecture.max_positions: {} is not a sequence of 1 to {MAX_SEQUENCE} tokens, the longest FAMILY_DEBERTA_V2 takes",
                a.max_positions
            )));
        }
        let Some(r) = a.relative.filter(|_| a.position_embedding == PositionEmbedding::Relative) else {
            return Err(invalid(
                "manifest.json: architecture.relative: FAMILY_DEBERTA_V2 takes POSITION_RELATIVE and architecture.relative",
            ));
        };
        if a.token_types != 0 || a.position_offset != 0 {
            return Err(invalid(
                "manifest.json: architecture.token_types: FAMILY_DEBERTA_V2 has no token types or position table (0)",
            ));
        }
        let b = r.position_buckets;
        if !(4..=MAX_POSITION_BUCKETS).contains(&b) || b % 2 != 0 {
            return Err(invalid(format!(
                "manifest.json: architecture.relative.position_buckets: {b} is not an even count from 4 to {MAX_POSITION_BUCKETS}"
            )));
        }
        let m = r.max_relative_positions;
        if m <= b / 2 + 1 || m > MAX_RELATIVE_POSITIONS {
            return Err(invalid(format!(
                "manifest.json: architecture.relative.max_relative_positions: {m} is not past {} and at most {MAX_RELATIVE_POSITIONS}",
                b / 2 + 1
            )));
        }
        Ok(())
    }
}
