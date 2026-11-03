use super::*;

/// The longest sequence the rotary families take: the core builds a
/// cos/sin table of max_positions rows per rotary base.
const MAX_SEQUENCE: u32 = 32768;

impl Manifest {
    /// The rotary families' fields, FAMILY_MODERNBERT and FAMILY_NOMIC_BERT
    /// as their checkpoints are: anything else a manifest could write for
    /// them is refused, so it never claims more than the encoders were
    /// sealed against. Every other family is held to having none of them.
    pub(super) fn validate_rotary(&self, a: &Architecture) -> Result<()> {
        let (name, attention, mlp, activation, norm) = match a.family {
            Family::Modernbert => ("FAMILY_MODERNBERT", true, Mlp::Gated, Activation::GeluErf, Norm::LayerNoBias),
            Family::NomicBert => ("FAMILY_NOMIC_BERT", false, Mlp::Gated, Activation::Silu, Norm::Layer),
            Family::Bert | Family::Roberta | Family::DebertaV2 => return self.refuse_rotary(a),
        };
        let field = |f: &str, want: &str| Err(invalid(format!("manifest.json: architecture.{f}: {name} takes {want}")));
        if a.position_embedding != PositionEmbedding::Rotary {
            return field("position_embedding", "POSITION_ROTARY");
        }
        if a.relative.is_some() || a.lookahead.is_some() || a.position_offset != 0 {
            return field("position_embedding", "no relative, lookahead or position_offset");
        }
        if a.mlp != mlp || a.activation != activation {
            return field(
                "mlp",
                if activation == Activation::Silu {
                    "MLP_GATED with ACTIVATION_SILU"
                } else {
                    "MLP_GATED with ACTIVATION_GELU_ERF"
                },
            );
        }
        if a.norm == Norm::Rms {
            return Err(invalid("manifest.json: architecture.norm: no backend runs NORM_RMS yet"));
        }
        if a.norm != norm {
            return field("norm", if norm == Norm::Layer { "NORM_LAYER" } else { "NORM_LAYER_NO_BIAS" });
        }
        if a.attention.is_some() != attention {
            return field("attention", if attention { "attention { window, global_every }" } else { "no attention" });
        }
        let Some(rope) = a.rope else {
            return field("rope", "rope { global_theta }");
        };
        if rope.local_theta.is_some() != attention {
            return field("rope.local_theta", if attention { "a local_theta with attention" } else { "none" });
        }
        for theta in [Some(rope.global_theta), rope.local_theta].into_iter().flatten() {
            if !(theta.is_finite() && theta > 1.0) {
                return Err(invalid(format!("manifest.json: architecture.rope: {theta} is not a base above 1")));
            }
        }
        if a.max_positions > MAX_SEQUENCE {
            return Err(invalid(format!(
                "manifest.json: architecture.max_positions: {} is over {MAX_SEQUENCE}, the longest {name} takes",
                a.max_positions
            )));
        }
        if !(a.hidden / a.heads.max(1)).is_multiple_of(2) {
            return Err(invalid("manifest.json: architecture.heads: a rotary head's width is even"));
        }
        if let Some(w) = a.attention
            && (w.window < 2 || !w.window.is_multiple_of(2) || w.window / 2 >= a.max_positions || w.global_every == 0)
        {
            return Err(invalid(format!(
                "manifest.json: architecture.attention: window {} is not even, from 2, with half under max_positions {}, or global_every {} is 0",
                w.window, a.max_positions, w.global_every
            )));
        }
        match a.family {
            Family::Modernbert if a.token_types != 0 => return field("token_types", "0, no token type table"),
            Family::NomicBert => positive("architecture.token_types", a.token_types)?,
            _ => {}
        }
        if self.classifier.is_some() || self.projection.is_some() || self.qa.is_some() {
            return Err(invalid(format!("manifest.json: {name} carries no classifier, projection or QA head yet")));
        }
        if self.embed.as_ref().is_some_and(|e| e.sparse.is_some()) {
            return Err(invalid(format!("manifest.json: embed.sparse: {name} has no MLM head yet")));
        }
        // Only the backends' raw-weights path holds the families, and a
        // tensor the encoder never reads is refused, not ignored.
        for art in &self.artifacts {
            if art.format != Format::Safetensors {
                return Err(invalid(format!(
                    "manifest.json: artifacts {:?}: {name} is loaded from FORMAT_SAFETENSORS only",
                    art.name
                )));
            }
            for &role in art.tensor_names.keys() {
                if !takes(a.family, role) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts {:?}.tensor_names: {name} has no {}",
                        art.name,
                        role_name(role)
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Whether a rotary family's encoder reads a tensor of `role`.
fn takes(family: Family, role: TensorRole) -> bool {
    use TensorRole::*;
    match family {
        Family::Modernbert => matches!(
            role,
            WordEmbeddings
                | EmbeddingsLnWeight
                | FinalNormWeight
                | AttnLnWeight
                | QkvWeight
                | AttnOutWeight
                | FfnLnWeight
                | FfnInWeight
                | FfnOutWeight
        ),
        Family::NomicBert => matches!(
            role,
            WordEmbeddings
                | TokenTypeEmbeddings
                | EmbeddingsLnWeight
                | EmbeddingsLnBias
                | QkvWeight
                | AttnOutWeight
                | AttnLnWeight
                | AttnLnBias
                | FfnGateWeight
                | FfnUpWeight
                | FfnOutWeight
                | FfnLnWeight
                | FfnLnBias
        ),
        Family::Bert | Family::Roberta | Family::DebertaV2 => {
            !matches!(role, QkvWeight | FinalNormWeight | FfnGateWeight | FfnUpWeight)
        }
    }
}

impl Manifest {
    /// BERT, RoBERTa and DeBERTa-v2 run none of the rotary families' parts.
    fn refuse_rotary(&self, a: &Architecture) -> Result<()> {
        if a.rope.is_some()
            || a.attention.is_some()
            || a.mlp != Mlp::Dense
            || a.norm != Norm::Layer
            || a.activation != Activation::GeluErf
        {
            return Err(invalid(
                "manifest.json: architecture: rope, attention, MLP_GATED, a norm other than NORM_LAYER and ACTIVATION_SILU are FAMILY_MODERNBERT's and FAMILY_NOMIC_BERT's",
            ));
        }
        for art in &self.artifacts {
            if let Some(&role) = art.tensor_names.keys().find(|&&r| !takes(a.family, r)) {
                return Err(invalid(format!(
                    "manifest.json: artifacts {:?}.tensor_names: {} is FAMILY_MODERNBERT's and FAMILY_NOMIC_BERT's",
                    art.name,
                    role_name(role)
                )));
            }
        }
        Ok(())
    }
}
