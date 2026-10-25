use super::*;
impl Manifest {
    pub(super) fn validate_architecture(
        &self,
        e: &Embed,
        listed_path: &dyn Fn(&str, &str) -> Result<()>,
    ) -> Result<()> {
        let t = &self.tokenizer;
        if let Some(a) = &self.architecture {
            for (field, v) in [
                ("layers", a.layers),
                ("hidden", a.hidden),
                ("heads", a.heads),
                ("intermediate", a.intermediate),
                ("max_positions", a.max_positions),
                ("token_types", a.token_types),
                ("vocab_size", a.vocab_size),
            ] {
                positive(&format!("architecture.{field}"), v)?;
            }
            if a.hidden % a.heads != 0 {
                return Err(invalid(format!(
                    "manifest.json: architecture.heads: {} does not divide hidden {}",
                    a.heads, a.hidden
                )));
            }
            if !(a.layer_norm_eps > 0.0 && a.layer_norm_eps.is_finite()) {
                return Err(invalid("manifest.json: architecture.layer_norm_eps: not a positive number"));
            }
            // The vectors are pooled hidden states, and no role projects them.
            if e.dim != a.hidden {
                return Err(invalid(format!(
                    "manifest.json: embed.dim: {} is not architecture.hidden {}, the width of the pooled hidden states",
                    e.dim, a.hidden
                )));
            }
            match a.family {
                Family::Bert if a.position_offset != 0 => {
                    return Err(invalid(
                        "manifest.json: architecture.position_offset: FAMILY_BERT counts positions from 0",
                    ));
                }
                Family::Roberta if a.position_offset == 0 => {
                    return Err(invalid(
                        "manifest.json: architecture.position_offset: required for FAMILY_ROBERTA (its padding id plus one)",
                    ));
                }
                _ => {}
            }
            if let Some(n) = a.lookahead {
                if !matches!(a.family, Family::Bert | Family::Roberta) {
                    return Err(invalid(
                        "manifest.json: architecture.lookahead: only FAMILY_BERT and FAMILY_ROBERTA attention takes a band",
                    ));
                }
                // 0 would be causal attention, which no checkpoint here
                // uses; the backend description keeps 0 for no band.
                if n == 0 || n == u32::MAX {
                    return Err(invalid(format!(
                        "manifest.json: architecture.lookahead: {n} is not a band of 1 key or more"
                    )));
                }
            }
            if e.max_seq as u64 + a.position_offset as u64 > a.max_positions as u64 {
                return Err(invalid(format!(
                    "manifest.json: embed.max_seq: {} from position {} is over architecture.max_positions {}",
                    e.max_seq, a.position_offset, a.max_positions
                )));
            }
            for s in &t.special_tokens {
                if s.id >= a.vocab_size {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.special_tokens: id {} of {:?} is not under architecture.vocab_size {}",
                        s.id, s.content, a.vocab_size
                    )));
                }
            }
        }

        // static_embedding
        if let Some(st) = &self.static_embedding {
            if self.architecture.is_some() {
                return Err(invalid(
                    "manifest.json: static_embedding: a static model has no architecture; the bundle names one of them",
                ));
            }
            positive("static_embedding.vocab_size", st.vocab_size)?;
            positive("static_embedding.max_length", st.max_length)?;
            if st.max_length > e.max_seq {
                return Err(invalid(format!(
                    "manifest.json: static_embedding.max_length: {} is over embed.max_seq {}, the longest row a session takes",
                    st.max_length, e.max_seq
                )));
            }
            // A static model tokenizes the text alone and cuts it on the
            // right, and nothing else.
            if t.template != ["$TEXT"] {
                return Err(invalid(
                    "manifest.json: tokenizer.template: a static model's rows are the text's tokens alone, [\"$TEXT\"]",
                ));
            }
            if t.truncation != Truncation::Right {
                return Err(invalid(
                    "manifest.json: tokenizer.truncation: a static model cuts on the right, TRUNCATE_RIGHT",
                ));
            }
            match (&st.distilled_from, &st.quality) {
                (Some(d), Some(q)) => {
                    required("static_embedding.distilled_from.model_id", &d.model_id)?;
                    required("static_embedding.distilled_from.revision", &d.revision)?;
                    if !is_hex64(&d.manifest_sha256) {
                        return Err(invalid(
                            "manifest.json: static_embedding.distilled_from.manifest_sha256: not 64 lowercase hex digits",
                        ));
                    }
                    listed_path("static_embedding.quality.texts", &q.texts)?;
                    for (field, v, lo) in [
                        ("base_top1", q.base_top1, 0.0),
                        ("static_top1", q.static_top1, 0.0),
                        ("similarity_spearman", q.similarity_spearman, -1.0),
                    ] {
                        if !(lo..=1.0).contains(&v) {
                            return Err(invalid(format!(
                                "manifest.json: static_embedding.quality.{field}: {v} is not in {lo}..=1"
                            )));
                        }
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(invalid(
                        "manifest.json: static_embedding: distilled_from and quality come together, or neither",
                    ));
                }
            }
            for (i, a) in self.artifacts.iter().enumerate() {
                let names = |r: TensorRole| a.tensor_names.contains_key(&r);
                if !names(TensorRole::StaticEmbeddings) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: no static_embeddings, which a static model needs"
                    )));
                }
                if names(TensorRole::StaticMapping) != (st.rows != 0) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: static_mapping and static_embedding.rows come together, or neither"
                    )));
                }
            }
            for s in &t.special_tokens {
                if s.id >= st.vocab_size {
                    return Err(invalid(format!(
                        "manifest.json: tokenizer.special_tokens: id {} of {:?} is not under static_embedding.vocab_size {}",
                        s.id, s.content, st.vocab_size
                    )));
                }
            }
        }

        // classifier
        if let Some(c) = &self.classifier {
            if self.architecture.is_none() {
                return Err(invalid("manifest.json: classifier: requires an encoder architecture"));
            }
            if c.labels.is_empty() {
                return Err(invalid("manifest.json: classifier.labels: empty"));
            }
            for (i, l) in c.labels.iter().enumerate() {
                required(&format!("classifier.labels[{i}]"), l)?;
            }
            for (i, a) in self.artifacts.iter().enumerate().filter(|(_, a)| a.format == Format::Safetensors) {
                for role in [TensorRole::ClassifierWeight, TensorRole::ClassifierBias] {
                    if !a.tensor_names.contains_key(&role) {
                        return Err(invalid(format!(
                            "manifest.json: artifacts[{i}].tensor_names: no {}, which a classifier needs",
                            role_name(role)
                        )));
                    }
                }
                let pooler_w = a.tensor_names.contains_key(&TensorRole::PoolerWeight);
                let pooler_b = a.tensor_names.contains_key(&TensorRole::PoolerBias);
                if pooler_w != pooler_b {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: pooler_weight and pooler_bias come together, or neither"
                    )));
                }
                if c.kind == ClassifierKind::Token && pooler_w {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: CLASSIFIER_TOKEN has no pooler"
                    )));
                }
            }
        }

        // embed.sparse: the learned sparse output (SPLADE, arXiv 2403.06789)
        // needs the encoder's hidden states, so an architecture, and every
        // raw-weights artifact names the MLM head, as a classifier's is
        // named. The decoder weight is optional: absent, it is tied to the
        // word embeddings, as BERT ties it.
        if e.sparse.is_some() {
            if self.architecture.is_none() {
                return Err(invalid("manifest.json: embed.sparse: requires an encoder architecture"));
            }
            for (i, a) in self.artifacts.iter().enumerate().filter(|(_, a)| a.format == Format::Safetensors) {
                for role in [
                    TensorRole::MlmTransformWeight,
                    TensorRole::MlmTransformBias,
                    TensorRole::MlmTransformLnWeight,
                    TensorRole::MlmTransformLnBias,
                    TensorRole::MlmDecoderBias,
                ] {
                    if !a.tensor_names.contains_key(&role) {
                        return Err(invalid(format!(
                            "manifest.json: artifacts[{i}].tensor_names: no {}, which embed.sparse needs",
                            role_name(role)
                        )));
                    }
                }
            }
        }

        // projection
        if let Some(p) = &self.projection {
            if self.architecture.is_none() {
                return Err(invalid("manifest.json: projection: requires an encoder architecture"));
            }
            positive("projection.dim", p.dim)?;
            for (i, a) in self.artifacts.iter().enumerate().filter(|(_, a)| a.format == Format::Safetensors) {
                if !a.tensor_names.contains_key(&TensorRole::ProjectionWeight) {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].tensor_names: no projection_weight, which a projection needs"
                    )));
                }
            }
        }

        Ok(())
    }
}
