use super::*;
impl Manifest {
    pub(super) fn validate_artifacts(&self, listed_path: &dyn Fn(&str, &str) -> Result<()>) -> Result<()> {
        if self.artifacts.is_empty() {
            return Err(invalid("manifest.json: artifacts: empty"));
        }
        let mut names = HashSet::new();
        for (i, a) in self.artifacts.iter().enumerate() {
            let at = |f: &str| format!("artifacts[{i}].{f}");
            required(&at("name"), &a.name)?;
            if !names.insert(a.name.as_str()) {
                return Err(invalid(format!("manifest.json: {}: {:?} is used twice", at("name"), a.name)));
            }
            if a.files.is_empty() {
                return Err(invalid(format!("manifest.json: {}: empty", at("files"))));
            }
            for (j, p) in a.files.iter().enumerate() {
                listed_path(&at(&format!("files[{j}]")), p)?;
            }
            for (j, b) in a.backends.iter().enumerate() {
                required(&at(&format!("backends[{j}]")), b)?;
                fits(&at(&format!("backends[{j}]")), b, 32)?;
            }
            fits(&at("target"), &a.target, 32)?;
            if a.format == Format::Hef {
                if a.target.is_empty() {
                    return Err(invalid(format!("manifest.json: {}: required for a compiled HEF", at("target"))));
                }
                // The core hands a backend a HEF as one block of bytes, over
                // the architecture raw weights would describe.
                if a.files.len() != 1 {
                    return Err(invalid(format!("manifest.json: {}: a HEF is one file", at("files"))));
                }
                if a.compute_dtype.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a compiled HEF",
                        at("compute_dtype")
                    )));
                }
                if self.architecture.is_none() {
                    return Err(invalid(format!("manifest.json: architecture: required by the HEF in {}", at("name"))));
                }
                // A HEF's shape is compiled in; its backend has no dynamic one.
                for (field, v) in [("fixed_seq", a.fixed_seq), ("fixed_batch", a.fixed_batch)] {
                    if v == 0 {
                        return Err(invalid(format!("manifest.json: {}: required for a compiled HEF", at(field))));
                    }
                }
            }
            if a.format == Format::OpenvinoIr {
                // The core hands a backend the IR as two blocks of bytes:
                // the xml, then its weights.
                if a.files.len() != 2 {
                    return Err(invalid(format!(
                        "manifest.json: {}: an OpenVINO IR is two files, the xml then its weights",
                        at("files")
                    )));
                }
                if a.compute_dtype.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a compiled OpenVINO IR",
                        at("compute_dtype")
                    )));
                }
                if self.architecture.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: architecture: required by the OpenVINO IR in {}",
                        at("name")
                    )));
                }
                // The NPU compiles a static shape; the manifest names it,
                // the same way a HEF's compiled frame is named.
                for (field, v) in [("fixed_seq", a.fixed_seq), ("fixed_batch", a.fixed_batch)] {
                    if v == 0 {
                        return Err(invalid(format!(
                            "manifest.json: {}: required for a compiled OpenVINO IR",
                            at(field)
                        )));
                    }
                }
            }
            if a.graph_input == GraphInput::Embeddings && a.host_weights.is_empty() {
                return Err(invalid(format!(
                    "manifest.json: {}: required when graph_input is INPUT_EMBEDDINGS",
                    at("host_weights")
                )));
            }
            if self.static_embedding.is_some() && a.format != Format::Safetensors {
                return Err(invalid(format!(
                    "manifest.json: {}: a static model's artifacts are raw weights, FORMAT_SAFETENSORS",
                    at("format")
                )));
            }
            if a.format == Format::Safetensors {
                if a.tensor_names.is_empty() {
                    return Err(invalid(format!("manifest.json: {}: required for raw weights", at("tensor_names"))));
                }
                if self.architecture.is_none() && self.static_embedding.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: architecture: required by raw weights in {}",
                        at("name")
                    )));
                }
                // Raw weights compute in what the session's precision says,
                // and start where the weights do: at token ids.
                if a.compute_dtype.is_some() {
                    return Err(invalid(format!(
                        "manifest.json: {}: fixed by a compilation; raw weights have none",
                        at("compute_dtype")
                    )));
                }
                if a.graph_input != GraphInput::TokenIds {
                    return Err(invalid(format!(
                        "manifest.json: {}: raw weights start at INPUT_TOKEN_IDS",
                        at("graph_input")
                    )));
                }
                if a.graph_output != GraphOutput::HiddenStates {
                    return Err(invalid(format!(
                        "manifest.json: {}: raw weights stop at OUTPUT_HIDDEN_STATES; the backend pools",
                        at("graph_output")
                    )));
                }
            }
            for (role, name) in &a.tensor_names {
                let field = at(&format!("tensor_names.{}", role_name(*role)));
                required(&field, name)?;
                if role.is_static() != self.static_embedding.is_some() {
                    let whose = if role.is_static() { "a static model's" } else { "an encoder's" };
                    return Err(invalid(format!("manifest.json: {field}: {whose} role, and the bundle is not one")));
                }
                if role.is_classifier() && self.classifier.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {field}: a classifier role, and the bundle has no classifier"
                    )));
                }
                if role.is_projection() && self.projection.is_none() {
                    return Err(invalid(format!(
                        "manifest.json: {field}: a projection role, and the bundle has no projection"
                    )));
                }
                if role.per_layer() != name.contains("{layer}") {
                    let want = if role.per_layer() { "must" } else { "must not" };
                    return Err(invalid(format!("manifest.json: {field}: {want} contain {{layer}}")));
                }
            }
            if let Some(p) = &a.produced_by {
                self.check_produced_by(&at("produced_by"), p, &listed_path)?;
                // Raw weights the tool wrote from the upstream checkpoint
                // come from the upstream model, as the reference does.
                if p.from.is_empty() && a.format != Format::Safetensors {
                    return Err(invalid(format!(
                        "manifest.json: {}: required for a converted artifact",
                        at("produced_by.from")
                    )));
                }
            }
        }
        for (i, a) in self.artifacts.iter().enumerate() {
            if !a.host_weights.is_empty() {
                let Some(h) = self.artifacts.iter().find(|h| h.name == a.host_weights) else {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].host_weights: no artifact named {:?}",
                        a.host_weights
                    )));
                };
                // The lookup reads embedding tensors, which raw weights name.
                if h.format != Format::Safetensors {
                    return Err(invalid(format!(
                        "manifest.json: artifacts[{i}].host_weights: {:?} is not FORMAT_SAFETENSORS",
                        a.host_weights
                    )));
                }
            }
            if let Some(p) = &a.produced_by
                && !p.from.is_empty()
                && !names.contains(p.from.as_str())
            {
                return Err(invalid(format!(
                    "manifest.json: artifacts[{i}].produced_by.from: no artifact named {:?}",
                    p.from
                )));
            }
        }

        Ok(())
    }
}
