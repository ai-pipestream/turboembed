use super::*;
impl Manifest {
    pub(super) fn validate(&self) -> Result<()> {
        if self.bundle_version != BUNDLE_VERSION {
            return Err(invalid(format!(
                "manifest.json: bundle_version: {} is not {BUNDLE_VERSION}",
                self.bundle_version
            )));
        }
        if self.task != "TASK_EMBED" {
            if is_task_name(&self.task) {
                return Err(unsupported_task(&self.task));
            }
            return Err(invalid(format!("manifest.json: task: unknown value {:?}", self.task)));
        }

        // files[]: the list every other path must be in.
        let mut listed = HashSet::new();
        for (i, f) in self.files.iter().enumerate() {
            let field = format!("files[{i}].path");
            check_path(&field, &f.path)?;
            if f.path == "manifest.json" {
                return Err(invalid(format!("manifest.json: {field}: the manifest does not list itself")));
            }
            if !listed.insert(f.path.as_str()) {
                return Err(invalid(format!("manifest.json: {field}: {:?} is listed twice", f.path)));
            }
            if !is_hex64(&f.sha256) {
                return Err(invalid(format!("manifest.json: files[{i}].sha256: not 64 lowercase hex digits")));
            }
        }
        let listed_path = |field: &str, p: &str| -> Result<()> {
            check_path(field, p)?;
            if !listed.contains(p) {
                return Err(invalid(format!("manifest.json: {field}: {p:?} is not in files")));
            }
            Ok(())
        };

        // model
        required("model.id", &self.model.id)?;
        fits("model.id", &self.model.id, 128)?;
        required("model.revision", &self.model.revision)?;
        fits("model.revision", &self.model.revision, 64)?;
        required("model.source.repository", &self.model.source.repository)?;
        required("model.source.commit", &self.model.source.commit)?;
        required("model.license", &self.model.license)?;
        if let Some(p) = &self.model.license_file {
            listed_path("model.license_file", p)?;
        }

        // embed
        let Some(e) = &self.embed else {
            return Err(invalid("manifest.json: embed: required for TASK_EMBED"));
        };
        positive("embed.dim", e.dim)?;
        positive("embed.max_seq", e.max_seq)?;
        positive("embed.max_batch", e.max_batch)?;
        fits("embed.prefix_query", &e.prefix_query, 128)?;
        fits("embed.prefix_document", &e.prefix_document, 128)?;
        if e.output_dims.len() > crate::TURBO_OUTPUT_DIMS_MAX {
            return Err(invalid(format!(
                "manifest.json: embed.output_dims: {} widths, and turbo_model_info holds {}",
                e.output_dims.len(),
                crate::TURBO_OUTPUT_DIMS_MAX
            )));
        }
        let mut dims = HashSet::new();
        for (i, &d) in e.output_dims.iter().enumerate() {
            if d == 0 || d > e.dim {
                return Err(invalid(format!("manifest.json: embed.output_dims[{i}]: {d} is not in 1..={}", e.dim)));
            }
            if !dims.insert(d) {
                return Err(invalid(format!("manifest.json: embed.output_dims[{i}]: {d} is listed twice")));
            }
        }

        self.validate_tokenizer(e, &listed_path)?;
        self.validate_architecture(e, &listed_path)?;
        self.validate_artifacts(&listed_path)?;
        listed_path("reference.file", &self.reference.file)?;
        if self.reference.cases.is_empty() {
            return Err(invalid("manifest.json: reference.cases: empty"));
        }
        self.check_produced_by("reference.produced_by", &self.reference.produced_by, &listed_path)?;
        if !self.reference.produced_by.from.is_empty() {
            return Err(invalid(
                "manifest.json: reference.produced_by.from: the reference is made from the upstream model, not an artifact",
            ));
        }
        Ok(())
    }
}
