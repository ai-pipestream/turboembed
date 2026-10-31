use super::*;
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProducedBy {
    pub tool: String,
    pub tool_version: String,
    pub container: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub from: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inputs: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub args: Vec<String>,
    pub reproducible: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Reference {
    pub file: String,
    pub cases: Vec<Case>,
    pub produced_by: ProducedBy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Case {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passage: Option<String>,
    pub text: String,
    pub prompt_role: PromptRole,
    /// Context for TASK_EXTRACTIVE_QA; text is the question.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PromptRole {
    #[serde(rename = "PROMPT_NONE")]
    None,
    #[serde(rename = "PROMPT_QUERY")]
    Query,
    #[serde(rename = "PROMPT_DOCUMENT")]
    Document,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileEntry {
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// The training receipt provenance and corpus attribution recorded for a
/// task head in `heads` (#1726).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadProvenance {
    pub head: String,
    pub commit: String,
    pub encoder_manifest_sha256: String,
    pub corpus_id: String,
    pub corpus_name: String,
    pub license: String,
    pub spdx: String,
    pub terms: String,
    pub attribution: String,
    pub corpus_sha256: String,
    pub recorded: String,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub recipe_only: bool,
    pub best_epoch: usize,
    pub metric: String,
    pub dev_score: f64,
    pub test_score: f64,
    pub baseline_score: f64,
}

impl Manifest {
    pub(super) fn validate_head_provenance(&self) -> Result<()> {
        for (i, p) in self.head_provenance.iter().enumerate() {
            let at = |field: &str| format!("head_provenance[{i}].{field}");
            required(&at("head"), &p.head)?;
            if !self.heads.iter().any(|h| h.name == p.head) {
                return Err(invalid(format!("manifest.json: {}: {:?} is not in heads", at("head"), p.head)));
            }
            if self.head_provenance[..i].iter().any(|o| o.head == p.head) {
                return Err(invalid(format!(
                    "manifest.json: {}: {:?} is listed twice in head_provenance",
                    at("head"),
                    p.head
                )));
            }
            required(&at("commit"), &p.commit)?;
            fits(&at("commit"), &p.commit, 64)?;
            if !is_hex64(&p.encoder_manifest_sha256) {
                return Err(invalid(format!(
                    "manifest.json: {}: not 64 lowercase hex digits",
                    at("encoder_manifest_sha256")
                )));
            }
            required(&at("corpus_id"), &p.corpus_id)?;
            fits(&at("corpus_id"), &p.corpus_id, 128)?;
            required(&at("corpus_name"), &p.corpus_name)?;
            fits(&at("corpus_name"), &p.corpus_name, 256)?;
            required(&at("license"), &p.license)?;
            fits(&at("license"), &p.license, 256)?;
            required(&at("spdx"), &p.spdx)?;
            fits(&at("spdx"), &p.spdx, 64)?;
            required(&at("terms"), &p.terms)?;
            fits(&at("terms"), &p.terms, 512)?;
            required(&at("attribution"), &p.attribution)?;
            fits(&at("attribution"), &p.attribution, 2048)?;
            if !is_hex64(&p.corpus_sha256) {
                return Err(invalid(format!("manifest.json: {}: not 64 lowercase hex digits", at("corpus_sha256"))));
            }
            required(&at("recorded"), &p.recorded)?;
            fits(&at("recorded"), &p.recorded, 64)?;
            required(&at("metric"), &p.metric)?;
            fits(&at("metric"), &p.metric, 64)?;
            for (name, val) in
                [("dev_score", p.dev_score), ("test_score", p.test_score), ("baseline_score", p.baseline_score)]
            {
                if !val.is_finite() {
                    return Err(invalid(format!("manifest.json: {}: {val} is not finite", at(name))));
                }
            }
        }
        Ok(())
    }
}
