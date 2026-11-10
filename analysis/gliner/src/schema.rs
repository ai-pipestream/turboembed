use turbo_analysis::Fnv1a;

/// Request limits are enforced before tokenization or candidate allocation.
pub const MAX_LABELS: usize = 32;
pub const MAX_TOKENS: usize = 384;
pub const MAX_LABEL_BYTES: usize = 128;
pub const MAX_PROMPT_BYTES: usize = 4096;
pub const MAX_TOKEN_BYTES: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decode {
    Flat,
    Nested,
}
/// The entity portion of an extraction schema. Classification and structured
/// records have different checkpoint heads and are not approximated here.
#[derive(Clone, Debug)]
pub struct ExtractionSchema {
    labels: Vec<String>,
    prompts: Vec<String>,
    threshold: f32,
    mode: Decode,
}
impl ExtractionSchema {
    pub fn new(labels: Vec<String>, threshold: f32, mode: Decode) -> Result<Self, String> {
        let prompts = labels.clone();
        Self::with_prompts(labels, prompts, threshold, mode)
    }

    /// Uses canonical labels for output and scoring, and aliases or guidelines
    /// for the model prompt. The two lists are parallel.
    pub fn with_prompts(
        labels: Vec<String>,
        prompts: Vec<String>,
        threshold: f32,
        mode: Decode,
    ) -> Result<Self, String> {
        if labels.is_empty() || labels.len() > MAX_LABELS {
            return Err("max_labels limit: 1..=32".into());
        }
        if labels.len() != prompts.len() {
            return Err("each entity label needs exactly one prompt".into());
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("threshold must be finite in [0,1]".into());
        }
        for (i, (label, prompt)) in labels.iter().zip(&prompts).enumerate() {
            if label.trim().is_empty() || label.len() > MAX_LABEL_BYTES {
                return Err("max_label_bytes limit: nonempty, <=128 bytes".into());
            }
            if labels[..i].contains(label) {
                return Err("duplicate entity label".into());
            }
            if prompt.trim().is_empty() || prompt.len() > MAX_PROMPT_BYTES {
                return Err("max_prompt_bytes limit: nonempty, <=4096 bytes".into());
            }
            if prompts[..i].contains(prompt) {
                return Err("duplicate entity prompt".into());
            }
            for text in [label, prompt] {
                if ["<<ENT>>", "<<SEP>>", "[FLERT]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
                    .iter()
                    .any(|s| text.contains(s))
                {
                    return Err("reserved prompt token in label or prompt".into());
                }
            }
        }
        Ok(Self { labels, prompts, threshold, mode })
    }

    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    pub fn prompts(&self) -> &[String] {
        &self.prompts
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    pub fn mode(&self) -> Decode {
        self.mode
    }

    /// Stable fingerprint of the output labels, prompt text, threshold, and mode.
    pub fn digest(&self) -> String {
        Self::digest_parts(&self.labels, &self.prompts, self.threshold, self.mode)
    }

    /// Fingerprint a full schema independently of the per-batch model label limit.
    pub fn digest_parts(labels: &[String], prompts: &[String], threshold: f32, mode: Decode) -> String {
        let mut h = Fnv1a::new();
        h.bytes(&(labels.len() as u64).to_le_bytes()).bytes(&(prompts.len() as u64).to_le_bytes());
        for text in labels.iter().chain(prompts) {
            h.bytes(&(text.len() as u64).to_le_bytes()).bytes(text.as_bytes());
        }
        h.bytes(&(threshold + 0.0).to_bits().to_le_bytes()).bytes(&[mode as u8]);
        format!("{:016x}", h.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plain_labels_keep_their_prompts_and_digest() {
        let plain = ExtractionSchema::new(vec!["person".into(), "location".into()], 0.5, Decode::Flat).unwrap();
        assert_eq!(plain.labels(), plain.prompts());
        assert_eq!(plain.digest(), ExtractionSchema::new(plain.labels().to_vec(), 0.5, Decode::Flat).unwrap().digest());
    }

    #[test]
    fn guidelines_change_prompts_and_digest_not_output_labels() {
        let plain = ExtractionSchema::new(vec!["VisualWork".into()], 0.8, Decode::Flat).unwrap();
        let guided = ExtractionSchema::with_prompts(
            vec!["VisualWork".into()],
            vec!["visual artwork, including a painting or photograph".into()],
            0.8,
            Decode::Flat,
        )
        .unwrap();
        assert_eq!(guided.labels(), ["VisualWork"]);
        assert_eq!(guided.prompts(), ["visual artwork, including a painting or photograph"]);
        assert_ne!(plain.digest(), guided.digest());
    }

    #[test]
    fn digest_frames_label_and_prompt_lists() {
        let plain =
            ExtractionSchema::digest_parts(&["a".into(), "b".into()], &["a".into(), "b".into()], 0.5, Decode::Flat);
        let guided = ExtractionSchema::digest_parts(&["a".into()], &["b".into()], 0.5, Decode::Flat);
        assert_ne!(plain, guided);
    }

    #[test]
    fn prompt_validation_rejects_mismatch_duplicates_and_reserved_text() {
        assert!(ExtractionSchema::with_prompts(vec!["person".into()], Vec::new(), 0.5, Decode::Flat).is_err());
        assert!(
            ExtractionSchema::with_prompts(
                vec!["person".into(), "organization".into()],
                vec!["same".into(), "same".into()],
                0.5,
                Decode::Flat,
            )
            .is_err()
        );
        assert!(
            ExtractionSchema::with_prompts(vec!["person".into()], vec!["<<ENT>>".into()], 0.5, Decode::Flat).is_err()
        );
    }
}
