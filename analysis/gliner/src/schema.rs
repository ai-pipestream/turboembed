/// Request limits are enforced before tokenization or candidate allocation.
pub const MAX_LABELS: usize = 32;
pub const MAX_TOKENS: usize = 384;
pub const MAX_LABEL_BYTES: usize = 128;
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
    threshold: f32,
    mode: Decode,
}
impl ExtractionSchema {
    pub fn new(labels: Vec<String>, threshold: f32, mode: Decode) -> Result<Self, String> {
        if labels.is_empty() || labels.len() > MAX_LABELS {
            return Err("max_labels limit: 1..=32".into());
        }
        if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
            return Err("threshold must be finite in [0,1]".into());
        }
        for (i, l) in labels.iter().enumerate() {
            if l.trim().is_empty() || l.len() > MAX_LABEL_BYTES {
                return Err("max_label_bytes limit: nonempty, <=128 bytes".into());
            }
            if labels[..i].contains(l) {
                return Err("duplicate entity label".into());
            }
            if ["<<ENT>>", "<<SEP>>", "[FLERT]", "[CLS]", "[SEP]", "[MASK]", "[PAD]", "[UNK]"]
                .iter()
                .any(|s| l.contains(s))
            {
                return Err("reserved prompt token in label".into());
            }
        }
        Ok(Self { labels, threshold, mode })
    }
    pub fn labels(&self) -> &[String] {
        &self.labels
    }
    pub fn threshold(&self) -> f32 {
        self.threshold
    }
    pub fn mode(&self) -> Decode {
        self.mode
    }
}
