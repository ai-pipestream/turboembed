use serde::{Deserialize, Serialize};

/// Supported GLiNER markerV0 profile: first-subword pooling, one bidirectional
/// LSTM, ReLU projection heads, and dot-product sigmoid span scores.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub hidden: usize,
    pub max_width: usize,
    pub entity_token: u32,
    pub separator_token: u32,
    pub cls_token: u32,
    pub eos_token: u32,
}
impl Config {
    pub fn check(&self, encoder: &turbo::deberta::Config) -> Result<(), String> {
        encoder.check().map_err(|e| e.message)?;
        if self.hidden == 0 || self.hidden > 1024 || !self.hidden.is_multiple_of(2) {
            return Err("gliner hidden limit: positive even width <= 1024".into());
        }
        if self.max_width == 0 || self.max_width > 32 {
            return Err("max_width limit: 1..=32".into());
        }
        let ids = [self.entity_token, self.separator_token, self.cls_token, self.eos_token];
        if ids.iter().any(|&i| i as usize >= encoder.vocabulary)
            || (0..ids.len()).any(|i| ids[i + 1..].contains(&ids[i]))
        {
            return Err("gliner special token IDs must be distinct and inside vocabulary".into());
        }
        Ok(())
    }
    /// Tensor names and exact shapes in the upstream checkpoint.
    pub fn tensor_shapes(&self, encoder: &turbo::deberta::Config) -> Result<Vec<(String, Vec<u64>)>, String> {
        self.check(encoder)?;
        let h = self.hidden as u64;
        let mut out = Vec::new();
        let mut linear = |stem: &str, input, output| {
            out.push((format!("{stem}.weight"), vec![output, input]));
            out.push((format!("{stem}.bias"), vec![output]));
        };
        linear("token_rep_layer.projection", encoder.hidden as u64, h);
        for stem in [
            "prompt_rep_layer",
            "span_rep_layer.span_rep_layer.project_start",
            "span_rep_layer.span_rep_layer.project_end",
            "span_rep_layer.span_rep_layer.out_project",
        ] {
            linear(&format!("{stem}.0"), if stem.ends_with("out_project") { 2 * h } else { h }, 4 * h);
            linear(&format!("{stem}.3"), 4 * h, h);
        }
        for suffix in ["", "_reverse"] {
            for (name, shape) in [
                ("weight_ih", vec![2 * h, h]),
                ("weight_hh", vec![2 * h, h / 2]),
                ("bias_ih", vec![2 * h]),
                ("bias_hh", vec![2 * h]),
            ] {
                out.push((format!("rnn.lstm.{name}_l0{suffix}"), shape));
            }
        }
        Ok(out)
    }
}
