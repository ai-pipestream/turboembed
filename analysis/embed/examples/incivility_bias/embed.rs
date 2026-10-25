//! A vector per unit from the bundle's encoder, kept in a cache file so a
//! second run reads them back.

use std::fs;
use std::path::Path;

use turbo::TURBO_POOLING_MODEL;
use turbo::bundle::Bundle;
use turbo::manifest::{PromptRole, Truncation};
use turbo::tokenizer::{Encode, Tokenizer};
use turbo_analysis::ModelRef;
use turbo_analysis_embed::{Encoder, EncoderOptions, Rows};

/// The encoder, its tokenizer and the model it names.
pub struct Embedder {
    encoder: Encoder,
    tokenizer: Tokenizer,
    pub model: ModelRef,
    pub dim: usize,
}

impl Embedder {
    pub fn open(dir: &Path, o: &EncoderOptions) -> Result<Embedder, String> {
        let bundle = Bundle::open(dir).map_err(|e| e.to_string())?;
        let tokenizer = Tokenizer::load(&bundle).map_err(|e| e.to_string())?;
        let m = &bundle.manifest.model;
        let model = ModelRef { id: m.id.clone(), revision: m.revision.clone(), sha256: bundle.manifest_sha256.clone() };
        let encoder = Encoder::open(dir, o).map_err(|e| e.to_string())?;
        let dim = encoder.info().dim as usize;
        Ok(Embedder { encoder, tokenizer, model, dim })
    }

    /// The vectors of `texts`, `dim` values each, each text cut to the
    /// model's longest row; from `cache` when it holds them.
    pub fn embed(&mut self, texts: &[&str], cache: &Path) -> Result<Vec<f32>, String> {
        let file = cache.join(format!("{}-{:016x}.f32", self.model.sha256.get(..16).unwrap_or("model"), fnv(texts)));
        if let Ok(bytes) = fs::read(&file)
            && bytes.len() == texts.len() * self.dim * 4
        {
            return Ok(bytes.as_chunks::<4>().0.iter().map(|&b| f32::from_le_bytes(b)).collect());
        }
        let info = self.encoder.info().clone();
        let opts = Encode {
            add_special_tokens: true,
            truncation: Truncation::Right,
            max_tokens: info.max_seq,
            prompt: PromptRole::None,
        };
        let mut rows = Rows::new(info.max_seq, self.tokenizer.fill_id());
        let (mut ids, mut batch, mut out) = (Vec::new(), Vec::new(), Vec::with_capacity(texts.len() * self.dim));
        for (n, chunk) in texts.chunks(info.max_batch as usize).enumerate() {
            rows.clear();
            for t in chunk {
                ids.clear();
                self.tokenizer.encode_into(t, opts, &mut ids).map_err(|e| e.to_string())?;
                rows.push(&[&ids]);
            }
            self.encoder.run_rows_pooled(&rows, TURBO_POOLING_MODEL, info.l2, &mut batch).map_err(|e| e.to_string())?;
            out.extend_from_slice(&batch);
            if n % 200 == 0 {
                eprintln!("  embedded {} of {}", (n * info.max_batch as usize).min(texts.len()), texts.len());
            }
        }
        fs::create_dir_all(cache).map_err(|e| e.to_string())?;
        let bytes: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
        fs::write(&file, bytes).map_err(|e| format!("{}: {e}", file.display()))?;
        Ok(out)
    }
}

/// FNV-1a over the texts and their lengths: the cache file's name.
fn fnv(texts: &[&str]) -> u64 {
    let mut h = turbo_analysis::Fnv1a::new();
    for t in texts {
        h.str(t);
    }
    h.finish()
}
