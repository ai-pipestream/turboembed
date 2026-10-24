//! Strict conversion of the GLiNER small-v2.1 and multi-v2.1 markerV0 profiles.
//! Accepts either an F32 safetensors checkpoint or a `torch.save` zip archive
//! (`pytorch_model.bin`).
use crate::Result;
use serde_json::{Value, json};
use std::{borrow::Cow, collections::BTreeMap, fs, path::Path};
use turbo::deberta::package::digest;
use turbo::safetensors::{Dtype, File};
use turbo_gliner::{
    Config,
    package::{Manifest, read},
};

pub fn from_huggingface(v: &Value, vocab: usize) -> Result<Config> {
    if v.get("span_mode").and_then(Value::as_str) != Some("markerV0")
        || v.get("subtoken_pooling").and_then(Value::as_str) != Some("first")
    {
        return Err("gliner requires v2.1 markerV0/first profile".into());
    }
    let expected_vocab = match v.get("model_name").and_then(Value::as_str) {
        Some("microsoft/deberta-v3-small") => 128_004,
        Some("microsoft/mdeberta-v3-base") => 250_105,
        _ => return Err("gliner requires small-v2.1 or multi-v2.1 markerV0/first profile".into()),
    };
    for (name, expected) in [
        ("has_rnn", json!(true)),
        ("embed_ent_token", json!(true)),
        ("labels_encoder", Value::Null),
        ("post_fusion_schema", Value::Null),
    ] {
        if v.get(name).is_some_and(|x| x != &expected) {
            return Err(format!("gliner unsupported {name}"));
        }
    }
    if vocab != expected_vocab {
        return Err(format!("gliner vocabulary must have {expected_vocab} entries"));
    }
    Ok(Config {
        hidden: v.get("hidden_size").and_then(Value::as_u64).ok_or("gliner missing hidden_size")? as usize,
        max_width: v.get("max_width").and_then(Value::as_u64).ok_or("gliner missing max_width")? as usize,
        entity_token: (vocab - 2) as u32,
        separator_token: (vocab - 1) as u32,
        cls_token: 1,
        eos_token: 2,
    })
}
pub fn convert_heads(c: &Config, ec: &turbo::deberta::Config, checkpoint: &[u8]) -> Result<Vec<u8>> {
    let f = File::parse("gliner checkpoint", checkpoint).map_err(|e| e.message)?;
    let mut header = BTreeMap::new();
    let mut data = Vec::new();
    for (name, shape) in c.tensor_shapes(ec)? {
        let t = f.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
        if t.shape != shape || t.data.as_chunks::<4>().0.iter().any(|v| !f32::from_le_bytes(*v).is_finite()) {
            return Err(format!("gliner {name}: shape or finite-weight validation failed"));
        }
        let start = data.len();
        data.extend_from_slice(t.data);
        header.insert(name, json!({"dtype":"F32","shape":shape,"data_offsets":[start,data.len()]}));
    }
    let mut h = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    h.resize(h.len().next_multiple_of(8), b' ');
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend(h);
    out.extend(data);
    Ok(out)
}
pub fn convert_files(
    encoder_config: &Path,
    gliner_config: &Path,
    checkpoint: &Path,
    spm: &Path,
    source: (&str, &str, &str),
    out: &Path,
) -> Result<()> {
    let mut ec: Value = serde_json::from_slice(&read(encoder_config, 65536)?).map_err(|e| e.to_string())?;
    let gc: Value = serde_json::from_slice(&read(gliner_config, 65536)?).map_err(|e| e.to_string())?;
    let raw = read(checkpoint, 2 << 30)?;
    let bytes: Cow<'_, [u8]> = if raw.starts_with(b"PK\x03\x04") {
        let tensors = crate::torch::read(&raw)?;
        Cow::Owned(crate::torch::to_safetensors(&tensors, |k| Some(k.to_owned())))
    } else {
        Cow::Borrowed(raw.as_slice())
    };
    let f = File::parse("gliner checkpoint", &bytes).map_err(|e| e.message)?;
    let embedding = f
        .get("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight", Dtype::F32, 2)
        .map_err(|e| e.message)?;
    ec["vocab_size"] = embedding.shape[0].into();
    let encoder = crate::deberta::from_huggingface(&ec)?;
    let config = from_huggingface(&gc, encoder.vocabulary)?;
    let heads = convert_heads(&config, &encoder, &bytes)?;
    let tokenizer = read(spm, 16 << 20)?;
    let parsed_spm = turbo::sentencepiece::SentencePiece::parse(&tokenizer).map_err(|e| e.message)?;
    if parsed_spm.len() + 4 != encoder.vocabulary
        || parsed_spm.id("[CLS]") != Some(config.cls_token)
        || parsed_spm.id("[SEP]") != Some(config.eos_token)
    {
        return Err("gliner spm.model does not match checkpoint vocabulary".into());
    }
    // Validate encoder before creating the destination; publication uses a new
    // directory and an incomplete package cannot pass the digest loader.
    crate::deberta::convert(&encoder, &bytes, "token_rep_layer.bert_layer.model.")?;
    fs::create_dir(out).map_err(|e| e.to_string())?;
    crate::deberta::write_package(encoder, &bytes, "token_rep_layer.bert_layer.model.", source, &out.join("encoder"))?;
    let m = Manifest {
        version: 1,
        config,
        encoder_sha256: digest(&read(&out.join("encoder/deberta.json"), 65536)?),
        heads_sha256: digest(&heads),
        tokenizer_sha256: digest(&tokenizer),
    };
    fs::write(out.join("heads.safetensors"), heads).map_err(|e| e.to_string())?;
    fs::write(out.join("spm.model"), tokenizer).map_err(|e| e.to_string())?;
    fs::write(out.join("gliner.json"), serde_json::to_vec_pretty(&m).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    Manifest::load(out)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conversion_rejects_unsupported_profiles() {
        let base = json!({"span_mode":"markerV0", "subtoken_pooling":"first",
            "model_name":"microsoft/deberta-v3-small", "hidden_size":512, "max_width":12});
        assert_eq!(from_huggingface(&base, 128004).unwrap().hidden, 512);
        assert!(from_huggingface(&base, 128003).is_err());
        let multi = json!({"span_mode":"markerV0", "subtoken_pooling":"first",
            "model_name":"microsoft/mdeberta-v3-base", "hidden_size":512, "max_width":12});
        let multi_cfg = from_huggingface(&multi, 250105).unwrap();
        assert_eq!((multi_cfg.entity_token, multi_cfg.separator_token), (250103, 250104));
        assert!(from_huggingface(&multi, 128004).is_err());
        for (key, value) in [
            ("has_rnn", json!(false)),
            ("embed_ent_token", json!(false)),
            ("labels_encoder", json!("other")),
            ("span_mode", json!("markerV1")),
            ("subtoken_pooling", json!("mean")),
            ("model_name", json!("microsoft/deberta-v3-large")),
        ] {
            let mut bad = base.clone();
            bad[key] = value;
            assert!(from_huggingface(&bad, 128004).is_err(), "{key}");
        }
    }
    #[test]
    fn canonical_head_conversion_is_deterministic_and_rejects_nonfinite() {
        let c = Config { hidden: 16, max_width: 3, entity_token: 12, separator_token: 13, cls_token: 1, eos_token: 2 };
        let ec = turbo::deberta::Config {
            hidden: 32,
            heads: 2,
            intermediate: 48,
            layers: 2,
            vocabulary: 16,
            max_sequence: 32,
            position_buckets: 8,
            max_relative_positions: 32,
            layer_norm_eps: 1e-7,
        };
        let input = include_bytes!("../../analysis/gliner/tests/fixtures/heads.safetensors");
        let first = convert_heads(&c, &ec, input).unwrap();
        assert_eq!(first, convert_heads(&c, &ec, &first).unwrap());
        let mut bad = input.to_vec();
        let len = bad.len();
        bad[len - 4..].copy_from_slice(&f32::INFINITY.to_le_bytes());
        assert!(convert_heads(&c, &ec, &bad).is_err());
        assert!(convert_heads(&c, &ec, &input[..10]).is_err());
    }
}
