#![cfg(feature = "cpu")]
use serde::Deserialize;
use turbo::{
    deberta::{Config, Encoder, TokenPromptProjections},
    head::Linear,
};

fn config() -> Config {
    Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    }
}
const WEIGHTS: &[u8] = include_bytes!("fixtures/deberta/weights.safetensors");
#[derive(Deserialize)]
struct Case {
    ids: Vec<u32>,
    mask: Vec<u8>,
    layers: Vec<Vec<f32>>,
    token: Vec<f32>,
    prompt: Vec<f32>,
}

#[test]
fn real_checkpoint_slices_match_transformers_at_every_layer() {
    let model = Encoder::from_safetensors(config(), WEIGHTS).unwrap();
    let cases: Vec<Case> = serde_json::from_slice(include_bytes!("fixtures/deberta/reference.json")).unwrap();
    for c in cases {
        let mask: Vec<_> = c.mask.iter().map(|x| *x == 1).collect();
        let actual = model.encode_layers(&c.ids, &mask).unwrap();
        assert_eq!(actual.len(), c.layers.len());
        for (layer, (a, b)) in actual.iter().zip(&c.layers).enumerate() {
            assert_eq!(a.len(), b.len());
            let error = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            println!("tokens={} layer={layer} max_abs={error:e}", c.ids.len());
            assert!(error < 2e-5, "layer {layer}: {error}");
        }
        let final_only = model.encode(&c.ids, &mask).unwrap();
        assert_eq!(final_only.data(), actual.last().unwrap());
        assert_eq!(final_only.mask(), mask);
        let heads = TokenPromptProjections::from_safetensors(
            32,
            WEIGHTS,
            "encoder.layer.0.attention.self.query_proj",
            "encoder.layer.0.attention.self.key_proj",
        )
        .unwrap();
        let indices: Vec<_> = mask.iter().enumerate().filter_map(|(i, &m)| m.then_some(i)).collect();
        for (got, reference) in [
            (heads.tokens(&final_only, &indices).unwrap(), &c.token),
            (heads.prompts(&final_only, &indices).unwrap(), &c.prompt),
        ] {
            let expected: Vec<_> =
                indices.iter().flat_map(|&i| reference[i * 32..(i + 1) * 32].iter().copied()).collect();
            let error = got.iter().zip(expected).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            assert!(error < 2e-5, "projection error {error}");
        }
    }
}

#[test]
fn rejects_invalid_config_inputs_weights_and_outputs() {
    let mut c = config();
    c.heads = 3;
    assert!(Encoder::from_safetensors(c, WEIGHTS).is_err());
    let mut c = config();
    c.layer_norm_eps = f64::NAN;
    assert!(c.check().is_err());
    let mut c = config();
    c.position_buckets = 2;
    assert!(c.check().is_err());
    let mut c = config();
    c.hidden = usize::MAX;
    assert!(c.check().is_err());
    let model = Encoder::from_safetensors(config(), WEIGHTS).unwrap();
    for (ids, mask) in [
        (vec![], vec![]),
        (vec![0], vec![]),
        (vec![16], vec![true]),
        (vec![0], vec![false]),
        (vec![0; 33], vec![true; 33]),
        (vec![0, 16], vec![true, false]),
    ] {
        assert!(model.encode(&ids, &mask).is_err());
    }
    let mut bytes = WEIGHTS.to_vec();
    let start = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    bytes[start..start + 4].copy_from_slice(&f32::NAN.to_le_bytes());
    assert!(Encoder::from_safetensors(config(), &bytes).is_err());
    assert!(Encoder::from_safetensors(config(), &WEIGHTS[..WEIGHTS.len() - 4]).is_err());
    let mut c = config();
    c.hidden = 64;
    assert!(Encoder::from_safetensors(c, WEIGHTS).is_err());
}

#[test]
fn token_and_prompt_heads_select_checked_visible_rows() {
    assert!(TokenPromptProjections::from_safetensors(0, WEIGHTS, "a", "b").is_err());
    assert!(TokenPromptProjections::from_safetensors(32, WEIGHTS, "missing", "missing").is_err());
    let model = Encoder::from_safetensors(config(), WEIGHTS).unwrap();
    let states = model.encode(&[1, 2, 3], &[true, true, false]).unwrap();
    let mut w = vec![0.0; 64];
    w[0] = 1.0;
    w[33] = 1.0;
    let token = Linear::new(32, 2, w.clone(), vec![0.0; 2]).unwrap();
    let prompt = Linear::new(32, 2, w, vec![1.0; 2]).unwrap();
    let heads = TokenPromptProjections::new(token, prompt).unwrap();
    assert_eq!(
        heads.tokens(&states, &[1, 0]).unwrap(),
        [states.data()[32], states.data()[33], states.data()[0], states.data()[1]]
    );
    assert_eq!(heads.prompts(&states, &[0]).unwrap(), [states.data()[0] + 1.0, states.data()[1] + 1.0]);
    assert!(heads.tokens(&states, &[2]).is_err());
    assert!(heads.prompts(&states, &[usize::MAX]).is_err());
    assert!(heads.tokens(&states, &[0; 4]).is_err());
    let bad = Linear::new(16, 2, vec![0.0; 32], vec![0.0; 2]).unwrap();
    let heads = TokenPromptProjections::new(bad.clone(), bad).unwrap();
    assert!(heads.tokens(&states, &[0]).is_err());
}
