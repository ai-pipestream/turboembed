use super::*;
use crate::rng::Rng;

const EPS: f64 = 1e-6;

fn random(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(0.8) - 0.4).collect()
}

fn weighted(y: &[f64], r: &[f64]) -> f64 {
    y.iter().zip(r).map(|(a, b)| a * b).sum()
}

fn check(what: &str, x: &mut [f64], analytic: &[f64], f: &mut dyn FnMut(&[f64]) -> f64) {
    assert_eq!(x.len(), analytic.len(), "{what}");
    for i in 0..x.len() {
        let keep = x[i];
        x[i] = keep + EPS;
        let up = f(x);
        x[i] = keep - EPS;
        let down = f(x);
        x[i] = keep;
        let numeric = (up - down) / (2.0 * EPS);
        let tolerance = 1e-5 * (1.0 + numeric.abs().max(analytic[i].abs()));
        assert!((numeric - analytic[i]).abs() <= tolerance, "{what}[{i}]: numeric {numeric}, analytic {}", analytic[i]);
    }
}

#[test]
fn disentangled_attention_gradients_match_finite_differences() {
    let (n, hidden, heads, buckets, maximum) = (4, 4, 2, 4, 16);
    let mut rng = Rng::new(41);
    let qkv = random(&mut rng, n * 3 * hidden);
    let pos = random(&mut rng, 2 * buckets * 2 * hidden);
    let r = random(&mut rng, n * hidden);
    let loss = |qkv: &[f64], pos: &[f64]| {
        let (mut ctx, mut cache) = (vec![0.0; n * hidden], Cache::default());
        forward(qkv, pos, hidden, heads, buckets, maximum, &mut ctx, &mut cache);
        weighted(&ctx, &r)
    };
    let (mut ctx, mut cache) = (vec![0.0; n * hidden], Cache::default());
    forward(&qkv, &pos, hidden, heads, buckets, maximum, &mut ctx, &mut cache);
    let (mut dqkv, mut dpos) = (vec![0.0; qkv.len()], vec![0.0; pos.len()]);
    backward(&qkv, &pos, hidden, heads, buckets, maximum, &cache, &r, &mut dqkv, &mut dpos);
    let (qkv0, pos0) = (qkv.clone(), pos.clone());
    check("qkv", &mut qkv.clone(), &dqkv, &mut |q| loss(q, &pos0));
    check("pos", &mut pos.clone(), &dpos, &mut |p| loss(&qkv0, p));
}

#[test]
fn relative_bucket_boundaries_match_the_signed_convention() {
    assert_eq!(bucket(-16, 4, 16), 0);
    assert_eq!(bucket(16, 4, 16), 7);
    assert_eq!(bucket(-1, 4, 16), 3);
    assert_eq!(bucket(0, 4, 16), 4);
    assert_eq!(bucket(1, 4, 16), 5);
}

#[test]
fn one_layer_matches_the_native_deberta_encoder() {
    use crate::encoder::{matmul, norm};
    use std::collections::BTreeMap;
    use turbo::deberta::{Config, Encoder};

    let config = Config {
        hidden: 4,
        intermediate: 6,
        heads: 2,
        layers: 1,
        vocabulary: 12,
        max_sequence: 8,
        position_buckets: 4,
        max_relative_positions: 16,
        layer_norm_eps: 1e-5,
    };
    let mut values = BTreeMap::new();
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut rng = Rng::new(42);
    for (name, shape) in config.tensor_shapes().unwrap() {
        let start = data.len();
        let weights: Vec<f32> = if name.ends_with("LayerNorm.weight") {
            vec![1.0; shape[0] as usize]
        } else if name.ends_with("LayerNorm.bias") {
            vec![0.0; shape[0] as usize]
        } else {
            random(&mut rng, shape.iter().product::<u64>() as usize).into_iter().map(|x| x as f32 * 0.2).collect()
        };
        for &value in &weights {
            data.extend_from_slice(&value.to_le_bytes());
        }
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [start, data.len()]}),
        );
        values.insert(name, weights);
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let bytes = (header.len() as u64).to_le_bytes().into_iter().chain(header).chain(data).collect::<Vec<u8>>();
    let encoder = Encoder::from_safetensors(config.clone(), &bytes).unwrap().with_threads(1).unwrap();
    let ids = [1, 4, 7, 2];
    let reference = encoder.encode_layers(&ids, &vec![true; ids.len()]).unwrap();
    let get = |name: &str| values.get(name).map(Vec::as_slice).unwrap();
    let project = |x: &[f32], stem: &str, input: usize, output: usize| {
        let mut y = vec![0.0; x.len() / input * output];
        matmul::forward(x, get(&format!("{stem}.weight")), input, output, &mut y);
        let bias = get(&format!("{stem}.bias"));
        for row in y.chunks_exact_mut(output) {
            for (y, &b) in row.iter_mut().zip(bias) {
                *y += b;
            }
        }
        y
    };
    let normalize = |x: &[f32], stem: &str| {
        let mut y = vec![0.0; x.len()];
        norm::forward(
            x,
            get(&format!("{stem}.weight")),
            get(&format!("{stem}.bias")),
            config.layer_norm_eps,
            &mut y,
            &mut norm::Cache::default(),
        );
        y
    };
    let qkv = |stem: String| {
        let names = ["query_proj", "key_proj", "value_proj"];
        let weights = names.map(|name| get(&format!("{stem}.{name}.weight")).to_vec()).concat();
        let bias = names.map(|name| get(&format!("{stem}.{name}.bias")).to_vec()).concat();
        (weights, bias)
    };
    let (qw, qb) = qkv("encoder.layer.0.attention.self".into());
    let relative = normalize(get("encoder.rel_embeddings.weight"), "encoder.LayerNorm");
    let mut pos = vec![0.0; 2 * config.position_buckets * 3 * config.hidden];
    matmul::forward(&relative, &qw, config.hidden, 3 * config.hidden, &mut pos);
    for row in pos.chunks_exact_mut(3 * config.hidden) {
        for (y, &b) in row.iter_mut().zip(&qb) {
            *y += b;
        }
    }
    let x = &reference[0];
    let mut packed = vec![0.0; ids.len() * 3 * config.hidden];
    matmul::forward(x, &qw, config.hidden, 3 * config.hidden, &mut packed);
    for row in packed.chunks_exact_mut(3 * config.hidden) {
        for (y, &b) in row.iter_mut().zip(&qb) {
            *y += b;
        }
    }
    let mut ctx = vec![0.0; ids.len() * config.hidden];
    forward(
        &packed,
        &pos,
        config.hidden,
        config.heads,
        config.position_buckets,
        config.max_relative_positions,
        &mut ctx,
        &mut Cache::default(),
    );
    let mut mid = project(&ctx, "encoder.layer.0.attention.output.dense", config.hidden, config.hidden);
    for (y, &x) in mid.iter_mut().zip(x) {
        *y += x;
    }
    let mid = normalize(&mid, "encoder.layer.0.attention.output.LayerNorm");
    let pre = project(&mid, "encoder.layer.0.intermediate.dense", config.hidden, config.intermediate);
    let act: Vec<f32> = pre.into_iter().map(crate::encoder::math::gelu).collect();
    let mut out = project(&act, "encoder.layer.0.output.dense", config.intermediate, config.hidden);
    for (y, &x) in out.iter_mut().zip(&mid) {
        *y += x;
    }
    let out = normalize(&out, "encoder.layer.0.output.LayerNorm");
    let maximum =
        out.iter().zip(&reference[1]).map(|(&actual, &expected)| (actual - expected).abs()).fold(0.0, f32::max);
    assert!(maximum < 0.002, "maximum state error {maximum}");
}
