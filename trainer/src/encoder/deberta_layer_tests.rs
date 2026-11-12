use super::*;

const EPS: f64 = 1e-6;

fn random(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n).map(|_| rng.uniform(0.8) - 0.4).collect()
}

fn fixture() -> (DebertaLayer<f64>, Vec<f64>, Vec<f64>) {
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
    let (h, i) = (config.hidden, config.intermediate);
    let mut rng = Rng::new(51);
    let mut tensor = |n| random(&mut rng, n);
    let base = Base {
        qkv_w: tensor(3 * h * h),
        qkv_b: tensor(3 * h),
        out_w: tensor(h * h),
        out_b: tensor(h),
        attn_norm_w: vec![1.0; h],
        attn_norm_b: vec![0.0; h],
        ffn_in_w: tensor(i * h),
        ffn_in_b: tensor(i),
        ffn_out_w: tensor(h * i),
        ffn_out_b: tensor(h),
        output_norm_w: vec![1.0; h],
        output_norm_b: vec![0.0; h],
    };
    let mut params = Vec::new();
    for (inp, out) in [(h, 3 * h), (h, h), (h, i), (i, h)] {
        params.extend([tensor(inp), tensor(out)]);
    }
    let layer = DebertaLayer { config, base, params, rank: 1, scale: 0.75 };
    let x = random(&mut rng, 4 * h);
    let relative = random(&mut rng, 2 * layer.config.position_buckets * h);
    (layer, x, relative)
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
fn layer_gradients_match_finite_differences() {
    let (mut layer, x, relative) = fixture();
    let mut rng = Rng::new(52);
    let r = random(&mut rng, x.len());
    let loss = |layer: &DebertaLayer<f64>, x: &[f64]| {
        let mut work = Work::default();
        layer.forward(x, &relative, &mut work);
        weighted(&work.out, &r)
    };
    let mut work = Work::default();
    layer.forward(&x, &relative, &mut work);
    let mut grads = layer.zero_grads();
    let dx = layer.backward(&work, &relative, &r, &mut grads);
    let layer0 = layer.clone();
    check("input", &mut x.clone(), &dx, &mut |x| loss(&layer0, x));
    for (tensor, analytic) in grads.iter().enumerate() {
        let mut params = layer.params[tensor].clone();
        let analytic = analytic.clone();
        let layer0 = layer.clone();
        check("adapter", &mut params, &analytic, &mut |values| {
            let mut layer = layer0.clone();
            layer.params[tensor].copy_from_slice(values);
            loss(&layer, &x)
        });
    }
    layer.params[0][0] += 1.0;
}

#[test]
fn qkv_adapters_also_move_relative_projections() {
    let (layer, _, relative) = fixture();
    let x = vec![0.0; 4 * layer.config.hidden];
    let mut baseline = Work::default();
    let zero = vec![0.0; 4 * layer.config.hidden];
    let mut base = layer.clone();
    base.params[0].fill(0.0);
    base.params[1].fill(0.0);
    base.forward(&zero, &relative, &mut baseline);
    let mut adapted = Work::default();
    layer.forward(&x, &relative, &mut adapted);
    assert!(baseline.out.iter().zip(&adapted.out).any(|(a, b)| a != b));
}

#[test]
fn an_optimizer_step_moves_only_the_adapters() {
    let (mut layer, x, relative) = fixture();
    let base = layer.base.clone();
    let before = layer.params.clone();
    let settings = crate::adamw::Settings { lr: 0.01, total: 1, ..crate::adamw::Settings::default() };
    let lens: Vec<usize> = layer.params.iter().map(Vec::len).collect();
    let mut optimizer = crate::adamw::AdamW::new(settings, &lens).unwrap();
    layer.step(&mut optimizer, &mut layer.zero_grads()).unwrap();
    assert!(layer.params.iter().zip(&before).any(|(a, b)| a != b));
    assert_eq!(layer.base.qkv_w, base.qkv_w);
    let mut work = Work::default();
    layer.forward(&x, &relative, &mut work);
    assert!(work.out.iter().all(|x| x.is_finite()));
}
