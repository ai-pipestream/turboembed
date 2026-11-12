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

#[test]
#[ignore = "loads real small-v2.5 tensors for layer parity"]
fn real_layer_inputs_match_native_products() {
    let root = std::env::var_os("GLINER_V2_5_PACKAGE").expect("GLINER_V2_5_PACKAGE");
    let (metadata, core) =
        turbo::deberta::package::PackageManifest::load(&turbo_pins::require(&root).join("encoder")).unwrap();
    let original = std::fs::read(turbo_pins::require(&root).join("encoder/weights.safetensors")).unwrap();
    let file = turbo::safetensors::File::parse("encoder", &original).unwrap();
    let layer =
        DebertaLayer::load(metadata.config.clone(), &file, metadata.config.layers - 1, 1, 2.0, &mut Rng::new(17))
            .unwrap();
    let ids = [101, 30000, 12000, 30001, 12001, 12002, 12003, 12004, 12005, 12006, 12007, 12008, 102];
    let trace = core.encode_layers(&ids, &vec![true; ids.len()]).unwrap();
    let x = &trace[metadata.config.layers - 1];
    let relative = DebertaLayer::relative(&metadata.config, &file).unwrap();
    let mut work = Work::default();
    layer.forward(x, &relative, &mut work);
    let h = metadata.config.hidden;
    let names = ["query_proj", "key_proj", "value_proj"];
    let prefix = format!("encoder.layer.{}", metadata.config.layers - 1);
    let weights: Vec<Vec<f32>> = names
        .iter()
        .map(|name| {
            file.get(&format!("{prefix}.attention.self.{name}.weight"), turbo::safetensors::Dtype::F32, 2)
                .unwrap()
                .f32s()
        })
        .collect();
    let parts: Vec<&[f32]> = weights.iter().map(Vec::as_slice).collect();
    let matrix = turbo::cpu::Matrix::from_rows(&parts, h).unwrap();
    let mut expected = vec![0.0; work.qkv.len()];
    matrix.apply(x, h, ids.len(), Some(&layer.base.qkv_b), &mut expected, 3 * h, turbo::cpu::Mode::Plain);
    let qkv_error = expected.iter().zip(&work.qkv).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    let mut expected_pos = vec![0.0; 2 * metadata.config.position_buckets * 3 * h];
    matrix.apply(
        &relative,
        h,
        2 * metadata.config.position_buckets,
        Some(&layer.base.qkv_b),
        &mut expected_pos,
        3 * h,
        turbo::cpu::Mode::Plain,
    );
    let pos_error = expected_pos
        .chunks_exact(3 * h)
        .zip(work.pos.chunks_exact(2 * h))
        .flat_map(|(a, b)| a[..2 * h].iter().zip(b).map(|(a, b)| (a - b).abs()))
        .fold(0.0, f32::max);
    let mut expected_ctx = vec![0.0; work.ctx.len()];
    let d = h / metadata.config.heads;
    let rr = 2 * metadata.config.position_buckets;
    for head in 0..metadata.config.heads {
        let o = head * d;
        let mut mk = turbo::cpu::Matrix::empty();
        let mut mv = turbo::cpu::Matrix::empty();
        let mut mpk = turbo::cpu::Matrix::empty();
        let mut mpq = turbo::cpu::Matrix::empty();
        mk.pack_rows(&work.qkv[h + o..], 3 * h, ids.len(), d).unwrap();
        mv.pack_cols(&work.qkv[2 * h + o..], 3 * h, d, ids.len()).unwrap();
        mpk.pack_rows(&work.pos[h + o..], 2 * h, rr, d).unwrap();
        mpq.pack_rows(&work.pos[o..], 2 * h, rr, d).unwrap();
        let mut p2c = vec![0.0; ids.len() * rr];
        mpq.apply(&work.qkv[h + o..], 3 * h, ids.len(), None, &mut p2c, rr, turbo::cpu::Mode::Plain);
        let our_p2c = deberta::p2c(&work.qkv, &work.pos, h, head, d, rr);
        let p2c_error = our_p2c.iter().zip(&p2c).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        if head == 0 {
            println!("head {head} p2c error {p2c_error}");
        }
        for i in 0..ids.len() {
            let q = &work.qkv[i * 3 * h + o..];
            let mut scores = vec![0.0; ids.len()];
            mk.apply(q, 3 * h, 1, None, &mut scores, ids.len(), turbo::cpu::Mode::Plain);
            let mut c2p = vec![0.0; rr];
            mpk.apply(q, 3 * h, 1, None, &mut c2p, rr, turbo::cpu::Mode::Plain);
            let our_c2p = deberta::c2p(&work.qkv, &work.pos, h, i, head, d, rr);
            if head == 0 && i == 0 {
                let error = our_c2p.iter().zip(&c2p).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                println!("c2p error {error}");
            }
            for (j, score) in scores.iter_mut().enumerate() {
                let r = deberta::bucket(
                    i as isize - j as isize,
                    metadata.config.position_buckets,
                    metadata.config.max_relative_positions,
                );
                *score = (*score + c2p[r] + p2c[j * rr + r]) * (1.0 / (3.0 * d as f64).sqrt()) as f32;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            for score in &mut scores {
                *score /= sum;
            }
            if head == 0 && i == 0 {
                let mut row = vec![0.0; ids.len()];
                deberta::probabilities(
                    &work.qkv,
                    &our_p2c,
                    &our_c2p,
                    h,
                    i,
                    head,
                    d,
                    metadata.config.position_buckets,
                    metadata.config.max_relative_positions,
                    &mut row,
                );
                let error = row.iter().zip(&scores).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
                println!("probability error {error}: ours {:?} expected {:?}", &row[..4], &scores[..4]);
            }
            for (j, &p) in scores.iter().enumerate() {
                for k in 0..d {
                    expected_ctx[i * h + o + k] += p * work.qkv[j * 3 * h + 2 * h + o + k];
                }
            }
        }
    }
    let ctx_error = expected_ctx.iter().zip(&work.ctx).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("attention context error {ctx_error}");
    let out_error = trace[metadata.config.layers].iter().zip(&work.out).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
    println!("qkv error {qkv_error}, relative error {pos_error}, output error {out_error}");
    assert!(ctx_error < 0.002);
    assert!(qkv_error < 0.002);
    assert!(pos_error < 0.002);
    assert!(out_error < 0.002);
}
