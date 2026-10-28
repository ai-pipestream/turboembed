use super::*;

#[test]
fn pca_recovers_the_dominant_axis() {
    // Rows lie mostly along (1, 1, 0): the first component must
    // point that way up to sign.
    let mut x = Vec::new();
    for i in 0..50 {
        let t = i as f32 / 10.0;
        x.extend([t, t + 0.01 * ((i % 3) as f32), 0.02 * ((i % 5) as f32)]);
    }
    let (comps, _) = pca_components(&x, 50, 3, 1);
    let d = (comps[0] + comps[1]).abs() / std::f32::consts::SQRT_2;
    assert!(d > 0.98, "first component {comps:?}");
    assert!(comps[2].abs() < 0.2, "first component {comps:?}");
}

#[test]
fn training_lowers_the_loss_on_toy_pairs() {
    // 40 tokens in 8 latent groups; documents are two sentences
    // drawn from the same group, so the table can only fit by
    // grouping cosines.
    let groups = 8;
    let dim = 16;
    let vocab = 64;
    let mut table: Vec<f32> = (0..vocab * dim).map(|i| ((i * 7919) % 100) as f32 / 250.0 - 0.2).collect();
    let mut docs = Vec::new();
    let mut rng = Rng(7);
    for _ in 0..120 {
        let g = rng.below(groups);
        let sent = |rng: &mut Rng| (0..6).map(|_| (g * 8 + rng.below(8)) as i32).collect::<Vec<_>>();
        docs.push(vec![sent(&mut rng), sent(&mut rng)]);
    }
    let cfg = Config { dim, batch: 32, steps: 900, ..Config::default() };
    let (first, last) = train(&mut table, vocab, &docs, &cfg);
    assert!(last < first * 0.7, "loss {first} -> {last}");
    // Fresh same-group sentence means end up closer than
    // cross-group ones (what the objective actually trains).
    let sent_vec = |rng: &mut Rng, g: usize| {
        let ids: Vec<i32> = (0..6).map(|_| (g * 8 + rng.below(8)) as i32).collect();
        let mut v = vec![0.0f32; dim];
        encode_mean(&table, dim, &ids, &mut v);
        v
    };
    let mut rng = Rng(99);
    let (mut within, mut across) = (0.0f32, 0.0f32);
    for g in 0..8 {
        let a = sent_vec(&mut rng, g);
        let b = sent_vec(&mut rng, g);
        let c = sent_vec(&mut rng, (g + 1) % 8);
        within += a.iter().zip(&b).map(|(p, q)| p * q).sum::<f32>();
        across += a.iter().zip(&c).map(|(p, q)| p * q).sum::<f32>();
    }
    assert!(within / 8.0 > across / 8.0 + 0.15, "within {} across {}", within / 8.0, across / 8.0);
}

#[test]
fn safetensors_reader_reads_what_it_writes() {
    let bytes = write_safetensors(&[("embeddings", &[3usize, 2usize], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])]);
    let (v, shape) = read_tensor(&bytes, "embeddings").unwrap();
    assert_eq!(shape, [3, 2]);
    assert_eq!(v, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
