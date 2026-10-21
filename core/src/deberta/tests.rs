use super::{Config, Encoder};

const WEIGHTS: &[u8] = include_bytes!("../../tests/fixtures/deberta/weights.safetensors");

/// Relative projections made per run, for configurations too large to
/// keep them, give the same bits as the ones kept at load.
#[test]
fn uncached_relative_projections_give_the_same_bits() {
    let config = Config {
        hidden: 32,
        intermediate: 48,
        heads: 2,
        layers: 2,
        vocabulary: 16,
        max_sequence: 32,
        position_buckets: 8,
        max_relative_positions: 32,
        layer_norm_eps: 1e-7,
    };
    let cached = Encoder::from_safetensors(config.clone(), WEIGHTS).unwrap();
    let mut uncached = Encoder::from_safetensors(config, WEIGHTS).unwrap();
    assert!(cached.weights.layers.iter().all(|l| l.pos.is_some()));
    for l in &mut uncached.weights.layers {
        l.pos = None;
    }
    let ids: Vec<u32> = (0..32).map(|i| (i * 5 % 16) as u32).collect();
    for mask in [vec![true; 32], (0..32).map(|i| i % 3 != 1).collect(), (0..32).map(|i| i == 9).collect()] {
        let a = cached.encode_layers(&ids, &mask).unwrap();
        assert_eq!(a, uncached.encode_layers(&ids, &mask).unwrap());
    }
}
