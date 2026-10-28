use turbo_langdetect::ContextGenerator;

#[test]
fn cleanup_keeps_feature_iterators_valid_and_supports_reuse() {
    let snapshots = |g: &ContextGenerator| {
        (
            g.features().map(<[u8]>::to_vec).collect::<Vec<_>>(),
            g.training_features().map(<[u8]>::to_vec).collect::<Vec<_>>(),
        )
    };
    let mut g = ContextGenerator::new(1, 3);
    for text in [
        (0..4000).map(|i| char::from_u32(0x4e00 + i).unwrap()).collect(),
        "a".repeat(20_000),
        (0..6000).map(|i| char::from_u32(0x4e00 + i).unwrap()).collect(),
    ] {
        g.generate_prepared(&text);
        g.shrink_excess();
        let (features, training) = snapshots(&g);
        assert!(features.is_empty() && training.is_empty());
        assert!(g.is_empty());
        assert!(g.exact_order() && g.training_exact_order());
        g.shrink_excess();
        g.generate_prepared("x y");
        let mut fresh = ContextGenerator::new(1, 3);
        fresh.generate_prepared("x y");
        assert_eq!(snapshots(&g), snapshots(&fresh));
        g.shrink_excess();
        assert_eq!(snapshots(&g), snapshots(&fresh));
    }
}
