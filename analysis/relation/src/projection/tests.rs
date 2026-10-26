use super::*;

#[test]
fn partial_tiles_match_scalar_products_in_reused_output() {
    let mut state = 9u64;
    let mut next = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    let mut output = Vec::new();
    for (inp, out_dim, n) in [(48, 64, 19), (40, 96, 8), (33, 32, 3), (16, 40, 11), (24, 48, 7), (8, 8, 0)] {
        let weights: Vec<_> = (0..inp * out_dim).map(|_| next()).collect();
        let bias: Vec<_> = (0..out_dim).map(|_| next()).collect();
        let linear = Linear::new(inp, out_dim, weights, bias).unwrap();
        let projection = Projection::new(&linear).unwrap();
        let input: Vec<_> = (0..n * inp).map(|_| next()).collect();
        projection.rows_into(&input, n, &mut output);
        assert_eq!(output.len(), n * out_dim);
        for (row, x) in input.chunks_exact(inp).enumerate() {
            for (col, w) in linear.weight().chunks_exact(inp).enumerate() {
                let expected = w.iter().zip(x).map(|(&a, &b)| f64::from(a) * f64::from(b)).sum::<f64>()
                    + f64::from(linear.bias()[col]);
                assert!((f64::from(output[row * out_dim + col]) - expected).abs() < 1e-4);
            }
        }
        // Reapplying overwrites old contents rather than accumulating them.
        let expected = output.clone();
        output.fill(999.0);
        projection.rows_into(&input, n, &mut output);
        assert_eq!(output, expected);
    }
}
