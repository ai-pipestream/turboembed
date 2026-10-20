use super::*;
#[test]
fn analytic_gradients_match_finite_differences() {
    for activation in [HiddenActivation::Relu, HiddenActivation::Gelu] {
        for output in [OutputKind::Softmax, OutputKind::Sigmoid] {
            let mut layers = vec![
                DenseLayer::new(2, 2, vec![0.2, 0.4, -0.3, 0.6], vec![0.1, 0.2]).unwrap(),
                DenseLayer::new(2, 2, vec![0.5, -0.2, 0.3, 0.7], vec![0.1, -0.1]).unwrap(),
            ];
            let x = [0.4, -0.2];
            let y = [0., 1.];
            let mut gradients = optimizer::zeros(&layers);
            optimizer::gradient(&layers, activation, output, &x, &y, &mut gradients).unwrap();
            for layer in 0..2 {
                let weights = layers[layer].weights.len();
                for (j, &analytic) in gradients[layer].iter().enumerate() {
                    let set = |layers: &mut [DenseLayer], delta: f64| {
                        if j < weights {
                            layers[layer].weights[j] += delta;
                        } else {
                            layers[layer].bias[j - weights] += delta;
                        }
                    };
                    let loss = |layers: &[DenseLayer]| {
                        let (a, _) = optimizer::activations(layers, activation, &x).unwrap();
                        math::loss(a.last().unwrap(), &y, output, 1.).unwrap()
                    };
                    let eps = 1e-5;
                    set(&mut layers, eps);
                    let plus = loss(&layers);
                    set(&mut layers, -2. * eps);
                    let minus = loss(&layers);
                    set(&mut layers, eps);
                    let numeric = (plus - minus) / (2. * eps);
                    assert!(
                        (numeric - analytic).abs() < 1e-5,
                        "{activation:?} {output:?} {layer}/{j}: {numeric} vs {}",
                        analytic
                    );
                }
            }
        }
    }
}
