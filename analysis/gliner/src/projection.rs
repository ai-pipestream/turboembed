pub(crate) fn forward(params: &[Vec<f32>], inputs: &[f32], h: usize) -> Vec<f32> {
    let k = params[0].len() / h;
    let rows = inputs.len() / k;
    let mut out = vec![0.0; rows * h];
    for row in 0..rows {
        for o in 0..h {
            let mut z = params[1][o];
            for j in 0..k {
                z += params[0][o * k + j] * inputs[row * k + j];
            }
            out[row * h + o] = z;
        }
    }
    out
}

pub(crate) fn backward(params: &[Vec<f32>], inputs: &[f32], dout: &[f32], grads: &mut [Vec<f32>], h: usize) {
    let k = params[0].len() / h;
    for row in 0..inputs.len() / k {
        for o in 0..h {
            let d = dout[row * h + o];
            for j in 0..k {
                grads[0][o * k + j] += d * inputs[row * k + j];
            }
            grads[1][o] += d;
        }
    }
}
