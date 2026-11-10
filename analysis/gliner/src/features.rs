pub(crate) struct Boundary {
    pub(crate) mid: Vec<f32>,
    pub(crate) out: Vec<f32>,
}

impl Boundary {
    pub(crate) fn forward(params: &[Vec<f32>], words: &[f32], n: usize, h: usize, f: usize) -> Self {
        let mut out = Self { mid: vec![0.0; n * f * h], out: vec![0.0; n * h] };
        for row in 0..n {
            for i in 0..f * h {
                let mut z = params[1][i];
                for j in 0..h {
                    z += params[0][i * h + j] * words[row * h + j];
                }
                out.mid[row * f * h + i] = z.max(0.0);
            }
            for o in 0..h {
                let mut z = params[3][o];
                for i in 0..f * h {
                    z += params[2][o * f * h + i] * out.mid[row * f * h + i];
                }
                out.out[row * h + o] = z.max(0.0);
            }
        }
        out
    }

    pub(crate) fn backward(
        &self,
        params: &[Vec<f32>],
        words: &[f32],
        dout: &[f32],
        grads: &mut [Vec<f32>],
        h: usize,
        f: usize,
    ) {
        for row in 0..words.len() / h {
            let mut dmid = vec![0.0; f * h];
            for o in 0..h {
                let d = if self.out[row * h + o] > 0.0 { dout[row * h + o] } else { 0.0 };
                for i in 0..f * h {
                    grads[2][o * f * h + i] += d * self.mid[row * f * h + i];
                    dmid[i] += d * params[2][o * f * h + i];
                }
                grads[3][o] += d;
            }
            for i in 0..f * h {
                let d = if self.mid[row * f * h + i] > 0.0 { dmid[i] } else { 0.0 };
                for j in 0..h {
                    grads[0][i * h + j] += d * words[row * h + j];
                }
                grads[1][i] += d;
            }
        }
    }
}

pub(crate) struct Prompt {
    mid: Vec<f32>,
}

impl Prompt {
    pub(crate) fn forward(params: &[Vec<f32>], inputs: &[f32], h: usize, f: usize) -> (Self, Vec<f32>) {
        let n = inputs.len() / h;
        let mut out = Self { mid: vec![0.0; n * f * h] };
        let mut prompts = vec![0.0; n * h];
        for row in 0..n {
            for i in 0..f * h {
                let mut z = params[1][i];
                for j in 0..h {
                    z += params[0][i * h + j] * inputs[row * h + j];
                }
                out.mid[row * f * h + i] = z.max(0.0);
            }
            for o in 0..h {
                let mut z = params[3][o];
                for i in 0..f * h {
                    z += params[2][o * f * h + i] * out.mid[row * f * h + i];
                }
                prompts[row * h + o] = z;
            }
        }
        (out, prompts)
    }

    pub(crate) fn backward(
        &self,
        params: &[Vec<f32>],
        inputs: &[f32],
        dout: &[f32],
        grads: &mut [Vec<f32>],
        h: usize,
        f: usize,
    ) {
        for row in 0..inputs.len() / h {
            let mut dmid = vec![0.0; f * h];
            for o in 0..h {
                let d = dout[row * h + o];
                for i in 0..f * h {
                    grads[2][o * f * h + i] += d * self.mid[row * f * h + i];
                    dmid[i] += d * params[2][o * f * h + i];
                }
                grads[3][o] += d;
            }
            for i in 0..f * h {
                let d = if self.mid[row * f * h + i] > 0.0 { dmid[i] } else { 0.0 };
                for j in 0..h {
                    grads[0][i * h + j] += d * inputs[row * h + j];
                }
                grads[1][i] += d;
            }
        }
    }
}
