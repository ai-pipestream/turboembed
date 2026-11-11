use turbo_trainer::lstm::{Cache, Dir, Grad};

pub(crate) struct Recurrent {
    pub(crate) words: Vec<f32>,
    forward: Cache<f32>,
    reverse: Cache<f32>,
}

impl Recurrent {
    pub(crate) fn forward(params: &[Vec<f32>], words: &[f32], h: usize) -> Self {
        let n = words.len() / h;
        let mut out = Self { words: vec![0.0; n * h], forward: Cache::default(), reverse: Cache::default() };
        direction_forward(&params[..4], words, h, 0, false, &mut out.words, &mut out.forward);
        direction_forward(&params[4..], words, h, h / 2, true, &mut out.words, &mut out.reverse);
        out
    }

    pub(crate) fn backward(&self, params: &[Vec<f32>], words: &[f32], dout: &[f32], grads: &mut [Vec<f32>], h: usize) {
        direction_backward(&params[..4], &self.forward, words, dout, h, 0, false, &mut grads[..4]);
        direction_backward(&params[4..], &self.reverse, words, dout, h, h / 2, true, &mut grads[4..]);
    }
}

fn direction_forward(
    params: &[Vec<f32>],
    words: &[f32],
    h: usize,
    offset: usize,
    reverse: bool,
    out: &mut [f32],
    cache: &mut Cache<f32>,
) {
    let bias: Vec<f32> = params[2].iter().zip(&params[3]).map(|(a, b)| a + b).collect();
    turbo_trainer::lstm::forward(
        Dir { w: &params[0], u: &params[1], b: &bias, reverse },
        words,
        h,
        h / 2,
        out,
        h,
        offset,
        cache,
    );
}

#[allow(clippy::too_many_arguments)]
fn direction_backward(
    params: &[Vec<f32>],
    cache: &Cache<f32>,
    words: &[f32],
    dout: &[f32],
    h: usize,
    offset: usize,
    reverse: bool,
    grads: &mut [Vec<f32>],
) {
    let bias: Vec<f32> = params[2].iter().zip(&params[3]).map(|(a, b)| a + b).collect();
    let mut db = vec![0.0; bias.len()];
    let (w, rest) = grads.split_at_mut(1);
    let (u, _) = rest.split_at_mut(1);
    turbo_trainer::lstm::backward(
        Dir { w: &params[0], u: &params[1], b: &bias, reverse },
        words,
        h,
        h / 2,
        dout,
        h,
        offset,
        cache,
        Grad { w: &mut w[0], u: &mut u[0], b: &mut db },
        None,
    );
    for (a, b) in grads[2].iter_mut().zip(&db) {
        *a += b;
    }
    for (a, b) in grads[3].iter_mut().zip(&db) {
        *a += b;
    }
}
