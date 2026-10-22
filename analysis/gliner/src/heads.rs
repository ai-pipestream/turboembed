use crate::Config;
use turbo::{
    head::Linear,
    safetensors::{Dtype, File},
};

pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}
pub(crate) fn apply(l: &Linear, x: &[f32]) -> Result<Vec<f32>, String> {
    let mut y = vec![0.0; l.out_dim];
    l.apply(x, &mut y);
    if y.iter().any(|v| !v.is_finite()) {
        return Err("gliner non-finite head activation".into());
    }
    Ok(y)
}
type Rows = Vec<Vec<f32>>;

struct Mlp {
    first: Linear,
    last: Linear,
}
impl Mlp {
    fn run(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        let mut y = apply(&self.first, x)?;
        y.iter_mut().for_each(|v| *v = v.max(0.0));
        apply(&self.last, &y)
    }
}
pub(crate) struct Heads {
    projection: Linear,
    prompt: Mlp,
    start: Mlp,
    end: Mlp,
    span: Mlp,
    forward: (Linear, Linear),
    reverse: (Linear, Linear),
    hidden: usize,
}
impl Heads {
    pub fn load(c: &Config, ec: &turbo::deberta::Config, bytes: &[u8]) -> Result<Self, String> {
        let f = File::parse("gliner heads", bytes).map_err(|e| e.message)?;
        for (name, shape) in c.tensor_shapes(ec)? {
            let t = f.get(&name, Dtype::F32, shape.len()).map_err(|e| e.message)?;
            if t.shape != shape || t.data.as_chunks::<4>().0.iter().any(|v| !f32::from_le_bytes(*v).is_finite()) {
                return Err(format!("gliner {name}: shape or finite-weight validation failed"));
            }
        }
        let load = |w: &str, b: &str| -> Result<Linear, String> {
            let t = f.get(w, Dtype::F32, 2).map_err(|e| e.message)?;
            let bias = f.get(b, Dtype::F32, 1).map_err(|e| e.message)?.f32s();
            Linear::new(t.shape[1] as usize, t.shape[0] as usize, t.f32s(), bias).map_err(|e| e.message)
        };
        let linear = |s: &str| load(&format!("{s}.weight"), &format!("{s}.bias"));
        let mlp = |s: &str| -> Result<Mlp, String> {
            Ok(Mlp { first: linear(&format!("{s}.0"))?, last: linear(&format!("{s}.3"))? })
        };
        let recurrent = |suffix: &str| -> Result<(Linear, Linear), String> {
            Ok((
                load(&format!("rnn.lstm.weight_ih_l0{suffix}"), &format!("rnn.lstm.bias_ih_l0{suffix}"))?,
                load(&format!("rnn.lstm.weight_hh_l0{suffix}"), &format!("rnn.lstm.bias_hh_l0{suffix}"))?,
            ))
        };
        Ok(Self {
            projection: linear("token_rep_layer.projection")?,
            prompt: mlp("prompt_rep_layer")?,
            start: mlp("span_rep_layer.span_rep_layer.project_start")?,
            end: mlp("span_rep_layer.span_rep_layer.project_end")?,
            span: mlp("span_rep_layer.span_rep_layer.out_project")?,
            forward: recurrent("")?,
            reverse: recurrent("_reverse")?,
            hidden: c.hidden,
        })
    }
    pub fn project(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        apply(&self.projection, x)
    }
    pub fn prompt(&self, x: &[f32]) -> Result<Vec<f32>, String> {
        self.prompt.run(x)
    }
    pub fn words(&self, words: &[Vec<f32>]) -> Result<Vec<Vec<f32>>, String> {
        let h = self.hidden / 2;
        let mut out = vec![vec![0.0; self.hidden]; words.len()];
        for (back, (input, recurrent)) in [(false, &self.forward), (true, &self.reverse)] {
            let mut state = vec![0.0; h];
            let mut cell = vec![0.0; h];
            for step in 0..words.len() {
                let i = if back { words.len() - step - 1 } else { step };
                let a = apply(input, &words[i])?;
                let b = apply(recurrent, &state)?;
                for j in 0..h {
                    let gate = |k| a[k * h + j] + b[k * h + j];
                    if (0..4).any(|k| !gate(k).is_finite()) {
                        return Err("gliner non-finite recurrent activation".into());
                    }
                    cell[j] = sigmoid(gate(1)) * cell[j] + sigmoid(gate(0)) * gate(2).tanh();
                    state[j] = sigmoid(gate(3)) * cell[j].tanh();
                    if !state[j].is_finite() {
                        return Err("gliner non-finite recurrent state".into());
                    }
                    out[i][j + if back { h } else { 0 }] = state[j];
                }
            }
        }
        Ok(out)
    }
    pub fn endpoints(&self, words: &[Vec<f32>]) -> Result<(Rows, Rows), String> {
        Ok((
            words.iter().map(|x| self.start.run(x)).collect::<Result<_, _>>()?,
            words.iter().map(|x| self.end.run(x)).collect::<Result<_, _>>()?,
        ))
    }
    pub fn span(&self, start: &[f32], end: &[f32]) -> Result<Vec<f32>, String> {
        let x: Vec<_> = start.iter().chain(end).map(|x| x.max(0.0)).collect();
        self.span.run(&x)
    }
}
