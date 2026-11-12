//! One frozen DeBERTa-v2 layer with trainable LoRA projections.

use turbo::deberta::Config;
use turbo::safetensors::{Dtype, File};
use turbo::status::{Result, invalid};

use crate::real::Real;
use crate::rng::Rng;

use super::{deberta, lora, math, matmul, norm};

#[derive(Clone)]
struct Base<R> {
    qkv_w: Vec<R>,
    qkv_b: Vec<R>,
    out_w: Vec<R>,
    out_b: Vec<R>,
    attn_norm_w: Vec<R>,
    attn_norm_b: Vec<R>,
    ffn_in_w: Vec<R>,
    ffn_in_b: Vec<R>,
    ffn_out_w: Vec<R>,
    ffn_out_b: Vec<R>,
    output_norm_w: Vec<R>,
    output_norm_b: Vec<R>,
}

impl<R: Real> Base<R> {
    fn load(config: &Config, file: &File<'_>, layer: usize) -> Result<Self> {
        let (h, i) = (config.hidden, config.intermediate);
        let prefix = format!("encoder.layer.{layer}");
        let get = |stem: &str, shape: &[u64]| -> Result<Vec<R>> {
            let name = format!("{prefix}.{stem}");
            let tensor = file.get(&name, Dtype::F32, shape.len())?;
            if tensor.shape != shape {
                return Err(invalid(format!("deberta {name}: shape mismatch")));
            }
            Ok(tensor.f32s().into_iter().map(|x| R::from_f64(x as f64)).collect())
        };
        let mut qkv_w = Vec::new();
        let mut qkv_b = Vec::new();
        for stem in ["query_proj", "key_proj", "value_proj"] {
            qkv_w.extend(get(&format!("attention.self.{stem}.weight"), &[h as u64, h as u64])?);
            qkv_b.extend(get(&format!("attention.self.{stem}.bias"), &[h as u64])?);
        }
        Ok(Self {
            qkv_w,
            qkv_b,
            out_w: get("attention.output.dense.weight", &[h as u64, h as u64])?,
            out_b: get("attention.output.dense.bias", &[h as u64])?,
            attn_norm_w: get("attention.output.LayerNorm.weight", &[h as u64])?,
            attn_norm_b: get("attention.output.LayerNorm.bias", &[h as u64])?,
            ffn_in_w: get("intermediate.dense.weight", &[i as u64, h as u64])?,
            ffn_in_b: get("intermediate.dense.bias", &[i as u64])?,
            ffn_out_w: get("output.dense.weight", &[h as u64, i as u64])?,
            ffn_out_b: get("output.dense.bias", &[h as u64])?,
            output_norm_w: get("output.LayerNorm.weight", &[h as u64])?,
            output_norm_b: get("output.LayerNorm.bias", &[h as u64])?,
        })
    }
}

#[derive(Default)]
pub struct Work<R> {
    x: Vec<R>,
    qkv: Vec<R>,
    qkv_h: Vec<R>,
    pos: Vec<R>,
    pos_h: Vec<R>,
    attn: deberta::Cache<R>,
    ctx: Vec<R>,
    x_mid: Vec<R>,
    n1: norm::Cache,
    xn1: Vec<R>,
    out_h: Vec<R>,
    pre: Vec<R>,
    in_h: Vec<R>,
    act: Vec<R>,
    out_ffn_h: Vec<R>,
    n2: norm::Cache,
    xn2: Vec<R>,
    pub out: Vec<R>,
}

#[derive(Clone)]
pub struct DebertaLayer<R> {
    config: Config,
    base: Base<R>,
    params: Vec<Vec<R>>,
    rank: usize,
    scale: R,
}

impl<R: Real> DebertaLayer<R> {
    pub fn load(config: Config, file: &File<'_>, layer: usize, rank: usize, alpha: f64, rng: &mut Rng) -> Result<Self> {
        config.check()?;
        if rank == 0 || !alpha.is_finite() || alpha <= 0.0 {
            return Err(invalid("deberta LoRA rank and alpha must be positive"));
        }
        let base = Base::load(&config, file, layer)?;
        let (h, i) = (config.hidden, config.intermediate);
        let mut params = Vec::new();
        for (inp, out) in [(h, 3 * h), (h, h), (h, i), (i, h)] {
            let (a, b) = lora::init(inp, out, rank, rng);
            params.extend([a, b]);
        }
        Ok(Self { config, base, params, rank, scale: R::from_f64(alpha / rank as f64) })
    }

    pub fn relative(config: &Config, file: &File<'_>) -> Result<Vec<R>> {
        config.check()?;
        let (h, rows) = (config.hidden, 2 * config.position_buckets);
        let get = |name: &str, shape: &[u64]| -> Result<Vec<R>> {
            let tensor = file.get(name, Dtype::F32, shape.len())?;
            if tensor.shape != shape {
                return Err(invalid(format!("deberta {name}: shape mismatch")));
            }
            Ok(tensor.f32s().into_iter().map(|x| R::from_f64(x as f64)).collect())
        };
        let input = get("encoder.rel_embeddings.weight", &[rows as u64, h as u64])?;
        let weight = get("encoder.LayerNorm.weight", &[h as u64])?;
        let bias = get("encoder.LayerNorm.bias", &[h as u64])?;
        let mut out = vec![R::ZERO; input.len()];
        norm::forward(&input, &weight, &bias, config.layer_norm_eps, &mut out, &mut norm::Cache::default());
        Ok(out)
    }

    pub fn write_merged(&self, original: &[u8], layer: usize) -> Result<Vec<u8>> {
        let file = File::parse("deberta tuned encoder", original)?;
        let (h, i) = (self.config.hidden, self.config.intermediate);
        let prefix = format!("encoder.layer.{layer}");
        let specs = [
            (0, 0, "attention.self.query_proj", h, h),
            (0, h, "attention.self.key_proj", h, h),
            (0, 2 * h, "attention.self.value_proj", h, h),
            (1, 0, "attention.output.dense", h, h),
            (2, 0, "intermediate.dense", h, i),
            (3, 0, "output.dense", i, h),
        ];
        let mut out = original.to_vec();
        for (which, row, stem, inp, width) in specs {
            let name = format!("{prefix}.{stem}.weight");
            let tensor = file.get(&name, Dtype::F32, 2)?;
            if tensor.shape != [width as u64, inp as u64] {
                return Err(invalid(format!("deberta {name}: shape mismatch")));
            }
            let offset = tensor.data.as_ptr() as usize - original.as_ptr() as usize;
            let (a, b) = (&self.params[2 * which], &self.params[2 * which + 1]);
            for o in 0..width {
                for r in 0..self.rank {
                    for j in 0..inp {
                        let delta = self.scale * b[(row + o) * self.rank + r] * a[r * inp + j];
                        let at = offset + (o * inp + j) * 4;
                        let old = f32::from_le_bytes(out[at..at + 4].try_into().expect("f32 bytes"));
                        out[at..at + 4].copy_from_slice(&(old + delta.to_f64() as f32).to_le_bytes());
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn params(&self) -> &[Vec<R>] {
        &self.params
    }

    pub fn zero_grads(&self) -> Vec<Vec<R>> {
        self.params.iter().map(|p| vec![R::ZERO; p.len()]).collect()
    }

    pub fn step(&mut self, optimizer: &mut crate::adamw::AdamW<R>, grads: &mut [Vec<R>]) -> Result<f64> {
        let decay = vec![true; self.params.len()];
        optimizer.step(&mut self.params, grads, &decay)
    }

    fn project(&self, which: usize, x: &[R], inp: usize, out: usize, y: &mut [R], h: &mut Vec<R>) {
        let (w, b) = match which {
            0 => (&self.base.qkv_w, &self.base.qkv_b),
            1 => (&self.base.out_w, &self.base.out_b),
            2 => (&self.base.ffn_in_w, &self.base.ffn_in_b),
            _ => (&self.base.ffn_out_w, &self.base.ffn_out_b),
        };
        matmul::forward(x, w, inp, out, y);
        for row in y.chunks_exact_mut(out) {
            for (y, &b) in row.iter_mut().zip(b) {
                *y += b;
            }
        }
        let (a, b) = (&self.params[2 * which], &self.params[2 * which + 1]);
        lora::forward(x, a, b, inp, self.rank, self.scale, false, y, h);
    }

    fn project_back(&self, which: usize, x: &[R], h: &[R], dy: &[R], grads: &mut [Vec<R>], dx: &mut [R]) {
        let out = match which {
            0 => 3 * self.config.hidden,
            1 => self.config.hidden,
            2 => self.config.intermediate,
            _ => self.config.hidden,
        };
        let inp = if which == 3 { self.config.intermediate } else { self.config.hidden };
        let w = match which {
            0 => &self.base.qkv_w,
            1 => &self.base.out_w,
            2 => &self.base.ffn_in_w,
            _ => &self.base.ffn_out_w,
        };
        matmul::grad_x(w, dy, inp, out, dx);
        let [a_grad, b_grad] = grads.get_disjoint_mut([2 * which, 2 * which + 1]).expect("distinct LoRA slots");
        lora::backward(
            x,
            &self.params[2 * which],
            &self.params[2 * which + 1],
            h,
            inp,
            self.rank,
            self.scale,
            dy,
            a_grad,
            b_grad,
            Some(dx),
        );
    }

    pub fn forward(&self, x: &[R], relative: &[R], work: &mut Work<R>) {
        let (h, i) = (self.config.hidden, self.config.intermediate);
        let rows = 2 * self.config.position_buckets;
        work.x.clear();
        work.x.extend_from_slice(x);
        work.qkv.resize(x.len() * 3, R::ZERO);
        self.project(0, x, h, 3 * h, &mut work.qkv, &mut work.qkv_h);
        let mut pos = vec![R::ZERO; rows * 3 * h];
        self.project(0, relative, h, 3 * h, &mut pos, &mut work.pos_h);
        work.pos.clear();
        work.pos.extend(pos.chunks_exact(3 * h).flat_map(|row| row[..2 * h].iter().copied()));
        work.ctx.resize(x.len(), R::ZERO);
        deberta::forward(
            &work.qkv,
            &work.pos,
            h,
            self.config.heads,
            self.config.position_buckets,
            self.config.max_relative_positions,
            &mut work.ctx,
            &mut work.attn,
        );
        work.x_mid.resize(x.len(), R::ZERO);
        self.project(1, &work.ctx, h, h, &mut work.x_mid, &mut work.out_h);
        for (y, &x) in work.x_mid.iter_mut().zip(x) {
            *y += x;
        }
        work.xn1.resize(x.len(), R::ZERO);
        norm::forward(
            &work.x_mid,
            &self.base.attn_norm_w,
            &self.base.attn_norm_b,
            self.config.layer_norm_eps,
            &mut work.xn1,
            &mut work.n1,
        );
        work.pre.resize(x.len() / h * i, R::ZERO);
        self.project(2, &work.xn1, h, i, &mut work.pre, &mut work.in_h);
        work.act.clear();
        work.act.extend(work.pre.iter().map(|&x| math::gelu(x)));
        work.xn2.resize(x.len(), R::ZERO);
        self.project(3, &work.act, i, h, &mut work.xn2, &mut work.out_ffn_h);
        for (y, &x) in work.xn2.iter_mut().zip(&work.xn1) {
            *y += x;
        }
        work.out.resize(x.len(), R::ZERO);
        norm::forward(
            &work.xn2,
            &self.base.output_norm_w,
            &self.base.output_norm_b,
            self.config.layer_norm_eps,
            &mut work.out,
            &mut work.n2,
        );
    }

    pub fn backward(&self, work: &Work<R>, relative: &[R], dout: &[R], grads: &mut [Vec<R>]) -> Vec<R> {
        let h = self.config.hidden;
        let mut dxn2 = vec![R::ZERO; dout.len()];
        norm::backward(&work.xn2, &self.base.output_norm_w, &work.n2, dout, &mut dxn2, None, None);
        let mut dact = vec![R::ZERO; work.act.len()];
        self.project_back(3, &work.act, &work.out_ffn_h, &dxn2, grads, &mut dact);
        let mut dpre = vec![R::ZERO; work.pre.len()];
        for ((dp, &da), &p) in dpre.iter_mut().zip(&dact).zip(&work.pre) {
            *dp = da * math::gelu_grad(p);
        }
        let mut dxn1 = vec![R::ZERO; dout.len()];
        self.project_back(2, &work.xn1, &work.in_h, &dpre, grads, &mut dxn1);
        for (d, &r) in dxn1.iter_mut().zip(&dxn2) {
            *d += r;
        }
        let mut dx_mid = vec![R::ZERO; dout.len()];
        norm::backward(&work.x_mid, &self.base.attn_norm_w, &work.n1, &dxn1, &mut dx_mid, None, None);
        let mut dctx = vec![R::ZERO; work.ctx.len()];
        self.project_back(1, &work.ctx, &work.out_h, &dx_mid, grads, &mut dctx);
        let mut dqkv = vec![R::ZERO; work.qkv.len()];
        let mut dpos = vec![R::ZERO; work.pos.len()];
        deberta::backward(
            &work.qkv,
            &work.pos,
            h,
            self.config.heads,
            self.config.position_buckets,
            self.config.max_relative_positions,
            &work.attn,
            &dctx,
            &mut dqkv,
            &mut dpos,
        );
        let mut dpos_qkv = vec![R::ZERO; 2 * self.config.position_buckets * 3 * h];
        for (out, input) in dpos_qkv.chunks_exact_mut(3 * h).zip(dpos.chunks_exact(2 * h)) {
            out[..2 * h].copy_from_slice(input);
        }
        let mut dx = vec![R::ZERO; work.x.len()];
        self.project_back(0, &work.x, &work.qkv_h, &dqkv, grads, &mut dx);
        let mut drelative = vec![R::ZERO; relative.len()];
        self.project_back(0, relative, &work.pos_h, &dpos_qkv, grads, &mut drelative);
        for (d, &r) in dx.iter_mut().zip(&dx_mid) {
            *d += r;
        }
        dx
    }
}

#[cfg(test)]
#[path = "deberta_layer_tests.rs"]
mod tests;
