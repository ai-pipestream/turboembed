//! One LSTM direction over one sequence, forward and back through time,
//! with the core's numerics (`turbo::cpu::lstm`): gates (i, f, g, o),
//! W [4h, in], U [4h, h], one bias [4h], zero initial state and cell,
//! c = s(f) c' + s(i) tanh(g), h = s(o) tanh(c); a reverse direction reads
//! the sequence from its last row. A bidirectional layer writes each row
//! as [forward state, backward state], as the core does.

use crate::real::Real;

/// One direction's weights.
#[derive(Debug, Clone, Copy)]
pub struct Dir<'a, R> {
    pub w: &'a [R],
    pub u: &'a [R],
    pub b: &'a [R],
    pub reverse: bool,
}

/// One direction's gradients, added to.
pub struct Grad<'a, R> {
    pub w: &'a mut [R],
    pub u: &'a mut [R],
    pub b: &'a mut [R],
}

/// What backward needs from forward, by step (not by row).
#[derive(Debug, Clone, Default)]
pub struct Cache<R> {
    /// [n, 4h]: s(i), s(f), tanh(g), s(o).
    gates: Vec<R>,
    /// [n, h] cells and states.
    cell: Vec<R>,
    state: Vec<R>,
}

fn row(reverse: bool, n: usize, t: usize) -> usize {
    if reverse { n - 1 - t } else { t }
}

/// Runs `d` over x [n, inp], writing each row's state to
/// `out[row * stride + offset..][..h]`.
#[allow(clippy::too_many_arguments)]
pub fn forward<R: Real>(
    d: Dir<R>,
    x: &[R],
    inp: usize,
    h: usize,
    out: &mut [R],
    stride: usize,
    offset: usize,
    cache: &mut Cache<R>,
) {
    let n = x.len() / inp;
    let g4 = 4 * h;
    cache.gates.clear();
    cache.gates.resize(n * g4, R::ZERO);
    cache.cell.clear();
    cache.cell.resize(n * h, R::ZERO);
    cache.state.clear();
    cache.state.resize(n * h, R::ZERO);
    let mut a = vec![R::ZERO; g4];
    for t in 0..n {
        let xr = &x[row(d.reverse, n, t) * inp..][..inp];
        for (k, ak) in a.iter_mut().enumerate() {
            let mut s = d.b[k];
            for (wv, xv) in d.w[k * inp..][..inp].iter().zip(xr) {
                s += *wv * *xv;
            }
            if t > 0 {
                for (uv, hv) in d.u[k * h..][..h].iter().zip(&cache.state[(t - 1) * h..][..h]) {
                    s += *uv * *hv;
                }
            }
            *ak = s;
        }
        for j in 0..h {
            let (i, f, g, o) = (a[j].sigmoid(), a[h + j].sigmoid(), a[2 * h + j].tanh(), a[3 * h + j].sigmoid());
            let prev = if t > 0 { cache.cell[(t - 1) * h + j] } else { R::ZERO };
            let c = f * prev + i * g;
            let gs = &mut cache.gates[t * g4..][..g4];
            (gs[j], gs[h + j], gs[2 * h + j], gs[3 * h + j]) = (i, f, g, o);
            cache.cell[t * h + j] = c;
            cache.state[t * h + j] = o * c.tanh();
        }
        let r = row(d.reverse, n, t);
        out[r * stride + offset..][..h].copy_from_slice(&cache.state[t * h..][..h]);
    }
}

/// Adds the gradients of [`forward`] to `grad` and to `dx` [n, inp], from
/// `dout`, the gradient of the loss with respect to `out`.
#[allow(clippy::too_many_arguments)]
pub fn backward<R: Real>(
    d: Dir<R>,
    x: &[R],
    inp: usize,
    h: usize,
    dout: &[R],
    stride: usize,
    offset: usize,
    cache: &Cache<R>,
    grad: Grad<R>,
    dx: &mut [R],
) {
    let n = x.len() / inp;
    let g4 = 4 * h;
    let mut dh_next = vec![R::ZERO; h];
    let mut dc_next = vec![R::ZERO; h];
    let mut da = vec![R::ZERO; g4];
    for t in (0..n).rev() {
        let r = row(d.reverse, n, t);
        let gs = &cache.gates[t * g4..][..g4];
        for j in 0..h {
            let dh = dout[r * stride + offset + j] + dh_next[j];
            let c = cache.cell[t * h + j];
            let tc = c.tanh();
            let (i, f, g, o) = (gs[j], gs[h + j], gs[2 * h + j], gs[3 * h + j]);
            let dc = dc_next[j] + dh * o * (R::ONE - tc * tc);
            let prev = if t > 0 { cache.cell[(t - 1) * h + j] } else { R::ZERO };
            da[j] = dc * g * i * (R::ONE - i);
            da[h + j] = dc * prev * f * (R::ONE - f);
            da[2 * h + j] = dc * i * (R::ONE - g * g);
            da[3 * h + j] = dh * tc * o * (R::ONE - o);
            dc_next[j] = dc * f;
        }
        let xr = &x[r * inp..][..inp];
        let dxr = &mut dx[r * inp..][..inp];
        dh_next.fill(R::ZERO);
        for (k, &g) in da.iter().enumerate() {
            grad.b[k] += g;
            for (dw, xv) in grad.w[k * inp..][..inp].iter_mut().zip(xr) {
                *dw += g * *xv;
            }
            for (dxv, wv) in dxr.iter_mut().zip(&d.w[k * inp..][..inp]) {
                *dxv += g * *wv;
            }
            if t > 0 {
                let prev = &cache.state[(t - 1) * h..][..h];
                for ((du, uv), (hv, dn)) in
                    grad.u[k * h..][..h].iter_mut().zip(&d.u[k * h..][..h]).zip(prev.iter().zip(dh_next.iter_mut()))
                {
                    *du += g * *hv;
                    *dn += g * *uv;
                }
            }
        }
    }
}
