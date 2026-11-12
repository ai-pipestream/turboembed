//! DeBERTa-v2 disentangled attention over one fully visible sequence.

use crate::real::{Real, dot};

#[derive(Debug, Default, Clone)]
pub struct Cache<R> {
    row: Vec<R>,
    n: usize,
}

pub fn bucket(distance: isize, buckets: usize, maximum: usize) -> usize {
    let mid = buckets / 2;
    let abs = distance.unsigned_abs();
    let magnitude = if abs <= mid {
        abs
    } else {
        (((abs as f32 / mid as f32).ln() / ((maximum - 1) as f32 / mid as f32).ln() * (mid - 1) as f32).ceil() as usize)
            + mid
    };
    let signed = if distance < 0 { -(magnitude as isize) } else { magnitude as isize };
    (signed + buckets as isize).clamp(0, (2 * buckets - 1) as isize) as usize
}

fn p2c<R: Real>(qkv: &[R], pos: &[R], h: usize, head: usize, d: usize, rows: usize) -> Vec<R> {
    let n = qkv.len() / (3 * h);
    let mut out = vec![R::ZERO; n * rows];
    for j in 0..n {
        let key = &qkv[j * 3 * h + h + head * d..j * 3 * h + h + (head + 1) * d];
        for r in 0..rows {
            let query = &pos[r * 2 * h + head * d..r * 2 * h + (head + 1) * d];
            out[j * rows + r] = dot(key, query);
        }
    }
    out
}

fn c2p<R: Real>(qkv: &[R], pos: &[R], h: usize, i: usize, head: usize, d: usize, rows: usize) -> Vec<R> {
    let query = &qkv[i * 3 * h + head * d..i * 3 * h + (head + 1) * d];
    (0..rows).map(|r| dot(query, &pos[r * 2 * h + h + head * d..r * 2 * h + h + (head + 1) * d])).collect()
}

#[allow(clippy::too_many_arguments)]
fn probabilities<R: Real>(
    qkv: &[R],
    p2c: &[R],
    c2p: &[R],
    h: usize,
    i: usize,
    head: usize,
    d: usize,
    buckets: usize,
    maximum: usize,
    row: &mut [R],
) {
    let n = row.len();
    let rows = 2 * buckets;
    let query = &qkv[i * 3 * h + head * d..i * 3 * h + (head + 1) * d];
    let scale = R::from_f64(1.0 / (3.0 * d as f64).sqrt());
    let mut max = R::from_f64(f64::NEG_INFINITY);
    for j in 0..n {
        let key = &qkv[j * 3 * h + h + head * d..j * 3 * h + h + (head + 1) * d];
        let r = bucket(i as isize - j as isize, buckets, maximum);
        let score = (dot(query, key) + c2p[r] + p2c[j * rows + r]) * scale;
        row[j] = score;
        if score > max {
            max = score;
        }
    }
    let mut sum = R::ZERO;
    for score in row.iter_mut() {
        *score = (*score - max).exp();
        sum += *score;
    }
    let inverse = R::ONE / sum;
    for score in row {
        *score *= inverse;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn forward<R: Real>(
    qkv: &[R],
    pos: &[R],
    hidden: usize,
    heads: usize,
    buckets: usize,
    maximum: usize,
    ctx: &mut [R],
    cache: &mut Cache<R>,
) {
    let n = qkv.len() / (3 * hidden);
    let d = hidden / heads;
    cache.n = n;
    cache.row.resize(n, R::ZERO);
    ctx.fill(R::ZERO);
    for head in 0..heads {
        let p2c = p2c(qkv, pos, hidden, head, d, 2 * buckets);
        for i in 0..n {
            let c2p = c2p(qkv, pos, hidden, i, head, d, 2 * buckets);
            probabilities(qkv, &p2c, &c2p, hidden, i, head, d, buckets, maximum, &mut cache.row);
            let out = &mut ctx[i * hidden + head * d..i * hidden + (head + 1) * d];
            for j in 0..n {
                let value = &qkv[j * 3 * hidden + 2 * hidden + head * d..j * 3 * hidden + 2 * hidden + (head + 1) * d];
                for (o, &v) in out.iter_mut().zip(value) {
                    *o += cache.row[j] * v;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn backward<R: Real>(
    qkv: &[R],
    pos: &[R],
    hidden: usize,
    heads: usize,
    buckets: usize,
    maximum: usize,
    cache: &Cache<R>,
    dctx: &[R],
    dqkv: &mut [R],
    dpos: &mut [R],
) {
    let n = cache.n;
    let d = hidden / heads;
    let mut row = vec![R::ZERO; n];
    let mut dp = vec![R::ZERO; n];
    for head in 0..heads {
        let p2c = p2c(qkv, pos, hidden, head, d, 2 * buckets);
        for i in 0..n {
            let c2p = c2p(qkv, pos, hidden, i, head, d, 2 * buckets);
            probabilities(qkv, &p2c, &c2p, hidden, i, head, d, buckets, maximum, &mut row);
            let dout = &dctx[i * hidden + head * d..i * hidden + (head + 1) * d];
            let mut sum = R::ZERO;
            for j in 0..n {
                let value = &qkv[j * 3 * hidden + 2 * hidden + head * d..j * 3 * hidden + 2 * hidden + (head + 1) * d];
                dp[j] = dot(dout, value);
                sum += row[j] * dp[j];
                for (dv, &g) in dqkv
                    [j * 3 * hidden + 2 * hidden + head * d..j * 3 * hidden + 2 * hidden + (head + 1) * d]
                    .iter_mut()
                    .zip(dout)
                {
                    *dv += row[j] * g;
                }
            }
            let query = &qkv[i * 3 * hidden + head * d..i * 3 * hidden + (head + 1) * d];
            for j in 0..n {
                let key = &qkv[j * 3 * hidden + hidden + head * d..j * 3 * hidden + hidden + (head + 1) * d];
                let r = bucket(i as isize - j as isize, buckets, maximum);
                let ds = row[j] * (dp[j] - sum) * R::from_f64(1.0 / (3.0 * d as f64).sqrt());
                let pos_query = &pos[r * 2 * hidden + head * d..r * 2 * hidden + (head + 1) * d];
                let pos_key = &pos[r * 2 * hidden + hidden + head * d..r * 2 * hidden + hidden + (head + 1) * d];
                for o in 0..d {
                    dqkv[i * 3 * hidden + head * d + o] += ds * (key[o] + pos_key[o]);
                    dqkv[j * 3 * hidden + hidden + head * d + o] += ds * (query[o] + pos_query[o]);
                    dpos[r * 2 * hidden + head * d + o] += ds * key[o];
                    dpos[r * 2 * hidden + hidden + head * d + o] += ds * query[o];
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "deberta_tests.rs"]
mod tests;
