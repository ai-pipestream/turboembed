//! All eigenpairs of a real symmetric matrix by Householder reduction to
//! tridiagonal form and the implicit QL method, after the public-domain
//! JAMA `EigenvalueDecomposition` (tred2 and tql2). It is O(n^3) with a
//! small constant: at the 512-row cap it replaces cyclic Jacobi sweeps
//! that took seconds. Every loop runs in a fixed order and only `+`, `*`,
//! `/` and `sqrt` are used, so the result is the same bits on any machine.

/// `sqrt(a^2 + b^2)` without `hypot`, whose last bit can differ between
/// C libraries.
fn hyp(a: f64, b: f64) -> f64 {
    (a * a + b * b).sqrt()
}

/// Eigenvalues and eigenvectors (as columns, `v[row * n + col]`) of the
/// symmetric `n x n` matrix `a`, in no particular order. `a` is read only
/// through its lower triangle and left as it was.
pub(super) fn symmetric_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let mut v: Vec<f64> = a.to_vec();
    let mut d = vec![0f64; n];
    let mut e = vec![0f64; n];
    tred2(&mut v, &mut d, &mut e, n);
    tql2(&mut v, &mut d, &mut e, n);
    (d, v)
}

/// Householder reduction to tridiagonal form: `d` the diagonal, `e` the
/// subdiagonal, `v` the accumulated transformation.
fn tred2(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |i: usize, j: usize| i * n + j;
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
    }
    for i in (1..n).rev() {
        let mut scale = 0f64;
        let mut h = 0f64;
        for x in &d[..i] {
            scale += x.abs();
        }
        if scale == 0.0 {
            e[i] = d[i - 1];
            for j in 0..i {
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
                v[at(j, i)] = 0.0;
            }
        } else {
            for x in &mut d[..i] {
                *x /= scale;
                h += *x * *x;
            }
            let f = d[i - 1];
            let mut g = h.sqrt();
            if f > 0.0 {
                g = -g;
            }
            e[i] = scale * g;
            h -= f * g;
            d[i - 1] = f - g;
            for x in &mut e[..i] {
                *x = 0.0;
            }
            for j in 0..i {
                let f = d[j];
                v[at(j, i)] = f;
                let mut g = e[j] + v[at(j, j)] * f;
                for k in j + 1..i {
                    g += v[at(k, j)] * d[k];
                    e[k] += v[at(k, j)] * f;
                }
                e[j] = g;
            }
            let mut f = 0f64;
            for j in 0..i {
                e[j] /= h;
                f += e[j] * d[j];
            }
            let hh = f / (h + h);
            for j in 0..i {
                e[j] -= hh * d[j];
            }
            for j in 0..i {
                let (f, g) = (d[j], e[j]);
                for k in j..i {
                    v[at(k, j)] -= f * e[k] + g * d[k];
                }
                d[j] = v[at(i - 1, j)];
                v[at(i, j)] = 0.0;
            }
        }
        d[i] = h;
    }
    for i in 0..n - 1 {
        v[at(n - 1, i)] = v[at(i, i)];
        v[at(i, i)] = 1.0;
        let h = d[i + 1];
        if h != 0.0 {
            for k in 0..=i {
                d[k] = v[at(k, i + 1)] / h;
            }
            for j in 0..=i {
                let mut g = 0f64;
                for k in 0..=i {
                    g += v[at(k, i + 1)] * v[at(k, j)];
                }
                for k in 0..=i {
                    v[at(k, j)] -= g * d[k];
                }
            }
        }
        for k in 0..=i {
            v[at(k, i + 1)] = 0.0;
        }
    }
    for j in 0..n {
        d[j] = v[at(n - 1, j)];
        v[at(n - 1, j)] = 0.0;
    }
    v[at(n - 1, n - 1)] = 1.0;
    e[0] = 0.0;
}

/// Implicit QL on the tridiagonal matrix, accumulating into `v`.
fn tql2(v: &mut [f64], d: &mut [f64], e: &mut [f64], n: usize) {
    let at = |i: usize, j: usize| i * n + j;
    for i in 1..n {
        e[i - 1] = e[i];
    }
    e[n - 1] = 0.0;
    let mut f = 0f64;
    let mut tst1 = 0f64;
    let eps = f64::EPSILON;
    for l in 0..n {
        tst1 = tst1.max(d[l].abs() + e[l].abs());
        let mut m = l;
        while m < n - 1 && e[m].abs() > eps * tst1 {
            m += 1;
        }
        if m > l {
            // Converges in a few iterations; the cap only guards a
            // non-finite input from looping forever.
            for _ in 0..64 {
                let g = d[l];
                let mut p = (d[l + 1] - g) / (2.0 * e[l]);
                let mut r = hyp(p, 1.0);
                if p < 0.0 {
                    r = -r;
                }
                d[l] = e[l] / (p + r);
                d[l + 1] = e[l] * (p + r);
                let dl1 = d[l + 1];
                let mut h = g - d[l];
                for x in &mut d[l + 2..n] {
                    *x -= h;
                }
                f += h;
                p = d[m];
                let (mut c, mut c2, mut c3) = (1f64, 1f64, 1f64);
                let el1 = e[l + 1];
                let (mut s, mut s2) = (0f64, 0f64);
                for i in (l..m).rev() {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    let g = c * e[i];
                    h = c * p;
                    r = hyp(p, e[i]);
                    e[i + 1] = s * r;
                    s = e[i] / r;
                    c = p / r;
                    p = c * d[i] - s * g;
                    d[i + 1] = h + s * (c * g + s * d[i]);
                    for k in 0..n {
                        let h = v[at(k, i + 1)];
                        v[at(k, i + 1)] = s * v[at(k, i)] + c * h;
                        v[at(k, i)] = c * v[at(k, i)] - s * h;
                    }
                }
                p = -s * s2 * c3 * el1 * e[l] / dl1;
                e[l] = s * p;
                d[l] = c * p;
                if e[l].abs() <= eps * tst1 {
                    break;
                }
            }
        }
        d[l] += f;
        e[l] = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::symmetric_eigen;

    #[test]
    fn eigenpairs_satisfy_a_v_equals_lambda_v_and_are_orthonormal() {
        for n in [1usize, 2, 3, 7, 40] {
            let mut x = 0x2545_f491_4f6c_dd1du64;
            let mut a = vec![0f64; n * n];
            for i in 0..n {
                for j in 0..=i {
                    x ^= x << 13;
                    x ^= x >> 7;
                    x ^= x << 17;
                    let val = (x % 2001) as f64 / 1000.0 - 1.0;
                    a[i * n + j] = val;
                    a[j * n + i] = val;
                }
            }
            let (values, v) = symmetric_eigen(&a, n);
            for c in 0..n {
                for r in 0..n {
                    let av: f64 = (0..n).map(|k| a[r * n + k] * v[k * n + c]).sum();
                    assert!((av - values[c] * v[r * n + c]).abs() < 1e-9, "n={n} col {c}");
                }
                for c2 in 0..n {
                    let dot: f64 = (0..n).map(|k| v[k * n + c] * v[k * n + c2]).sum();
                    let want = if c == c2 { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-9, "n={n} cols {c},{c2}");
                }
            }
        }
    }

    #[test]
    fn a_diagonal_and_a_zero_matrix_come_back_unchanged() {
        let (values, _) = symmetric_eigen(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], 3);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert_eq!(sorted, [1.0, 2.0, 3.0]);
        let (values, _) = symmetric_eigen(&[0.0; 4], 2);
        assert_eq!(values, [0.0, 0.0]);
    }
}
