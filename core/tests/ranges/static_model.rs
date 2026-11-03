//! The same write on a static model.

use std::fs;
use std::path::Path;

use crate::common::*;
use crate::support::*;
use turbo::safetensors::{Dtype, File};
use turbo::*;

/// The static bundle's table, token weights and token mapping, widened.
struct Table {
    values: Vec<f64>,
    weights: Option<Vec<f64>>,
    mapping: Option<Vec<usize>>,
    dim: usize,
}

impl Table {
    fn read(dir: &Path) -> Table {
        let bytes = fs::read(dir.join("weights/static.safetensors")).unwrap();
        let f = File::parse("static.safetensors", &bytes).unwrap();
        let wide = |name: &str| -> Option<(Vec<f64>, Vec<u64>)> {
            let t = f.tensor(name)?;
            let v = match t.dtype {
                Dtype::F16 => t.data.chunks(2).map(|c| half_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64).collect(),
                Dtype::F32 => t.f32s().into_iter().map(f64::from).collect(),
                Dtype::F64 => t.data.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
                Dtype::I64 => t.data.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f64).collect(),
                other => panic!("{other:?}"),
            };
            Some((v, t.shape.clone()))
        };
        let (values, shape) = wide("embeddings").unwrap();
        Table {
            values,
            weights: wide("weights").map(|w| w.0),
            mapping: wide("mapping").map(|m| m.0.iter().map(|&v| v as usize).collect()),
            dim: shape[1] as usize,
        }
    }

    /// The vector token `id` contributes: its table row times its weight.
    fn token(&self, id: i32) -> Vec<f64> {
        let id = id as usize;
        let row = self.mapping.as_ref().map_or(id, |m| m[id]);
        let w = self.weights.as_ref().map_or(1.0, |w| w[id]);
        self.values[row * self.dim..(row + 1) * self.dim].iter().map(|v| v * w).collect()
    }
}

/// An F16 value widened; the table of the bundle holds normal numbers and 0.
fn half_to_f32(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 {
        let v = man as f32 * 2f32.powi(-24);
        return if sign != 0 { -v } else { v };
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

#[test]
fn a_static_models_range_is_the_mean_of_its_tokens_rows() {
    let dir = testdata().join("tiny-static-bundle");
    let table = Table::read(&dir);
    let l = Loaded::load(&dir).unwrap_or_else(|e| panic!("{e:?}"));
    let mut d = session_desc(0, 0, TURBO_PRECISION_MODEL);
    d.max_ranges = 64;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let texts = ["The quick brown fox jumps over the lazy dog.", "how do I reset a password", "a"];
    let (t, lens) = tokens_of(&dir, &texts);
    assert!(lens.iter().all(|&n| n >= 1));
    let o = normalized(TURBO_NORMALIZE_NONE);

    s.write_tokens(&t.batch(), Some(&o)).unwrap();
    let rows = s.run().unwrap().rows();
    let whole: Vec<_> = (0..3).map(|r| range(r, 0, t.seq)).collect();
    let got = run_ranges(&s, &t, &whole, &o);
    for (r, (g, w)) in got.iter().zip(&rows).enumerate() {
        assert_eq!(g.len(), table.dim);
        assert!(max_abs_diff(g, w) <= 1e-6, "row {r}");
    }

    // A range of one token is that token's table row, times its weight.
    let mut singles = Vec::new();
    for (r, &n) in lens.iter().enumerate() {
        for p in 0..n {
            singles.push((r, p, range(r as u32, p as u32, p as u32 + 1)));
        }
    }
    assert!(singles.len() <= 64);
    let ranges: Vec<_> = singles.iter().map(|s| s.2).collect();
    let got = run_ranges(&s, &t, &ranges, &o);
    for ((r, p, _), g) in singles.iter().zip(&got) {
        let want = table.token(t.ids[r * t.seq as usize + p]);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() <= 1e-6, "row {r} token {p}: {a} vs {b}");
        }
    }
}
