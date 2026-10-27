//! Output widths and weight dtypes, on the small BERT of tests/common.

use crate::common::*;
use crate::support::*;
use serde_json::json;
use turbo::status::*;
use turbo::*;

#[test]
fn an_output_dim_the_bundle_lists_is_cut_then_normalized() {
    let mut f = Fixture::new("output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load().unwrap();
    let s = session(&l);
    let e = s.write_text(&["a"], Some(&opts(|o| o.output_dim = 3))).err().unwrap();
    assert!(e.is(UNSUPPORTED_OPTION, "cut to [4]"), "{e:?}");
    let o = opts(|o| o.output_dim = 4);
    let got = s.embed(&TEXTS, Some(&o)).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        assert_eq!(g.len(), 4);
        let ids = tok.row(t, None).unwrap();
        let ones = vec![1; ids.len()];
        let want = plain.embed(&ids, &ones, &vec![0; ids.len()], TURBO_POOLING_MEAN, 4, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
    let r = {
        s.write_text(&TEXTS, Some(&o)).unwrap();
        s.run().unwrap()
    };
    assert_eq!((r.info().dim, r.info().bytes), (4, 3 * 4 * 4));
}

/// f32 to the nearest half, ties to even; for the small BERT's values,
/// which are normal halves or 0.
fn to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if x == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    assert!((1..31).contains(&exp), "{x} is not a normal half");
    let man = b & 0x7f_ffff;
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rest = man & 0x1fff;
    if rest > 0x1000 || (rest == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

fn from_f16(h: u16) -> f32 {
    let (sign, exp, man) = ((h as u32 & 0x8000) << 16, (h >> 10) & 0x1f, h as u32 & 0x3ff);
    if exp == 0 && man == 0 {
        return f32::from_bits(sign);
    }
    f32::from_bits(sign | ((exp as u32 + 112) << 23) | (man << 13))
}

fn to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

/// The small BERT stored as `dtype`, and the same values widened to F32.
fn narrowed(dtype: &'static str) -> (Vec<Tensor>, Vec<Tensor>) {
    type Narrow = fn(f32) -> u16;
    type Widen = fn(u16) -> f32;
    let (narrow, wide): (Narrow, Widen) = match dtype {
        "F16" => (to_f16, from_f16),
        _ => (to_bf16, |h| f32::from_bits((h as u32) << 16)),
    };
    tiny_weights(0)
        .into_iter()
        .map(|t| {
            let halves: Vec<u16> =
                t.data.chunks(4).map(|c| narrow(f32::from_le_bytes(c.try_into().unwrap()))).collect();
            let n = Tensor {
                name: t.name.clone(),
                dtype,
                shape: t.shape.clone(),
                data: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
            };
            let w = Tensor { data: halves.iter().flat_map(|&h| wide(h).to_le_bytes()).collect(), ..t };
            (n, w)
        })
        .unzip()
}

#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    for dtype in ["F16", "BF16"] {
        let (narrow, wide) = narrowed(dtype);
        let mut f = Fixture::model(&format!("half-{dtype}"));
        f.weights("weights/model.safetensors", &narrow);
        let l = f.load().unwrap();
        assert_ne!(l.info().dtype, TURBO_DTYPE_F32);
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(e.message.contains("computes in F32 only"), "{e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");

        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        assert_eq!(a.info().compute_dtype, TURBO_DTYPE_F32);
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");

        // The same values stored as F32 give the same vectors, bit for bit.
        let mut g = Fixture::model(&format!("half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &wide);
        let lw = g.load().unwrap();
        let want = session(&lw).embed(&TEXTS, None).unwrap();
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert_eq!(b.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are read in place");
    }
}
