//! Models and sessions: shared contexts, F16 and BF16 weights, threads, output_dim and failed appends.

use super::*;

/// A BERT of hidden width 32, whose linear layers run on the matrix
/// engines at FASTEST, with LayerNorm epsilon `eps`.
fn wide_enough(name: &str, eps: f64, seed: u32) -> Fixture {
    let mut m = model_manifest();
    m["architecture"]["hidden"] = json!(32);
    m["architecture"]["intermediate"] = json!(64);
    m["architecture"]["layer_norm_eps"] = json!(eps);
    m["embed"]["dim"] = json!(32);
    let mut f = Fixture::new(name, m);
    f.weights("weights/model.safetensors", &bert_weights(2, 32, 64, seed, 1.0));
    f
}

/// Two models of one width whose LayerNorms differ only in epsilon, on one
/// context, each at FASTEST after the other at the same token counts:
/// each matches the CPU, so neither takes the other's LayerNorm, which on
/// oneDNN the context keeps for both.
#[test]
fn models_of_one_width_and_different_epsilon_share_a_context() {
    let _t = turn();
    let Some(_) = gpu_device("models_of_one_width_and_different_epsilon_share_a_context") else { return };
    let (fa, fb) = (wide_enough("levelzero-eps-a", 1e-12, 3), wide_enough("levelzero-eps-b", 100.0, 3));
    let a = fa.load_on(gpu).unwrap();
    fb.write();
    let mut mb = ptr::null_mut();
    let mut err = new_error();
    let rc = unsafe { turbo_model_load(a.ctx, text(fb.dir.to_str().unwrap()), &mut mb, &mut err) };
    assert_eq!(rc, 0, "{:?}", failure(rc, &err));
    let b = Loaded::model(mb);
    let (ca, cb) = (fa.load().unwrap(), fb.load().unwrap());
    // 4 rows of 20 tokens: past the few-token kernels.
    let rows: Vec<Vec<i32>> = (0..4i32).map(|r| (0..20).map(|p| 1000 + r * 131 + p * 17).collect()).collect();
    let t = Tokens::new(&rows, 0);
    let embed = |m: *mut turbo_model, precision: u32| {
        let s = Session::create(m, Some(&session_desc(0, 0, precision))).unwrap();
        s.write_tokens(&t.batch(), None).unwrap();
        (s.run().unwrap().rows(), s.info().compute_dtype)
    };
    let (want_a, want_b) = (embed(ca.m, TURBO_PRECISION_EXACT).0, embed(cb.m, TURBO_PRECISION_EXACT).0);
    let apart = want_a.iter().zip(&want_b).map(|(x, y)| cosine(x, y)).fold(1f64, f64::min);
    assert!(apart < 0.99, "the two epsilons give vectors this test tells apart: cosine {apart}");
    for (name, m, want) in [("1e-12", a.m, &want_a), ("100", b.m, &want_b), ("1e-12", a.m, &want_a)] {
        let (got, dtype) = embed(m, TURBO_PRECISION_FASTEST);
        assert_eq!(dtype, TURBO_DTYPE_F16);
        for (r, (g, w)) in got.iter().zip(want).enumerate() {
            let cos = cosine(g, w);
            assert!(cos >= 0.999, "epsilon {name} row {r}: cosine {cos} with the cpu");
        }
    }
}

/// Sessions at FASTEST made at once on a new model make its F16 weights,
/// and oneDNN's packed copy, once: when the sessions and the model are
/// released, every allocation the backend made for them is freed.
#[test]
fn fastest_sessions_made_at_once_leave_nothing_allocated() {
    let _t = turn();
    let Some(_) = gpu_device("fastest_sessions_made_at_once_leave_nothing_allocated") else { return };
    for round in 0..8 {
        let live = turbo::levelzero::live_allocations();
        {
            let l = on_gpu(&tiny_bundle());
            let m = l.m as usize;
            let go = std::sync::Barrier::new(2);
            std::thread::scope(|scope| {
                let made: Vec<_> = (0..2)
                    .map(|_| {
                        scope.spawn(|| {
                            go.wait();
                            let s = Session::create(
                                m as *mut turbo_model,
                                Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST)),
                            );
                            let s = s.unwrap();
                            assert_eq!(s.info().compute_dtype, TURBO_DTYPE_F16);
                            s.embed(&TEXTS, None).unwrap()
                        })
                    })
                    .collect();
                let got: Vec<_> = made.into_iter().map(|h| h.join().unwrap()).collect();
                assert_eq!(got[0], got[1], "round {round}");
            });
        }
        assert_eq!(turbo::levelzero::live_allocations(), live, "round {round}: allocations left after the release");
    }
}

/// A model stored in F16 or BF16: MODEL is refused, EXACT and FASTEST
/// share one F32 copy on the device, and give the vectors of the same
/// values stored as F32.
#[test]
fn half_weights_compute_in_f32_from_one_shared_copy() {
    let _t = turn();
    let Some(_) = gpu_device("half_weights_compute_in_f32_from_one_shared_copy") else { return };
    for (dtype, narrow) in [("F16", to_f16 as fn(f32) -> u16), ("BF16", to_bf16)] {
        let (n, w): (Vec<Tensor>, Vec<Tensor>) = tiny_weights(0)
            .into_iter()
            .map(|t| {
                let halves: Vec<u16> =
                    t.data.chunks(4).map(|c| narrow(f32::from_le_bytes(c.try_into().unwrap()))).collect();
                let wide = |h: u16| if dtype == "F16" { from_f16(h) } else { f32::from_bits((h as u32) << 16) };
                let nt = Tensor {
                    name: t.name.clone(),
                    dtype,
                    shape: t.shape.clone(),
                    data: halves.iter().flat_map(|h| h.to_le_bytes()).collect(),
                };
                let wt = Tensor { data: halves.iter().flat_map(|&h| wide(h).to_le_bytes()).collect(), ..t };
                (nt, wt)
            })
            .unzip();
        let mut f = Fixture::model(&format!("levelzero-half-{dtype}"));
        f.weights("weights/model.safetensors", &n);
        let l = f.load_on(gpu).unwrap();
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");
        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copy, "one copy, shared");
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F32);

        let mut g = Fixture::model(&format!("levelzero-half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &w);
        let lw = g.load_on(gpu).unwrap();
        let want = Session::create(lw.m, None).unwrap().embed(&TEXTS, None).unwrap();
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert_eq!(b.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are used as loaded");

        // The same F16 or BF16 bundle on the CPU, which widens it too.
        let lc = f.load().unwrap();
        let cpu = Session::create(lc.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        for (r, (g, c)) in want.iter().zip(cpu.embed(&TEXTS, None).unwrap()).enumerate() {
            let cos = cosine(g, &c);
            assert!(cos >= 0.9999, "{dtype} row {r}: cosine {cos} with the cpu");
        }
    }
}

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

/// Sessions on one context, run from several threads at once, each get
/// their own vectors: the context's queue is shared under its lock.
#[test]
fn sessions_on_one_context_run_from_many_threads() {
    let _t = turn();
    let Some(_) = gpu_device("sessions_on_one_context_run_from_many_threads") else { return };
    let l = on_gpu(&tiny_bundle());
    let want: Vec<Vec<Vec<f32>>> = {
        let s = Session::create(l.m, None).unwrap();
        TEXTS.iter().map(|t| s.embed(&[t], None).unwrap()).collect()
    };
    let m = l.m as usize;
    std::thread::scope(|scope| {
        for (i, text) in TEXTS.iter().enumerate() {
            let want = &want[i];
            scope.spawn(move || {
                let s = Session::create(m as *mut turbo_model, None).unwrap();
                for _ in 0..20 {
                    assert_eq!(&s.embed(&[text], None).unwrap(), want, "text {i}");
                }
            });
        }
    });
}

/// An output_dim the bundle lists is cut, then normalized, on the device.
#[test]
fn an_output_dim_is_cut_then_normalized_on_the_device() {
    let _t = turn();
    let Some(_) = gpu_device("an_output_dim_is_cut_then_normalized_on_the_device") else { return };
    let mut f = Fixture::new("levelzero-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(gpu).unwrap();
    let s = Session::create(l.m, None).unwrap();
    let got = s.embed(&TEXTS, Some(&opts(|o| o.output_dim = 4))).unwrap();
    let plain = PlainBert::new(&f.dir);
    let tok = Tok::create(&f.dir).unwrap();
    for (t, g) in TEXTS.iter().zip(&got) {
        assert_eq!(g.len(), 4);
        let ids = tok.row(t, None).unwrap();
        let want = plain.embed(&ids, &vec![1; ids.len()], &vec![0; ids.len()], TURBO_POOLING_MEAN, 4, true);
        for (a, b) in g.iter().zip(&want) {
            assert!((*a as f64 - b).abs() < 1e-5, "{a} vs {b}");
        }
    }
}

/// After an append the driver refuses, its immediate list never finishes.
/// The call that saw the failure waits for the work appended before it,
/// which arrives whole, and the context gets a queue that works.
#[test]
fn a_failed_append_waits_for_what_came_before_and_replaces_the_queue() {
    let _t = turn();
    let Some(_) = gpu_device("a_failed_append_waits_for_what_came_before_and_replaces_the_queue") else { return };
    turbo::levelzero::append_failure_recovers().unwrap();
}
