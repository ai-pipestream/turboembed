//! CUDA runs: every option against the f64 arithmetic of tests/common and
//! the CPU backend's vectors, a run's vectors left on the device and the
//! bytes it moves, pinned rows, a warm run's allocation count, half weights,
//! threads, and output dims.
//!
//! Built with the `cuda` feature only. A test that needs a device says it
//! was skipped, and passes, when the backend lists none; nothing is run
//! on anything else in its place. With TURBO_TEST_REQUIRE_CUDA=1 it fails
//! instead, so a run on a GPU machine cannot pass by finding no GPU.
//! docs/cuda.md says how to run them; tests/cuda_support/ holds what the
//! files share.

#![cfg(feature = "cuda")]

mod common;
#[path = "cuda_support/mod.rs"]
mod cuda_support;

use std::ptr;

use common::*;
use cuda_support::*;
use serde_json::json;
use turbo::status::{UNSUPPORTED, UNSUPPORTED_OPTION};
use turbo::*;

// ---- Runs ------------------------------------------------------------------------

/// Every pooling, with and without normalization, against the f64
/// encoder of tests/common and against the CPU backend.
#[test]
fn every_option_matches_the_arithmetic_and_the_cpu() {
    let _t = turn();
    let Some(_) = cuda_device("every_option_matches_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_cuda(&dir), Loaded::load(&dir).unwrap());
    // EXACT: F32 FMAs throughout, which the f64 encoder's bound needs.
    let exact = session_desc(0, 0, TURBO_PRECISION_EXACT);
    let (gs, cs) = (Session::create(g.m, Some(&exact)).unwrap(), Session::create(c.m, None).unwrap());
    let t = awkward_rows(&dir);
    let seq = t.seq as usize;
    let types = t.types.clone().unwrap();
    let (mut worst, mut lowest) = (0f64, 1f64);
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            gs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = gs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let cpu = cs.run().unwrap().rows();
            for (r, row) in got.iter().enumerate() {
                let at = r * seq..(r + 1) * seq;
                let want = plain.embed(
                    &t.ids[at.clone()],
                    &t.mask[at.clone()],
                    &types[at],
                    pooling,
                    32,
                    normalize == TURBO_NORMALIZE_L2,
                );
                for (a, w) in row.iter().zip(&want) {
                    let d = (*a as f64 - w).abs();
                    worst = worst.max(d);
                    assert!(d < 1e-5 * (1.0 + w.abs()), "pooling {pooling} normalize {normalize} row {r}: {a} vs {w}");
                }
                let c = cosine(row, &cpu[r]);
                lowest = lowest.min(c);
                assert!(c >= 0.9999, "pooling {pooling} normalize {normalize} row {r}: cosine {c} with the cpu");
            }
        }
    }
    println!(
        "largest difference from the f64 encoder {worst:.3e}; 1 - lowest cosine with the cpu {:.3e}",
        1.0 - lowest
    );

    // Text, and the same rows as tokens at a wider stride, give the same vectors.
    let from_text = gs.embed(&TEXTS, None).unwrap();
    let tok = Tok::create(&dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let packed = Tokens::new(&rows, 0);
    let seq = packed.seq as usize;
    let mut wide = Tokens::new(&rows, 0);
    wide.stride = seq as u32 + 5;
    wide.ids = vec![-9; rows.len() * wide.stride as usize];
    wide.mask = vec![-9; rows.len() * wide.stride as usize];
    for r in 0..rows.len() {
        let at = r * wide.stride as usize;
        wide.ids[at..at + seq].copy_from_slice(&packed.ids[r * seq..(r + 1) * seq]);
        wide.mask[at..at + seq].copy_from_slice(&packed.mask[r * seq..(r + 1) * seq]);
    }
    gs.write_tokens(&wide.batch(), None).unwrap();
    assert_eq!(gs.run().unwrap().rows(), from_text, "the stride's gap is never read");
}

/// The vectors stay on the device: the result's buffer is device memory
/// whose CUDA pointer holds what turbo_result_read copies back, and the
/// counts are the bytes that crossed.
#[test]
fn a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed() {
    let _t = turn();
    let Some(_) = cuda_device("a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed") else { return };
    let l = on_cuda(&tiny_bundle());
    let mi = l.info();
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let seq = TEXTS.iter().map(|t| tok.row(t, None).unwrap().len()).max().unwrap() as u64;

    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_DEVICE));
    assert_eq!(i.device, cuda(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    assert_eq!(i.h2d_bytes, 2 * 3 * seq * 4, "ids and mask, [3, {seq}] int32 each");
    assert_eq!(i.d2h_bytes, 0, "nothing came back yet");
    assert_eq!((i.host_allocs, i.device_allocs), (0, 0));
    let (h, d, f, u) = (TURBO_STAGE_HOST, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, d, d, d, d, f, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "cuda");
    assert_eq!(field(&i.manifest_sha256), field(&mi.manifest_sha256));

    let read = r.rows().concat();
    assert_eq!(r.info().d2h_bytes, i.bytes, "a read is counted");
    r.rows();
    assert_eq!(r.info().d2h_bytes, 2 * i.bytes, "each time");

    let mut buf = ptr::null_mut();
    assert_eq!(unsafe { turbo_result_buffer(r.0, &mut buf, null_err()) }, 0);
    let buf = Buf(buf);
    let mut bd: turbo_buffer_desc = unsafe { std::mem::zeroed() };
    bd.struct_size = size_of::<turbo_buffer_desc>() as u32;
    assert_eq!(unsafe { turbo_buffer_get_desc(buf.0, &mut bd, null_err()) }, 0);
    assert_eq!((bd.placement, bd.shape, bd.bytes), (TURBO_PLACE_DEVICE, [3, 32], 384));
    assert!(buf.host().unwrap_err().is(UNSUPPORTED, "no host pointer"));
    let h = buf.export(TURBO_HANDLE_CUDA_PTR).unwrap();
    assert_eq!(h.aux, rt_ordinal(&l), "the device's own ordinal");
    assert_eq!(from_device(h.handle, read.len()), read, "the device pointer holds the vectors, no copy made");
    drop(buf);
    drop(r);

    // Tokens the caller wrote, with types: three arrays cross; nothing
    // is tokenized, and an unnormalized vector is not normalized.
    let mut b = Tokens::new(&[vec![101, 7592, 102]], 0);
    b.types = Some(vec![0, 1, 1]);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.h2d_bytes, 3 * 3 * 4);
    assert_eq!(i.stage[..7], [u, d, d, d, d, u, u]);
    assert_eq!((i.batch, i.d2h_bytes), (1, 0), "a run's count starts again");
}

fn rt_ordinal(l: &Loaded) -> u64 {
    let mut info: turbo_device_info = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<turbo_device_info>() as u32;
    assert_eq!(unsafe { turbo_runtime_device_info(l.rt, cuda(l.rt), &mut info, null_err()) }, 0);
    info.ordinal as u64
}

/// Rows in page-locked memory the caller got from a PINNED buffer go to
/// the device from where they are, and give the same vectors.
#[test]
fn rows_in_pinned_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = cuda_device("rows_in_pinned_memory_give_the_same_vectors") else { return };
    let l = on_cuda(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let t = awkward_rows(&tiny_bundle());
    s.write_tokens(&t.batch(), None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(TURBO_PLACE_PINNED, n).unwrap()).collect();
    let at: Vec<*mut i32> = bufs.iter().map(|b| b.host().unwrap() as *mut i32).collect();
    let types = t.types.as_ref().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(t.ids.as_ptr(), at[0], t.ids.len());
        std::ptr::copy_nonoverlapping(t.mask.as_ptr(), at[1], t.mask.len());
        std::ptr::copy_nonoverlapping(types.as_ptr(), at[2], types.len());
    }
    let mut b = t.batch();
    b.ids = at[0];
    b.mask = at[1];
    b.types = at[2];
    s.write_tokens(&b, None).unwrap();
    let r = s.run().unwrap();
    assert_eq!(r.rows(), want);
    assert_eq!(r.info().h2d_bytes, 3 * n * 4);
}

/// A warm run allocates nothing on the host or the device, in F32 and in
/// F16: the backend's own count does not move, the result says 0, and the
/// device's free memory is what it was.
#[test]
fn a_warm_run_allocates_nothing() {
    let _t = turn();
    let Some(_) = cuda_device("a_warm_run_allocates_nothing") else { return };
    for p in [TURBO_PRECISION_MODEL, TURBO_PRECISION_FASTEST] {
        warm_runs_allocate_nothing_at(p);
    }
}

fn warm_runs_allocate_nothing_at(precision: u32) {
    let l = on_cuda(&tiny_bundle());
    let mut d = session_desc(0, 0, precision);
    d.max_ranges = 128;
    let s = Session::create(l.m, Some(&d)).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    // Two ranges on each of the 64 rows: more vectors than rows.
    let halves: Vec<_> = (0..64).flat_map(|r| [range(r, 0, 32), range(r, 32, 64)]).collect();
    let mut dst = vec![0f32; 128 * 32];
    let mut run = |t: &Tokens, ranges: Option<&[turbo_token_range]>| {
        match ranges {
            Some(g) => s.write_ranges(&t.batch(), g, None).unwrap(),
            None => s.write_tokens(&t.batch(), None).unwrap(),
        }
        let r = s.run().unwrap();
        let i = r.info();
        let rc = unsafe {
            turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, (dst.len() * 4) as u64, ptr::null_mut(), null_err())
        };
        assert_eq!(rc, 0);
        (i.host_allocs, i.device_allocs)
    };
    // Cold: the first runs of each shape, and of ranges.
    run(&small, None);
    run(&long, None);
    run(&long, Some(&halves));
    let free = || {
        assert_eq!(unsafe { cudaSetDevice(rt_ordinal(&l) as i32) }, 0);
        let (mut f, mut t) = (0usize, 0usize);
        assert_eq!(unsafe { cudaMemGetInfo(&mut f, &mut t) }, 0);
        f
    };
    // What this process holds on the device, when NVML reports it: the
    // device's free memory also moves with every other process on a
    // shared GPU.
    let mut bus = [0u8; 32];
    assert_eq!(unsafe { cudaDeviceGetPCIBusId(bus.as_mut_ptr().cast(), 32, rt_ordinal(&l) as i32) }, 0);
    let nvml = nvml::Nvml::open(&bus);
    let own = || nvml.as_ref().and_then(nvml::Nvml::own_bytes);
    let (before, free_before, own_before) = (turbo::cuda::allocations(), free(), own());
    let warm = [(&small, None), (&long, None), (&long, Some(&halves[..])), (&small, None), (&long, Some(&halves[..]))];
    for (i, (t, g)) in warm.into_iter().enumerate() {
        assert_eq!(run(t, g), (0, 0), "precision {precision} warm run {i}: the result's count");
    }
    assert_eq!(turbo::cuda::allocations(), before, "precision {precision}: the backend allocated nothing in warm runs");
    // The count above is exact for the backend's own allocations. This
    // catches what it cannot count, cuBLAS or the driver growing a pool:
    // the driver reserves device memory in pages of 2 MiB, so the memory
    // moves in those steps when it moves.
    let page = 2usize << 20;
    match (own_before, own()) {
        (Some(was), Some(now)) => {
            println!("this process's device memory before the warm runs {was}, after {now}");
            assert!(now < was + page as u64, "this process's device memory grew by {} bytes", now - was);
            // The count sees this process's own allocations: 8 MiB more
            // shows.
            let mut p = ptr::null_mut();
            assert_eq!(unsafe { cudaMalloc(&mut p, 4 * page) }, 0);
            let grown = own().unwrap();
            assert_eq!(unsafe { cudaFree(p) }, 0);
            assert!(grown >= now + 2 * page as u64, "an allocation of {} bytes left the count at {grown}", 4 * page);
        }
        // No per-process count: the device's free memory, which another
        // process can move.
        _ => {
            let after = free();
            println!("free device memory before the warm runs {free_before}, after {after}");
            assert!(after + page > free_before, "the device's free memory fell by {} bytes", free_before - after);
        }
    }
}

/// A model stored in F16 or BF16: MODEL is refused; EXACT computes in F32
/// from one F32 copy on the device, which FASTEST shares for all but its
/// GEMMs, and gives the vectors of the same values stored as F32; FASTEST
/// computes in F16, from the stored weights of an F16 model and from one
/// F16 copy of a BF16 model's, within the F16 bound.
#[test]
fn half_weights_compute_from_one_shared_copy_per_dtype() {
    let _t = turn();
    let Some(_) = cuda_device("half_weights_compute_from_one_shared_copy_per_dtype") else { return };
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
        let mut f = Fixture::model(&format!("cuda-half-{dtype}"));
        f.weights("weights/model.safetensors", &n);
        let l = f.load_on(cuda).unwrap();
        let e = Session::create(l.m, None).err().unwrap();
        assert_eq!((e.code, e.field), (UNSUPPORTED_OPTION, 3), "{dtype}: {e:?}");
        assert!(unsafe { model_converted_weights(l.m) }.is_none(), "a refused session makes no copy");
        let a = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
        let copy = unsafe { model_converted_weights(l.m) }.expect("the first F32 session made the copy");
        assert_eq!(copy.len(), 1, "{dtype}: the F32 copy alone");
        let b = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(b.info().compute_dtype, TURBO_DTYPE_F16);
        let copies = unsafe { model_converted_weights(l.m) }.unwrap();
        assert_eq!(copies[0], copy[0], "{dtype}: one F32 copy, shared");
        // An F16 model's GEMMs read the weights as loaded; a BF16 model's
        // read one F16 copy, made once.
        assert_eq!(copies.len(), if dtype == "F16" { 1 } else { 2 }, "{dtype}");
        let b2 = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
        assert_eq!(unsafe { model_converted_weights(l.m) }.unwrap(), copies, "{dtype}: shared");

        let mut g = Fixture::model(&format!("cuda-half-{dtype}-wide"));
        g.weights("weights/model.safetensors", &w);
        let lw = g.load_on(cuda).unwrap();
        let want = Session::create(lw.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT)))
            .unwrap()
            .embed(&TEXTS, None)
            .unwrap();
        assert!(unsafe { model_converted_weights(lw.m) }.is_none(), "F32 weights are used as loaded");
        assert_eq!(a.embed(&TEXTS, None).unwrap(), want, "{dtype}");
        let f16 = record::tolerance(TURBO_DTYPE_F16).unwrap();
        for s in [&b, &b2] {
            for (r, (g, w)) in s.embed(&TEXTS, None).unwrap().iter().zip(&want).enumerate() {
                let cos = cosine(g, w);
                assert!(cos >= f16.min_cosine, "{dtype} FASTEST row {r}: cosine {cos} with EXACT");
            }
        }

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
/// their own vectors: the context's stream and cuBLAS handle are shared
/// under its lock.
#[test]
fn sessions_on_one_context_run_from_many_threads() {
    let _t = turn();
    let Some(_) = cuda_device("sessions_on_one_context_run_from_many_threads") else { return };
    let l = on_cuda(&tiny_bundle());
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
    let Some(_) = cuda_device("an_output_dim_is_cut_then_normalized_on_the_device") else { return };
    let mut f = Fixture::new("cuda-output-dims", {
        let mut m = model_manifest();
        m["embed"]["output_dims"] = json!([4]);
        m
    });
    f.weights("weights/model.safetensors", &tiny_weights(0));
    let l = f.load_on(cuda).unwrap();
    let s = Session::create(l.m, Some(&session_desc(0, 0, TURBO_PRECISION_EXACT))).unwrap();
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
