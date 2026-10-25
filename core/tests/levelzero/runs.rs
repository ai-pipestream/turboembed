//! Runs: every option against the arithmetic and the CPU, the bytes a run moves and its allocations.

use super::*;

// ---- Runs ----------------------------------------------------------------------------------

/// Rows with padding on either side, a masked token inside a row and token
/// types, as tests/sessions.rs writes them for the CPU.
pub(crate) fn awkward_rows(dir: &std::path::Path) -> Tokens {
    let tok = Tok::create(dir).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let seq = rows.iter().map(Vec::len).max().unwrap() + 2;
    let mut t = Tokens::new(&[vec![0; seq], vec![0; seq], vec![0; seq]], 0);
    t.mask.fill(0);
    let mut types = vec![0; 3 * seq];
    for (r, row) in rows.iter().enumerate() {
        let start = if r == 1 { seq - row.len() } else { 0 };
        for (p, &id) in row.iter().enumerate() {
            t.ids[r * seq + start + p] = id;
            t.mask[r * seq + start + p] = 1;
            if r == 2 && p > 0 {
                types[r * seq + start + p] = 1;
            }
        }
    }
    t.mask[2 * seq + 1] = 0;
    t.types = Some(types);
    t
}

/// Every pooling, with and without normalization, against the f64
/// encoder of tests/common and against the CPU backend.
#[test]
fn every_option_matches_the_arithmetic_and_the_cpu() {
    let _t = turn();
    let Some(_) = gpu_device("every_option_matches_the_arithmetic_and_the_cpu") else { return };
    let dir = tiny_bundle();
    let plain = PlainBert::new(&dir);
    let (g, c) = (on_gpu(&dir), Loaded::load(&dir).unwrap());
    let (gs, cs) = (Session::create(g.m, None).unwrap(), Session::create(c.m, None).unwrap());
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

    // FASTEST: the linear layers on the matrix engines in F16, every
    // option, held to the F16 floor against the cpu.
    let fs = Session::create(g.m, Some(&session_desc(0, 0, TURBO_PRECISION_FASTEST))).unwrap();
    assert_eq!(fs.info().compute_dtype, TURBO_DTYPE_F16);
    let mut lowest = 1f64;
    for pooling in [TURBO_POOLING_MEAN, TURBO_POOLING_CLS, TURBO_POOLING_LAST] {
        for normalize in [TURBO_NORMALIZE_NONE, TURBO_NORMALIZE_L2] {
            let o = opts(|o| {
                o.pooling = pooling;
                o.normalize = normalize;
            });
            fs.write_tokens(&t.batch(), Some(&o)).unwrap();
            let got = fs.run().unwrap().rows();
            cs.write_tokens(&t.batch(), Some(&o)).unwrap();
            for (r, (a, b)) in got.iter().zip(cs.run().unwrap().rows()).enumerate() {
                let c = cosine(a, &b);
                lowest = lowest.min(c);
                assert!(c >= 0.999, "FASTEST pooling {pooling} normalize {normalize} row {r}: cosine {c} with the cpu");
            }
        }
    }
    println!("FASTEST: 1 - lowest cosine with the cpu {:.3e}", 1.0 - lowest);

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
/// whose USM pointer holds what turbo_result_read copies back, and the
/// counts are the bytes that crossed.
#[test]
fn a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed() {
    let _t = turn();
    let Some(_) = gpu_device("a_run_leaves_its_vectors_on_the_device_and_counts_what_crossed") else { return };
    let l = on_gpu(&tiny_bundle());
    let mi = l.info();
    let s = Session::create(l.m, None).unwrap();
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let live: u64 = TEXTS.iter().map(|t| tok.row(t, None).unwrap().len() as u64).sum();

    s.write_text(&TEXTS, None).unwrap();
    let r = s.run().unwrap();
    let i = r.info();
    assert_eq!((i.task, i.batch, i.dim), (TURBO_TASK_EMBED, 3, 32));
    assert_eq!((i.dtype, i.compute_dtype, i.placement), (TURBO_DTYPE_F32, TURBO_DTYPE_F32, TURBO_PLACE_DEVICE));
    assert_eq!(i.device, gpu(l.rt));
    assert_eq!(i.bytes, 3 * 32 * 4);
    // The packed rows the lookup kernel reads over the bus: ids, positions
    // and mask for each live token, and the row table.
    assert_eq!(i.h2d_bytes, 3 * live * 4 + 3 * 8, "{live} live tokens");
    assert_eq!(i.d2h_bytes, 0, "nothing came back yet");
    // The context's first run at these tokens builds oneDNN's primitives
    // for them, where the build has it.
    assert_eq!(i.device_allocs, 0);
    if !cfg!(feature = "levelzero-onednn") {
        assert_eq!(i.host_allocs, 0);
    }
    let (h, d, f, u) = (TURBO_STAGE_HOST, TURBO_STAGE_DEVICE, TURBO_STAGE_FUSED, TURBO_STAGE_UNUSED);
    assert_eq!(i.stage[..7], [h, f, d, d, d, d, u], "tokenize, upload, lookup, encode, pool, normalize, download");
    assert!(i.stage[7..].iter().all(|&s| s == u));
    assert_eq!(field(&i.backend), "levelzero");
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
    let h = buf.export(TURBO_HANDLE_ZE_USM).unwrap();
    assert_eq!(ze().kind(h.aux, h.handle).0, MEMORY_TYPE_DEVICE);
    assert_eq!(ze().read(&h, read.len()), read, "the device pointer holds the vectors, no copy made");
    drop(buf);
    drop(r);

    // Tokens the caller wrote, with types: three arrays cross; nothing
    // is tokenized, and an unnormalized vector is not normalized.
    let mut b = Tokens::new(&[vec![101, 7592, 102]], 0);
    b.types = Some(vec![0, 1, 1]);
    s.write_tokens(&b.batch(), Some(&opts(|o| o.normalize = TURBO_NORMALIZE_NONE))).unwrap();
    let i = s.run().unwrap().info();
    assert_eq!(i.h2d_bytes, 4 * 3 * 4 + 8, "types too");
    assert_eq!(i.stage[..7], [u, f, d, d, d, u, u]);
    assert_eq!((i.batch, i.d2h_bytes), (1, 0), "a run's count starts again");
}

/// Rows in memory the driver allocated (a PINNED or SHARED buffer's) go to
/// the device from where they are, and give the same vectors.
#[test]
fn rows_in_driver_memory_give_the_same_vectors() {
    let _t = turn();
    let Some(_) = gpu_device("rows_in_driver_memory_give_the_same_vectors") else { return };
    let l = on_gpu(&tiny_bundle());
    let s = Session::create(l.m, None).unwrap();
    let t = awkward_rows(&tiny_bundle());
    s.write_tokens(&t.batch(), None).unwrap();
    let want = s.run().unwrap().rows();

    let ctx = Ctx(l.ctx, false);
    let n = t.ids.len() as u64;
    // Each row's positions through its last live token.
    let packed: u64 =
        t.mask.chunks(t.seq as usize).map(|m| m.iter().rposition(|&v| v != 0).map_or(0, |p| p + 1) as u64).sum();
    for placement in [TURBO_PLACE_PINNED, TURBO_PLACE_SHARED] {
        let bufs: Vec<Buf> = (0..3).map(|_| ctx.alloc(placement, n).unwrap()).collect();
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
        assert_eq!(r.rows(), want, "placement {placement}");
        assert_eq!(r.info().h2d_bytes, 4 * packed * 4 + 3 * 8);
    }
}

/// A run allocates nothing on the device, cold or warm, at each precision
/// and at token counts that land in different bins of oneDNN's (where the
/// build has it): this binary's allocator counts nothing on the running
/// thread and the backend's own count does not move. On the host the
/// result says 0, but for oneDNN's primitives the first time the context
/// runs a bin; a second run at the same tokens says 0.
#[test]
fn a_run_allocates_nothing() {
    let _t = turn();
    let Some(_) = gpu_device("a_run_allocates_nothing") else { return };
    let l = on_gpu(&tiny_bundle());
    let tok = Tok::create(&tiny_bundle()).unwrap();
    let rows: Vec<Vec<i32>> = TEXTS.iter().map(|t| tok.row(t, None).unwrap()).collect();
    let small = Tokens::new(&rows, 0);
    let long = Tokens::new(&vec![(0..64).map(|i| 1000 + i).collect(); 64], 0);
    // 9 to 2580 tokens: past the few-token kernels, in bins of 16 and,
    // past 2048, of 32.
    let odd: Vec<Tokens> = [(1, 9), (3, 17), (5, 41), (13, 50), (27, 63), (43, 60)]
        .iter()
        .map(|&(batch, len)| Tokens::new(&vec![(0..len).map(|i| 1000 + i).collect(); batch], 0))
        .collect();
    let mut dst = vec![0f32; 64 * 32];
    for precision in [TURBO_PRECISION_MODEL, TURBO_PRECISION_EXACT, TURBO_PRECISION_FASTEST] {
        let s = Session::create(l.m, Some(&session_desc(0, 0, precision))).unwrap();
        let mut run = |t: &Tokens| {
            let batch = t.batch();
            s.write_tokens(&batch, None).unwrap();
            let mut r = ptr::null_mut();
            let before = COUNT.with(Cell::get);
            assert_eq!(unsafe { turbo_session_run(s.0, &mut r, null_err()) }, 0);
            let counted = COUNT.with(Cell::get) - before;
            let r = Outcome(r);
            let i = r.info();
            let rc = unsafe {
                turbo_result_read(r.0, dst.as_mut_ptr() as *mut _, (dst.len() * 4) as u64, ptr::null_mut(), null_err())
            };
            assert_eq!(rc, 0);
            (i.host_allocs, i.device_allocs, counted)
        };
        let before = turbo::levelzero::allocations();
        for (i, t) in [&small, &long].into_iter().chain(&odd).enumerate() {
            let (host, device, counted) = run(t);
            assert_eq!(device, 0, "precision {precision} run {i}: the result's device count");
            assert_eq!(counted, 0, "precision {precision} run {i}: what this thread's allocator counted");
            if !cfg!(feature = "levelzero-onednn") {
                assert_eq!(host, 0, "precision {precision} run {i}: the result's host count");
            }
            let again = run(t);
            assert_eq!(
                again,
                (0, 0, 0),
                "precision {precision} run {i} again: the result's counts and the allocator's"
            );
        }
        assert_eq!(turbo::levelzero::allocations(), before, "precision {precision}: the backend allocated nothing");
    }
}
